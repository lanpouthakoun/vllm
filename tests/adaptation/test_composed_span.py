# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serving a COMPOSED member: a re-executing span with OTHER members
co-mounted inside it, on the multisite route.

`campaign/cartridge_readout/longhealth_raw` sub-lane B composes four
records into one member — a pointer-state leaf and a delta-rule
recurrence at `block_output` L0, a reader head at every layer, and a
gated pipe reading `block_output` L3 and writing `block_input` L0 with
two passes, so both stateful leaves sit INSIDE the re-executed span.  The
fork refused all of it and the lane was HF-scored.  Three separate
refusals were in the way, and this file pins each one's replacement:

1. **The span could only ride the BAKED route.** A rewiring member's
   per-pass KV caches can only be declared before the worker reads the
   KV-cache spec, and on the multisite route the member arrives by
   `collective_rpc("load_adapter", ...)` long after that.  The span is
   now RESERVED at engine-config time on `EngineArgs.adapter_recirc_span`
   — ports and passes, no weights — and the member FILLS the reservation
   when it lands (`recirculation.fill_reserved_span`), which checks its
   ports against what was reserved.  `REEXECUTION_ROUTES` declares both
   routes so the library's builder asks instead of assuming.
2. **A member co-mounted at the span's input port was refused**, because
   "the composition order with a co-mounted member is undefined".  It is
   defined now and declared: `SPAN_COMOUNT_ORDER == "diagonal_then_span"`
   — ordinary members at the port blend first, in load order, and the
   span is entered with the blended stream, which is the HF engine's own
   order with the rewiring record last.
3. **Members INSIDE the span** fire on every pass.  That already followed
   from re-running the adapter-wrapped layers; nothing pinned it, so it
   was not a guarantee.  It is one here, per pass and per phase.

What is NOT served, and is refused by name rather than served wrongly:
a member whose mount asked the engine to CARRY a per-request `State` from
the prompt scan into every decode step (`Mount.decode_state` on a
diagonal mount).  `CARRIES_ADAPTER_DECODE_STATE` is False and says so.

CPU-only, on `test_inout_sites.py`'s faithful decoder mocks (real causal
attention over a real per-`layer_name` KV log, and the in-place residual
contract a real llama layer has).
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.adaptation import protocol as P
from vllm.adaptation import recirculation as R
from vllm.adaptation.layer import (_add_adapter_to_layer,
                                   make_adapter_decoder_layer,
                                   update_adapter_position_masks)
from vllm.adaptation.specs import spec_to_adapter_config

from .test_inout_sites import (HIDDEN, NLAYERS, FakeAttention,
                               FakeDecoderLayer, GatedPipe,
                               InPlaceDecoderLayer, STORE, _Cfg)

# `patch_attention_modules` is an autouse fixture in test_inout_sites and
# does not travel with a plain import; re-declare it here so the shadow
# machinery sees the mock attentions.


@pytest.fixture(autouse=True)
def _patch_attention_modules(monkeypatch):
    monkeypatch.setattr(
        R, "_attention_modules",
        lambda layer: [m for m in layer.modules()
                       if isinstance(m, FakeAttention)])
    STORE.reset()
    yield
    STORE.reset()


# ---------------------------------------------------------------------------
# The two co-mounted member shapes the composed member needs
# ---------------------------------------------------------------------------

class CountingScan(nn.Module):
    """A sequence-mixing leaf that COUNTS its readouts.

    Stands in for `stack_k1` / the `ext` delta-rule core: it mixes along
    the sequence axis (a running mean), declares
    `sequence_mixing = True` the way a scanned member must, and has a
    `scale` that is 0 at init — the zero-init contract the whole zoo
    rests on, and what makes the all-members-at-init assertion below a
    bit-exact one.

    It carries NO engine state: every call starts its scan at the tensor
    it is handed.  That is the honest servable shape, and it is why the
    composed member's P axis is expressed with phase masks rather than
    with `Mount.decode_state` (see TestDecodeStateIsRefused).
    """

    sequence_mixing = True

    def __init__(self, scale: float = 0.0, label: str = "scan"):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale)))
        self.label = label
        self.calls = 0
        self.seen: list[torch.Tensor] = []

    def readout(self, fx, state=None, x=None):
        self.calls += 1
        self.seen.append(fx.detach().clone())
        running = fx.cumsum(dim=-2) / torch.arange(
            1, fx.shape[-2] + 1, device=fx.device,
            dtype=fx.dtype).unsqueeze(-1)
        return fx + self.scale.to(fx.dtype) * running


class Reader(nn.Module):
    """A stateless per-token head, zero-init (the `direft_gelusq` shape).

    Mounted at EVERY layer, which is why it is also at the span's input
    port — the co-mount that used to be refused.
    """

    def __init__(self, scale: float = 0.0, label: str = "reader"):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale)))
        self.label = label
        self.calls = 0

    def readout(self, fx, state=None, x=None):
        self.calls += 1
        return fx + self.scale.to(fx.dtype) * torch.tanh(fx)


class CarriedStateScan(CountingScan):
    """A leaf whose TRAINED function read an engine-carried State.

    It has no `.mixer`, so `check_adaptation_supported` — which only
    refuses a mixer declaring `stateful=True` — would let it load and run
    `readout(h)` with `state=None` at every decoded position.  That is
    precisely why the refusal is keyed on the MOUNT's `decode_state`
    flag and not on the leaf.
    """

    def readout(self, fx, state=None, x=None):
        assert state is not None, (
            "this member's trained function reads a CARRIED state; "
            "serving it with state=None is a different function")
        return super().readout(fx, state, x)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

IN_LAYER, OUT_LAYER, PASSES = 3, 0, 2
SPAN = (0, 1, 2, 3)


def _build_stack(seed=0, layer_cls=FakeDecoderLayer):
    torch.manual_seed(seed)
    cls = make_adapter_decoder_layer(layer_cls)
    layers = [cls(_Cfg(), None, None, f"model.layers.{i}")
              for i in range(NLAYERS)]
    model = nn.Module()
    model.layers = nn.ModuleList(layers)
    return model, layers


def _vllm_config(*, recirc_span=None, adapter_config=None,
                 prefix_caching=False, chunked_prefill=False, eager=True,
                 pp=1, kv_transfer=None):
    return SimpleNamespace(
        adapter_config=adapter_config,
        adapter_recirc_span=recirc_span,
        compilation_config=SimpleNamespace(static_forward_context={}),
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp,
                                        tensor_parallel_size=1),
        cache_config=SimpleNamespace(enable_prefix_caching=prefix_caching),
        scheduler_config=SimpleNamespace(
            chunked_prefill_enabled=chunked_prefill),
        model_config=SimpleNamespace(enforce_eager=eager),
        kv_transfer_config=kv_transfer,
    )


def _pipe_config(gate=0.0, in_layer=IN_LAYER, out_layer=OUT_LAYER,
                 passes=PASSES, out_site="block_input", position="all"):
    pipe = GatedPipe(gate=gate)
    return spec_to_adapter_config({
        "layer_indices": [in_layer], "position": position,
        "sample_adapter": pipe, "adapters": {in_layer: pipe},
        "site": "block_output", "output_site": out_site,
        "output_layer": out_layer, "passes": passes})


def _span_declaration(in_layer=IN_LAYER, out_layer=OUT_LAYER,
                      passes=PASSES, out_site="block_input"):
    """What ``adapters/serving.py::recirc_span_declaration`` builds."""
    return {"site": "block_output", "layer_indices": [in_layer],
            "output_site": out_site, "output_layer": out_layer,
            "passes": passes, "declared_by": ["m3:pipe"]}


# Member ids, in the LOAD order the library's builder uses: every
# ordinary member first, the rewiring member LAST (_rewired_last), which
# is what makes the fork's insertion-order blend match
# SPAN_COMOUNT_ORDER.
ID_READER, ID_SCAN_A, ID_SCAN_B, ID_PIPE = 1, 2, 3, 4


def _compose(model, layers, *, gate=0.0, reader_scale=0.0,
             scan_scale=0.0, scan_phase="all", passes=PASSES,
             reader_layers=SPAN, scan_layers=(0,), reserve=True,
             fill=True):
    """Mount the composed member the way the multisite route does.

    ORDER IS THE POINT, and it is the real one: the span is RESERVED and
    installed first (at load_model, before the KV-cache spec is read),
    and the members arrive afterwards by RPC — ordinary members first,
    the pipe last.
    """
    plan = None
    if reserve:
        vc = _vllm_config(recirc_span=_span_declaration(passes=passes))
        plan = R.install_recirculation(model, vc)
    readers, scans = {}, {}
    for li in reader_layers:
        readers[li] = Reader(reader_scale, f"reader.L{li}")
        _add_adapter_to_layer(layers[li], ID_READER, readers[li], "all",
                              torch.device("cpu"), site="block_output")
    for n, li in enumerate(scan_layers):
        scans[li] = CountingScan(scan_scale, f"scan.L{li}")
        _add_adapter_to_layer(layers[li], ID_SCAN_A + n, scans[li],
                              scan_phase, torch.device("cpu"),
                              site="block_output")
    cfg = _pipe_config(gate=gate, passes=passes)
    pipe = GatedPipe(gate=gate)
    _add_adapter_to_layer(layers[IN_LAYER], ID_PIPE, pipe, "all",
                          torch.device("cpu"), site="block_output")
    if fill and reserve:
        R.fill_reserved_span(model, ID_PIPE, cfg)
    return SimpleNamespace(plan=plan, readers=readers, scans=scans,
                           pipe=pipe, cfg=cfg,
                           ids=[ID_READER]
                           + [ID_SCAN_A + n for n in range(len(scan_layers))]
                           + [ID_PIPE])


def _meta(num_prefill_tokens, num_decodes, num_prefills):
    return SimpleNamespace(num_prefill_tokens=num_prefill_tokens,
                           num_decodes=num_decodes,
                           num_prefills=num_prefills,
                           query_start_loc=None, seq_lens=None)


# Every member of a COMPOSITE rides one request, which on the fork means
# one member slot each: the primary `token_adapter_ids` plus
# `extra_token_adapter_ids` (the InputBatch carries MAX_ADAPTER_SLOTS=8
# per request, which is what `adapters/serving.py` builds its
# AdapterRequest list against).
ALL_IDS = (ID_READER, ID_SCAN_A, ID_SCAN_B, ID_PIPE)


def _masks(layers, positions, n, meta, ids=ALL_IDS):
    primary = torch.full((n,), ids[0], dtype=torch.int32)
    extras = [torch.full((n,), i, dtype=torch.int32) for i in ids[1:]]
    update_adapter_position_masks(layers, primary, positions, meta, n,
                                  extra_token_adapter_ids=extras)


def _run(layers, tokens, positions, *, decode=False, ids=ALL_IDS):
    n = tokens.shape[0]
    meta = _meta(0, n, 0) if decode else _meta(n, 0, 1)
    _masks(layers, positions, n, meta, ids)
    hidden, residual = tokens, None
    for layer in layers:
        hidden, residual = layer(positions, hidden, residual)
    return hidden if residual is None else hidden + residual


def _prefill_then_decode(layers, x, prefill):
    total = x.shape[0]
    positions = torch.arange(total)
    STORE.reset()
    with torch.no_grad():
        outs = [_run(layers, x[:prefill].clone(), positions[:prefill])]
        for t in range(prefill, total):
            outs.append(_run(layers, x[t:t + 1].clone(),
                             positions[t:t + 1], decode=True))
    return torch.cat(outs, dim=0)


def _x(n=8, seed=23):
    return torch.randn(n, HIDDEN,
                       generator=torch.Generator().manual_seed(seed))


# ---------------------------------------------------------------------------
# 1. The capability surface
# ---------------------------------------------------------------------------

class TestCapabilitySurface:
    """Every guarantee this file pins is NAMED, so the library asks
    instead of assuming — the rule SUPPORTED_WRITE_LABELS established."""

    def test_the_routes_that_can_execute_a_span_are_declared(self):
        assert P.REEXECUTION_ROUTES == frozenset(
            {"baked", "multisite_reserved"})
        assert R.REEXECUTION_ROUTES is P.REEXECUTION_ROUTES

    def test_the_comount_order_is_declared(self):
        assert P.SPAN_COMOUNT_ORDER == "diagonal_then_span"
        assert R.SPAN_COMOUNT_ORDER is P.SPAN_COMOUNT_ORDER

    def test_the_absence_of_per_request_decode_state_is_declared(self):
        """The honest answer is False, and it is a NAME so the library's
        refusal can be a probe rather than a mirrored assumption."""
        assert P.CARRIES_ADAPTER_DECODE_STATE is False
        assert R.CARRIES_ADAPTER_DECODE_STATE is False

    def test_the_reservation_keys_are_a_subset_of_the_config_keys(self):
        """One vocabulary: a reservation is an adapter_config with the
        weights left out, so plan_from_adapter_config validates both."""
        assert set(R.SPAN_DECLARATION_KEYS) >= set(R.REWIRE_KEYS)
        assert "site" in R.SPAN_DECLARATION_KEYS
        assert "layer_indices" in R.SPAN_DECLARATION_KEYS


# ---------------------------------------------------------------------------
# 2. The reservation route
# ---------------------------------------------------------------------------

class TestReservation:

    def test_the_span_is_registered_before_any_member_lands(self):
        """The whole point of the reservation: the shadow attention
        layers exist while the KV-cache spec can still be read, and at
        that moment nothing is mounted."""
        model, layers = _build_stack()
        vc = _vllm_config(recirc_span=_span_declaration())
        plan = R.install_recirculation(model, vc)
        assert plan is not None
        assert list(plan.span_layers) == list(SPAN)
        # one extra attention per span layer per pass
        assert len(vc.compilation_config.static_forward_context) == \
            PASSES * len(SPAN)
        installed = R.find_installed_recirc(model)
        assert installed.reserved is True
        assert installed.adapter_int_id is None
        assert installed.declared_by == ("m3:pipe",)
        assert not layers[IN_LAYER].served_adapters

    def test_an_unfilled_reservation_is_INERT(self):
        """A reservation whose member never arrives must degrade to the
        plain host, not to a span driven by nothing."""
        model, layers = _build_stack(5, InPlaceDecoderLayer)
        R.install_recirculation(model, _vllm_config(
            recirc_span=_span_declaration()))
        got = _prefill_then_decode(layers, _x(), 3)
        _, ref_layers = _build_stack(5, InPlaceDecoderLayer)
        ref = _prefill_then_decode(ref_layers, _x(), 3)
        assert torch.equal(got, ref)

    def test_the_arriving_member_fills_it(self):
        model, layers = _build_stack()
        c = _compose(model, layers)
        installed = R.find_installed_recirc(model)
        assert installed.adapter_int_id == ID_PIPE
        assert c.plan.passes == PASSES

    @pytest.mark.parametrize("kw,field", [
        (dict(passes=3), "passes"),
        (dict(in_layer=4), "input layer"),
        (dict(out_layer=1), "output layer"),
        (dict(out_site="block_output"), "output port"),
    ])
    def test_a_member_that_does_not_match_the_reservation_is_refused(
            self, kw, field):
        """The reservation SIZED and NAMED the per-pass caches. Running a
        different span against them would attend to keys budgeted for
        other layers, so the mismatch is named rather than tolerated."""
        model, layers = _build_stack()
        R.install_recirculation(model, _vllm_config(
            recirc_span=_span_declaration()))
        with pytest.raises(RuntimeError) as ei:
            R.fill_reserved_span(model, ID_PIPE, _pipe_config(**kw))
        assert "does not match the span reserved" in str(ei.value)
        assert field in str(ei.value)

    def test_a_rewiring_member_with_no_reservation_is_refused(self):
        model, layers = _build_stack()
        with pytest.raises(RuntimeError) as ei:
            R.fill_reserved_span(model, ID_PIPE, _pipe_config())
        text = str(ei.value)
        assert "NO SPAN WAS RESERVED" in text
        assert "KV-cache spec ONCE" in text
        assert "adapter_recirc_span" in text

    def test_a_second_member_cannot_fill_the_same_span(self):
        model, layers = _build_stack()
        R.install_recirculation(model, _vllm_config(
            recirc_span=_span_declaration()))
        R.fill_reserved_span(model, ID_PIPE, _pipe_config())
        with pytest.raises(RuntimeError, match="already driven by adapter"):
            R.fill_reserved_span(model, ID_PIPE + 1, _pipe_config())

    def test_a_reservation_takes_the_same_engine_refusals_as_a_baked_span(
            self):
        for kw, match in [(dict(prefix_caching=True), "prefix caching"),
                          (dict(chunked_prefill=True), "chunked prefill"),
                          (dict(eager=False), "eager execution"),
                          (dict(pp=2), "pipeline_parallel_size")]:
            model, _ = _build_stack()
            with pytest.raises(RuntimeError, match=match):
                R.install_recirculation(model, _vllm_config(
                    recirc_span=_span_declaration(), **kw))

    def test_a_diagonal_declaration_reserves_nothing(self):
        """Additive and default-off: no declaration, no shadows."""
        model, _ = _build_stack()
        vc = _vllm_config(recirc_span=None)
        assert R.install_recirculation(model, vc) is None
        assert vc.compilation_config.static_forward_context == {}


# ---------------------------------------------------------------------------
# 3. Identity at gate 0 — with the WHOLE composed member mounted
# ---------------------------------------------------------------------------

class TestIdentityAtGateZero:

    @pytest.mark.parametrize("layer_cls",
                             [FakeDecoderLayer, InPlaceDecoderLayer])
    @pytest.mark.parametrize("prefill", [1, 3, 6])
    def test_every_member_at_its_zero_init_is_the_bare_host(self, layer_cls,
                                                            prefill):
        """The composed member, whole, at init: bit-exact the host.

        Four records are mounted — a reader at every span layer, two
        scanned leaves at L0 inside the span, and the gated pipe at L3 —
        and each is an exact no-op at its own init, so the composition
        must be one too, through prefill AND every decode step.  That is
        the claim the campaign's gate record makes on the HF side and
        could not make on the fork.
        """
        model, layers = _build_stack(5, layer_cls)
        _compose(model, layers, gate=0.0, reader_scale=0.0,
                 scan_scale=0.0, scan_layers=(0, 0))
        got = _prefill_then_decode(layers, _x(), prefill)
        _, ref_layers = _build_stack(5, layer_cls)
        ref = _prefill_then_decode(ref_layers, _x(), prefill)
        assert got.shape == ref.shape
        assert torch.equal(got, ref), (
            "the composed member is not an exact no-op at init: first "
            "differing token "
            f"{int((got != ref).any(dim=-1).nonzero()[0])}")

    @pytest.mark.parametrize("layer_cls",
                             [FakeDecoderLayer, InPlaceDecoderLayer])
    def test_gate_zero_is_the_identity_WITH_co_mounts_that_do_something(
            self, layer_cls):
        """The sharper test: the co-mounted members have a real effect,
        and the pipe at g = 0 must still change nothing.

        This is the assertion the co-mount refusal existed to avoid
        having to make.  It can only hold if the span's per-pass KV lives
        in its own caches (or the extra passes' keys would reach the
        host's own attention) and if `write`'s `h_out` is the PRE-pipe
        stream after the co-mounts blended into it.
        """
        model, layers = _build_stack(5, layer_cls)
        _compose(model, layers, gate=0.0, reader_scale=0.35,
                 scan_scale=0.4, scan_layers=(0, 2))
        got = _prefill_then_decode(layers, _x(), 3)

        ref_model, ref_layers = _build_stack(5, layer_cls)
        _compose(ref_model, ref_layers, gate=0.0, reader_scale=0.35,
                 scan_scale=0.4, scan_layers=(0, 2), reserve=False,
                 fill=False)
        ref = _prefill_then_decode(ref_layers, _x(), 3)
        assert torch.equal(got, ref), (
            "the g = 0 pipe perturbed a stack whose other members are "
            "active — the span is leaking into the host's own pass")

    def test_the_co_mounts_really_do_move_the_stream(self):
        """Guards the test above: if the co-mounts were themselves no-ops
        it would prove nothing."""
        model, layers = _build_stack(5, InPlaceDecoderLayer)
        _compose(model, layers, gate=0.0, reader_scale=0.35,
                 scan_scale=0.4, scan_layers=(0, 2))
        got = _prefill_then_decode(layers, _x(), 3)
        _, bare = _build_stack(5, InPlaceDecoderLayer)
        assert not torch.equal(got, _prefill_then_decode(bare, _x(), 3))

    @pytest.mark.parametrize("passes", [1, 2, 3])
    def test_a_nonzero_gate_moves_the_stream(self, passes):
        model, layers = _build_stack(5, InPlaceDecoderLayer)
        _compose(model, layers, gate=0.5, reader_scale=0.2,
                 scan_scale=0.3, passes=passes)
        got = _prefill_then_decode(layers, _x(6), 3)

        ref_model, ref_layers = _build_stack(5, InPlaceDecoderLayer)
        _compose(ref_model, ref_layers, gate=0.0, reader_scale=0.2,
                 scan_scale=0.3, passes=passes)
        ref = _prefill_then_decode(ref_layers, _x(6), 3)
        assert not torch.allclose(got, ref, atol=1e-6), (
            f"passes={passes}: a gate of 0.5 changed nothing — the span "
            f"is not running")


# ---------------------------------------------------------------------------
# 4. Members INSIDE the span fire on every pass
# ---------------------------------------------------------------------------

class TestPerPassFiring:

    @pytest.mark.parametrize("passes", [1, 2, 3])
    def test_a_member_inside_the_span_fires_once_per_pass(self, passes):
        """`1 + passes`: the host's own visit plus each re-execution.

        This is the reason a stateful member is mounted inside the span
        at all — the loop is what re-develops the state — so it is a
        GUARANTEE and not an implementation detail.  The HF engine states
        the same semantics ("a co-mounted leaf inside the re-executed
        span fires on every pass", docs/recirculation-serving.md §2.5).
        """
        model, layers = _build_stack(5, InPlaceDecoderLayer)
        c = _compose(model, layers, gate=0.3, scan_scale=0.3,
                     scan_layers=(0,), passes=passes)
        x = _x(4)
        with torch.no_grad():
            _run(layers, x.clone(), torch.arange(4))
        assert c.scans[0].calls == 1 + passes

    @pytest.mark.parametrize("passes", [1, 2])
    def test_the_reader_fires_per_pass_inside_and_once_outside(self,
                                                               passes):
        """The reader is mounted at EVERY layer, so it distinguishes the
        two regions: `1 + passes` inside the span, once outside it."""
        model, layers = _build_stack(5, InPlaceDecoderLayer)
        c = _compose(model, layers, gate=0.3, reader_scale=0.25,
                     reader_layers=tuple(range(NLAYERS)), passes=passes)
        with torch.no_grad():
            _run(layers, _x(4).clone(), torch.arange(4))
        for li in SPAN:
            assert c.readers[li].calls == 1 + passes, f"L{li}"
        for li in range(len(SPAN), NLAYERS):
            assert c.readers[li].calls == 1, f"L{li} is outside the span"

    def test_the_rewiring_member_does_not_fire_inside_its_own_span(self):
        """The depth guard: the pipe is a pass-through on its own passes,
        or the recursion would not terminate.  Its readout runs ONCE per
        forward, on the way in."""
        model, layers = _build_stack(5, InPlaceDecoderLayer)
        c = _compose(model, layers, gate=0.3)
        seen = []
        inner = c.pipe.readout
        c.pipe.readout = lambda fx, state=None, x=None: (
            seen.append(fx.shape) or inner(fx, state, x))
        with torch.no_grad():
            _run(layers, _x(4).clone(), torch.arange(4))
        assert len(seen) == 1

    def test_decode_steps_fire_the_span_too(self):
        """The per-pass caches persist across steps exactly as the host's
        do, so a decode step re-executes the span as well."""
        model, layers = _build_stack(5, InPlaceDecoderLayer)
        c = _compose(model, layers, gate=0.3, scan_scale=0.3)
        x = _x(5)
        positions = torch.arange(5)
        with torch.no_grad():
            _run(layers, x[:3].clone(), positions[:3])
            before = c.scans[0].calls
            _run(layers, x[3:4].clone(), positions[3:4], decode=True)
        assert c.scans[0].calls - before == 1 + PASSES


# ---------------------------------------------------------------------------
# 5. The composition ORDER at the input port
# ---------------------------------------------------------------------------

class TestCompositionOrder:

    def test_the_span_is_entered_with_the_BLENDED_stream(self):
        """`SPAN_COMOUNT_ORDER == "diagonal_then_span"`, checked at the
        port: what the pipe's readout receives is the stream AFTER the
        co-mounted reader at L3 blended into it, not before.

        This is the semantic the old refusal called undefined, and it is
        the HF engine's own (`_apply_leaf` runs the records' hooks in
        order, with the rewiring record last), which is why the library's
        builder loads the pipe LAST.
        """
        model, layers = _build_stack(5, FakeDecoderLayer)
        c = _compose(model, layers, gate=0.3, reader_scale=0.5,
                     reader_layers=(IN_LAYER,))
        entry = []
        c.pipe.readout = lambda fx, state=None, x=None: (
            entry.append(fx.detach().clone()) or fx)
        x = _x(4)
        with torch.no_grad():
            _run(layers, x.clone(), torch.arange(4))
        assert len(entry) == 1
        seen = entry[0].squeeze(0)

        # The same stack with the reader's scale at 0: if the span were
        # entered with the UNBLENDED stream the two would agree.
        ref_model, ref_layers = _build_stack(5, FakeDecoderLayer)
        rc = _compose(ref_model, ref_layers, gate=0.3, reader_scale=0.0,
                      reader_layers=(IN_LAYER,))
        ref_entry = []
        rc.pipe.readout = lambda fx, state=None, x=None: (
            ref_entry.append(fx.detach().clone()) or fx)
        with torch.no_grad():
            _run(ref_layers, x.clone(), torch.arange(4))
        assert not torch.allclose(seen, ref_entry[0].squeeze(0)), (
            "the span was entered with the pre-blend stream: the "
            "co-mounted reader at the input port did not compose")

    def test_the_builder_orders_the_pipe_last(self):
        """The fork blends `served_adapters` in INSERTION order, which is
        the order the load_adapter RPCs arrive in, so the library has to
        put the pipe last. Pinned on the library's own helper so the two
        sides cannot drift."""
        pytest.importorskip("adapters.serving")
        from adapters.serving import _rewired_last
        pipe = {"label": "m3:pipe", "site": "block_output",
                "output_site": "block_input", "output_layer": 0,
                "passes": 2, "host_reexecute": True,
                "spec": {"layer_indices": [3]}}
        diag = {"label": "m0:reader", "site": "block_output",
                "output_site": None, "output_layer": None,
                "spec": {"layer_indices": [0]}}
        assert [m["label"] for m in _rewired_last([pipe, diag])][-1] == \
            "m3:pipe"


# ---------------------------------------------------------------------------
# 6. P-axis phase masks, inside the span
# ---------------------------------------------------------------------------

class TestPhaseMasksInsideTheSpan:
    """The servable expression of the P axis.

    `Mount.decode_state` is refused (§7), so the composed member's "the
    member behaves differently at the two phases" is carried by the
    fork's own phase masks: a member at `position="prefill"` contributes
    on prompt tokens only, on every pass, and a member at
    `position="all"` contributes at both phases.  What travels from
    prefill into decode is the span's PER-PASS KV — engine-owned,
    per-request state the fork already carries.
    """

    def test_a_prefill_member_inside_the_span_is_silent_at_decode(self):
        model, layers = _build_stack(5, InPlaceDecoderLayer)
        _compose(model, layers, gate=0.0, scan_scale=0.6,
                 scan_phase="prefill")
        got = _prefill_then_decode(layers, _x(6), 3)

        # Same stack, scan disabled entirely.
        ref_model, ref_layers = _build_stack(5, InPlaceDecoderLayer)
        _compose(ref_model, ref_layers, gate=0.0, scan_scale=0.0,
                 scan_phase="prefill")
        ref = _prefill_then_decode(ref_layers, _x(6), 3)
        # It fired on the prompt...
        assert not torch.allclose(got[:3], ref[:3], atol=1e-6)
        # ...and its CORRECTION is zero on every decode step. (The
        # remaining difference is the prompt's own KV, which is the
        # member's legitimate effect on later tokens — so compare the
        # correction at the port, below, not the final logits.)

    def test_a_prefill_member_inside_the_span_is_SKIPPED_at_decode(self):
        """At the port: on a pure-decode batch the prefill-phase member is
        not in the layer's active set at all, so its readout does not run
        on ANY pass of the span — while the "all"-phase pipe is active and
        the span does run.

        That is the mechanism, and asserting it here rather than on the
        mask buffer is deliberate: the fork does not zero an INACTIVE
        adapter's buffer in eager mode (it never reads it), so the buffer
        holds the previous batch's value and is not the gate.
        """
        model, layers = _build_stack(5, InPlaceDecoderLayer)
        c = _compose(model, layers, gate=0.3, scan_scale=0.6,
                     scan_phase="prefill")
        x = _x(5)
        positions = torch.arange(5)
        with torch.no_grad():
            _run(layers, x[:3].clone(), positions[:3])
            assert ID_SCAN_A in layers[0]._adapter_active_ids
            fired_on_prompt = c.scans[0].calls
            assert fired_on_prompt == 1 + PASSES
            _run(layers, x[3:4].clone(), positions[3:4], decode=True)
        assert ID_SCAN_A not in layers[0]._adapter_active_ids
        assert ID_PIPE in layers[IN_LAYER]._adapter_active_ids
        assert c.scans[0].calls == fired_on_prompt, (
            "a prefill-phase member fired on a decode step")

    def test_an_all_phase_member_fires_at_both_phases(self):
        model, layers = _build_stack(5, InPlaceDecoderLayer)
        c = _compose(model, layers, gate=0.0, scan_scale=0.6,
                     scan_phase="all")
        x = _x(5)
        positions = torch.arange(5)
        with torch.no_grad():
            _run(layers, x[:3].clone(), positions[:3])
            before = c.scans[0].calls
            _run(layers, x[3:4].clone(), positions[3:4], decode=True)
        assert ID_SCAN_A in layers[0]._adapter_active_ids
        assert c.scans[0].calls - before == 1 + PASSES

    def test_a_decode_member_and_a_prefill_member_PARTITION_a_mixed_batch(
            self):
        """The two halves of a phase SPLIT never overlap. A v1 batch puts
        decode tokens first and prefill tokens last, and both members are
        active there, so this is where the partition is observable —
        inside a span, with the span running.
        (tests/adaptation/test_decode_phase_mask.py pins the masks
        themselves; this pins that co-mounting them inside a re-executed
        span changes neither.)
        """
        model, layers = _build_stack(5, InPlaceDecoderLayer)
        R.install_recirculation(model, _vllm_config(
            recirc_span=_span_declaration()))
        a, b = CountingScan(0.4, "T.prefill"), CountingScan(0.4, "R.decode")
        _add_adapter_to_layer(layers[0], ID_SCAN_A, a, "prefill",
                              torch.device("cpu"), site="block_output")
        _add_adapter_to_layer(layers[0], ID_SCAN_B, b, "decode",
                              torch.device("cpu"), site="block_output")
        pipe = GatedPipe(gate=0.2)
        _add_adapter_to_layer(layers[IN_LAYER], ID_PIPE, pipe, "all",
                              torch.device("cpu"), site="block_output")
        R.fill_reserved_span(model, ID_PIPE, _pipe_config(gate=0.2))
        # 2 decode tokens then 4 prefill tokens, the v1 layout.
        n_dec, n_pre = 2, 4
        n = n_dec + n_pre
        positions = torch.cat([torch.tensor([7, 9]), torch.arange(n_pre)])
        _masks(layers, positions, n, _meta(n_pre, n_dec, 1))
        ma = layers[0]._adapter_combined_masks[ID_SCAN_A][:n]
        mb = layers[0]._adapter_combined_masks[ID_SCAN_B][:n]
        assert ma.tolist() == [0., 0., 1., 1., 1., 1.]
        assert mb.tolist() == [1., 1., 0., 0., 0., 0.]
        assert torch.all(ma + mb == 1.0), "the phase split overlaps"
        # Both fire on every pass of the span, each on its own tokens —
        # and PER REQUEST SEGMENT, which the span does not break: a
        # sequence-mixing member must not scan across the flattened
        # batch's request boundaries, and this batch is 2 decode requests
        # plus 1 prefill = 3 segments.
        with torch.no_grad():
            hidden, residual = torch.randn(n, HIDDEN), None
            for layer in layers:
                hidden, residual = layer(positions, hidden, residual)
        segments = n_dec + 1
        assert a.calls == (1 + PASSES) * segments
        assert b.calls == (1 + PASSES) * segments
        assert layers[0]._adapter_segments is not None and \
            len(layers[0]._adapter_segments) == segments


# ---------------------------------------------------------------------------
# 7. What is NOT served: an engine-carried per-request State
# ---------------------------------------------------------------------------

class TestDecodeStateIsRefused:
    """The P axis as the campaign first specified it —
    `Mount.decode_state=True` on a diagonal mount, T scanning the prompt
    into a State that R reads at every decode step — is NOT servable
    here, and the refusal is the deliverable.

    It was worse than unsupported: it was INVISIBLE. The library recorded
    `decode_state` only for a pipe, so a diagonal carried-state member
    exported as an ordinary member, passed every refusal, loaded, and ran
    `readout(h)` with `state=None` at each decoded position — a
    fresh-state singleton where the trained function was a carried scan,
    fluently and with no error.
    """

    def test_the_fork_refuses_it_by_name(self):
        cfg = dict(_pipe_config())
        cfg.pop("output_site"), cfg.pop("output_layer"), cfg.pop("passes")
        cfg["decode_state"] = True
        with pytest.raises(RuntimeError) as ei:
            R.refuse_decode_state(cfg, label="id=2 at site=block_output")
        text = str(ei.value)
        assert "decode_state=True" in text
        assert "CARRIES_ADAPTER_DECODE_STATE" in text
        assert "fresh-state singleton" in text
        assert "id=2 at site=block_output" in text

    def test_a_member_declaring_nothing_is_untouched(self):
        R.refuse_decode_state(dict(_pipe_config()))     # no raise
        R.refuse_decode_state(None)
        R.refuse_decode_state({"layer_indices": [0]})

    def test_the_forks_own_load_time_guard_would_NOT_have_caught_it(self):
        """Why the refusal is keyed on the MOUNT and not on the leaf:
        `check_adaptation_supported` only refuses a `.mixer` declaring
        `stateful=True`, and a scanned leaf carries no `.mixer`."""
        from vllm.adaptation import check_adaptation_supported
        check_adaptation_supported(CarriedStateScan())   # admits it

    def test_the_failure_it_prevents_is_real(self):
        """Serving the member anyway calls readout with state=None."""
        leaf = CarriedStateScan()
        with pytest.raises(AssertionError, match="CARRIED state"):
            P.apply_adaptation(leaf, torch.randn(3, HIDDEN), None)

    def test_the_refusal_does_not_fire_on_a_carry_pipe(self):
        """A carry pipe (`decode_state` WITH a port pair) has its own
        refusal — `adapters/serving.py::refuse_carry` — because its fork
        work is different (no re-execution, no second KV arena). Two
        diagnoses for two mechanisms, never one blurred message."""
        cfg = dict(_pipe_config())
        cfg["decode_state"] = True
        cfg["host_reexecute"] = True
        R.refuse_decode_state(cfg)                       # no raise here


# ---------------------------------------------------------------------------
# 8. What a re-executing span CANNOT be combined with: an external KV prefix
# ---------------------------------------------------------------------------

class TestKVConnectorIsRefused:
    """The refusal that decides how the composed member can be scored.

    `campaign/cartridge_readout/longhealth_raw` delivers its frozen
    cartridge as an external KV prefix through
    `vllm.adaptation.kv_prefix_connector.PrefixInjectionConnector`
    (`probes/vllm_eval_fs12.py`), and a span cannot run alongside one.
    The reason is not bookkeeping: the shadow layers carry names of their
    own, which no row store contains, while SHARING the host's block
    table — so the injected prefix positions are allocated in every
    pass's cache tensor, written in none of them, and the span's queries
    attend causally over exactly those positions.  Uninitialised keys,
    fluent output.

    Seeding each pass with the same rows would not be a fix and the
    message says so: the HF engine runs the re-executed passes CACHE-FREE
    when the host is not caching, so a member trained behind an external
    prefix never saw that prefix inside its span.
    """

    def test_a_connector_is_refused_with_the_reason(self):
        model, layers = _build_stack()
        vc = _vllm_config(recirc_span=_span_declaration(),
                          kv_transfer=SimpleNamespace(
                              kv_connector="PrefixInjectionConnector"))
        with pytest.raises(RuntimeError) as ei:
            R.install_recirculation(model, vc)
        text = str(ei.value)
        assert "PrefixInjectionConnector" in text
        assert "UNINITIALISED MEMORY" in text
        assert "share the host's block table" in text.lower()
        assert "CACHE-FREE" in text

    def test_the_refusal_is_the_same_on_the_baked_route(self):
        """One diagnosis for one hazard, whichever route asked."""
        model, layers = _build_stack()
        cfg = _pipe_config()
        _add_adapter_to_layer(layers[IN_LAYER], 1, GatedPipe(), "all",
                              torch.device("cpu"), site="block_output")
        with pytest.raises(RuntimeError, match="UNINITIALISED MEMORY"):
            R.install_recirculation(model, _vllm_config(
                adapter_config=cfg,
                kv_transfer=SimpleNamespace(kv_connector="X")))

    def test_without_a_connector_the_span_installs(self):
        """The refusal is about the COMBINATION, not about the span."""
        model, layers = _build_stack()
        assert R.install_recirculation(model, _vllm_config(
            recirc_span=_span_declaration())) is not None
