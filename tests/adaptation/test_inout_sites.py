# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Off-diagonal mount sites: F = (F_in, F_out) with host re-execution.

The site axis is a PAIR.  A mount whose output port PRECEDES its input
port makes the engine re-execute the enclosed decoder layers ``passes``
times from the write port, with the member's ``recombine`` folding the
result back in.  See ``vllm/adaptation/recirculation.py`` and
``docs/inout-sites.md``.

Everything here is CPU-only.  The decoder layer is a faithful mock: a
real causal attention over a real per-``layer_name`` KV cache, so the
tests can assert the property that actually matters — that each
re-executed pass keeps its OWN cache, and that a cached decode step
reproduces the full-prefix forward through the span.
"""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.adaptation import protocol as P
from vllm.adaptation import recirculation as R
from vllm.adaptation.layer import (_add_adapter_to_layer,
                                   make_adapter_decoder_layer,
                                   update_adapter_position_masks)
from vllm.adaptation.specs import (adapter_config_rewires,
                                   adapter_config_to_spec,
                                   spec_to_adapter_config)

HIDDEN = 8
NLAYERS = 6


def _meta(num_prefill_tokens, num_decodes, num_prefills):
    return SimpleNamespace(num_prefill_tokens=num_prefill_tokens,
                           num_decodes=num_decodes,
                           num_prefills=num_prefills,
                           query_start_loc=None,
                           seq_lens=None)


class _Cfg:
    hidden_size = HIDDEN
    torch_dtype = torch.float32


# ---------------------------------------------------------------------------
# A faithful-enough decoder layer: real causal attention, real KV cache
# ---------------------------------------------------------------------------

class KVStore:
    """Stands in for vLLM's paged cache: one K/V log per layer_name.

    The engine gives each pass its own cache TENSOR while sharing the
    block table, so pass p writes token t's K/V at the same slot in its
    own tensor.  Appending to a per-name log in token order is the same
    semantics, which is what makes the decode-consistency test below a
    real test of the engine's scheme rather than of the mock.
    """

    def __init__(self):
        self.k: dict[str, torch.Tensor] = {}
        self.v: dict[str, torch.Tensor] = {}
        self.calls: list[str] = []

    def append(self, name, k, v):
        self.calls.append(name)
        if name in self.k:
            self.k[name] = torch.cat([self.k[name], k], dim=0)
            self.v[name] = torch.cat([self.v[name], v], dim=0)
        else:
            self.k[name], self.v[name] = k, v
        return self.k[name], self.v[name]

    def reset(self):
        self.k.clear()
        self.v.clear()
        self.calls.clear()


# The store the mock attention writes to.  A module-level handle keeps
# the mock's signature identical to the real Attention's.
STORE = KVStore()
# None => no cache (single whole-sequence forward); a tensor => absolute
# positions of the queries in this call, i.e. the cached/decode path.
USE_CACHE = True


class FakeAttention(nn.Module):
    """Enough of ``vllm.attention.Attention`` for the recirculation path.

    Carries the attributes ``get_kv_cache_spec`` reads (so a shallow copy
    is a valid KV-cache twin) plus the two the span executor swaps.
    """

    def __init__(self, prefix: str):
        super().__init__()
        self.qkv = nn.Linear(HIDDEN, 3 * HIDDEN, bias=False)
        self.o = nn.Linear(HIDDEN, HIDDEN, bias=False)
        # --- the Attention surface the engine reads ---
        self.layer_name = prefix
        self.kv_cache = [torch.tensor([])]
        self.attn_type = "decoder"
        self.num_kv_heads = 1
        self.head_size = HIDDEN
        self.sliding_window = None
        self.kv_sharing_target_layer_name = None

    def forward(self, positions, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        if USE_CACHE:
            k_all, v_all = STORE.append(self.layer_name, k, v)
        else:
            STORE.calls.append(self.layer_name)
            k_all, v_all = k, v
        # Causal mask on ABSOLUTE positions: a query at position p sees
        # every cached key at position <= p.  The re-executed pass is
        # handed the host's positions unchanged, so this is the same mask
        # the host's own pass used.
        n_cached = k_all.shape[0]
        key_pos = torch.arange(n_cached)
        allowed = key_pos.unsqueeze(0) <= positions.unsqueeze(1)
        scores = (q @ k_all.transpose(0, 1)) / math.sqrt(HIDDEN)
        scores = scores.masked_fill(~allowed, float("-inf"))
        return self.o(torch.softmax(scores, dim=-1) @ v_all)


class FakeDecoderLayer(nn.Module):
    """llama-style ``(positions, hidden, residual) -> (hidden, residual)``."""

    def __init__(self, config, cache_config=None, quant_config=None,
                 prefix: str = ""):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = nn.Module()
        self.self_attn.attn = FakeAttention(f"{prefix}.self_attn.attn")
        self.mlp = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.forwards = 0

    def forward(self, positions, hidden_states, residual):
        self.forwards += 1
        stream = (hidden_states if residual is None else
                  hidden_states + residual)
        mid = stream + self.self_attn.attn(positions, stream)
        out = mid + 0.5 * torch.tanh(self.mlp(mid))
        # Deferred-residual contract: hidden + residual == out.
        return out - mid, mid


class GatedPipe(nn.Module):
    """The recirculation member: identity T, identity R, gated recombine.

    ``out = fx + g*(piped - fx)`` — bit-exact identity at ``g == 0``,
    which is the zero-init contract the whole zoo rests on.
    """

    def __init__(self, gate: float = 0.0, gated: bool = True):
        super().__init__()
        self.gated = gated
        self.gate = nn.Parameter(torch.tensor(float(gate)))

    def readout(self, fx, state=None, x=None):
        return fx

    def recombine(self, fx, piped):
        if not self.gated:
            return piped
        return fx + self.gate.to(fx.dtype) * (piped - fx)


# ---------------------------------------------------------------------------
# Fixtures: a stack of adapter-wrapped fake layers + a fake VllmConfig
# ---------------------------------------------------------------------------

def _build_stack(seed: int = 0):
    torch.manual_seed(seed)
    cls = make_adapter_decoder_layer(FakeDecoderLayer)
    layers = [cls(_Cfg(), None, None, f"model.layers.{i}")
              for i in range(NLAYERS)]
    model = nn.Module()
    model.layers = nn.ModuleList(layers)
    return model, layers


def _fake_vllm_config(adapter_config, *, prefix_caching=False,
                      chunked_prefill=False, eager=True, pp=1,
                      kv_transfer=None):
    return SimpleNamespace(
        adapter_config=adapter_config,
        compilation_config=SimpleNamespace(static_forward_context={}),
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp,
                                        tensor_parallel_size=1),
        cache_config=SimpleNamespace(enable_prefix_caching=prefix_caching),
        scheduler_config=SimpleNamespace(
            chunked_prefill_enabled=chunked_prefill),
        model_config=SimpleNamespace(enforce_eager=eager),
        kv_transfer_config=kv_transfer,
    )


def _pipe_config(in_layer=3, out_layer=0, out_site="block_input", passes=2,
                 gate=0.0, gated=True, position="all"):
    """A baked adapter_config for a gated pipe, as serving.py would build."""
    pipe = GatedPipe(gate=gate, gated=gated)
    spec = {
        "layer_indices": [in_layer],
        "position": position,
        "sample_adapter": pipe,
        "adapters": {in_layer: pipe},
        "site": "block_output",
        "output_site": out_site,
        "output_layer": out_layer,
        "passes": passes,
    }
    return spec_to_adapter_config(spec)


def _mount(layers, adapter_config, in_layer):
    """Install the member on its layer, the way the baked route does."""
    spec = adapter_config_to_spec(adapter_config)
    adapter = spec["adapters"][in_layer]
    _add_adapter_to_layer(layers[in_layer], 1, adapter,
                          adapter_config["position"], torch.device("cpu"),
                          site=adapter_config.get("site", "block_output"))
    return adapter


def _install(model, layers, adapter_config, **cfg_kw):
    vllm_config = _fake_vllm_config(adapter_config, **cfg_kw)
    return R.install_recirculation(model, vllm_config), vllm_config


# ---------------------------------------------------------------------------
# Forward-compat shim: the pair-port API lands with the adapters library's
# `recirculation` work, and the fork imports it OPTIONALLY (it refuses to
# invent a private copy of the port order — that is the drift bug class
# vllm/adaptation/protocol.py's hard site import exists to kill).  When
# the installed library predates it, these tests supply the definitions so
# the FORK's own logic can still be exercised.  The shim mirrors
# adapters/sites.py verbatim, and
# TestPairPortContract::test_library_definitions_win pins that the fork
# uses the library's objects — never these — once they exist.
# ---------------------------------------------------------------------------

_SHIM_WRITE_PORTS = ("block_input", "block_output")
_SHIM_PORT_ORDER = {"block_input": 0, "post_attn": 1, "post_mlp": 2,
                    "block_output": 3}


def _shim_validate_write_port(site):
    from adapters.sites import ENGINE_SITES, validate_site as _vs
    _vs(site)
    if site not in _SHIM_WRITE_PORTS:
        raise ValueError(
            f"{site!r} is not a write port. A mount whose output port "
            f"differs from its input port must write at a whole-block "
            f"boundary {_SHIM_WRITE_PORTS} — the engine resumes the "
            f"host's layer loop there, and a decoder layer cannot be "
            f"resumed from the middle of its own forward "
            f"(post_attn/post_mlp are read taps) nor from an engine site "
            f"({ENGINE_SITES}) or a weights: site.")


def _shim_port_order(layer_idx, site):
    from adapters.sites import validate_site as _vs
    _vs(site)
    if site not in _SHIM_PORT_ORDER:
        raise ValueError(f"{site!r} has no place in the port order")
    return (int(layer_idx), _SHIM_PORT_ORDER[site])


@pytest.fixture(autouse=True)
def _pair_ports(monkeypatch):
    if P.HAVE_PAIR_PORTS:
        yield
        return
    monkeypatch.setattr(P, "HAVE_PAIR_PORTS", True)
    monkeypatch.setattr(P, "WRITE_PORTS", _SHIM_WRITE_PORTS)
    monkeypatch.setattr(P, "PORT_ORDER", _SHIM_PORT_ORDER)
    monkeypatch.setattr(P, "_shared_validate_write_port",
                        _shim_validate_write_port)
    monkeypatch.setattr(P, "_shared_port_order", _shim_port_order)
    yield


class TestPairPortContract:

    def test_library_definitions_win_when_present(self):
        """Same rule as tests/test_site_contract.py in the adapters repo:
        the fork must USE the library's objects, not re-implement them."""
        import adapters.sites as shared
        if not hasattr(shared, "port_order"):
            pytest.skip("installed adapters library predates the pair axis")
        assert P.HAVE_PAIR_PORTS
        assert P._shared_port_order is shared.port_order
        assert P._shared_validate_write_port is shared.validate_write_port
        assert P.WRITE_PORTS == shared.WRITE_PORTS
        assert P.PORT_ORDER == shared.PORT_ORDER

    def test_rewiring_raises_without_the_library_api(self, monkeypatch):
        """No private fallback: a rewiring config against an old library
        must fail loudly, not guess the port order."""
        monkeypatch.setattr(P, "HAVE_PAIR_PORTS", False)
        with pytest.raises(ImportError, match="adapters.sites must export"):
            R.plan_from_adapter_config(_pipe_config(), NLAYERS)


@pytest.fixture(autouse=True)
def _attn_seam(monkeypatch):
    """Let the mock's attention modules be found by the installer.

    ``_attention_modules`` isinstance-checks the real ``Attention``,
    which cannot be constructed on CPU without a full engine; the seam
    keeps the rest of the production path (shadow creation, name
    allocation, registration, swapping) exactly as it ships.
    """
    monkeypatch.setattr(
        R, "_attention_modules",
        lambda layer: [m for m in layer.modules()
                       if isinstance(m, FakeAttention)])
    STORE.reset()
    global USE_CACHE
    USE_CACHE = True
    yield
    STORE.reset()


def _run(model, layers, tokens, positions, num_tokens=None):
    """One forward through the whole stack."""
    num_tokens = num_tokens if num_tokens is not None else tokens.shape[0]
    token_ids = torch.ones(num_tokens, dtype=torch.int32)
    update_adapter_position_masks(layers, token_ids, positions,
                                  _meta(num_tokens, 0, 1), num_tokens)
    hidden, residual = tokens, None
    for layer in layers:
        hidden, residual = layer(positions, hidden, residual)
    return hidden if residual is None else hidden + residual


# ---------------------------------------------------------------------------
# 1. Manifest / adapter_config round-trip and byte-compatibility
# ---------------------------------------------------------------------------

class TestManifestRoundTrip:

    def test_diagonal_config_carries_no_pair_keys(self):
        """The byte-compat guarantee: a diagonal member's config is
        exactly what it was before the pair axis existed."""
        spec = {"layer_indices": [2], "position": "prefill",
                "sample_adapter": GatedPipe(), "adapters": {}}
        cfg = spec_to_adapter_config(spec)
        for key in R.REWIRE_KEYS:
            assert key not in cfg, f"{key} leaked into a diagonal config"
        assert not R.config_rewires(cfg)
        assert not adapter_config_rewires(cfg)

    def test_explicit_none_pair_keys_are_still_diagonal(self):
        """``output_site=None`` is the library's "same port" encoding and
        must not be mistaken for a rewiring (adapters/serving.py emits it
        for every member on the v2 path)."""
        spec = {"layer_indices": [2], "position": "all",
                "sample_adapter": GatedPipe(), "adapters": {},
                "output_site": None, "output_layer": None, "passes": 1}
        cfg = spec_to_adapter_config(spec)
        assert not R.config_rewires(cfg)
        assert R.plan_from_adapter_config(cfg, NLAYERS) is None

    def test_pair_keys_round_trip(self):
        cfg = _pipe_config(in_layer=3, out_layer=0, passes=3)
        assert cfg["output_site"] == "block_input"
        assert cfg["output_layer"] == 0
        assert cfg["passes"] == 3
        assert cfg["site"] == "block_output"
        spec = adapter_config_to_spec(cfg)
        assert spec["output_site"] == "block_input"
        assert spec["output_layer"] == 0
        assert spec["passes"] == 3
        # and a second round trip is stable
        again = spec_to_adapter_config(spec)
        for key in ("site", *R.REWIRE_KEYS):
            assert again[key] == cfg[key]

    def test_output_layer_alone_is_a_rewiring(self):
        """Mirrors Mount.rewires(): the same port KIND at an earlier
        layer is still off-diagonal."""
        spec = {"layer_indices": [3], "position": "all",
                "sample_adapter": GatedPipe(), "adapters": {},
                "output_layer": 1}
        cfg = spec_to_adapter_config(spec)
        assert R.config_rewires(cfg)
        plan = R.plan_from_adapter_config(cfg, NLAYERS)
        assert (plan.out_layer, plan.out_site) == (1, "block_output")
        # block_output of layer 1 resumes at layer 2, not layer 1
        assert list(plan.span_layers) == [2, 3]

    def test_plan_geometry_block_input_is_inclusive(self):
        plan = R.plan_from_adapter_config(
            _pipe_config(in_layer=3, out_layer=0,
                         out_site="block_input", passes=2), NLAYERS)
        assert list(plan.span_layers) == [0, 1, 2, 3]
        assert plan.span_len == 4
        assert plan.extra_attention_layers() == 8


# ---------------------------------------------------------------------------
# 2. Refusals — each with its own reason
# ---------------------------------------------------------------------------

class TestRefusals:

    def test_multisite_route_is_refused(self):
        cfg = _pipe_config()
        with pytest.raises(RuntimeError, match="REFUSING to serve"):
            R.refuse_rewired(cfg, "multisite")
        with pytest.raises(RuntimeError, match="KV-cache spec ONCE"):
            R.refuse_rewired(cfg, "multisite")

    def test_lora_view_route_is_refused(self):
        with pytest.raises(RuntimeError, match="lora_view"):
            R.refuse_rewired(_pipe_config(), "lora_view")

    def test_refusal_is_a_no_op_for_diagonal_members(self):
        spec = {"layer_indices": [2], "position": "all",
                "sample_adapter": GatedPipe(), "adapters": {}}
        R.refuse_rewired(spec_to_adapter_config(spec), "multisite")

    def test_forward_write_port_is_refused(self):
        cfg = _pipe_config(in_layer=1, out_layer=4)
        with pytest.raises(RuntimeError, match="at or AFTER the input port"):
            R.plan_from_adapter_config(cfg, NLAYERS)

    def test_same_port_pair_is_refused(self):
        """block_output of layer j writing to block_output of layer j is
        the diagonal wearing the pair's clothes: the span is empty."""
        cfg = _pipe_config(in_layer=3, out_layer=3, out_site="block_output")
        with pytest.raises(RuntimeError, match="at or AFTER the input port"):
            R.plan_from_adapter_config(cfg, NLAYERS)

    def test_illegal_write_ports_are_refused(self):
        for bad in ("post_attn", "post_mlp", "attn_kv", "rope"):
            cfg = _pipe_config(out_site=bad)
            with pytest.raises(ValueError, match="not a write port"):
                R.plan_from_adapter_config(cfg, NLAYERS)

    def test_non_block_output_input_port_is_refused(self):
        cfg = _pipe_config()
        cfg["site"] = "post_attn"
        with pytest.raises(RuntimeError,
                           match="input port .* not implemented"):
            R.plan_from_adapter_config(cfg, NLAYERS)

    def test_multiple_input_layers_are_refused(self):
        cfg = _pipe_config()
        cfg["layer_indices"] = [2, 3]
        with pytest.raises(RuntimeError, match="exactly ONE input layer"):
            R.plan_from_adapter_config(cfg, NLAYERS)

    def test_out_of_range_layers_are_refused(self):
        cfg = _pipe_config(in_layer=3, out_layer=0)
        cfg["layer_indices"] = [99]
        with pytest.raises(RuntimeError, match="outside the host"):
            R.plan_from_adapter_config(cfg, NLAYERS)

    @pytest.mark.parametrize("passes", [0, -1, 1.5, True, "2"])
    def test_bad_pass_counts_are_refused(self, passes):
        cfg = _pipe_config()
        cfg["passes"] = passes
        with pytest.raises(RuntimeError, match="passes must be an int"):
            R.plan_from_adapter_config(cfg, NLAYERS)

    @pytest.mark.parametrize("kw,match", [
        ({"pp": 2}, "pipeline_parallel_size"),
        ({"prefix_caching": True}, "prefix caching must be disabled"),
        ({"chunked_prefill": True}, "chunked prefill must be disabled"),
        ({"eager": False}, "eager execution is required"),
        ({"kv_transfer": object()}, "KV-transfer connector"),
    ])
    def test_unsupported_engine_configs_are_refused(self, kw, match):
        model, layers = _build_stack()
        cfg = _pipe_config()
        _mount(layers, cfg, 3)
        with pytest.raises(RuntimeError, match=match):
            _install(model, layers, cfg, **kw)

    def test_co_mounted_member_at_the_input_port_is_refused(self):
        model, layers = _build_stack()
        cfg = _pipe_config()
        _mount(layers, cfg, 3)
        _add_adapter_to_layer(layers[3], 2, GatedPipe(), "all",
                              torch.device("cpu"), site="block_output")
        with pytest.raises(RuntimeError, match="must be the only one at its"):
            _install(model, layers, cfg)

    def test_install_is_a_no_op_without_a_rewiring_config(self):
        model, layers = _build_stack()
        plan, vc = _install(model, layers, None)
        assert plan is None
        assert vc.compilation_config.static_forward_context == {}
        assert not any(hasattr(l, "_adapter_recirc") for l in layers)


# ---------------------------------------------------------------------------
# 3. KV allocation: one extra cache set per pass
# ---------------------------------------------------------------------------

class TestKVAllocation:

    def test_one_shadow_set_per_pass_per_span_layer(self):
        model, layers = _build_stack()
        cfg = _pipe_config(in_layer=3, out_layer=0, passes=2)
        _mount(layers, cfg, 3)
        plan, vc = _install(model, layers, cfg)
        ctx = vc.compilation_config.static_forward_context
        assert plan.span_len == 4
        assert len(ctx) == 2 * 4 == plan.extra_attention_layers()

    @pytest.mark.parametrize("passes", [1, 2, 3])
    def test_shadow_count_scales_with_passes(self, passes):
        model, layers = _build_stack()
        cfg = _pipe_config(in_layer=2, out_layer=0, passes=passes)
        _mount(layers, cfg, 2)
        _, vc = _install(model, layers, cfg)
        assert len(vc.compilation_config.static_forward_context) == passes * 3

    def test_shadow_names_are_unique_and_extract_one_index(self):
        """``bind_kv_cache`` parses the layer index out of the name and
        asserts there is exactly one integer in it; two names mapping to
        one index would drop a cache from the runner's list."""
        from vllm.model_executor.models.utils import extract_layer_index
        model, layers = _build_stack()
        cfg = _pipe_config(in_layer=3, out_layer=0, passes=2)
        _mount(layers, cfg, 3)
        _, vc = _install(model, layers, cfg)
        names = list(vc.compilation_config.static_forward_context)
        assert len(set(names)) == len(names)
        indices = [extract_layer_index(n) for n in names]
        assert len(set(indices)) == len(indices), "shadow indices collide"
        # and none collides with a real decoder layer's index
        assert min(indices) >= NLAYERS

    def test_shadows_are_kv_spec_twins_of_their_originals(self):
        model, layers = _build_stack()
        cfg = _pipe_config(in_layer=2, out_layer=0, passes=2)
        _mount(layers, cfg, 2)
        _, vc = _install(model, layers, cfg)
        for shadow in vc.compilation_config.static_forward_context.values():
            # exactly the fields get_kv_cache_spec reads
            assert shadow.attn_type == "decoder"
            assert shadow.num_kv_heads == 1
            assert shadow.head_size == HIDDEN
            assert shadow.sliding_window is None
            assert shadow.kv_sharing_target_layer_name is None

    def test_shadow_creation_does_not_disturb_the_original(self):
        model, layers = _build_stack()
        cfg = _pipe_config(in_layer=2, out_layer=0, passes=2)
        _mount(layers, cfg, 2)
        originals = {id(m): (m.layer_name, m.kv_cache)
                     for l in layers for m in l.modules()
                     if isinstance(m, FakeAttention)}
        _, vc = _install(model, layers, cfg)
        for l in layers:
            for m in l.modules():
                if isinstance(m, FakeAttention):
                    name, cache = originals[id(m)]
                    assert m.layer_name == name
                    assert m.kv_cache is cache
        # the shadows share weights with their originals (no copies)
        shadows = list(vc.compilation_config.static_forward_context.values())
        real_qkv = {id(m.qkv) for l in layers for m in l.modules()
                    if isinstance(m, FakeAttention)}
        assert all(id(s.qkv) in real_qkv for s in shadows)

    def test_each_pass_writes_its_own_cache(self):
        """The property the whole design exists for: pass 0 and pass 1
        must never touch the same cache, and neither may touch the
        host's."""
        model, layers = _build_stack()
        cfg = _pipe_config(in_layer=2, out_layer=0, passes=2, gate=0.7)
        _mount(layers, cfg, 2)
        _install(model, layers, cfg)
        _run(model, layers, torch.randn(5, HIDDEN), torch.arange(5))

        host = [f"model.layers.{i}.self_attn.attn" for i in range(NLAYERS)]
        assert set(host) <= set(STORE.k)
        pass0 = sorted(n for n in STORE.k if "pass0" in n)
        pass1 = sorted(n for n in STORE.k if "pass1" in n)
        assert len(pass0) == len(pass1) == 3      # span layers 0,1,2
        assert not set(pass0) & set(pass1)
        assert not (set(pass0) | set(pass1)) & set(host)
        # every cache saw the same 5 tokens: same positions, own tensor
        for name in list(pass0) + list(pass1) + host:
            assert STORE.k[name].shape[0] == 5

    def test_span_layers_execute_once_plus_k_times(self):
        model, layers = _build_stack()
        cfg = _pipe_config(in_layer=2, out_layer=0, passes=3, gate=0.3)
        _mount(layers, cfg, 2)
        _install(model, layers, cfg)
        _run(model, layers, torch.randn(4, HIDDEN), torch.arange(4))
        # span = layers 0..2: the host's own pass plus 3 re-executions
        for i in range(3):
            assert layers[i].forwards == 4, f"layer {i}"
        for i in range(3, NLAYERS):
            assert layers[i].forwards == 1, f"layer {i}"


# ---------------------------------------------------------------------------
# 4. Numerics: identity at gate 0, real effect otherwise
# ---------------------------------------------------------------------------

class TestNumerics:

    def _base_and_piped(self, gate, passes=2, in_layer=2, out_layer=0,
                        seed=1, tokens=5):
        gen = torch.Generator().manual_seed(9)
        x = torch.randn(tokens, HIDDEN, generator=gen)
        positions = torch.arange(tokens)

        model_b, layers_b = _build_stack(seed)
        STORE.reset()
        base = _run(model_b, layers_b, x, positions)

        model_p, layers_p = _build_stack(seed)
        cfg = _pipe_config(in_layer=in_layer, out_layer=out_layer,
                           passes=passes, gate=gate)
        _mount(layers_p, cfg, in_layer)
        _install(model_p, layers_p, cfg)
        STORE.reset()
        out = _run(model_p, layers_p, x, positions)
        return base, out

    def test_zero_gate_is_bit_exact_identity(self):
        """The zero-init contract: mounting the pipe at g=0 must change
        NOTHING, bit for bit, even though the span really ran."""
        base, out = self._base_and_piped(gate=0.0)
        assert torch.equal(base, out)

    def test_nonzero_gate_changes_the_output(self):
        base, out = self._base_and_piped(gate=0.9)
        assert not torch.allclose(base, out, atol=1e-6)

    def test_more_passes_change_the_result(self):
        _, k2 = self._base_and_piped(gate=0.9, passes=2)
        _, k3 = self._base_and_piped(gate=0.9, passes=3)
        assert not torch.allclose(k2, k3, atol=1e-6)

    def test_ungated_pipe_replaces_the_stream(self):
        """An identity-recombine pipe hands the re-executed stream on
        verbatim, so it must differ from the base."""
        x = torch.randn(4, HIDDEN, generator=torch.Generator().manual_seed(3))
        positions = torch.arange(4)
        model_b, layers_b = _build_stack(2)
        STORE.reset()
        base = _run(model_b, layers_b, x, positions)

        model_p, layers_p = _build_stack(2)
        cfg = _pipe_config(in_layer=2, out_layer=0, passes=1, gated=False)
        _mount(layers_p, cfg, 2)
        _install(model_p, layers_p, cfg)
        STORE.reset()
        out = _run(model_p, layers_p, x, positions)
        assert not torch.allclose(base, out, atol=1e-6)

    def test_gate_receives_gradient_at_init(self):
        """g=0 is an identity, not a saddle: dout/dg != 0 there, or the
        member could never learn to switch itself on."""
        model, layers = _build_stack(4)
        cfg = _pipe_config(in_layer=2, out_layer=0, passes=2, gate=0.0)
        adapter = _mount(layers, cfg, 2)
        _install(model, layers, cfg)
        out = _run(model, layers, torch.randn(4, HIDDEN), torch.arange(4))
        out.sum().backward()
        assert adapter.gate.grad is not None
        assert float(adapter.gate.grad.abs()) > 0.0

    def test_positions_are_not_shifted(self):
        """The re-executed pass sees the host's positions unchanged, so
        every cache holds one entry per position and the causal mask is
        the same one the host used."""
        model, layers = _build_stack()
        cfg = _pipe_config(in_layer=2, out_layer=0, passes=2, gate=0.5)
        _mount(layers, cfg, 2)
        _install(model, layers, cfg)
        n = 6
        _run(model, layers, torch.randn(n, HIDDEN), torch.arange(n))
        for name, k in STORE.k.items():
            assert k.shape[0] == n, name


# ---------------------------------------------------------------------------
# 5. Decode consistency: cached decode == full-prefix forward
# ---------------------------------------------------------------------------

class TestDecodeConsistency:
    """The property the per-pass caches exist to preserve.

    ``adapters``' HF-side test asserts exactly this
    (``test_cached_decode_equals_full_forward``); it is the contract the
    engine route has to match, and the reason each pass needs its own
    cache rather than a shared one.
    """

    def _stack_with_pipe(self, gate, passes, in_layer=2, out_layer=0):
        model, layers = _build_stack(7)
        cfg = _pipe_config(in_layer=in_layer, out_layer=out_layer,
                           passes=passes, gate=gate)
        _mount(layers, cfg, in_layer)
        _install(model, layers, cfg)
        return model, layers

    @pytest.mark.parametrize("prefill", [1, 3, 6])
    @pytest.mark.parametrize("passes", [1, 2])
    def test_cached_decode_equals_full_forward(self, prefill, passes):
        total = 7
        x = torch.randn(total, HIDDEN,
                        generator=torch.Generator().manual_seed(11))
        positions = torch.arange(total)

        # (a) one whole-sequence forward
        model, layers = self._stack_with_pipe(0.8, passes)
        STORE.reset()
        full = _run(model, layers, x, positions)

        # (b) prefill a prefix, then step the rest one token at a time,
        #     reusing every cache (host's and per-pass) across steps
        model, layers = self._stack_with_pipe(0.8, passes)
        STORE.reset()
        outs = [_run(model, layers, x[:prefill], positions[:prefill])]
        for t in range(prefill, total):
            outs.append(_run(model, layers, x[t:t + 1], positions[t:t + 1]))
        stepped = torch.cat(outs, dim=0)

        assert stepped.shape == full.shape
        torch.testing.assert_close(stepped, full, atol=1e-5, rtol=1e-4)

    def test_decode_without_per_pass_caches_would_differ(self):
        """Guards the test itself: if every pass shared one cache the
        decode would NOT reproduce the full forward, so the assertion
        above is really testing the separation."""
        total = 5
        x = torch.randn(total, HIDDEN,
                        generator=torch.Generator().manual_seed(13))
        positions = torch.arange(total)

        model, layers = self._stack_with_pipe(0.8, 2)
        STORE.reset()
        full = _run(model, layers, x, positions)

        # collapse pass 1's shadows onto pass 0's names
        model, layers = self._stack_with_pipe(0.8, 2)
        inst = layers[2]._adapter_recirc
        for (span_pos, p), twins in inst.shadows.items():
            if p == 0:
                continue
            for twin, ref in zip(twins, inst.shadows[(span_pos, 0)]):
                twin.layer_name = ref.layer_name
        STORE.reset()
        outs = [_run(model, layers, x[:1], positions[:1])]
        for t in range(1, total):
            outs.append(_run(model, layers, x[t:t + 1], positions[t:t + 1]))
        collapsed = torch.cat(outs, dim=0)
        assert not torch.allclose(collapsed, full, atol=1e-4)

    def test_causality_holds_through_the_span(self):
        """Editing the last token must leave every earlier position bit
        equal: the span must not leak future tokens backwards."""
        total = 6
        x = torch.randn(total, HIDDEN,
                        generator=torch.Generator().manual_seed(17))
        positions = torch.arange(total)

        model, layers = self._stack_with_pipe(0.8, 2)
        STORE.reset()
        a = _run(model, layers, x, positions)

        y = x.clone()
        y[-1] = torch.randn(HIDDEN)
        model, layers = self._stack_with_pipe(0.8, 2)
        STORE.reset()
        b = _run(model, layers, y, positions)

        torch.testing.assert_close(a[:-1], b[:-1], atol=0, rtol=0)


# ---------------------------------------------------------------------------
# 6. Regression: the diagonal path is untouched
# ---------------------------------------------------------------------------

class TestDiagonalRegression:

    def test_diagonal_member_output_is_unchanged_by_the_new_code_path(self):
        """A plain block_output member must produce bit-identical output
        whether or not the recirculation machinery is importable/armed."""

        class Const(nn.Module):

            def __init__(self, v=0.25):
                super().__init__()
                self.v = v
                self.marker = nn.Linear(1, 1)

            def readout(self, fx, state=None, x=None):
                return fx + self.v

        x = torch.randn(4, HIDDEN, generator=torch.Generator().manual_seed(5))
        positions = torch.arange(4)

        model, layers = _build_stack(6)
        _add_adapter_to_layer(layers[2], 1, Const(), "all",
                              torch.device("cpu"), site="block_output")
        STORE.reset()
        with_member = _run(model, layers, x, positions)

        model_b, layers_b = _build_stack(6)
        STORE.reset()
        base = _run(model_b, layers_b, x, positions)

        # the member fires (output differs) and no recirc state was armed
        assert not torch.allclose(base, with_member)
        assert not hasattr(layers[2], "_adapter_recirc")
        assert not any("pass" in n for n in STORE.k)

    def test_no_adapter_at_all_is_untouched(self):
        x = torch.randn(4, HIDDEN, generator=torch.Generator().manual_seed(8))
        positions = torch.arange(4)
        model, layers = _build_stack(6)
        STORE.reset()
        a = _run(model, layers, x, positions)
        model, layers = _build_stack(6)
        STORE.reset()
        b = _run(model, layers, x, positions)
        assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# 7. Re-entrancy
# ---------------------------------------------------------------------------

class TestReentrancy:

    def test_the_pipe_is_a_passthrough_inside_its_own_span(self):
        """Layer j is IN its own span, so without the depth guard the
        span would recurse until the stack blew."""
        model, layers = _build_stack()
        cfg = _pipe_config(in_layer=2, out_layer=0, passes=2, gate=0.5)
        _mount(layers, cfg, 2)
        _install(model, layers, cfg)
        out = _run(model, layers, torch.randn(3, HIDDEN), torch.arange(3))
        assert torch.isfinite(out).all()
        assert layers[2]._adapter_recirc.depth == 0

    def test_depth_is_restored_after_an_exception_in_the_span(self):
        model, layers = _build_stack()
        cfg = _pipe_config(in_layer=2, out_layer=0, passes=2, gate=0.5)
        _mount(layers, cfg, 2)
        _install(model, layers, cfg)
        inst = layers[2]._adapter_recirc
        boom = layers[1]

        original = boom.forward

        def explode(*a, **k):
            raise ValueError("boom")

        boom.forward = explode
        with pytest.raises(ValueError, match="boom"):
            _run(model, layers, torch.randn(3, HIDDEN), torch.arange(3))
        boom.forward = original
        assert inst.depth == 0

    def test_attention_names_are_restored_after_an_exception(self):
        """A mid-span failure must not leave the host's own layers
        pointing at a pass cache."""
        model, layers = _build_stack()
        cfg = _pipe_config(in_layer=2, out_layer=0, passes=2, gate=0.5)
        _mount(layers, cfg, 2)
        _install(model, layers, cfg)
        names = {id(m): m.layer_name for l in layers for m in l.modules()
                 if isinstance(m, FakeAttention)}
        boom = layers[1]
        original = boom.forward
        boom.forward = lambda *a, **k: (_ for _ in ()).throw(ValueError("x"))
        with pytest.raises(ValueError):
            _run(model, layers, torch.randn(3, HIDDEN), torch.arange(3))
        boom.forward = original
        for l in layers:
            for m in l.modules():
                if isinstance(m, FakeAttention):
                    assert m.layer_name == names[id(m)]
