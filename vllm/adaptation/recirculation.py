# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Off-diagonal mount sites: F = (F_in, F_out) with host re-execution.

Background
----------
Until now every adaptation's site axis was a single port: the member
read the residual stream at some port and wrote it back at the SAME
port (``F_in == F_out``, the "diagonal").  ``adapters.mounting.Mount``
now carries the pair explicitly::

    Mount(site="block_output", layers=[j],          # F_in  = (j, block_output)
          output_site="block_input", output_layer=i,# F_out = (i, block_input)
          passes=k)

When the output port PRECEDES the input port in the host's forward
order, the mount is not a computation at a point — it is a SCHEDULE
REWIRING.  The engine re-executes the enclosed decoder layers::

    span  = layers[start .. j]        # start = i for block_input,
                                      #         i+1 for block_output
    piped = (L_j o ... o L_start)^k ( R(h_j) )
    h_j'  = W(h_j, piped)             # the member's own write at F_out:
                                      # bare pipe (default W): piped
                                      # gated pipe (InterpolateWrite):
                                      #     h + g*(piped - h)

and the host then continues at layer j+1 with ``h_j'``.  This module
implements that for the BAKED serving route (a single member baked into
``VllmConfig.adapter_config`` at engine construction).

Why the engine and not the adapter
----------------------------------
``vllm/adaptation/protocol.py::apply_adaptation`` is handed a FLATTENED
``(num_tokens, dim)`` tensor inside a per-layer hook: no decoder-layer
list, no attention metadata, no KV blocks.  A span re-execution needs
all three, so it lives in the layer's own forward (which has them) and
in the model runner (which owns the KV cache spec).

The KV-cache crux
-----------------
Each re-executed pass visits the span's attention layers with the SAME
token positions as the host's own pass.  If they shared one KV cache,
pass 2 would attend to pass 1's keys at the same slots — the caches must
be separate, one set per pass.

vLLM v1 collects the KV-cache spec ONCE, at engine start, by walking
``compilation_config.static_forward_context`` for ``Attention`` modules
(``gpu_model_runner.get_kv_cache_spec``).  So the extra caches have to
exist as extra ENTRIES in that dict before the worker asks for the
spec.  :func:`install_recirculation` registers, for every span layer and
every pass, a SHADOW ``Attention``: a shallow copy of the real one
(same impl, same head counts, same scales — so it produces an identical
KV-cache spec) with its own ``layer_name`` and its own ``kv_cache``
slot.  From there the engine's own machinery does the rest:

  * ``get_kv_cache_spec`` emits one ``FullAttentionSpec`` per shadow, so
    the memory profiler divides the KV budget over ``L + k*|span|``
    layers instead of ``L`` — the extra caches are BUDGETED, not stolen;
  * shadows have a spec identical to their originals, so they land in
    the same KV-cache group and share the group's block table and
    ``slot_mapping``.  Pass p therefore writes token t's K/V at the same
    slot in ITS OWN cache tensor, and a later decode step reading that
    block table sees exactly pass p's history;
  * ``bind_kv_cache`` binds each shadow's tensor by layer name.

At pass time :func:`run_recirculation` swaps ``layer_name`` and
``kv_cache`` on the real ``Attention`` modules for the pass's shadows,
runs the span, and restores.  Both attention call paths follow: the
direct call reads the swapped ``self.kv_cache``, and the custom-op path
re-resolves the module by name out of the forward context (which is the
same dict the shadows were registered in).

Positions and masks are untouched: the re-executed pass is handed the
SAME ``positions`` tensor (no shift) and the same attention metadata, so
rotary embeddings and the causal mask are identical to the host's pass.

What this route refuses
-----------------------
See :func:`plan_from_adapter_config` and :func:`refuse_rewired`.  In
short: the multisite and lora_view routes, chunked prefill, CUDA graphs
/ compilation, prefix caching, pipeline parallelism and KV-transfer
connectors.  Each refusal names the reason.
"""

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from torch import nn

from vllm.adaptation.protocol import (CARRIES_ADAPTER_DECODE_STATE,
                                      MULTISITE_HONOURS_UNCHUNKED_PREFILL,
                                      REEXECUTION_ROUTES,
                                      SPAN_COMOUNT_ORDER,
                                      SUPPORTED_WRITE_LABELS, apply_write,
                                      carrier_dtype_of, port_order,
                                      require_pair_ports, validate_site,
                                      validate_write_port)

logger = logging.getLogger("vllm.adaptation.recirculation")

__all__ = [
    "REWIRE_KEYS",
    # Re-exported so the library's serving builder has ONE name to probe
    # for "which W_phi does the installed fork apply?" — the write port
    # this module owns is where a non-default W actually lands.  The
    # definition (and the reasoning for what is in the set) lives next
    # to apply_write in vllm/adaptation/protocol.py.
    "SUPPORTED_WRITE_LABELS",
    # Second capability the serving builder probes on this module: does
    # the fork honour a member's unchunked-prefill declaration on the
    # MULTISITE route (adapter_config unset at engine-config time)?  The
    # definition, and why the answer is not simply "chunked prefill is
    # off", lives beside SUPPORTED_WRITE_LABELS in protocol.py.
    "MULTISITE_HONOURS_UNCHUNKED_PREFILL",
    # Third and fourth capabilities the serving builder probes here:
    # which routes can execute a span (baked only, or a reserved
    # multisite span too), the composition order at a span's input
    # port, and whether this engine carries a per-request adapter
    # State across decode steps.  Defined beside the others in
    # protocol.py; re-exported so the builder has ONE module to ask.
    "REEXECUTION_ROUTES",
    "SPAN_COMOUNT_ORDER",
    "CARRIES_ADAPTER_DECODE_STATE",
    "RecircPlan",
    "SPAN_DECLARATION_KEYS",
    "fill_reserved_span",
    "find_installed_recirc",
    "refuse_decode_state",
    "span_declaration_matches",
    "config_rewires",
    "install_recirculation",
    "plan_from_adapter_config",
    "refuse_rewired",
    "run_recirculation",
]

# The three manifest/adapter_config keys that carry the pair axis.  They
# mirror adapters.mounting.Mount's serialization EXACTLY (mounting.py
# writes them only when the mount rewires), which is what keeps every
# pre-existing manifest byte-identical: a diagonal mount emits none of
# them, and their absence here reconstructs the diagonal.
REWIRE_KEYS = ("output_site", "output_layer", "passes")

# The keys of an ``EngineArgs.adapter_recirc_span`` RESERVATION.  They
# are a strict SUBSET of the adapter_config keys above plus the input
# port, on purpose: a reservation is the same vocabulary with the
# weights left out, so ``plan_from_adapter_config`` validates both and
# there is exactly one place that decides what a span means.
#
#   {"site": "block_output", "layer_indices": [3],
#    "output_site": "block_input", "output_layer": 0, "passes": 2,
#    "declared_by": ["m3:...RecirculationAdapter:block_output:all"]}
#
# ``declared_by`` is diagnostic only (it names the member the builder
# reserved for, so a mismatch at fill time can say which).
SPAN_DECLARATION_KEYS = ("site", "layer_indices", "output_site",
                         "output_layer", "passes")


def config_rewires(adapter_config: Optional[dict[str, Any]]) -> bool:
    """Whether *adapter_config* declares an output port of its own.

    Mirrors ``adapters.mounting.Mount.rewires()``: EITHER an explicit
    ``output_site`` or an explicit ``output_layer`` makes the mount
    off-diagonal (a different layer at the same kind of port is still a
    rewiring).  ``passes`` alone does not — a diagonal member may loop
    internally without touching the host schedule.
    """
    if not adapter_config:
        return False
    return (adapter_config.get("output_site") is not None
            or adapter_config.get("output_layer") is not None)


@dataclass
class RecircPlan:
    """A validated, static off-diagonal route for one baked member."""

    in_layer: int
    """Layer whose port the member READS (F_in)."""
    in_site: str
    """Port kind read at ``in_layer`` — ``block_output`` on this route."""
    out_layer: int
    """Layer whose port the member WRITES (F_out); precedes ``in_layer``."""
    out_site: str
    """Port kind written — ``block_input`` or ``block_output``."""
    passes: int
    """How many extra times the span is executed (k >= 1)."""
    start: int
    """First layer of the re-executed span (resolved from the write port)."""

    @property
    def span_layers(self) -> range:
        """Decoder layer indices re-executed, inclusive of both ends."""
        return range(self.start, self.in_layer + 1)

    @property
    def span_len(self) -> int:
        return self.in_layer + 1 - self.start

    def extra_attention_layers(self, attns_per_layer: int = 1) -> int:
        """Extra KV-cache-bearing layers this plan registers."""
        return self.passes * self.span_len * attns_per_layer

    def describe(self) -> str:
        return (f"reads {self.in_site} L{self.in_layer}, writes "
                f"{self.out_site} L{self.out_layer}, {self.passes} "
                f"passes over layers {self.start}..{self.in_layer}")


def _refuse(msg: str) -> "RuntimeError":
    return RuntimeError(f"adapter recirculation route: {msg}")


def plan_from_adapter_config(
    adapter_config: Optional[dict[str, Any]],
    num_layers: int,
) -> Optional[RecircPlan]:
    """Validate a baked ``adapter_config``'s pair axis into a plan.

    Returns ``None`` for every diagonal config (``F_in == F_out``) —
    including configs from a library predating the pair axis, which
    carry none of :data:`REWIRE_KEYS`.  That is the byte-compatibility
    guarantee: an existing manifest takes exactly the code path it took
    before.

    Raises with a specific message for a rewiring config this route does
    not implement.
    """
    if not config_rewires(adapter_config):
        return None
    assert adapter_config is not None
    require_pair_ports()

    layer_indices = sorted(int(i) for i in adapter_config["layer_indices"])
    if len(layer_indices) != 1:
        raise _refuse(
            f"a rewiring member mounts at exactly ONE input layer (the "
            f"span's far end); this config declares layer_indices="
            f"{layer_indices}. Serving N independent spans from one "
            f"baked config is not implemented — each span would need its "
            f"own per-pass KV sets and its own nesting order.")
    in_layer = layer_indices[0]

    in_site = adapter_config.get("site", "block_output")
    validate_site(in_site)
    if in_site != "block_output":
        raise _refuse(
            f"input port {in_site!r} is not implemented. The baked route "
            f"applies its member at the decoder block's output, so the "
            f"span can only be re-entered from a whole-block boundary; a "
            f"read tap inside the block (post_attn/post_mlp/linear:*) "
            f"would have to resume layer {in_layer} mid-forward.")

    out_site = adapter_config.get("output_site") or in_site
    validate_write_port(out_site)
    out_layer = adapter_config.get("output_layer")
    out_layer = in_layer if out_layer is None else int(out_layer)

    if not 0 <= out_layer < num_layers:
        raise _refuse(
            f"output_layer={out_layer} is outside the host's "
            f"{num_layers} decoder layers.")
    if not 0 <= in_layer < num_layers:
        raise _refuse(
            f"input layer {in_layer} is outside the host's {num_layers} "
            f"decoder layers.")

    if port_order(out_layer, out_site) >= port_order(in_layer, in_site):
        raise _refuse(
            f"output port ({out_layer}, {out_site}) is at or AFTER the "
            f"input port ({in_layer}, {in_site}). Writing forward past "
            f"the read point is not implemented: it would SKIP the "
            f"host's computation of the layers in between rather than "
            f"re-run them. Declare an output port that precedes the "
            f"input port, or leave output_site/output_layer unset for "
            f"the ordinary same-port mount.")

    passes = adapter_config.get("passes", 1)
    if not (isinstance(passes, int) and not isinstance(passes, bool)
            and passes >= 1):
        raise _refuse(f"passes must be an int >= 1, got {passes!r}.")

    # block_input of layer i resumes AT layer i; block_output of layer i
    # resumes at the next layer.  (adapters.mounting._install_rewire)
    start = out_layer if out_site == "block_input" else out_layer + 1
    if start > in_layer:
        raise _refuse(
            f"ports ({out_layer}, {out_site}) -> ({in_layer}, {in_site}) "
            f"enclose no decoder layer to re-execute.")

    plan = RecircPlan(in_layer=in_layer, in_site=in_site,
                      out_layer=out_layer, out_site=out_site,
                      passes=passes, start=start)
    return plan


def refuse_rewired(adapter_config: Optional[dict[str, Any]],
                   route: str) -> None:
    """Refuse a rewiring member on a route that cannot execute it.

    Two routes can drive a span (:data:`REEXECUTION_ROUTES`): ``baked``,
    and ``multisite_reserved`` — a multisite load whose span was
    RESERVED on ``EngineArgs.adapter_recirc_span`` before the engine
    existed.  This function is the refusal for everything else: a
    multisite load with no reservation, and the lora_view route.  The
    reason is the same one it always was and is about TIMING, not about
    the hook: their members are loaded AFTER the engine froze its
    KV-cache spec, so the per-pass caches can no longer be created.
    """
    if not config_rewires(adapter_config):
        return
    assert adapter_config is not None
    detail = (f"reads {adapter_config.get('site', 'block_output')} "
              f"L{sorted(adapter_config.get('layer_indices', []))}, writes "
              f"{adapter_config.get('output_site')} "
              f"L{adapter_config.get('output_layer')}, "
              f"{adapter_config.get('passes', 1)} passes")
    raise _refuse(
        f"REFUSING to serve a schedule-rewiring member ({detail}) on the "
        f"{route!r} route. Only the BAKED route implements re-execution, "
        f"for one structural reason: each re-executed pass needs its OWN "
        f"KV cache for every layer in the span, and vLLM v1 collects the "
        f"KV-cache spec ONCE, at engine start, from the attention modules "
        f"in compilation_config.static_forward_context "
        f"(gpu_model_runner.get_kv_cache_spec). The {route!r} route loads "
        f"its members through collective_rpc AFTER that spec is frozen "
        f"and the cache tensors are allocated, so the per-pass caches "
        f"cannot be created; re-executing the span without them would "
        f"make every pass overwrite the host's own K/V at the same slots "
        f"and silently corrupt the generation. Two ways out, both "
        f"declared in vllm.adaptation.recirculation.REEXECUTION_ROUTES: "
        f"RESERVE the span at engine-config time by passing "
        f"EngineArgs.adapter_recirc_span "
        f"(SPAN_DECLARATION_KEYS: site, layer_indices, output_site, "
        f"output_layer, passes) to LLM(...), which registers the "
        f"per-pass caches in the same window the baked route uses and "
        f"lets this member load alongside others; or bake it into "
        f"LLM(adapter_config=...) when it is the only member.")


def find_installed_recirc(model_or_layers, in_layer: Optional[int] = None):
    """The ``_InstalledRecirc`` armed on this model, or ``None``.

    Accepts the model, or anything iterable of decoder layers.  There is
    at most one span per engine today (``plan_from_adapter_config``
    refuses more than one input layer, and only one reservation rides
    the EngineArgs), so *in_layer* is only a cross-check.
    """
    modules = (model_or_layers.modules()
               if hasattr(model_or_layers, "modules")
               else list(model_or_layers))
    for module in modules:
        installed = getattr(module, "_adapter_recirc", None)
        if installed is None:
            continue
        if in_layer is not None and installed.plan.in_layer != in_layer:
            continue
        return installed
    return None


def span_declaration_matches(plan: RecircPlan,
                             adapter_config: dict[str, Any]) -> Optional[str]:
    """``None`` if *adapter_config*'s span is the one *plan* reserved,
    else a human sentence saying which field disagrees.

    Every field is compared, not a subset: a reservation that differs in
    ANY of them reserved a different number of shadow caches, or reserved
    them for different layers, and filling it would run the span against
    caches that were budgeted for something else.
    """
    layer_indices = sorted(int(i) for i in
                           adapter_config.get("layer_indices", []))
    want = {
        "input layer": (plan.in_layer, layer_indices[0]
                        if len(layer_indices) == 1 else layer_indices),
        "input port": (plan.in_site,
                       adapter_config.get("site", "block_output")),
        "output port": (plan.out_site,
                        adapter_config.get("output_site")
                        or adapter_config.get("site", "block_output")),
        "output layer": (plan.out_layer,
                         int(adapter_config["output_layer"])
                         if adapter_config.get("output_layer") is not None
                         else (adapter_config.get("layer_indices") or
                               [plan.in_layer])[0]),
        "passes": (plan.passes, adapter_config.get("passes", 1)),
    }
    bad = [f"{k}: reserved {r!r}, member declares {g!r}"
           for k, (r, g) in want.items() if r != g]
    return "; ".join(bad) if bad else None


def fill_reserved_span(model_or_layers, adapter_int_id: int,
                       adapter_config: dict[str, Any]) -> bool:
    """Bind an arriving multisite member to the span reserved for it.

    Returns True when the span was filled.  Raises — never returns False
    quietly — when a rewiring member arrives and there is no reservation
    that fits it, because the alternative is a member that loads, is
    never driven, and produces fluent output from a computation that is
    missing its whole loop.
    """
    installed = find_installed_recirc(model_or_layers)
    detail = (f"reads {adapter_config.get('site', 'block_output')} "
              f"L{sorted(adapter_config.get('layer_indices', []))}, writes "
              f"{adapter_config.get('output_site')} "
              f"L{adapter_config.get('output_layer')}, "
              f"{adapter_config.get('passes', 1)} passes")
    if installed is None:
        raise _refuse(
            f"REFUSING to serve a schedule-rewiring member (id="
            f"{adapter_int_id}, {detail}) on the multisite route: NO SPAN "
            f"WAS RESERVED for it. Each re-executed pass needs its own KV "
            f"cache for every layer in the span, and vLLM v1 collects the "
            f"KV-cache spec ONCE, at engine start "
            f"(gpu_model_runner.get_kv_cache_spec); this RPC runs after "
            f"that spec is frozen and the tensors are allocated, so the "
            f"caches cannot be created now. Declare the span up front in "
            f"EngineArgs.adapter_recirc_span "
            f"(vllm.adaptation.recirculation.SPAN_DECLARATION_KEYS) BEFORE "
            f"LLM(...) — adapters/serving.py::build_vllm does this when "
            f"the fork declares 'multisite_reserved' in "
            f"vllm.adaptation.recirculation.REEXECUTION_ROUTES — or bake "
            f"the member into LLM(adapter_config=...).")
    if installed.adapter_int_id is not None:
        raise _refuse(
            f"REFUSING to serve a schedule-rewiring member (id="
            f"{adapter_int_id}, {detail}): the reserved span "
            f"({installed.plan.describe()}) is already driven by adapter "
            f"id={installed.adapter_int_id}. N independent spans are not "
            f"implemented — each would need its own per-pass KV sets and "
            f"a defined nesting order.")
    mismatch = span_declaration_matches(installed.plan, adapter_config)
    if mismatch:
        raise _refuse(
            f"REFUSING to serve a schedule-rewiring member (id="
            f"{adapter_int_id}, {detail}): it does not match the span "
            f"reserved at engine construction "
            f"({installed.plan.describe()}"
            + (f", declared for {list(installed.declared_by)}"
               if installed.declared_by else "") +
            f"). {mismatch}. The reservation sized and named the per-pass "
            f"KV caches; running a different span against them would "
            f"attend to keys budgeted for other layers.")
    installed.adapter_int_id = int(adapter_int_id)
    logger.info(
        "[adapter-recirc] reserved span %s FILLED by adapter id=%d; "
        "co-mounted members at the input port blend first "
        "(SPAN_COMOUNT_ORDER=%r) and members inside the span fire on "
        "all %d passes.", installed.plan.describe(), adapter_int_id,
        SPAN_COMOUNT_ORDER, installed.plan.passes + 1)
    return True


def refuse_decode_state(adapter_config: Optional[dict[str, Any]],
                        label: str = "") -> None:
    """Refuse a member whose mount asked the ENGINE to carry a State.

    ``adapters.mounting.Mount.decode_state`` is a property of the TRAINED
    function: T scans the prompt into a State and R reads that State at
    every decode step.  This engine carries no per-request adapter state
    (:data:`vllm.adaptation.protocol.CARRIES_ADAPTER_DECODE_STATE` is
    False), so such a member would run R with ``state=None`` on every
    decode step — a fresh-state singleton, a different function, and
    fluent output with no error.

    The library refuses this first (``adapters/serving.py::
    refuse_decode_state``), on the same capability name.  This is the
    re-check at the worker, for the same reason the serving-requirement
    re-check exists: a member that reaches here is past every
    library-side guard.
    """
    if not adapter_config:
        return
    if not adapter_config.get("decode_state"):
        return
    if adapter_config.get("host_reexecute"):
        # Mount.validate refuses decode_state together with
        # host_reexecute, so this cannot be a trained member; fall
        # through to the ordinary rewiring path rather than inventing a
        # second diagnosis for a config that cannot exist.
        return
    if CARRIES_ADAPTER_DECODE_STATE:  # pragma: no cover - False today
        return
    who = f" ({label})" if label else ""
    raise _refuse(
        f"REFUSING to serve a member that declares decode_state=True"
        f"{who}. Its mount asked the engine to carry a per-request State "
        f"from the prompt scan (T) into every decode step (R); this "
        f"engine does not have one and says so — "
        f"vllm.adaptation.protocol.CARRIES_ADAPTER_DECODE_STATE is "
        f"False. Serving it anyway would call readout() with state=None "
        f"at each decoded position, i.e. a fresh-state singleton rather "
        f"than the scan the member was trained as, and the output would "
        f"read fluently. Owning that state means a (max_num_seqs, ...) "
        f"slot per member, indexed by the running request's slot, "
        f"advanced exactly once per decoded position and zeroed on "
        f"request start, preemption and recompute. Express the phase "
        f"split with P-axis masks instead (a member at position="
        f"'prefill' and one at position='all' share the span's per-pass "
        f"KV, which the engine DOES carry per request from prefill into "
        f"decode), or score this member on the HF path.")


# ---------------------------------------------------------------------------
# Engine-construction side: shadow attention layers
# ---------------------------------------------------------------------------

def _attention_modules(layer: nn.Module) -> list[nn.Module]:
    """The KV-cache-bearing attention modules inside one decoder layer."""
    from vllm.attention import Attention
    return [m for m in layer.modules() if isinstance(m, Attention)]


def _shadow_name(pass_idx: int, virtual_idx: int, suffix: str) -> str:
    """Layer name for a shadow attention module.

    Constraints this satisfies, both load-bearing:

    * ``vllm.model_executor.models.utils.extract_layer_index`` (used by
      ``bind_kv_cache``) asserts the name contains EXACTLY ONE integer.
      ``pass{p}`` and the suffix contribute none.
    * that one integer must not collide with a real decoder layer's, or
      ``bind_kv_cache`` maps two names to one index and drops one from
      the runner's cache list.  ``virtual_idx`` is allocated ABOVE the
      host's layer count for exactly that reason.
    """
    return f"recirc.pass{pass_idx}.layers.{virtual_idx}.{suffix}"


def _make_shadow(attn: nn.Module, name: str,
                 pp_size: int) -> nn.Module:
    """A KV-cache twin of *attn* registered under a fresh layer name.

    A shallow copy shares ``_parameters``/``_buffers``/``_modules`` and
    ``impl`` with the original — deliberately: the shadow must produce a
    byte-identical KV-cache spec and identical attention numerics.  Only
    the two plain attributes that select WHICH cache and WHICH metadata
    the module uses are given fresh values, and both assignments land in
    the copy's own ``__dict__`` (nn.Module.__setattr__ routes non-tensor,
    non-module values there), so the original is untouched.
    """
    shadow = copy.copy(attn)
    shadow.layer_name = name
    shadow.kv_cache = [torch.tensor([]) for _ in range(pp_size)]
    return shadow


@dataclass
class _InstalledRecirc:
    """Everything the input layer needs to drive the span at run time."""

    plan: RecircPlan
    span: list[nn.Module]
    """The decoder layers to re-execute, in forward order."""
    span_attns: list[list[nn.Module]] = field(default_factory=list)
    """Per span layer, its live Attention modules."""
    shadows: dict = field(default_factory=dict)
    """``(span_pos, pass_idx) -> list[Attention]`` twins."""
    depth: int = 0
    """Re-entrancy counter: > 0 while our own passes are running."""
    adapter_int_id: Optional[int] = 1
    """The member driving the span (the baked route's id=1).

    ``None`` means the span is RESERVED and not yet filled: the shadow
    KV caches exist (they had to be declared before the spec froze) but
    no member has arrived to drive them.  An unfilled span is inert —
    :func:`run_recirculation` is never reached and the host's layers run
    exactly once — so an engine whose reserved member never loads
    degrades to the plain host, not to a silently wrong one.
    """
    reserved: bool = False
    """Whether the plan came from an ``adapter_recirc_span`` reservation
    (the multisite route) rather than from a baked ``adapter_config``."""
    declared_by: tuple = ()
    """Member labels the reservation named, for diagnostics only."""


def install_recirculation(model: nn.Module,
                          vllm_config) -> Optional[RecircPlan]:
    """Register per-pass KV caches and arm the span, once, at load time.

    Called from the model runner right after the model is loaded and
    BEFORE the worker asks for the KV-cache spec, which is the only
    window in which extra cache-bearing layers can still be declared.

    Returns the installed plan, or ``None`` when the engine carries no
    rewiring adapter (the overwhelmingly common case, and a pure no-op).
    """
    adapter_config = getattr(vllm_config, "adapter_config", None)
    span_decl = getattr(vllm_config, "adapter_recirc_span", None)
    if config_rewires(adapter_config):
        source, reserved = adapter_config, False
    elif config_rewires(span_decl):
        # The MULTISITE reservation (REEXECUTION_ROUTES's
        # "multisite_reserved"): ports and passes declared on the
        # EngineArgs at construction, so the shadow caches are registered
        # in the SAME window the baked route uses, and the member itself
        # arrives later by collective_rpc("load_adapter") and fills it.
        source, reserved = span_decl, True
    else:
        return None

    # Layers wrapped by make_adapter_decoder_layer carry their index.
    layers: dict[int, nn.Module] = {}
    for module in model.modules():
        idx = getattr(module, "_adapter_layer_idx", None)
        if idx is not None and idx >= 0 and hasattr(module,
                                                    "served_adapters"):
            layers[idx] = module
    if not layers:
        raise _refuse(
            "no adapter-wrapped decoder layers found on the loaded "
            "model, so the span cannot be re-executed. A rewiring member "
            "needs the fork's adapter decoder-layer wrapper on this "
            "architecture (vllm/adaptation/layer.py::maybe_adapter_layer_"
            "type); the model was built without it.")

    num_layers = max(layers) + 1
    plan = plan_from_adapter_config(source, num_layers)
    if plan is None:  # pragma: no cover - config_rewires already gated
        return None

    _refuse_unsupported_engine(vllm_config, plan, layers)

    span = [layers[i] for i in plan.span_layers]
    installed = _InstalledRecirc(
        plan=plan, span=span, reserved=reserved,
        declared_by=tuple(source.get("declared_by") or ()))

    ctx = vllm_config.compilation_config.static_forward_context
    pp_size = vllm_config.parallel_config.pipeline_parallel_size
    virtual_idx = num_layers
    for span_pos, layer in enumerate(span):
        attns = _attention_modules(layer)
        if not attns:
            raise _refuse(
                f"decoder layer {plan.start + span_pos} in the span has "
                f"no Attention module; this route allocates one extra KV "
                f"cache set per pass per attention layer and has nothing "
                f"to allocate for a non-attention layer (Mamba/linear "
                f"layers carry engine state this route does not "
                f"duplicate).")
        installed.span_attns.append(attns)

    for pass_idx in range(plan.passes):
        for span_pos, attns in enumerate(installed.span_attns):
            twins = []
            for attn_pos, attn in enumerate(attns):
                suffix = f"self_attn.attn{attn_pos}"
                name = _shadow_name(pass_idx, virtual_idx, suffix)
                if name in ctx:
                    raise _refuse(f"duplicate shadow layer name {name!r}")
                shadow = _make_shadow(attn, name, pp_size)
                ctx[name] = shadow
                twins.append(shadow)
            installed.shadows[(span_pos, pass_idx)] = twins
            virtual_idx += 1

    in_layer = layers[plan.in_layer]
    if reserved:
        # Nothing is mounted yet — members arrive by RPC.  The span is
        # armed but UNFILLED; fill_reserved_span() checks the arriving
        # member's ports against this plan and sets adapter_int_id.
        installed.adapter_int_id = None
    else:
        # BAKED: the one member in adapter_config is the span's driver.
        # Co-mounted members at the same port are no longer undefined —
        # SPAN_COMOUNT_ORDER declares the order ("diagonal_then_span":
        # ordinary members blend first, the span is entered with the
        # blended stream) and _multi_adapter_forward implements exactly
        # that.  What IS still undefined, and still refused, is a SECOND
        # rewiring member at the same port: two spans over one stream
        # have no defined nesting order and each would need its own
        # per-pass KV sets.
        sites = getattr(in_layer, "_adapter_adapter_sites", {}) or {}
        mounted = [i for i, site in sites.items() if site == plan.in_site]
        if not mounted:
            raise _refuse(
                f"layer {plan.in_layer} carries no member at "
                f"{plan.in_site!r}, so the baked rewiring config names a "
                f"port nothing is mounted at.")
        # The baked member is loaded as id=1 (the builtin adapter id);
        # anything else at this port arrived some other way and is a
        # co-mount, which blends first and does not drive the span.
        installed.adapter_int_id = 1 if 1 in mounted else (
            mounted[0] if len(mounted) == 1 else None)
        if installed.adapter_int_id is None:
            raise _refuse(
                f"layer {plan.in_layer} carries {len(mounted)} members at "
                f"{plan.in_site!r} ({sorted(mounted)}) and none of them "
                f"is the baked adapter id=1, so which one drives the span "
                f"is ambiguous. Co-mounted members at a span's input port "
                f"are supported (they blend first — "
                f"SPAN_COMOUNT_ORDER={SPAN_COMOUNT_ORDER!r}), but the "
                f"span's DRIVER has to be identifiable.")
    object.__setattr__(in_layer, "_adapter_recirc", installed)

    extra = sum(len(a) for a in installed.span_attns) * plan.passes
    logger.info(
        "[adapter-recirc] installed (%s): %s; registered %d extra "
        "attention layers (%d passes x %d span layers) — the KV budget "
        "is now divided over %d cache-bearing layers instead of %d, so "
        "expect max concurrency to fall by roughly that ratio.%s",
        "RESERVED for the multisite route" if reserved else "baked",
        plan.describe(), extra, plan.passes, plan.span_len,
        len(ctx), len(ctx) - extra,
        (f" Awaiting the member(s) {list(installed.declared_by)} by "
         f"load_adapter; the span is INERT until one fills it."
         if reserved else ""))
    return plan


def _refuse_unsupported_engine(vllm_config, plan: RecircPlan,
                               layers: dict) -> None:
    """Refuse engine configurations this route cannot honour."""
    parallel = vllm_config.parallel_config
    if parallel.pipeline_parallel_size > 1:
        raise _refuse(
            f"pipeline_parallel_size="
            f"{parallel.pipeline_parallel_size} is not supported. The "
            f"span {plan.start}..{plan.in_layer} is re-executed from "
            f"inside layer {plan.in_layer}'s forward, which can only "
            f"reach layers resident on the SAME rank; a span crossing a "
            f"pipeline stage boundary would need the re-entry to travel "
            f"back over the PP send/recv path. Serve with "
            f"pipeline_parallel_size=1 (tensor parallelism is fine — "
            f"every rank holds every layer).")

    missing = [i for i in plan.span_layers if i not in layers]
    if missing:
        raise _refuse(
            f"decoder layers {missing} of the span are not resident on "
            f"this rank.")

    cache_config = vllm_config.cache_config
    if getattr(cache_config, "enable_prefix_caching", False):
        raise _refuse(
            "prefix caching must be disabled. Prefix-cache blocks are "
            "keyed by a hash of the token prefix alone, which does not "
            "distinguish the host's own pass from the k re-executed "
            "passes; a block reused across requests would hand one "
            "request's per-pass K/V to another. Pass "
            "enable_prefix_caching=False (adapters/serving.py's "
            "build_vllm already defaults it off).")

    if getattr(vllm_config, "kv_transfer_config", None) is not None:
        connector = getattr(vllm_config.kv_transfer_config,
                            "kv_connector", "?")
        raise _refuse(
            f"a KV-transfer connector ({connector!r}) is configured, and "
            f"this is the sharpest refusal on the route — not a "
            f"bookkeeping gap but UNINITIALISED MEMORY.\n"
            f"A connector injects externally-computed K/V into a "
            f"request's paged cache BY LAYER NAME "
            f"(wait_for_kv_layer_from_connector / "
            f"maybe_save_kv_layer_to_connector in vllm/attention/"
            f"layer.py; PrefixInjectionConnector reads "
            f"'<store>/<layer_name>.safetensors'). The per-pass shadow "
            f"layers this route registers have names of their own "
            f"('recirc.pass{{p}}.layers.{{n}}...'), which no store "
            f"contains, so nothing is injected into them — while they "
            f"SHARE the host's block table and slot mapping. The "
            f"injected prefix positions are therefore allocated in every "
            f"pass's cache tensor and WRITTEN IN NONE of them, and the "
            f"span's queries attend causally over those positions: each "
            f"re-executed pass would read whatever those slots happen to "
            f"hold. Fluent output, uninitialised keys.\n"
            f"Injecting the same rows into each pass would not fix it "
            f"either, and that matters for the diagnosis: the HF engine "
            f"runs the re-executed passes CACHE-FREE when the host is "
            f"not caching (adapters/mounting.py::_rewire_span keys the "
            f"per-pass DynamicCache on the host's own use_cache), so a "
            f"member TRAINED with an external prefix never saw that "
            f"prefix inside its span. Prefix-seeded pass caches would be "
            f"a different function from the trained one; empty ones are "
            f"the faithful semantics and are what this route gives when "
            f"no connector is present.\n"
            f"So a member whose MEMORY arrives through a connector "
            f"cannot also re-execute a span here. Serve the span without "
            f"a connector (mount the memory as a member, or drop the "
            f"external prefix), or drop the span.")

    scheduler = vllm_config.scheduler_config
    if getattr(scheduler, "chunked_prefill_enabled", False):
        raise _refuse(
            "chunked prefill must be disabled. A prefill chunk re-enters "
            "the span with only that chunk's queries while the per-pass "
            "caches hold the previous chunks' keys, so pass p's chunk "
            "boundary sees a history assembled from a DIFFERENT pass's "
            "recirculated stream than the one that produced its queries. "
            "The engine forces this off for a rewiring adapter_config "
            "(vllm/engine/arg_utils.py::"
            "_enforce_unchunked_prefill_for_sequence_mixing); reaching "
            "this message means the config was overridden afterwards.")

    if not vllm_config.model_config.enforce_eager:
        raise _refuse(
            "eager execution is required (enforce_eager=True). The span "
            "loop is Python control flow that re-enters the decoder "
            "stack a data-independent but graph-visible number of times, "
            "and it swaps each attention module's layer_name between "
            "passes — exactly the control flow CUDA-graph capture and "
            "the general-shape compile strip. The engine forces eager "
            "for a rewiring adapter_config; reaching this message means "
            "the config was overridden afterwards.")


# ---------------------------------------------------------------------------
# Run-time side: executing the span
# ---------------------------------------------------------------------------

class _ShadowSwap:
    """Point the span's attention modules at one pass's KV caches.

    Swaps only ``layer_name`` and ``kv_cache``, which is enough for BOTH
    attention call paths:

    * ``use_direct_call`` reads ``self.kv_cache[ve]`` and
      ``attn_metadata[self.layer_name]`` off the module we swapped;
    * the custom-op path passes ``self.layer_name`` to
      ``torch.ops.vllm.unified_attention*``, which re-resolves the module
      out of ``forward_context.no_compile_layers`` — the very dict the
      shadows were registered in — and so reads the SHADOW's kv_cache.

    Restoring in ``finally`` matters: an exception mid-span would
    otherwise leave the host's own layers pointing at a pass cache.
    """

    def __init__(self, installed: "_InstalledRecirc", pass_idx: int):
        self._pairs = []
        for span_pos, attns in enumerate(installed.span_attns):
            twins = installed.shadows[(span_pos, pass_idx)]
            for attn, shadow in zip(attns, twins):
                self._pairs.append((attn, shadow))

    def __enter__(self):
        self._saved = [(a, a.layer_name, a.kv_cache) for a, _ in self._pairs]
        for attn, shadow in self._pairs:
            attn.layer_name = shadow.layer_name
            attn.kv_cache = shadow.kv_cache
        return self

    def __exit__(self, *exc):
        for attn, name, cache in self._saved:
            attn.layer_name = name
            attn.kv_cache = cache
        return False


def _run_span_once(installed: "_InstalledRecirc", positions, stream,
                   layer_kwargs: dict):
    """One pass of the span, threading the (hidden, residual) contract.

    *stream* is COPIED before the span touches it.  A real decoder layer
    keeps no promise about the tensor it is handed: llama's aliases it as
    ``residual`` when it is called with ``residual=None`` (which is how a
    pass starts), and every ``RMSNorm(x, residual)`` after that is
    ``ops.fused_add_rms_norm``, which writes the running residual sum
    back into that very tensor.  The caller's ``stream`` is the host
    layer's ``h_full`` — the value the member's ``write`` must see as
    ``h_out`` and the value the layer re-bases its deferred residual on — so
    letting the span write through it silently replaces the host's own
    block output with the span's first intermediate, and a gate of 0
    stops being a no-op.  One (num_tokens, dim) copy per pass, against
    ``span_len`` layer executions.
    """
    hidden, residual = stream.clone(), None
    for layer in installed.span:
        if getattr(type(layer), "_adapter_has_residual_arg", True):
            hidden, residual = layer(positions, hidden, residual,
                                     **layer_kwargs)
        else:
            # Residual-free contract (olmo2-style): the layer carries the
            # whole stream in `hidden`.
            hidden = layer(positions, hidden, **layer_kwargs)
            residual = None
    return hidden if residual is None else hidden + residual


def run_recirculation(layer: nn.Module, positions, stream,
                      adapter: Optional[nn.Module],
                      mask, layer_kwargs: Optional[dict] = None):
    """Re-execute the span and fold the result back into *stream*.

    Args:
        layer: the input-port decoder layer (the span's far end).
        positions: the host's position tensor — handed to every pass
            UNCHANGED, so the re-executed tokens keep their rotary
            phase and their place in the causal mask.
        stream: the residual stream at the input port,
            ``(num_tokens, dim)``.
        adapter: the mounted member.  Its ``readout`` (identity for a
            pipe) is applied on the way in, producing the payload the
            span re-executes; its ``write(h_out, payload)`` mixes the
            result back at the write port — the gated pipe's
            ``fx + g*(piped - fx)``, exactly ``fx`` at ``g == 0``.
        mask: the fork's per-token combined phase/membership mask, or
            ``None`` for all tokens.  The span always runs on the whole
            batch (it is a schedule, not a per-token computation); the
            mask selects which tokens KEEP the recirculated value.
    """
    installed: Optional[_InstalledRecirc] = getattr(layer, "_adapter_recirc",
                                                    None)
    if installed is None:
        return stream
    if installed.depth:
        # Our own passes are running: the rewiring member is a
        # pass-through inside its own span, or the span would recurse.
        return stream

    entry = stream
    if adapter is not None:
        readout = getattr(adapter, "readout", None)
        if readout is not None:
            # R runs at the member's CARRIER dtype and the payload comes
            # back in the stream's — the same two port casts the diagonal
            # route applies (protocol.readout_at_port).  What re-enters
            # the span has to be the stream's dtype: the host's own
            # layers are the model's.
            carrier = carrier_dtype_of(adapter)
            h_in = stream if (carrier is None
                              or carrier is stream.dtype) \
                else stream.to(carrier)
            entry = readout(h_in.unsqueeze(0)).squeeze(0).to(stream.dtype)

    layer_kwargs = layer_kwargs or {}
    piped = entry
    installed.depth += 1
    try:
        for pass_idx in range(installed.plan.passes):
            with _ShadowSwap(installed, pass_idx):
                piped = _run_span_once(installed, positions, piped,
                                       layer_kwargs)
    finally:
        installed.depth -= 1

    piped = piped.to(stream.dtype)
    # The write port.  *stream* is the host's own value there (the
    # PRE-pipe block output, which is why _run_span_once copies before
    # the span can alias it) and *piped* is the member's payload, so the
    # recombination is exactly W_phi: h_out <- write(h_out, payload).
    # The gated pipe's InterpolateWrite makes that fx + g*(piped - fx),
    # bit-exact fx at g == 0; the bare pipe's default W returns the
    # payload, so an ungated pipe still hands on the re-executed stream
    # unchanged.  (This used to call a private `recombine(fx, piped)`
    # hook on the leaf; that hook is retired and apply_write refuses a
    # leaf that still carries it rather than silently dropping its gate.)
    out = apply_write(adapter, stream, piped) if adapter is not None \
        else piped
    out = out.to(stream.dtype)

    if mask is None:
        return out
    return stream + (out - stream) * mask.unsqueeze(-1).to(out.dtype)
