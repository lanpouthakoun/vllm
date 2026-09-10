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

from vllm.adaptation.protocol import (MULTISITE_HONOURS_UNCHUNKED_PREFILL,
                                      SUPPORTED_WRITE_LABELS, apply_write,
                                      port_order, require_pair_ports,
                                      validate_site, validate_write_port)

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
    "RecircPlan",
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

    Only the BAKED route re-executes decoder layers.  The other two
    routes reach the adaptation through a per-layer hook with no layer
    list and no KV allocation of their own, and — decisively — their
    members are loaded AFTER the engine froze its KV-cache spec, so the
    per-pass caches can no longer be created.
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
        f"and silently corrupt the generation. Serve this member by "
        f"baking it into LLM(adapter_config=...) — the single-member "
        f"path adapters/serving.py already takes when the stream has one "
        f"block_output member.")


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
    adapter_int_id: int = 1
    """The member driving the span (the baked route's id=1)."""


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
    if not config_rewires(adapter_config):
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
    plan = plan_from_adapter_config(adapter_config, num_layers)
    if plan is None:  # pragma: no cover - config_rewires already gated
        return None

    _refuse_unsupported_engine(vllm_config, plan, layers)

    span = [layers[i] for i in plan.span_layers]
    installed = _InstalledRecirc(plan=plan, span=span)

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
    sites = getattr(in_layer, "_adapter_adapter_sites", {}) or {}
    mounted = [i for i, site in sites.items() if site == plan.in_site]
    if len(mounted) != 1:
        raise _refuse(
            f"layer {plan.in_layer} carries {len(mounted)} members at "
            f"{plan.in_site!r} ({sorted(mounted)}); a rewiring member "
            f"must be the only one at its input port, because the span "
            f"re-execution replaces the whole stream at that port and "
            f"the composition order with a co-mounted member is "
            f"undefined.")
    installed.adapter_int_id = mounted[0]
    object.__setattr__(in_layer, "_adapter_recirc", installed)

    extra = sum(len(a) for a in installed.span_attns) * plan.passes
    logger.info(
        "[adapter-recirc] installed: %s; registered %d extra attention "
        "layers (%d passes x %d span layers) — the KV budget is now "
        "divided over %d cache-bearing layers instead of %d, so expect "
        "max concurrency to fall by roughly that ratio.",
        plan.describe(), extra, plan.passes, plan.span_len,
        len(ctx), len(ctx) - extra)
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
        raise _refuse(
            "a KV-transfer connector is configured. The connector is "
            "driven by layer name (wait_for_kv_layer_from_connector / "
            "maybe_save_kv_layer_to_connector in vllm/attention/"
            "layer.py) and knows nothing of the shadow layer names this "
            "route registers, so the per-pass caches would be neither "
            "waited on nor saved.")

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
            entry = readout(stream.unsqueeze(0)).squeeze(0)

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
