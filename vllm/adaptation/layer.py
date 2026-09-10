"""vllm.adaptation.layer – CUDA-graph-compatible adapter decoder layer factories.

Overview
--------
For GRPO training we need vLLM to apply the adapter readout during
inference (prefill) while staying on-policy with the HF training model.

This module provides two things:

  1. Graph-safe cache helpers (_install_R_cache, _install_pinv_cache) –
     identical logic to the ones in the repo's local vllm_config/, but now
     living inside the fork so callers don't need to import project internals.

  2. Layer factory functions (make_adapter_qwen2_layer, make_adapter_llama_layer) –
     return nn.Module *classes* (not instances) that subclass the standard
     vLLM decoder layer and run the adapter readout in their forward pass.

The factories are called by Qwen2ForCausalLM / LlamaForCausalLM when a
``_adapter_spec`` is present (see vllm.adaptation.specs for the thread-local API).

CUDA-graph safety
-----------------
The forward logic uses only tensor ops; there is no Python branching on live
tensor values.  The adapter is stored as a plain Python attribute (bypassing
nn.Module) so vLLM's weight loader does not raise on unknown parameters.

Graph-safe caches
-----------------
  _R_cache      – precomputed Householder product (avoids 48 decompositions
                  per prefill for 24-layer models)
  _w2_pinv_cache – precomputed pseudo-inverse (avoids SVD inside graph)

Both are registered as nn.Buffers on the adapter instance so they stay on
the correct device after .to() calls.
"""

import copy
import inspect
import logging
import math
import os
import re
import types
from typing import Optional, Sequence

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
_adapter_mask_debug_counts: dict[tuple[str, int, str, str], int] = {}
_adapter_init_debug_count = 0
_adapter_nan_check: Optional[bool] = None


def _nan_check_enabled() -> bool:
    """Return True if VLLM_ADAPTER_NAN_CHECK=1 is set (cached after first call)."""
    global _adapter_nan_check
    if _adapter_nan_check is None:
        _adapter_nan_check = os.environ.get(
            "VLLM_ADAPTER_NAN_CHECK", "").lower() in {"1", "true", "yes"}
    return _adapter_nan_check


def _readout_correction(adapter, h2d: torch.Tensor) -> torch.Tensor:
    """R_phi spoken natively (v3 contract): a member's correction is
    readout(h) - h. Phase masking blends h + mask * (R(h) - h) — exact
    for additive members and the correct phase-gated blend for any R
    (multiplicative readouts included). The additive _compute_delta
    dialect is gone; leaves need only T/R."""
    return (adapter.readout(h2d.unsqueeze(0)) - h2d.unsqueeze(0)).squeeze(0)


def _check_nan(tensor: torch.Tensor, label: str, layer_idx: int) -> bool:
    """If tensor has NaN/Inf, log details and return True.

    Only called from the custom op implementation which runs eagerly
    (outside Dynamo / CUDA graphs), so .item() is safe.
    """
    if not _nan_check_enabled():
        return False
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    if has_nan or has_inf:
        num_nan = torch.isnan(tensor).sum().item()
        num_inf = torch.isinf(tensor).sum().item()
        total = tensor.numel()
        logger.error(
            "[adapter NaN check] layer=%d label=%s has_nan=%s has_inf=%s "
            "num_nan=%d num_inf=%d total=%d shape=%s dtype=%s "
            "abs_max=%.6g abs_mean=%.6g",
            layer_idx, label, has_nan, has_inf,
            num_nan, num_inf, total, list(tensor.shape), tensor.dtype,
            tensor[~torch.isnan(tensor) & ~torch.isinf(tensor)].abs().max().item()
            if (total - num_nan - num_inf) > 0 else float('nan'),
            tensor[~torch.isnan(tensor) & ~torch.isinf(tensor)].abs().mean().item()
            if (total - num_nan - num_inf) > 0 else float('nan'),
        )
        return True
    return False


def _adapter_mask_debug_enabled() -> bool:
    return os.environ.get("VLLM_ADAPTER_DEBUG_MASK", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _adapter_mask_debug_limit() -> int:
    return int(os.environ.get("VLLM_ADAPTER_DEBUG_LIMIT", "12"))


def _maybe_log_adapter_layer_init(
    *,
    arch: str,
    layer_idx: int,
    attached: bool,
    debug_enabled: bool,
    position: str,
    adapter: Optional[nn.Module],
) -> None:
    """Emit a small construction-time log for adapter-aware decoder layers."""
    global _adapter_init_debug_count

    if not (debug_enabled or _adapter_mask_debug_enabled()):
        return

    limit = int(os.environ.get("VLLM_ADAPTER_DEBUG_INIT_LIMIT", "64"))
    if _adapter_init_debug_count >= limit:
        return
    _adapter_init_debug_count += 1

    adapter_type = type(adapter).__name__ if adapter is not None else None
    logger.info(
        "[adapter-vLLM init debug] arch=%s layer=%d attached=%s debug_enabled=%s "
        "position=%s adapter_type=%s",
        arch,
        layer_idx,
        attached,
        debug_enabled,
        position,
        adapter_type,
    )

# ---------------------------------------------------------------------------
# Weight verification helper (prints to stderr for worker visibility)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Graph-safe cache helpers (mirrors vllm_config/vllm_adapter_layer.py)
# ---------------------------------------------------------------------------

def _install_adapter_caches(adapter: nn.Module,
                            model_dtype: torch.dtype = torch.bfloat16) -> None:
    """Install inference caches on an adapter via the standard protocol.

    Calls ``adapter.install_inference_caches()`` if the method exists.
    Each adapter class is responsible for defining what needs to be cached
    (e.g. ``_R_cache`` for the rotation matrix, ``_w2_pinv_cache`` for
    pseudo-inverses, ``_w2_ridge_cache`` for regularised solves, etc.).
    vLLM never needs to know about specific adapter formulas.
    """
    if hasattr(adapter, "install_inference_caches"):
        adapter.install_inference_caches(model_dtype=model_dtype)


# ---------------------------------------------------------------------------
# Layer index extraction from vLLM prefix string
# ---------------------------------------------------------------------------

def _extract_layer_idx(prefix: str) -> Optional[int]:
    """Extract the integer layer index from a vLLM prefix like 'model.layers.7'."""
    if not prefix:
        return None
    m = re.search(r"\.layers\.(\d+)", prefix)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Position masking (CUDA-graph-safe)
# ---------------------------------------------------------------------------

def _resolve_attn_metadata(attn_metadata):
    """Resolve attn_metadata to a single object.

    In vLLM V1, ForwardContext.attn_metadata is a dict mapping layer names
    to per-layer metadata objects.  All layers within a KV-cache group share
    the same metadata, so we just grab the first value.
    """
    if isinstance(attn_metadata, dict):
        for v in attn_metadata.values():
            return v
    return attn_metadata


def _get_prefill_info(attn_metadata) -> tuple[
    Optional[int], int, int, Optional[torch.Tensor], Optional[torch.Tensor]
]:
    """Extract prefill/decode split info from any V1 attention metadata.

    Returns (num_prefill_tokens, num_decodes, num_prefills,
             query_start_loc, seq_lens).

    Works with both FlashInfer (has explicit num_prefill_tokens) and
    Flash Attention (only has query_start_loc, need to derive the split).
    """
    # --- Try explicit fields first (FlashInfer) ---
    num_prefill_tokens = getattr(attn_metadata, "num_prefill_tokens", None)
    num_decodes = getattr(attn_metadata, "num_decodes", None)
    num_prefills = getattr(attn_metadata, "num_prefills", None)
    query_start_loc = getattr(attn_metadata, "query_start_loc", None)
    seq_lens = getattr(attn_metadata, "seq_lens", None)

    if num_prefill_tokens is not None:
        # FlashInfer or similar backend with explicit split info
        if num_decodes is None:
            num_decodes = 0
        if num_prefills is None:
            num_prefills = 0
        return (int(num_prefill_tokens), int(num_decodes), int(num_prefills),
                query_start_loc, seq_lens)

    # --- Derive from query_start_loc (Flash Attention and others) ---
    if query_start_loc is not None and query_start_loc.numel() > 1:
        # query_start_loc: (num_reqs + 1,) cumulative query token counts.
        # In V1, decode requests (query_len=1) come FIRST, then prefill
        # requests (query_len>1).
        query_lens = query_start_loc[1:] - query_start_loc[:-1]  # (num_reqs,)
        is_prefill = (query_lens > 1)
        num_reqs = query_lens.numel()

        # Since decodes come first and prefills last, find the split point.
        # The first prefill request is where query_len > 1 first appears.
        if is_prefill.any():
            # All decode requests have query_len=1 and come first.
            first_prefill_idx = int(is_prefill.to(torch.int32).argmax().item())
            n_decodes = first_prefill_idx
            n_prefills = num_reqs - n_decodes
            n_prefill_tokens = int(
                (query_start_loc[-1] - query_start_loc[n_decodes]).item())
        else:
            # All decode, no prefill
            n_decodes = num_reqs
            n_prefills = 0
            n_prefill_tokens = 0

        return (n_prefill_tokens, n_decodes, n_prefills,
                query_start_loc, seq_lens)

    # No usable metadata
    return (None, 0, 0, None, seq_lens)


# _prefill_request_indices moved to vllm.adaptation.positions; re-exported
# here for backward compatibility.
from vllm.adaptation.positions import _prefill_request_indices  # noqa: E402,F401
from vllm.adaptation.positions import (PhaseInfo,  # noqa: E402
                                       get_position_mask,
                                       position_active_in_decode)
from vllm.adaptation.protocol import (apply_adaptation,  # noqa: E402
                                      check_adaptation_supported,
                                      needs_sequence_segmentation,
                                      resolve_site_submodule_path,
                                      validate_site)


def _compute_position_mask(
    positions: torch.Tensor,
    position: str,
    dtype: torch.dtype,
    num_tokens: int,
    attn_metadata,
    prompt_lens: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Return the per-token mask for *position*, or ``None`` for all-tokens.

    Extracts the batch's prefill/decode split from *attn_metadata*
    (handles both vLLM V0 single objects and V1 dicts, any attention
    backend) and dispatches to the named position registry in
    :mod:`vllm.adaptation.positions`.  Builtins: "all", "all_tokens",
    "prefill", "decode", "first", "last", plus the parameterized
    families (``every_<k>``, ``every_<k>_decode``, ``first_<k>_decode``,
    ``decode_range_<a>_<b>``); custom positions are added via
    :func:`vllm.adaptation.register_position_mask`.

    ``prompt_lens`` (per-request, batch order) enables the
    generation-anchored families; without it those masks are inert.
    """
    attn_metadata = _resolve_attn_metadata(attn_metadata)
    (num_prefill_tokens, num_decodes, num_prefills,
     query_start_loc, seq_lens) = _get_prefill_info(attn_metadata)
    phase = PhaseInfo(
        num_prefill_tokens=num_prefill_tokens,
        num_decodes=num_decodes,
        num_prefills=num_prefills,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        prompt_lens=prompt_lens,
    )
    return get_position_mask(position, positions, dtype, num_tokens, phase)


def _apply_position_mask(
    correction: torch.Tensor,
    positions: torch.Tensor,
    position: str,
    dtype: torch.dtype,
    num_tokens: int,
    attn_metadata,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Mask a readout displacement; blending downstream is lerp(h, R(h), mask)."""
    mask = _compute_position_mask(positions, position, dtype, num_tokens, attn_metadata)
    if mask is None:
        return correction, None
    return (correction * mask.unsqueeze(-1)).contiguous(), mask


# ---------------------------------------------------------------------------
# Adapter registry & custom op (Dynamo splitting op)
# ---------------------------------------------------------------------------
# The custom op wraps both the adapter readout (R_phi) AND position masking
# in a single cudagraph_unsafe boundary.  This prevents CUDA graph capture
# from baking in stale adapter weights — after sync_weights() updates the
# adapter parameters, the custom op always uses the current weights.

_LEGACY_ADAPTER_REGISTRY: dict[int, nn.Module] = {}


def _adapter_apply_adapter_op(
    h_full: torch.Tensor,
    positions: torch.Tensor,
    position: str,
    num_tokens: int,
    layer_idx: int,
) -> torch.Tensor:
    """Readout displacement R(h)-h with position mask (in-process GRPO path)."""
    # CUDA graph capture is decode-only; adapter is a no-op for decode tokens.
    if torch.cuda.is_current_stream_capturing():
        return torch.zeros_like(h_full)

    adapter = _LEGACY_ADAPTER_REGISTRY.get(layer_idx)
    if adapter is None:
        return torch.zeros_like(h_full)

    _check_nan(h_full, "h_full_input", layer_idx)

    from vllm.forward_context import get_forward_context
    attn_metadata = get_forward_context().attn_metadata
    resolved = _resolve_attn_metadata(attn_metadata)
    (num_prefill_tokens, num_decodes, num_prefills,
     query_start_loc, seq_lens) = _get_prefill_info(resolved)

    # Pure decode batch: skip prefill-flavored adapters entirely
    # ("all" and "decode" adapters still apply to decode tokens).
    if (position not in ("all", "decode")
            and num_prefill_tokens is not None
            and num_prefill_tokens == 0):
        return torch.zeros_like(h_full)

    # Mixed batch: only run prefill-flavored adapters on prefill tokens
    # (at end of batch).  "decode" falls through to the full-batch path
    # below, where its position mask restricts it to the decode region.
    if (position not in ("all", "decode")
            and num_prefill_tokens is not None
            and 0 < num_prefill_tokens < num_tokens):
        num_decode_tokens = num_tokens - num_prefill_tokens
        h_prefill = h_full[num_decode_tokens:]
        _check_nan(h_prefill, "h_prefill_input", layer_idx)
        correction_prefill = _readout_correction(adapter, h_prefill)
        _check_nan(correction_prefill, "correction_after_readout(mixed)", layer_idx)

        if position == "prefill":
            correction = torch.zeros_like(h_full)
            correction[num_decode_tokens:] = correction_prefill
            return correction

        # first/last: apply mask within the prefill region.
        mask = _compute_position_mask(
            positions, position, h_full.dtype, num_tokens, attn_metadata)
        if mask is not None:
            correction_prefill = (
                correction_prefill * mask[num_decode_tokens:].unsqueeze(-1)
            ).contiguous()
        correction = torch.zeros_like(h_full)
        correction[num_decode_tokens:] = correction_prefill
        _check_nan(correction, "delta_after_mask(mixed)", layer_idx)
        return correction

    # Full batch (all prefill, or position="all").
    correction = _readout_correction(adapter, h_full)
    _check_nan(correction, "correction_after_readout(full)", layer_idx)
    mask = _compute_position_mask(
        positions, position, h_full.dtype, num_tokens, attn_metadata)
    if mask is not None:
        correction = (correction * mask.unsqueeze(-1)).contiguous()
    _check_nan(correction, "delta_final(full)", layer_idx)
    return correction


def _adapter_apply_adapter_fake(
    h_full: torch.Tensor,
    positions: torch.Tensor,
    position: str,
    num_tokens: int,
    layer_idx: int,
) -> torch.Tensor:
    return torch.empty_like(h_full).contiguous()


# NOTE: The adapter_apply_adapter custom op was removed.  Position masking
# now uses a pre-computed buffer updated by the model runner before each
# forward pass.  The adapter's readout runs inline in the compiled
# forward — no splitting ops, no graph breaks.
_LEGACY_CUSTOM_OP_AVAILABLE = False


def _maybe_log_mask_debug(
    *,
    arch: str,
    layer_idx: int,
    position: str,
    positions: torch.Tensor,
    mask: Optional[torch.Tensor],
    attn_metadata,
) -> None:
    """Log how many tokens receive a nonzero adapter mask in eager mode."""
    if not _adapter_mask_debug_enabled():
        return
    if hasattr(torch, "compiler") and torch.compiler.is_compiling():
        return

    num_tokens = int(positions.numel())
    if num_tokens == 0:
        return

    resolved_meta = _resolve_attn_metadata(attn_metadata)
    npt, _, _, _, _ = _get_prefill_info(resolved_meta)
    if npt is None:
        first_is_prefill = bool((positions[:1] == 0).all().item())
        prefill_tokens = num_tokens if first_is_prefill else 0
    else:
        prefill_tokens = int(npt)
    decode_tokens = max(0, num_tokens - prefill_tokens)

    if mask is None:
        masked_tokens = num_tokens
    else:
        masked_tokens = int((mask.detach().float() > 0).sum().item())

    if prefill_tokens > 0 and decode_tokens > 0:
        phase = "mixed"
    elif prefill_tokens > 0:
        phase = "prefill"
    else:
        phase = "decode"

    key = (arch, layer_idx, position, phase)
    count = _adapter_mask_debug_counts.get(key, 0)
    if count >= _adapter_mask_debug_limit():
        return
    _adapter_mask_debug_counts[key] = count + 1

    positions_head = positions[: min(8, num_tokens)].detach().cpu().tolist()
    logger.info(
        "[adapter-vLLM mask debug] arch=%s layer=%d position=%s phase=%s "
        "masked_tokens=%d total_tokens=%d prefill_tokens=%d decode_tokens=%d "
        "positions_head=%s",
        arch,
        layer_idx,
        position,
        phase,
        masked_tokens,
        num_tokens,
        prefill_tokens,
        decode_tokens,
        positions_head,
    )


_CAPTURE_MAX_TOKENS = 2048  # pre-allocated buffer size
_MASK_FALLBACK_SIZE = 131072  # fallback if model runner doesn't init


def _init_adapter_position_mask_buffer(module: nn.Module,
                                    max_tokens: int,
                                    device: torch.device) -> None:
    """Pre-allocate the position mask buffer used by the compiled forward.

    The buffer lives at a fixed address so CUDA graph replays read updated
    values after .copy_() / .fill_() calls.  The model runner updates it
    before each forward pass.
    """
    if hasattr(module, "_adapter_position_mask"):
        return
    module.register_buffer(
        "_adapter_position_mask",
        torch.zeros(max_tokens, dtype=torch.float32, device=device),
        persistent=False,
    )


def update_single_adapter_position_masks(
    adapter_layers: list[nn.Module],
    positions: torch.Tensor,
    attn_metadata,
    num_tokens: int,
) -> None:
    """Compute and store adapter position masks on each layer's buffer.

    Called from the model runner (not compiled) before each model forward.
    Uses the existing ``_compute_position_mask`` logic which handles all
    attention backends and chunked prefill correctly.

    Groups layers by position so each unique mask is computed only once
    (avoids 23 redundant GPU→CPU syncs for a 24-layer model).
    """
    if not adapter_layers:
        return

    # Group layers by position string.
    by_position: dict[str, list[nn.Module]] = {}
    for layer in adapter_layers:
        pos = getattr(layer, "_adapter_position", "prefill")
        by_position.setdefault(pos, []).append(layer)

    for pos, layers in by_position.items():
        if pos in ("all", "all_tokens"):
            mask = None
        else:
            mask = _compute_position_mask(
                positions, pos, torch.float32, num_tokens, attn_metadata)

        for layer in layers:
            buf = getattr(layer, "_adapter_position_mask", None)
            if buf is None:
                continue
            if mask is None:
                buf[:num_tokens].fill_(1.0)
            else:
                buf[:num_tokens].copy_(mask)
            # Zero the tail so stale data from a larger prior batch
            # doesn't leak if something indexes past num_tokens.
            buf[num_tokens:].zero_()


def _init_adapter_capture_buffers(module: nn.Module, hidden_size: int,
                               device: torch.device) -> None:
    """Pre-allocate buffers for per-token h_full/correction capture.

    Buffers are (max_tokens, hidden_size) — we always write to the full
    buffer and track the actual token count separately.  All operations
    are pure tensor ops, fully compatible with TorchDynamo/CUDA graphs.
    """
    if hasattr(module, "_adapter_cap_h"):
        return
    M = _CAPTURE_MAX_TOKENS
    H = hidden_size
    # Full per-token captures
    module.register_buffer(
        "_adapter_cap_h", torch.zeros(M, H, dtype=torch.float32,
                                   device=device), persistent=False)
    module.register_buffer(
        "_adapter_cap_delta", torch.zeros(M, H, dtype=torch.float32,
                                       device=device), persistent=False)
    # Per-token norms (1-D, one per token position)
    module.register_buffer(
        "_adapter_cap_h_norms", torch.zeros(M, dtype=torch.float32,
                                         device=device), persistent=False)
    module.register_buffer(
        "_adapter_cap_delta_norms", torch.zeros(M, dtype=torch.float32,
                                             device=device), persistent=False)
    # Scalar aggregates
    module.register_buffer(
        "_adapter_cap_num_tokens", torch.zeros((), dtype=torch.int64,
                                            device=device), persistent=False)
    module.register_buffer(
        "_adapter_cap_delta_abs_max", torch.zeros((), dtype=torch.float32,
                                               device=device), persistent=False)
    module.register_buffer(
        "_adapter_cap_h_mean_norm", torch.zeros((), dtype=torch.float32,
                                             device=device), persistent=False)
    module.register_buffer(
        "_adapter_cap_delta_mean_norm", torch.zeros((), dtype=torch.float32,
                                                 device=device), persistent=False)


def _capture_last_token(
    module: nn.Module,
    *,
    h_full: torch.Tensor,
    correction: torch.Tensor,
    mask: Optional[torch.Tensor],  # noqa: ARG001 — kept for call-site compat
) -> None:
    """Capture all tokens' h_full/correction into pre-allocated buffers.

    Pure tensor ops only — no data-dependent indexing, no Python scalars
    from tensor values. Fully compatible with TorchDynamo and CUDA graphs.

    The buffers are fixed-size (CAPTURE_MAX_TOKENS x hidden_size).
    We write min(num_tokens, max) into the buffer and zero the rest.
    """
    if not hasattr(module, "_adapter_cap_h"):
        return

    M = _CAPTURE_MAX_TOKENS
    n = h_full.shape[0]  # Python int from .shape — Dynamo-safe
    write_n = min(n, M)

    # Zero out then copy with implicit dtype cast — avoids allocating
    # temporary float32 tensors which break CUDA graph capture.
    module._adapter_cap_h.zero_()
    module._adapter_cap_delta.zero_()
    module._adapter_cap_h_norms.zero_()
    module._adapter_cap_delta_norms.zero_()

    module._adapter_cap_h[:write_n].copy_(h_full[:write_n].detach())
    module._adapter_cap_delta[:write_n].copy_(correction[:write_n].detach())
    module._adapter_cap_h_norms[:write_n].copy_(
        module._adapter_cap_h[:write_n].norm(dim=-1))
    module._adapter_cap_delta_norms[:write_n].copy_(
        module._adapter_cap_delta[:write_n].norm(dim=-1))

    # Store token count: zero then add n as a tensor to avoid SymInt in .fill_()
    module._adapter_cap_num_tokens.zero_()
    module._adapter_cap_num_tokens.add_(write_n)
    module._adapter_cap_delta_abs_max.copy_(
        module._adapter_cap_delta[:write_n].abs().amax())
    module._adapter_cap_h_mean_norm.copy_(
        module._adapter_cap_h[:write_n].norm(dim=-1).mean())
    module._adapter_cap_delta_mean_norm.copy_(
        module._adapter_cap_delta[:write_n].norm(dim=-1).mean())


def _init_adapter_debug_buffers(module: nn.Module, device: torch.device) -> None:
    """Initialize non-persistent buffers used for compiled-path mask stats."""
    if hasattr(module, "_adapter_debug_total_tokens"):
        return

    def _zero() -> torch.Tensor:
        return torch.zeros((), dtype=torch.int64, device=device)

    def _zero_f() -> torch.Tensor:
        return torch.zeros((), dtype=torch.float32, device=device)

    module.register_buffer("_adapter_debug_total_tokens", _zero(), persistent=False)
    module.register_buffer("_adapter_debug_masked_tokens", _zero(), persistent=False)
    module.register_buffer("_adapter_debug_prefill_tokens", _zero(), persistent=False)
    module.register_buffer("_adapter_debug_decode_tokens", _zero(), persistent=False)
    module.register_buffer("_adapter_debug_prefill_calls", _zero(), persistent=False)
    module.register_buffer("_adapter_debug_decode_calls", _zero(), persistent=False)
    module.register_buffer("_adapter_debug_mixed_calls", _zero(), persistent=False)
    module.register_buffer("_adapter_debug_delta_l2_sum", _zero_f(), persistent=False)
    module.register_buffer("_adapter_debug_delta_abs_sum", _zero_f(), persistent=False)
    module.register_buffer("_adapter_debug_delta_abs_max", _zero_f(), persistent=False)
    module.register_buffer("_adapter_debug_hidden_l2_sum", _zero_f(), persistent=False)


def _record_mask_debug_stats(
    module: nn.Module,
    *,
    correction: torch.Tensor,
    hidden_states: torch.Tensor,
    mask: Optional[torch.Tensor],
    positions: torch.Tensor,
    attn_metadata,
) -> None:
    """Accumulate compiled-path-safe adapter mask stats into layer buffers."""
    debug_enabled = getattr(module, "_adapter_debug_enabled", False) or _adapter_mask_debug_enabled()
    if not debug_enabled or not hasattr(module, "_adapter_debug_total_tokens"):
        return

    num_tokens_int = positions.shape[0]
    if num_tokens_int == 0:
        return
    stats_device = module._adapter_debug_total_tokens.device
    num_tokens = torch.tensor(num_tokens_int, dtype=torch.int64, device=stats_device)

    if mask is None:
        masked_tokens = num_tokens
    else:
        masked_tokens = (mask.to(device=stats_device) > 0).to(torch.int64).sum()

    resolved_meta = _resolve_attn_metadata(attn_metadata)
    npt, _, _, _, _ = _get_prefill_info(resolved_meta)
    if npt is not None:
        prefill_tokens = torch.tensor(int(npt), dtype=torch.int64, device=stats_device)
        prefill_tokens = torch.minimum(prefill_tokens, num_tokens)
    else:
        first_is_prefill = (positions[:1].to(device=stats_device) == 0).all()
        prefill_tokens = torch.where(first_is_prefill, num_tokens, torch.zeros_like(num_tokens))

    decode_tokens = num_tokens - prefill_tokens
    one = torch.ones((), dtype=torch.int64, device=stats_device)
    zero = torch.zeros((), dtype=torch.int64, device=stats_device)

    module._adapter_debug_total_tokens.add_(num_tokens)
    module._adapter_debug_masked_tokens.add_(masked_tokens)
    module._adapter_debug_prefill_tokens.add_(prefill_tokens)
    module._adapter_debug_decode_tokens.add_(decode_tokens)
    module._adapter_debug_prefill_calls.add_(
        torch.where((prefill_tokens > 0) & (decode_tokens == 0), one, zero)
    )
    module._adapter_debug_decode_calls.add_(
        torch.where((prefill_tokens == 0) & (decode_tokens > 0), one, zero)
    )
    module._adapter_debug_mixed_calls.add_(
        torch.where((prefill_tokens > 0) & (decode_tokens > 0), one, zero)
    )

    delta_f = correction.detach().to(device=stats_device, dtype=torch.float32)
    hidden_f = hidden_states.detach().to(device=stats_device, dtype=torch.float32)
    module._adapter_debug_delta_l2_sum.add_(delta_f.norm(dim=-1).sum())
    module._adapter_debug_delta_abs_sum.add_(delta_f.abs().sum())
    module._adapter_debug_delta_abs_max.copy_(
        torch.maximum(module._adapter_debug_delta_abs_max, delta_f.abs().amax())
    )
    module._adapter_debug_hidden_l2_sum.add_(hidden_f.norm(dim=-1).sum())


def _collect_layer_adapter_debug_stats(layer: nn.Module) -> Optional[dict]:
    """Return serializable adapter debug stats for one layer."""
    total_tokens = getattr(layer, "_adapter_debug_total_tokens", None)
    if total_tokens is None:
        return None
    total = int(total_tokens.detach().cpu().item())

    stats = {
        "layer_idx": int(getattr(layer, "_adapter_layer_idx", -1)),
        "has_adapter": getattr(layer, "adapter_adapter", None) is not None,
        "debug_enabled": bool(getattr(layer, "_adapter_debug_enabled", False)),
        "has_debug_buffers": True,
        "position": getattr(layer, "_adapter_position", None),
        "masked_tokens": int(layer._adapter_debug_masked_tokens.detach().cpu().item()),
        "total_tokens": total,
        "prefill_tokens": int(layer._adapter_debug_prefill_tokens.detach().cpu().item()),
        "decode_tokens": int(layer._adapter_debug_decode_tokens.detach().cpu().item()),
        "prefill_calls": int(layer._adapter_debug_prefill_calls.detach().cpu().item()),
        "decode_calls": int(layer._adapter_debug_decode_calls.detach().cpu().item()),
        "mixed_calls": int(layer._adapter_debug_mixed_calls.detach().cpu().item()),
        "delta_l2_sum": float(layer._adapter_debug_delta_l2_sum.detach().cpu().item()),
        "delta_abs_sum": float(layer._adapter_debug_delta_abs_sum.detach().cpu().item()),
        "delta_abs_max": float(layer._adapter_debug_delta_abs_max.detach().cpu().item()),
        "hidden_l2_sum": float(layer._adapter_debug_hidden_l2_sum.detach().cpu().item()),
    }

    # Include per-token capture data if available
    cap_n = getattr(layer, "_adapter_cap_num_tokens", None)
    if cap_n is not None:
        n = int(cap_n.item())
        if n > 0:
            stats["cap_num_tokens"] = n
            stats["cap_h"] = layer._adapter_cap_h[:n].detach().cpu().tolist()
            stats["cap_delta"] = layer._adapter_cap_delta[:n].detach().cpu().tolist()
            stats["cap_h_norms"] = layer._adapter_cap_h_norms[:n].detach().cpu().tolist()
            stats["cap_delta_norms"] = layer._adapter_cap_delta_norms[:n].detach().cpu().tolist()
            stats["cap_delta_abs_max"] = float(layer._adapter_cap_delta_abs_max.item())
            stats["cap_h_mean_norm"] = float(layer._adapter_cap_h_mean_norm.item())
            stats["cap_delta_mean_norm"] = float(layer._adapter_cap_delta_mean_norm.item())

    return stats


# ---------------------------------------------------------------------------
# Qwen2 adapter-aware decoder layer factory
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Shared multi-adapter forward logic
# ---------------------------------------------------------------------------

def _prepare_adapter(source: nn.Module, dev: torch.device,
                     model_dtype: torch.dtype) -> nn.Module:
    """Deep-copy an adapter, cast linears to model dtype, install caches."""
    check_adaptation_supported(source)
    adapter_copy = copy.deepcopy(source).to(dev)
    for child in adapter_copy.modules():
        if isinstance(child, nn.Linear):
            child.to(dtype=model_dtype)
    _install_adapter_caches(adapter_copy, model_dtype=model_dtype)
    return adapter_copy


def _init_multi_adapter_state(layer: nn.Module, dev: torch.device,
                           hidden_size: int) -> None:
    """Initialise the multi-adapter bookkeeping on a decoder layer.

    Sets up:
      - ``served_adapters``  (nn.ModuleDict) keyed by ``str(adapter_int_id)``
      - ``_adapter_adapter_positions`` (dict) adapter_id -> position string
      - ``_adapter_combined_masks`` (dict) adapter_id -> pre-allocated mask buffer
    """
    layer.served_adapters = nn.ModuleDict()
    layer._adapter_adapter_positions: dict[int, str] = {}
    layer._adapter_adapter_sites: dict[int, str] = {}
    layer._adapter_combined_masks: dict[int, torch.Tensor] = {}
    layer._adapter_site_hooks: dict[str, torch.utils.hooks.RemovableHandle] = {}
    _init_adapter_debug_buffers(layer, dev)
    _init_adapter_capture_buffers(layer, hidden_size, dev)


def _compute_batch_segments(resolved_meta,
                            num_tokens: int) -> Optional[list]:
    """Per-request (start, end) token spans of the flattened batch.

    Used to run sequence-mixing adaptations once per request so they
    never mix across request boundaries.  Returns ``None`` when the
    boundaries cannot be determined."""
    (num_prefill_tokens, num_decodes, num_prefills, query_start_loc,
     _) = _get_prefill_info(resolved_meta)
    if num_prefill_tokens is None:
        return None
    num_decode_tokens = num_tokens - num_prefill_tokens
    if query_start_loc is not None:
        bounds = [int(b) for b in query_start_loc.tolist()]
        return [(a, min(b, num_tokens))
                for a, b in zip(bounds[:-1], bounds[1:])
                if b > a and a < num_tokens]
    # No query_start_loc: decode tokens are one-per-request; the
    # prefill region is unambiguous only for a single prefill request.
    segments = [(i, i + 1) for i in range(num_decode_tokens)]
    if num_prefill_tokens > 0:
        if num_prefills > 1:
            return None
        segments.append((num_decode_tokens, num_tokens))
    return segments


_seq_mixing_graph_mode_warned = False
_seq_mixing_no_segments_warned = False


def _blend_one_adaptation(layer: nn.Module, int_id: int, adapter: nn.Module,
                          hidden: torch.Tensor) -> torch.Tensor:
    """Blend one adaptation into *hidden*, via the gather path when
    member indices were precomputed for this step.

    Composites (``adapter.members``: phase-gated submodules in one
    checkpoint, e.g. storage-on-prefill + steering-on-decode) blend each
    member under its own membership∧phase mask, precomputed by
    ``update_adapter_position_masks`` alongside the regular buffers.
    """
    members = getattr(adapter, "members", None)
    if members is not None:
        comp_masks = getattr(layer, "_adapter_composite_masks", {})
        num_tokens = hidden.shape[0]
        out = hidden
        for j, member in enumerate(members):
            mbuf = comp_masks.get((int_id, j))
            if mbuf is None:
                # Never seen a batch (e.g. first graph-mode step for an
                # inactive adapter): all-zero mask, correction contributes 0.
                mbuf = torch.zeros(
                    max(num_tokens, _MASK_FALLBACK_SIZE),
                    dtype=torch.float32, device=hidden.device)
                comp_masks = layer.__dict__.setdefault(
                    "_adapter_composite_masks", comp_masks)
                comp_masks[(int_id, j)] = mbuf
            out = apply_adaptation(member, out, mbuf[:num_tokens])
        return out

    # The gather path (use_gather=True in update_multi_adapter_position_
    # masks) runs the adaptation only on its member tokens and scatters
    # results back — O(members) instead of O(batch). Eager mode only:
    # the member count is dynamic, which CUDA graphs cannot capture.
    # Sequence-mixing adaptations are applied once per request span
    # (needs_sequence_segmentation) so they never mix across request
    # boundaries in the flattened batch.
    if needs_sequence_segmentation(adapter):
        segments = getattr(layer, "_adapter_segments", None)
        if segments is not None and len(segments) > 1:
            mask_buf = layer._adapter_combined_masks.get(int_id)
            num_tokens = hidden.shape[0]
            out = hidden.clone()
            for start, end in segments:
                end = min(end, num_tokens)
                if start >= end:
                    continue
                mask = mask_buf[start:end] if mask_buf is not None else None
                out[start:end] = apply_adaptation(adapter,
                                                  hidden[start:end], mask)
            return out
        # Single segment (or no boundary info): the full-batch masked
        # path below is the correct application — the mixer must see
        # the whole contiguous span, with the mask applied to the
        # blend only.  Never fall into the gather path: gathering
        # hands the mixer a non-contiguous subset of member tokens,
        # silently changing what "the sequence" is.
        mask_buf = layer._adapter_combined_masks.get(int_id)
        num_tokens = hidden.shape[0]
        mask = mask_buf[:num_tokens] if mask_buf is not None else None
        return apply_adaptation(adapter, hidden, mask)

    member_idx = getattr(layer, "_adapter_member_indices", {}).get(int_id)
    if member_idx is not None:
        if member_idx.numel() == 0:
            return hidden
        member_mask = layer._adapter_member_mask_values[int_id]
        sub = hidden.index_select(0, member_idx)
        sub_new = apply_adaptation(adapter, sub, member_mask)
        return hidden.index_copy(0, member_idx, sub_new.to(hidden.dtype))

    mask_buf = layer._adapter_combined_masks.get(int_id)
    num_tokens = hidden.shape[0]
    mask = mask_buf[:num_tokens] if mask_buf is not None else None
    return apply_adaptation(adapter, hidden, mask)


def _apply_site_adaptations(layer: nn.Module, site: str, output):
    """Blend all active adaptations mounted at *site* into *output*.

    *output* may be a bare tensor or a tuple whose first element is the
    tensor (vLLM's parallel linear layers return ``(out, bias)``).
    """
    if getattr(layer, "_adapter_all_masks_zero", False):
        return output
    active_ids = getattr(layer, "_adapter_active_ids", None)
    hidden = output[0] if isinstance(output, tuple) else output
    new_hidden = hidden
    for str_id, adapter in layer.served_adapters.items():
        int_id = int(str_id)
        if layer._adapter_adapter_sites.get(int_id, "block_output") != site:
            continue
        if active_ids is not None and int_id not in active_ids:
            continue
        new_hidden = _blend_one_adaptation(layer, int_id, adapter,
                                           new_hidden)
    if new_hidden is hidden:
        return output
    if isinstance(output, tuple):
        return (new_hidden, ) + tuple(output[1:])
    return new_hidden


def _install_site_hook(layer: nn.Module, site: str) -> None:
    """Install (once) the forward hook a submodule-mounted site needs."""
    if site in layer._adapter_site_hooks:
        return
    path = resolve_site_submodule_path(site)
    assert path is not None, f"site {site!r} does not use a hook"
    try:
        submodule = layer.get_submodule(path)
    except AttributeError as e:
        raise ValueError(
            f"Mount site {site!r}: decoder layer has no submodule "
            f"{path!r}") from e

    def hook(module, args, output, _layer=layer, _site=site):
        return _apply_site_adaptations(_layer, _site, output)

    layer._adapter_site_hooks[site] = submodule.register_forward_hook(hook)


def _add_adapter_to_layer(layer: nn.Module, adapter_int_id: int,
                          adapter: nn.Module, position: str,
                          dev: torch.device,
                          site: str = "block_output") -> None:
    """Register an adapter on a decoder layer (in-place)."""
    validate_site(site)
    key = str(adapter_int_id)
    layer.served_adapters[key] = adapter
    layer._adapter_adapter_positions[adapter_int_id] = position
    if not hasattr(layer, "_adapter_adapter_sites"):
        layer._adapter_adapter_sites = {}
    if not hasattr(layer, "_adapter_site_hooks"):
        layer._adapter_site_hooks = {}
    layer._adapter_adapter_sites[adapter_int_id] = site
    if resolve_site_submodule_path(site) is not None:
        _install_site_hook(layer, site)
    # Pre-allocate combined mask buffer
    layer._adapter_combined_masks[adapter_int_id] = torch.zeros(
        _MASK_FALLBACK_SIZE, dtype=torch.float32, device=dev)
    # Backward compat: keep adapter_adapter pointing to first loaded adapter
    if not hasattr(layer, "adapter_adapter") or layer.adapter_adapter is None:
        layer.adapter_adapter = adapter


def _remove_adapter_from_layer(layer: nn.Module, adapter_int_id: int) -> bool:
    """Remove an adapter from a decoder layer, tearing down its site
    hook when it was the site's last user.  Returns True if removed."""
    key = str(adapter_int_id)
    if not hasattr(layer, "served_adapters") or key not in layer.served_adapters:
        return False
    del layer.served_adapters[key]
    layer._adapter_adapter_positions.pop(adapter_int_id, None)
    layer._adapter_combined_masks.pop(adapter_int_id, None)
    sites = getattr(layer, "_adapter_adapter_sites", {})
    site = sites.pop(adapter_int_id, "block_output")
    hooks = getattr(layer, "_adapter_site_hooks", {})
    if site in hooks and site not in sites.values():
        hooks.pop(site).remove()
    # Update backward compat adapter_adapter reference.
    if getattr(layer, "adapter_adapter", None) is not None:
        remaining = list(layer.served_adapters.values())
        layer.adapter_adapter = remaining[0] if remaining else None
    return True


def _multi_adapter_forward(
    layer_self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual,
    *,
    super_forward,
    layer_kwargs: Optional[dict] = None,
):
    """Shared multi-adapter forward pass for Qwen2 and Llama layers.

    Only iterates adapters that are active in the current batch (set by
    ``update_adapter_position_masks`` before each forward).  Inactive
    adapters are skipped entirely — no readout, no masking.

    Pure-decode optimisation: if ``_adapter_all_masks_zero`` is set (e.g.
    decode-only batch with prefill adapters), skip everything.

    Sites: ``block_input`` adaptations run on the incoming residual
    stream before ``super_forward``; ``post_attn`` / ``post_mlp`` /
    ``linear:*`` adaptations run via forward hooks on submodules inside
    ``super_forward``; ``block_output`` adaptations run on the outgoing
    stream below.

    Off-diagonal members (``F_out`` preceding ``F_in``, installed by
    ``vllm.adaptation.recirculation``) run LAST, after the ordinary
    block_output blend: the span is re-executed from the write port and
    its result replaces the stream leaving this block.  ``layer_kwargs``
    is the host's own extra per-layer kwargs, replayed verbatim on every
    re-executed pass so the span sees an identical calling convention.
    """
    skip = (not hasattr(layer_self, "served_adapters")
            or len(layer_self.served_adapters) == 0
            or getattr(layer_self, "_adapter_all_masks_zero", False))

    # Only iterate adapters that have at least one token in this batch.
    # _adapter_active_ids is a Python set computed by
    # update_adapter_position_masks (outside compilation, no GPU sync).
    active_ids = getattr(layer_self, "_adapter_active_ids", None)

    # block_input site: adapt the incoming stream.  The incoming
    # (hidden_states, residual) pair represents stream = hidden +
    # residual (residual is None on the first layer); handing the
    # adapted stream forward with residual=None re-bases the deferred
    # residual add on the adapted value.
    if not skip:
        sites = getattr(layer_self, "_adapter_adapter_sites", {})
        has_active_input_site = any(
            sites.get(int(str_id), "block_output") == "block_input"
            and (active_ids is None or int(str_id) in active_ids)
            for str_id in layer_self.served_adapters)
        if has_active_input_site:
            in_stream = (hidden_states if residual is None else
                         hidden_states + residual)
            new_in_stream = _apply_site_adaptations(
                layer_self, "block_input", in_stream)
            if new_in_stream is not in_stream:
                hidden_states, residual = new_in_stream, None

    hidden_states, residual = super_forward(positions, hidden_states, residual)

    if skip:
        return hidden_states, residual

    # Residual-free layers (e.g. olmo2's norm-after-block ordering) hand
    # back the full stream directly with residual=None; llama-style
    # layers defer the add to (hidden, residual).  Either way h_full is
    # the value of the residual stream leaving the block.
    h_full = hidden_states if residual is None else hidden_states + residual
    new_stream = h_full

    # Sequentially blend each active block_output adaptation into the
    # stream (insertion order).  For the additive default this matches
    # the historical sum-of-masked-corrections whenever masks are disjoint
    # (which membership ∧ phase masks guarantee for paired adapters).
    adapter_sites = getattr(layer_self, "_adapter_adapter_sites", None)
    # An off-diagonal member's readout is applied ON THE WAY INTO the
    # span, not as a blend at this port, so it must not also run here —
    # applying it twice would square a non-identity readout.
    recirc = getattr(layer_self, "_adapter_recirc", None)
    recirc_id = recirc.adapter_int_id if recirc is not None else None
    for str_id, adapter in layer_self.served_adapters.items():
        int_id = int(str_id)
        if active_ids is not None and int_id not in active_ids:
            continue
        if int_id == recirc_id:
            continue  # handled by the span re-execution below
        if (adapter_sites is not None
                and adapter_sites.get(int_id, "block_output")
                != "block_output"):
            continue  # mounted elsewhere (hooked submodule / block_input)
        new_stream = _blend_one_adaptation(layer_self, int_id, adapter,
                                           new_stream)

    # Schedule rewiring: re-execute layers start..this one, `passes`
    # times, from the stream leaving this block.  Runs after every
    # same-port blend so the span is entered with the fully adapted
    # stream, matching adapters.mounting._apply_leaf's ordering.
    if recirc is not None and not recirc.depth and (
            active_ids is None or recirc_id in active_ids):
        from vllm.adaptation.recirculation import run_recirculation
        mask_buf = layer_self._adapter_combined_masks.get(recirc_id)
        num_tokens = new_stream.shape[0]
        mask = mask_buf[:num_tokens] if mask_buf is not None else None
        new_stream = run_recirculation(
            layer_self, positions, new_stream,
            layer_self.served_adapters.get(str(recirc_id)),
            mask, layer_kwargs)

    if residual is None:
        # Residual-free contract: the adapted stream is the output.
        return new_stream, None
    hidden_states = new_stream - h_full
    residual = h_full
    return hidden_states, residual


def update_adapter_position_masks(
    adapter_layers: list[nn.Module],
    token_adapter_ids: torch.Tensor,
    positions: torch.Tensor,
    attn_metadata,
    num_tokens: int,
    decode_token_adapter_ids: Optional[torch.Tensor] = None,
    use_gather: bool = False,
    graph_safe: bool = False,
    prompt_lens: Optional[torch.Tensor] = None,
    extra_token_adapter_ids: Sequence[torch.Tensor] = (),
) -> None:
    """Pre-compute combined masks for all adapters on all adapter layers.

    For each adapter on each layer, the combined mask is:
        (token belongs to this adapter) AND (position satisfies adapter's mode)

    A token belongs to an adapter when the token's request references it
    through either slot: the primary ``token_adapter_ids`` or the optional
    ``decode_token_adapter_ids`` (a decode-phase-paired adapter).  The
    adapter's own position mask then restricts it to its phase, so a
    prefill adapter and a decode adapter paired on the same request
    never overlap.

    With ``use_gather=True`` (eager mode only — member counts are
    dynamic shapes, incompatible with CUDA graph replay), each active
    adapter's member-token indices are additionally precomputed so the
    forward runs the adapter on O(members) tokens instead of the full
    batch.

    With ``graph_safe=True`` (CUDA graph mode), the Python-state skip
    optimisations are disabled so the forward's control flow is
    batch-agnostic — captured graphs bake Python branches, so every
    loaded adapter must compute unconditionally, with per-batch
    selection expressed only through the mask buffers.  Buffers of
    adapters with no tokens this step are zeroed (they'd otherwise be
    stale from an earlier batch).

    Called from the model runner before each forward pass.
    """
    if graph_safe:
        # Gather uses dynamic member counts — never in graph mode.
        use_gather = False
    if not adapter_layers:
        return

    # Detect pure-decode batch from metadata (no GPU sync needed).
    # If no prefill tokens and no adapter needs decode-time compute,
    # skip mask computation and adapter forward entirely.
    resolved_meta = _resolve_attn_metadata(attn_metadata)
    (num_prefill_tokens, _, _, _, _) = _get_prefill_info(resolved_meta)
    is_pure_decode = (num_prefill_tokens is not None
                      and num_prefill_tokens == 0)

    if is_pure_decode and not graph_safe:
        # Positions whose masks can be nonzero on decode tokens (e.g.
        # "all", "decode", custom positions) need decode-time compute.
        any_decode_active = any(
            position_active_in_decode(pos)
            for layer in adapter_layers
            if hasattr(layer, "_adapter_adapter_positions")
            for pos in layer._adapter_adapter_positions.values()
        )
        if not any_decode_active:
            # Pure decode + only prefill-flavored adapters → skip.
            for layer in adapter_layers:
                layer._adapter_all_masks_zero = True
            return

    # Per-request segments for sequence-mixing adaptations (eager only:
    # segment counts are dynamic shapes).
    any_seq_mixing = any(
        needs_sequence_segmentation(layer.served_adapters[str_id])
        for layer in adapter_layers if hasattr(layer, "served_adapters")
        for str_id in layer.served_adapters)
    batch_segments = None
    if any_seq_mixing:
        if graph_safe:
            global _seq_mixing_graph_mode_warned
            if not _seq_mixing_graph_mode_warned:
                _seq_mixing_graph_mode_warned = True
                logger.warning(
                    "Sequence-mixing adaptations are loaded while CUDA "
                    "graphs are enabled; per-request segmentation is "
                    "eager-only, so mixing may leak across request "
                    "boundaries in the flattened batch. Use "
                    "enforce_eager=True for sequence-mixing adaptations.")
        else:
            batch_segments = _compute_batch_segments(resolved_meta,
                                                     num_tokens)
            if batch_segments is None:
                (_npt, _nd, _np_, _qsl, _) = _get_prefill_info(resolved_meta)
                multi_request = (_npt is not None and _qsl is None
                                 and _np_ > 1)
                if multi_request:
                    global _seq_mixing_no_segments_warned
                    if not _seq_mixing_no_segments_warned:
                        _seq_mixing_no_segments_warned = True
                        logger.warning(
                            "Sequence-mixing adaptations are loaded but "
                            "request boundaries could not be determined "
                            "for a multi-prefill batch (no query_start_loc "
                            "in attention metadata); mixing may leak "
                            "across request boundaries this step.")

    # Compute the set of adapter IDs actually referenced in this batch.
    # One .unique() call on a small 1-D int tensor — fast, no GPU sync
    # needed since we only use the result as a Python set for membership
    # checks in the forward pass.
    batch_active_ids: set[int] = set(token_adapter_ids.unique().tolist())
    if decode_token_adapter_ids is not None:
        batch_active_ids |= set(decode_token_adapter_ids.unique().tolist())
    for extra in extra_token_adapter_ids:
        batch_active_ids |= set(extra.unique().tolist())
    batch_active_ids.discard(0)  # 0 = no adapter

    # Cache position masks by position string to avoid redundant computation.
    pos_mask_cache: dict[str, Optional[torch.Tensor]] = {}

    for layer in adapter_layers:
        if not hasattr(layer, "served_adapters"):
            continue
        # Reset per-step gather state; repopulated below when enabled.
        layer._adapter_member_indices = {}
        layer._adapter_member_mask_values = {}
        layer._adapter_segments = batch_segments
        # Intersect batch-active IDs with this layer's loaded adapters.
        layer_active_ids = batch_active_ids & {
            int(s) for s in layer.served_adapters}
        if is_pure_decode:
            # Prefill-flavored adapters can't fire on decode tokens.
            layer_active_ids = {
                i for i in layer_active_ids
                if position_active_in_decode(
                    layer._adapter_adapter_positions.get(i, "prefill"))
            }
        if graph_safe:
            # Batch-agnostic control flow: the forward iterates every
            # loaded adapter; selection lives in the mask buffers only.
            layer._adapter_active_ids = None
            layer._adapter_all_masks_zero = False
            for str_id in layer.served_adapters:
                inactive_id = int(str_id)
                if inactive_id not in layer_active_ids:
                    buf = layer._adapter_combined_masks.get(inactive_id)
                    if buf is not None:
                        buf.zero_()
                    for key, mbuf in getattr(
                            layer, "_adapter_composite_masks", {}).items():
                        if key[0] == inactive_id:
                            mbuf.zero_()
        else:
            layer._adapter_active_ids = layer_active_ids
            layer._adapter_all_masks_zero = len(layer_active_ids) == 0
            if layer._adapter_all_masks_zero:
                continue
        for str_id in layer.served_adapters:
            int_id = int(str_id)
            if int_id not in layer_active_ids:
                continue
            # Adapter membership mask (any slot).
            adapter_mask = (token_adapter_ids == int_id)
            if decode_token_adapter_ids is not None:
                adapter_mask = adapter_mask | (
                    decode_token_adapter_ids == int_id)
            for extra in extra_token_adapter_ids:
                adapter_mask = adapter_mask | (extra == int_id)
            adapter_mask = adapter_mask.float()
            # Position mask (cached by position string)
            pos = layer._adapter_adapter_positions.get(int_id, "prefill")
            if pos not in pos_mask_cache:
                if pos in ("all", "all_tokens"):
                    pos_mask_cache[pos] = None
                else:
                    pos_mask_cache[pos] = _compute_position_mask(
                        positions, pos, torch.float32, num_tokens,
                        attn_metadata, prompt_lens=prompt_lens)
            pos_mask = pos_mask_cache[pos]
            # Composite: additionally precompute membership ∧ member-
            # phase masks, one buffer per (adapter id, member index) —
            # _blend_one_adaptation applies each member under its own.
            comp_members = getattr(
                layer.served_adapters[str_id], "members", None)
            if comp_members is not None:
                comp_masks = layer.__dict__.setdefault(
                    "_adapter_composite_masks", {})
                for j, member in enumerate(comp_members):
                    mpos = getattr(member, "position", "all")
                    if mpos not in pos_mask_cache:
                        if mpos in ("all", "all_tokens"):
                            pos_mask_cache[mpos] = None
                        else:
                            pos_mask_cache[mpos] = _compute_position_mask(
                                positions, mpos, torch.float32, num_tokens,
                                attn_metadata, prompt_lens=prompt_lens)
                    mpos_mask = pos_mask_cache[mpos]
                    mcombined = (adapter_mask if mpos_mask is None
                                 else adapter_mask * mpos_mask)
                    mbuf = comp_masks.get((int_id, j))
                    if mbuf is None or mbuf.shape[0] < num_tokens:
                        mbuf = torch.zeros(
                            max(num_tokens, _MASK_FALLBACK_SIZE),
                            dtype=torch.float32, device=positions.device)
                        comp_masks[(int_id, j)] = mbuf
                    mbuf[:num_tokens].copy_(mcombined)
                    if mbuf.shape[0] > num_tokens:
                        mbuf[num_tokens:].zero_()
            # Combined mask
            if pos_mask is not None:
                combined = adapter_mask * pos_mask
            else:
                combined = adapter_mask
            # Write into pre-allocated buffer
            buf = layer._adapter_combined_masks.get(int_id)
            if buf is None or buf.shape[0] < num_tokens:
                buf = torch.zeros(max(num_tokens, _MASK_FALLBACK_SIZE),
                                  dtype=torch.float32,
                                  device=positions.device)
                layer._adapter_combined_masks[int_id] = buf
            buf[:num_tokens].copy_(combined[:num_tokens])
            buf[num_tokens:].zero_()
            if use_gather:
                member_idx = combined[:num_tokens].nonzero(
                    as_tuple=True)[0]
                layer._adapter_member_indices[int_id] = member_idx
                layer._adapter_member_mask_values[int_id] = combined[member_idx]


# ---------------------------------------------------------------------------
# Qwen2 adapter-aware decoder layer factory
# ---------------------------------------------------------------------------
# Architecture-generic adapter decoder-layer factory
# ---------------------------------------------------------------------------

def make_adapter_decoder_layer(base_layer_cls: type,
                            adapter_spec: Optional[dict] = None,
                            arch: Optional[str] = None) -> type:
    """Return a subclass of *base_layer_cls* with multi-adapter support.

    Works for any decoder layer following vLLM's conventions:

      - constructor arguments are passed through verbatim (both the
        legacy ``(config, cache_config, quant_config, prefix)`` style
        and the modern ``(vllm_config, prefix)`` style),
      - forward contract ``(positions, hidden_states, residual,
        **kwargs) -> (hidden_states, residual)`` (extra kwargs, e.g.
        gemma3's, are forwarded), OR the residual-free contract
        ``(positions, hidden_states, **kwargs) -> hidden_states`` used
        by norm-after-block architectures (olmo2/olmo3), detected from
        the base class's forward signature.  Both contracts run the
        same ``_multi_adapter_forward`` path; only the calling
        convention is adapted.

    If *adapter_spec* is provided (single-adapter backward compat), the
    initial adapter is installed at construction with ``adapter_int_id=1``
    on the spec's ``layer_indices``.  Otherwise layers are created with
    empty adapter dicts, ready for dynamic loading.
    """
    if adapter_spec is not None:
        layer_indices_set = frozenset(adapter_spec["layer_indices"])
        position = adapter_spec["position"]
        sample_adapter: Optional[nn.Module] = adapter_spec["sample_adapter"]
        per_layer_adapters: dict = adapter_spec.get("adapters", {})
        debug_mask_enabled = bool(adapter_spec.get("debug_mask", False))
    else:
        layer_indices_set = frozenset()
        position = "prefill"
        sample_adapter = None
        per_layer_adapters = {}
        debug_mask_enabled = False

    arch_name = arch or base_layer_cls.__name__

    # Detect the base layer's forward contract from its signature:
    # llama-style takes an explicit ``residual`` argument and returns a
    # (hidden, residual) pair; olmo2-style takes none and returns the
    # full stream directly.
    try:
        _has_residual_arg = "residual" in inspect.signature(
            base_layer_cls.forward).parameters
    except (TypeError, ValueError):
        _has_residual_arg = True

    def _find_prefix(args, kwargs) -> str:
        prefix = kwargs.get("prefix")
        if isinstance(prefix, str):
            return prefix
        for a in args:
            if isinstance(a, str) and ".layers." in a:
                return a
        return ""

    def _find_model_dtype(args, kwargs):
        for candidate in list(args) + list(kwargs.values()):
            model_config = getattr(candidate, "model_config", None)
            if model_config is not None:
                dtype = getattr(model_config, "dtype", None)
                if isinstance(dtype, torch.dtype):
                    return dtype
                hf_config = getattr(model_config, "hf_config", None)
                dtype = getattr(hf_config, "torch_dtype", None)
                if isinstance(dtype, torch.dtype):
                    return dtype
            dtype = getattr(candidate, "torch_dtype", None)
            if isinstance(dtype, torch.dtype):
                return dtype
        return torch.bfloat16

    def _find_hidden_size(self, args, kwargs) -> int:
        hidden = getattr(self, "hidden_size", None)
        if isinstance(hidden, int):
            return hidden
        for candidate in list(args) + list(kwargs.values()):
            hidden = getattr(candidate, "hidden_size", None)
            if isinstance(hidden, int):
                return hidden
            hf_config = getattr(getattr(candidate, "model_config", None),
                                "hf_config", None)
            hidden = getattr(hf_config, "hidden_size", None)
            if isinstance(hidden, int):
                return hidden
        return 4096

    class AdapterDecoderLayer(base_layer_cls):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            prefix = _find_prefix(args, kwargs)
            layer_idx = _extract_layer_idx(prefix)
            try:
                dev = next(self.parameters()).device
            except StopIteration:
                dev = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
            hidden_size = _find_hidden_size(self, args, kwargs)
            model_dtype = _find_model_dtype(args, kwargs)

            object.__setattr__(self, "_adapter_layer_idx",
                               layer_idx if layer_idx is not None else -1)
            object.__setattr__(self, "_adapter_debug_enabled",
                               debug_mask_enabled)
            self.adapter_adapter = None  # backward compat attribute

            _init_multi_adapter_state(self, dev, hidden_size)

            # Load initial adapter from adapter_spec (backward compat, id=1)
            if (layer_idx is not None and layer_idx in layer_indices_set
                    and sample_adapter is not None):
                source = per_layer_adapters.get(layer_idx, sample_adapter)
                adapter_copy = _prepare_adapter(source, dev, model_dtype)
                _add_adapter_to_layer(self, 1, adapter_copy, position, dev)
                _LEGACY_ADAPTER_REGISTRY[layer_idx] = adapter_copy
                logger.debug(
                    "[adapter-vLLM] Attached initial adapter to %s layer %s "
                    "(adapter_type=%s)", arch_name, layer_idx,
                    type(adapter_copy).__name__)

            object.__setattr__(self, "_adapter_position", position)
            _maybe_log_adapter_layer_init(
                arch=arch_name,
                layer_idx=getattr(self, "_adapter_layer_idx", -1),
                attached=getattr(self, "adapter_adapter", None) is not None,
                debug_enabled=bool(getattr(self, "_adapter_debug_enabled",
                                           False)),
                position=position,
                adapter=getattr(self, "adapter_adapter", None),
            )

        # Which calling convention the span executor must use when it
        # re-runs this layer (vllm/adaptation/recirculation.py).
        _adapter_has_residual_arg = _has_residual_arg

        if _has_residual_arg:

            def forward(self, positions, hidden_states, residual, **kwargs):
                if kwargs:

                    def super_forward(p, h, r):
                        return base_layer_cls.forward(self, p, h, r, **kwargs)
                else:

                    def super_forward(p, h, r):
                        return base_layer_cls.forward(self, p, h, r)

                return _multi_adapter_forward(self, positions, hidden_states,
                                           residual,
                                           super_forward=super_forward,
                                           layer_kwargs=kwargs)
        else:

            def forward(self, positions, hidden_states, **kwargs):
                # Residual-free contract (olmo2-style): the base forward
                # returns the full stream; thread it through the shared
                # multi-adapter path with residual=None on both sides.

                def super_forward(p, h, r):
                    del r  # base layer carries the stream in h alone
                    return base_layer_cls.forward(self, p, h, **kwargs), None

                out, _ = _multi_adapter_forward(self, positions, hidden_states,
                                             None,
                                             super_forward=super_forward,
                                             layer_kwargs=kwargs)
                return out

        def get_adapter_debug_stats(self) -> Optional[dict]:
            return _collect_layer_adapter_debug_stats(self)

    AdapterDecoderLayer.__name__ = f"adapter{base_layer_cls.__name__}"
    AdapterDecoderLayer.__qualname__ = AdapterDecoderLayer.__name__
    return AdapterDecoderLayer


_prefix_position_warned: set = set()


def warn_if_prefix_incompatible_position(position: str, vllm_config) -> None:
    """Warn (once per position) when a prompt-anchored position is used
    while the engine injects an external KV prefix.

    The PrefixInjectionConnector marks the first ``prefix_len`` prompt
    tokens as externally computed, so the first token entering the model
    sits at position ``prefix_len``: ``"first"`` masks are silently dead
    and ``every_<k>`` masks re-anchor to prefix-shifted position ids.
    This cannot be detected inside the mask functions (a prefix-injected
    first chunk is indistinguishable from a chunked-prefill continuation
    there), so the guard lives at adapter load/config time.
    """
    try:
        ktc = getattr(vllm_config, "kv_transfer_config", None)
        if ktc is None:
            return
        if getattr(ktc, "kv_connector", None) != "PrefixInjectionConnector":
            return
        prefix_len = int(ktc.get_from_extra_config("prefix_len", 0))
    except Exception:
        return
    if prefix_len <= 0:
        return
    from vllm.adaptation.positions import position_prompt_anchored
    if not position_prompt_anchored(position):
        return
    if position in _prefix_position_warned:
        return
    _prefix_position_warned.add(position)
    logger.warning(
        "Adapter position %r is anchored to absolute position 0, but "
        "PrefixInjectionConnector injects an external KV prefix of "
        "%d tokens: the first locally computed token sits at position "
        "%d, so this mask is dead ('first') or re-anchored "
        "('every_<k>'). Use a generation-anchored position "
        "(e.g. 'every_<k>_decode', 'first_<k>_decode') or a "
        "custom position that accounts for the prefix.",
        position, prefix_len, prefix_len)


def maybe_adapter_layer_type(vllm_config, default_cls: type,
                          arch: Optional[str] = None) -> type:
    """Resolve the decoder-layer class for a model given its adapter config.

    Returns a adapter-aware subclass of *default_cls* when ``adapter_config``
    is set (baked adapter) or ``enable_adapters`` is on (dynamic loading);
    otherwise returns *default_cls* unchanged.  This is the one-line
    hook new architectures use.
    """
    from vllm.adaptation.specs import get_adapter_spec, adapter_config_to_spec
    adapter_spec = adapter_config_to_spec(getattr(vllm_config, "adapter_config",
                                            None))
    if adapter_spec is None:
        adapter_spec = get_adapter_spec()
    if adapter_spec is not None:
        warn_if_prefix_incompatible_position(
            adapter_spec.get("position", "prefill"), vllm_config)
        return make_adapter_decoder_layer(default_cls, adapter_spec, arch=arch)
    if getattr(vllm_config, "enable_adapters", False):
        return make_adapter_decoder_layer(default_cls, None, arch=arch)
    return default_cls


# ---------------------------------------------------------------------------
# Per-architecture factories (backward-compat wrappers)
# ---------------------------------------------------------------------------

def make_adapter_qwen2_layer(adapter_spec: Optional[dict] = None) -> type:
    from vllm.model_executor.models.qwen2 import Qwen2DecoderLayer
    return make_adapter_decoder_layer(Qwen2DecoderLayer, adapter_spec,
                                   arch="qwen2")


def make_adapter_llama_layer(adapter_spec: Optional[dict] = None) -> type:
    from vllm.model_executor.models.llama import LlamaDecoderLayer
    return make_adapter_decoder_layer(LlamaDecoderLayer, adapter_spec,
                                   arch="llama")


def make_adapter_qwen3_moe_layer(adapter_spec: Optional[dict] = None) -> type:
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer
    return make_adapter_decoder_layer(Qwen3MoeDecoderLayer, adapter_spec,
                                   arch="qwen3_moe")


def make_adapter_olmo2_layer(adapter_spec: Optional[dict] = None) -> type:
    """Olmo2DecoderLayer serves both Olmo2ForCausalLM and Olmo3ForCausalLM
    (the registry aliases Olmo3 to olmo2.py)."""
    from vllm.model_executor.models.olmo2 import Olmo2DecoderLayer
    return make_adapter_decoder_layer(Olmo2DecoderLayer, adapter_spec,
                                   arch="olmo2")
