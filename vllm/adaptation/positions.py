# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Named position-mask registry.

A *position* names a per-token phase predicate ("which tokens of the
batch does this adaptation touch").  Each registered position maps to a
function::

    fn(positions, dtype, num_tokens, phase) -> Optional[torch.Tensor]

where ``positions`` is the per-token position-id tensor for the batch,
``phase`` is a :class:`PhaseInfo` describing the batch's decode/prefill
split, and the return value is a per-token float mask of shape
``(num_tokens,)`` — or ``None`` for "apply unconditionally".

Builtins mirror the historical adapter position strings: ``all``,
``all_tokens``, ``prefill``, ``decode``, ``first``, ``last``.  Custom
adaptations register their own with :func:`register_position_mask`.
"""

import re
from dataclasses import dataclass
from typing import Callable, Optional

import torch

__all__ = [
    "PhaseInfo",
    "get_position_mask",
    "position_active_in_decode",
    "position_prompt_anchored",
    "register_position_mask",
    "registered_positions",
]


@dataclass(frozen=True)
class PhaseInfo:
    """Prefill/decode split of the current batch.

    In a vLLM v1 batch, decode tokens come first and prefill tokens
    last.  ``num_prefill_tokens`` is ``None`` when no attention
    metadata was available (masks then fall back to position-id
    heuristics).
    """
    num_prefill_tokens: Optional[int]
    num_decodes: int
    num_prefills: int
    query_start_loc: Optional[torch.Tensor]
    seq_lens: Optional[torch.Tensor]
    prompt_lens: Optional[torch.Tensor] = None
    """Per-request prompt lengths in batch order (decode requests first,
    matching ``query_start_loc``).  Needed by generation-anchored
    positions; supplied by the model runner from the InputBatch."""


PositionMaskFn = Callable[
    [torch.Tensor, torch.dtype, int, PhaseInfo], Optional[torch.Tensor]]


@dataclass
class _PositionEntry:
    fn: PositionMaskFn
    # Whether tokens in the decode region can receive a nonzero mask.
    # Used by the pure-decode fast path to skip adapters that can never
    # fire.  Custom positions default to True (never skipped — safe).
    active_in_decode: bool = True


_POSITION_REGISTRY: dict[str, _PositionEntry] = {}


def register_position_mask(name: str,
                           fn: PositionMaskFn,
                           *,
                           active_in_decode: bool = True,
                           override: bool = False) -> None:
    """Register a named position mask.

    Args:
        name: Position name used in adapter configs.
        fn: ``fn(positions, dtype, num_tokens, phase)`` returning a
            per-token float mask or ``None`` (= all tokens).
        active_in_decode: Set False only if the mask is provably zero
            for every decode-region token; enables the pure-decode
            batch skip.
        override: Allow replacing an existing registration.
    """
    if name in _POSITION_REGISTRY and not override:
        raise ValueError(
            f"Position {name!r} is already registered; pass override=True "
            "to replace it.")
    _POSITION_REGISTRY[name] = _PositionEntry(
        fn=fn, active_in_decode=active_in_decode)


def registered_positions() -> dict[str, _PositionEntry]:
    """The live registry mapping (mutable; pop() to unregister)."""
    return _POSITION_REGISTRY


# Parameterized position families, resolved on demand (no explicit
# registration):
#   every_<k>[_offset_<j>]         — prompt-anchored: pos % k == j
#   every_<k>_decode[_offset_<j>]  — generation-anchored: gen % k == j
#   first_<k>_decode               — first k generated tokens
#   decode_range_<a>_<b>           — a <= gen < b
# where gen = position_id - prompt_len (0 = first sampled token).
_EVERY_K_RE = re.compile(r"^every_(\d+)(?:_offset_(\d+))?$")
_EVERY_K_DECODE_RE = re.compile(r"^every_(\d+)_decode(?:_offset_(\d+))?$")
_FIRST_K_DECODE_RE = re.compile(r"^first_(\d+)_decode$")
_DECODE_RANGE_RE = re.compile(r"^decode_range_(\d+)_(\d+)$")


def _token_generation_index(positions: torch.Tensor, num_tokens: int,
                            phase: PhaseInfo) -> Optional[torch.Tensor]:
    """Per-token generation index (position - prompt_len), or ``None``
    when per-request prompt lengths are unavailable.

    Prompt tokens get negative indices; the first sampled token is 0.
    """
    if phase.prompt_lens is None:
        return None
    prompt_lens = phase.prompt_lens.to(positions.device)

    if phase.num_prefill_tokens is None:
        return None
    num_decode_tokens = num_tokens - phase.num_prefill_tokens

    token_prompt_lens = torch.empty(num_tokens, device=positions.device,
                                    dtype=positions.dtype)
    # Decode region: one token per request, in request order.
    if num_decode_tokens > 0:
        token_prompt_lens[:num_decode_tokens] = \
            prompt_lens[:num_decode_tokens].to(positions.dtype)
    # Prefill region: expand per-request lengths across query spans.
    if phase.num_prefill_tokens > 0:
        prefill_positions = positions[num_decode_tokens:]
        req_idx = _prefill_request_indices(
            phase.num_prefill_tokens, phase.num_decodes,
            phase.num_prefills, phase.query_start_loc, prefill_positions)
        token_prompt_lens[num_decode_tokens:] = \
            prompt_lens[phase.num_decodes:][req_idx].to(positions.dtype)

    return positions - token_prompt_lens


def _make_gen_anchored_mask(predicate) -> PositionMaskFn:
    """Build a mask fn from a predicate over generation indices.

    Without prompt lengths the mask is all-zeros (inert) rather than
    misanchored."""

    def gen_mask(positions, dtype, num_tokens, phase):
        gen_idx = _token_generation_index(positions, num_tokens, phase)
        if gen_idx is None:
            return torch.zeros(num_tokens, dtype=dtype,
                               device=positions.device)
        return ((gen_idx >= 0) & predicate(gen_idx)).to(dtype)

    return gen_mask


def _make_every_k_mask(k: int, offset: int) -> PositionMaskFn:

    def every_k_mask(positions, dtype, num_tokens, phase):
        # Pure position-id arithmetic: exact under chunked prefill,
        # batching, and decode; needs no attention metadata.
        return (positions % k == offset).to(dtype)

    return every_k_mask


def _validate_every_k(name: str, k: int, offset: int) -> None:
    if k < 1:
        raise ValueError(
            f"Position {name!r}: requires k >= 1, got k={k}.")
    if offset >= k:
        raise ValueError(
            f"Position {name!r}: offset must be < k, got offset={offset} "
            f"with k={k} (the mask would never fire).")


def _resolve_position(name: str) -> Optional[_PositionEntry]:
    """Look up *name*, materializing parameterized families on demand."""
    entry = _POSITION_REGISTRY.get(name)
    if entry is not None:
        return entry

    # Generation-anchored families (checked before the prompt-anchored
    # every_<k> pattern, which would not match these anyway).
    match = _EVERY_K_DECODE_RE.match(name)
    if match is not None:
        k = int(match.group(1))
        offset = int(match.group(2) or 0)
        _validate_every_k(name, k, offset)
        register_position_mask(
            name,
            _make_gen_anchored_mask(lambda g, k=k, o=offset: g % k == o),
            active_in_decode=True)
        return _POSITION_REGISTRY[name]

    match = _FIRST_K_DECODE_RE.match(name)
    if match is not None:
        k = int(match.group(1))
        if k < 1:
            raise ValueError(
                f"Position {name!r}: requires k >= 1, got k={k}.")
        register_position_mask(
            name, _make_gen_anchored_mask(lambda g, k=k: g < k),
            active_in_decode=True)
        return _POSITION_REGISTRY[name]

    match = _DECODE_RANGE_RE.match(name)
    if match is not None:
        lo, hi = int(match.group(1)), int(match.group(2))
        if lo >= hi:
            raise ValueError(
                f"Position {name!r}: empty range [{lo}, {hi}).")
        register_position_mask(
            name,
            _make_gen_anchored_mask(
                lambda g, lo=lo, hi=hi: (g >= lo) & (g < hi)),
            active_in_decode=True)
        return _POSITION_REGISTRY[name]

    match = _EVERY_K_RE.match(name)
    if match is None:
        return None
    k = int(match.group(1))
    offset = int(match.group(2) or 0)
    _validate_every_k(name, k, offset)
    register_position_mask(name, _make_every_k_mask(k, offset),
                           active_in_decode=True)
    return _POSITION_REGISTRY[name]


def get_position_mask(name: str, positions: torch.Tensor,
                      dtype: torch.dtype, num_tokens: int,
                      phase: PhaseInfo) -> Optional[torch.Tensor]:
    """Compute the mask for position *name*, or ``None`` for all-tokens."""
    entry = _resolve_position(name)
    if entry is None:
        raise ValueError(
            f"Unknown position {name!r}. Registered positions: "
            f"{sorted(_POSITION_REGISTRY)} (plus the parameterized "
            "'every_<k>' / 'every_<k>_offset_<j>' family). Use "
            "vllm.adaptation.register_position_mask() to add custom ones.")
    return entry.fn(positions, dtype, num_tokens, phase)


def position_prompt_anchored(name: str) -> bool:
    """Whether this position's mask anchors to absolute position id 0.

    True for ``"first"`` (fires only where ``position == 0``) and the
    prompt-anchored ``every_<k>[_offset_<j>]`` family (``pos % k``).
    Under an externally injected KV prefix (kv_transfer connector), the
    first locally computed token sits at position ``prefix_len``, not 0:
    ``"first"`` masks are then dead (position 0 never enters the model)
    and ``every_<k>`` re-anchors to the prefix-shifted ids.  The
    generation-anchored families (``*_decode``, ``decode_range``) are
    unaffected — they subtract the per-request prompt length.
    """
    if name == "first":
        return True
    return _EVERY_K_RE.match(name) is not None


def position_active_in_decode(name: str) -> bool:
    """Whether an adaptation with this position can fire on decode tokens.

    Unknown names return True (never skip what we don't understand)."""
    try:
        entry = _resolve_position(name)
    except ValueError:
        return True
    return True if entry is None else entry.active_in_decode


# ---------------------------------------------------------------------------
# Builtin positions
# ---------------------------------------------------------------------------

def _prefill_request_indices(
    num_prefill_tokens: int,
    num_decodes: int,
    num_prefills: int,
    query_start_loc: Optional[torch.Tensor],
    prefill_positions: torch.Tensor,
) -> torch.Tensor:
    """Map each prefill token to its request index (0-based within prefills).

    Prefers ``query_start_loc`` (from attn_metadata) for an exact answer.
    Falls back to detecting position resets when metadata is unavailable.
    """
    if num_prefill_tokens <= 1:
        return torch.zeros(num_prefill_tokens,
                           device=prefill_positions.device,
                           dtype=torch.long)

    # --- Preferred path: use query_start_loc from metadata ---
    if query_start_loc is not None and num_prefills > 0:
        # query_start_loc covers all requests (decode + prefill), shape
        # (N+1,).  Prefill requests start at index num_decodes.  The
        # prefill-region offsets are relative to the start of prefill
        # tokens.
        prefill_qsl = (query_start_loc[num_decodes:]
                       - query_start_loc[num_decodes])
        prefill_tok_idx = torch.arange(num_prefill_tokens,
                                       device=prefill_positions.device)
        req_idx = torch.searchsorted(prefill_qsl[1:], prefill_tok_idx,
                                     right=True)
        return req_idx.clamp(max=num_prefills - 1)

    # --- Fallback: detect request boundaries from position resets ---
    is_boundary = torch.zeros(num_prefill_tokens,
                              device=prefill_positions.device,
                              dtype=torch.long)
    is_boundary[1:] = (
        prefill_positions[1:] <= prefill_positions[:-1]).long()
    return is_boundary.cumsum(0)


def _all_mask(positions, dtype, num_tokens, phase):
    return None


def _prefill_mask(positions, dtype, num_tokens, phase):
    if phase.num_prefill_tokens is not None:
        num_decode_tokens = num_tokens - phase.num_prefill_tokens
        token_idx = torch.arange(num_tokens, device=positions.device)
        return (token_idx >= num_decode_tokens).to(dtype)
    gate = (positions[0:1] == 0).to(dtype)
    return gate.expand(num_tokens)


def _decode_mask(positions, dtype, num_tokens, phase):
    # Exact complement of "prefill": the decode region at the start of a
    # v1 batch.  A request running its final prompt chunk is still in
    # the prefill region, so the decode adapter only kicks in from the
    # step after the prompt completes.
    if phase.num_prefill_tokens is not None:
        num_decode_tokens = num_tokens - phase.num_prefill_tokens
        token_idx = torch.arange(num_tokens, device=positions.device)
        return (token_idx < num_decode_tokens).to(dtype)
    # Fallback: whole batch is decode iff the first token is not at
    # position 0 (mirrors the "prefill" gate).
    gate = (positions[0:1] != 0).to(dtype)
    return gate.expand(num_tokens)


def _first_mask(positions, dtype, num_tokens, phase):
    if phase.num_prefill_tokens is not None:
        num_decode_tokens = num_tokens - phase.num_prefill_tokens
        # position == 0 only appears in the very first chunk of a
        # request, so a simple check is sufficient — no seq_lens needed.
        full_mask = torch.zeros(num_tokens, device=positions.device,
                                dtype=dtype)
        if phase.num_prefill_tokens > 0:
            prefill_positions = positions[num_decode_tokens:]
            full_mask[num_decode_tokens:] = (
                prefill_positions == 0).to(dtype)
        return full_mask
    in_prefill = (positions[0:1] == 0).to(dtype)
    return (positions == 0).to(dtype) * in_prefill


def _last_mask(positions, dtype, num_tokens, phase):
    if phase.num_prefill_tokens is not None:
        num_decode_tokens = num_tokens - phase.num_prefill_tokens
        full_mask = torch.zeros(num_tokens, device=positions.device,
                                dtype=dtype)
        if phase.num_prefill_tokens == 0:
            return full_mask

        prefill_positions = positions[num_decode_tokens:]  # (P,)

        # --- Identify the last token of each request's query span ---
        req_idx = _prefill_request_indices(
            phase.num_prefill_tokens, phase.num_decodes,
            phase.num_prefills, phase.query_start_loc, prefill_positions)

        # last-in-query-span: token where req_idx changes or final token
        is_last_in_span = torch.zeros(phase.num_prefill_tokens,
                                      device=positions.device, dtype=dtype)
        if phase.num_prefill_tokens > 1:
            is_last_in_span[:-1] = (req_idx[1:] != req_idx[:-1]).to(dtype)
        is_last_in_span[-1] = 1.0

        # --- Filter to only true last prefill tokens ---
        if phase.seq_lens is not None:
            # seq_lens is ordered [decode_reqs..., prefill_reqs...]
            prefill_seq_lens = phase.seq_lens[phase.num_decodes:]
            expected_last_pos = prefill_seq_lens[req_idx] - 1
            is_true_last = (
                prefill_positions == expected_last_pos).to(dtype)
            full_mask[num_decode_tokens:] = is_last_in_span * is_true_last
        else:
            # No seq_lens available — fall back to last-in-span only.
            full_mask[num_decode_tokens:] = is_last_in_span

        return full_mask

    # Fallback tensor-only "last" mask (no prefill/decode split info).
    is_last = torch.zeros_like(positions, dtype=dtype)
    is_last[-1] = 1.0
    next_is_zero = (torch.roll(positions, -1) == 0).to(dtype)
    is_last = is_last + next_is_zero * (1.0 - is_last)
    in_prefill = (positions[0:1] == 0).to(dtype)
    return is_last * in_prefill


register_position_mask("all", _all_mask, active_in_decode=True)
register_position_mask("all_tokens", _all_mask, active_in_decode=True)
register_position_mask("prefill", _prefill_mask, active_in_decode=False)
register_position_mask("decode", _decode_mask, active_in_decode=True)
register_position_mask(
    "decode_b",
    # Decode-plus-boundary: the decode region PLUS the final prompt
    # token of each prefill request (the position that produces the
    # first sampled token). Union of two existing masks.
    lambda positions, dtype, num_tokens, phase: (
        _decode_mask(positions, dtype, num_tokens, phase)
        + _last_mask(positions, dtype, num_tokens, phase)).clamp(max=1),
    active_in_decode=True)
register_position_mask("first", _first_mask, active_in_decode=False)
register_position_mask("last", _last_mask, active_in_decode=False)
