# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vllm.adaptation — generic phase-aware adaptations.

This package is the extension surface for custom "adaptations":
arbitrary per-token computations mounted at named points of the model
and restricted to named token positions (phases).  The adapter serving
machinery in :mod:`vllm.adaptation` is one consumer; anything satisfying the
protocol below can be served through the same request plumbing
(``adapter_request`` / ``decode_adapter_request``), LRU manager, and
training-time weight sync.

The adaptation contract (duck-typed; no base class required)
------------------------------------------------------------
An adaptation is an ``nn.Module`` that provides ONE of:

  - ``readout(h) -> h'`` (v3 T/R contract) — blended as h + mask*(R(h)-h):
    ``h' = h + mask * delta``.  All existing pyreft adapters use this.
  - ``apply_masked(h, mask) -> h'`` — full control of how the masked
    computation combines with the stream (replacement, gating, ...).
    ``mask`` is a per-token float tensor of shape ``(num_tokens,)``.

Optionally:

  - ``install_inference_caches()`` / ``refresh_inference_caches()`` —
    pre-compute graph-unsafe tensors (pinv, Householder products) as
    buffers.
  - ``position`` — a *registered* position name (see
    :func:`register_position_mask`).  Builtins: ``all``, ``prefill``,
    ``decode``, ``first``, ``last``.
  - ``site`` — a mount point (see :data:`MOUNT_SITES`).  Builtins:
    ``block_output`` (default, the residual stream after the decoder
    block), ``block_input``, ``post_attn``, ``post_mlp``, and
    ``linear:<submodule.path>`` for wrapping a named linear projection.

Not supported: adaptations whose mixers carry recurrent state across
decode steps (``mixer.stateful``) or need separate k/v streams
(``mixer.needs_kv``) — per-request state does not survive vLLM's
batching, reordering, and preemption.  Loading one raises.
"""

from vllm.adaptation.positions import (PhaseInfo, get_position_mask,
                                       position_active_in_decode,
                                       position_prompt_anchored,
                                       register_position_mask,
                                       registered_positions)
from vllm.adaptation.protocol import (MOUNT_SITES, apply_adaptation,
                                      check_adaptation_supported,
                                      needs_sequence_segmentation,
                                      resolve_site_submodule_path,
                                      validate_site)

__all__ = [
    "PhaseInfo",
    "get_position_mask",
    "position_active_in_decode",
    "position_prompt_anchored",
    "register_position_mask",
    "registered_positions",
    "MOUNT_SITES",
    "apply_adaptation",
    "check_adaptation_supported",
    "needs_sequence_segmentation",
    "resolve_site_submodule_path",
    "validate_site",
]
