# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adaptation protocol helpers: mount sites, blend, capability checks."""

from typing import Optional

import torch
import torch.nn as nn

__all__ = [
    "MOUNT_SITES",
    "LINEAR_SITE_PREFIX",
    "PORT_ORDER",
    "WRITE_PORTS",
    "HAVE_PAIR_PORTS",
    "apply_adaptation",
    "check_adaptation_supported",
    "port_order",
    "resolve_site_submodule_path",
    "validate_site",
    "validate_write_port",
]

# Mount-site vocabulary: imported from the adapters library — the
# SINGLE source of truth shared with HF-side training, so a site added
# there exists on both sides at once and cannot drift
# (ARCHITECTURE.md "Serving contract" in the adapters repo). The fork
# deliberately has no fallback copy: serving an adapter without the
# library that defines its placement vocabulary would reintroduce the
# silent-drift bug class this import exists to kill.
try:
    from adapters.sites import (LINEAR_SITE_PREFIX, MOUNT_SITES,
                                validate_site)
    from adapters.sites import \
        resolve_site_submodule_path as _shared_resolve
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "vllm.adaptation requires the 'adapters' library (the shared "
        "mount-site table, adapters.sites). Install it: "
        "pip install -e /path/to/adapters") from _e


# ---------------------------------------------------------------------------
# The site axis as a PAIR: (input port, output port)
# ---------------------------------------------------------------------------
# ``adapters.sites`` grew a total order on the residual-stream ports so a
# mount can name an OUTPUT port distinct from (and preceding) its INPUT
# port — the off-diagonal member the engine serves by re-executing the
# enclosed decoder layers (see vllm/adaptation/recirculation.py).
#
# These names are NEWER than the site table itself, so they are imported
# OPTIONALLY: a fork paired with a library predating the pair axis must
# keep serving every diagonal (F_in == F_out) adapter exactly as before.
# What it must NOT do is carry a private copy of the port order — that is
# the silent-drift bug class the hard import above exists to kill — so
# when the names are missing every entry point raises instead.
_PAIR_PORT_IMPORT_ERROR: Optional[str] = None
try:
    from adapters.sites import PORT_ORDER, WRITE_PORTS  # noqa: F401
    from adapters.sites import port_order as _shared_port_order
    from adapters.sites import validate_write_port as _shared_validate_write_port
    HAVE_PAIR_PORTS = True
except ImportError as _pair_e:  # pragma: no cover - depends on library age
    PORT_ORDER = None  # type: ignore[assignment]
    WRITE_PORTS = None  # type: ignore[assignment]
    _shared_port_order = None  # type: ignore[assignment]
    _shared_validate_write_port = None  # type: ignore[assignment]
    HAVE_PAIR_PORTS = False
    _PAIR_PORT_IMPORT_ERROR = str(_pair_e)


def require_pair_ports() -> None:
    """Raise unless the installed adapters library defines the port order.

    Called from every code path that reads a mount's ``output_site`` /
    ``output_layer`` / ``passes``.  Diagonal mounts never reach it.
    """
    if HAVE_PAIR_PORTS:
        return
    raise ImportError(
        "this adapter declares an output port distinct from its input "
        "port (output_site/output_layer/passes), which needs the port "
        "order from the adapters library: adapters.sites must export "
        "WRITE_PORTS, PORT_ORDER, validate_write_port and port_order. "
        "The installed adapters library does not "
        f"({_PAIR_PORT_IMPORT_ERROR}). Upgrade the adapters library; the "
        "fork deliberately keeps no private copy of the site/port table.")


def validate_write_port(site: str) -> None:
    """Raise unless *site* is a legal OUTPUT port (shared definition)."""
    require_pair_ports()
    _shared_validate_write_port(site)


def port_order(layer_idx: int, site: str) -> tuple:
    """Total order on residual ports across the stack (shared definition)."""
    require_pair_ports()
    return _shared_port_order(layer_idx, site)


def resolve_site_submodule_path(site: str) -> Optional[str]:
    """Submodule path (relative to the decoder layer) a site hooks onto.

    Returns ``None`` for the block-level sites, which are applied inside
    the layer's own forward rather than via a submodule hook. Resolution
    uses the shared table's "vllm" backend naming (q/k/v and gate/up are
    fused here, unlike HF).
    """
    return _shared_resolve(site, backend="vllm")


def apply_adaptation(adaptation: nn.Module, hidden: torch.Tensor,
                     mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Blend one adaptation's computation into *hidden*.

    If the adaptation defines ``apply_masked(h, mask) -> h'`` that wins;
    otherwise the general interpolation blend runs:
    ``lerp(h, R(h), mask)`` — written as ``h + mask*(R(h)-h)``. R is a
    MAP, not a correction: replacement and multiplicative readouts are
    first-class (mask=1 yields exactly R(h), whatever R is).

    Args:
        adaptation: The adaptation module.
        hidden: ``(num_tokens, dim)`` stream at the mount site.
        mask: Per-token float mask ``(num_tokens,)``, or ``None`` for
            all-tokens.
    """
    if mask is None:
        mask = torch.ones(hidden.shape[0], device=hidden.device,
                          dtype=torch.float32)
    apply_masked = getattr(adaptation, "apply_masked", None)
    if apply_masked is not None:
        return apply_masked(hidden, mask)
    out = adaptation.readout(hidden.unsqueeze(0)).squeeze(0)
    correction = out - hidden
    return hidden + correction * mask.unsqueeze(-1).to(correction.dtype)


def needs_sequence_segmentation(adaptation: nn.Module) -> bool:
    """Whether this adaptation mixes information along the sequence axis.

    Sequence-mixing computations (cnn/bigram mixers, chunked adapters)
    must run per request span — vLLM's flattened batch concatenates
    unrelated requests, and mixing across the boundary leaks one
    request's hiddens into another's correction.

    An explicit ``sequence_mixing`` attribute wins; otherwise the
    presence of a ``mixer`` implies sequence mixing.
    """
    explicit = getattr(adaptation, "sequence_mixing", None)
    if explicit is not None:
        return bool(explicit)
    return getattr(adaptation, "mixer", None) is not None


def check_adaptation_supported(adaptation: nn.Module) -> None:
    """Reject adaptations that cannot run correctly under vLLM serving.

    Mixers with recurrent state across decode steps (``stateful``) or
    that need separate k/v streams (``needs_kv``) require per-request
    state that does not survive vLLM's batching, reordering, and
    preemption — loading them would corrupt generations silently.

    Composites (``adaptation.members``) are checked member by member:
    a rejected mechanism does not become servable by being wrapped.
    """
    members = getattr(adaptation, "members", None)
    if members is not None:
        for member in members:
            check_adaptation_supported(member)
    mixer = getattr(adaptation, "mixer", None)
    if mixer is None:
        return
    if getattr(mixer, "stateful", False):
        raise ValueError(
            f"Adaptation {type(adaptation).__name__} uses a stateful mixer "
            f"({type(mixer).__name__}); recurrent decode-time state is not "
            "supported under vLLM serving (per-request state does not "
            "survive batching/reordering/preemption).")
    if getattr(mixer, "needs_kv", False):
        raise ValueError(
            f"Adaptation {type(adaptation).__name__} uses a needs_kv mixer "
            f"({type(mixer).__name__}); routing separate k/v streams into "
            "adaptations is not supported under vLLM serving.")
