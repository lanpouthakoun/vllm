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
    "SUPPORTED_WRITE_LABELS",
    "MULTISITE_HONOURS_UNCHUNKED_PREFILL",
    "SERVING_REQUIREMENT_ATTRS",
    "declared_serving_requirements",
    "apply_adaptation",
    "apply_write",
    "readout_then_write",
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
    from adapters.sites import \
        validate_write_port as _shared_validate_write_port
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


# ---------------------------------------------------------------------------
# W_phi — the member's WRITE at F_out
# ---------------------------------------------------------------------------
# A member is <(T, R, W), F, P>.  R reads at F_in and never has the
# stream at F_out in hand, so what the member does to that stream —
# replace it, add to it, gate it, norm-match it — is its own fourth
# method, ``write(h_out, payload)`` (adapters/_base.py::BaseAdapter.write,
# docs/recirculation-serving.md §1.1).  The engine's job at the write
# port is exactly::
#
#     payload = adaptation.readout(h_in, state)   # at F_in
#     h_out   = adaptation.write(h_out, payload)  # at F_out
#
# The DEFAULT ``write`` returns the payload unchanged, so every member
# that predates the W axis is byte-identical through this path (and the
# ``adapted is hidden`` fast path still fires, because the default
# returns the payload object itself).
#
# Prefer the library's own dispatch so the two sides cannot drift.  A
# library predating the W axis has no member with a non-default W, so
# the local fallback — which is the same duck-typed getattr — is exactly
# right for it and cannot be wrong.
try:
    from adapters._base import apply_write as _shared_apply_write
except ImportError:  # pragma: no cover - depends on library age
    _shared_apply_write = None  # type: ignore[assignment]


def apply_write(adaptation: nn.Module, h_out: torch.Tensor,
                payload: torch.Tensor) -> torch.Tensor:
    """W_phi(h_out, payload) -> the new stream at the write port.

    Args:
        adaptation: the mounted member.
        h_out: the host's own stream at F_out, BEFORE the member writes.
        payload: what the member produced at F_in (``readout``), carried
            to the write port (for a backward pipe, after the span has
            been re-executed over it).

    Raises:
        RuntimeError: if *adaptation* defines the RETIRED ``recombine``
            hook.  ``recombine(fx, piped)`` was this engine's private
            callback for the gated pipe until it became the member's
            ``write``; a leaf still carrying it would have its gate
            SILENTLY skipped here (gate 0 would stop being a no-op),
            which is the exact failure this refusal exists to make loud.
    """
    if getattr(adaptation, "recombine", None) is not None:
        raise RuntimeError(
            f"{type(adaptation).__name__} defines the retired "
            f"'recombine(fx, piped)' hook. The recombination at the "
            f"write port is now the member's own W_phi: rename it to "
            f"'write(self, h_out, payload)' (or inherit "
            f"adapters._base.InterpolateWrite / AddWrite / NormMixWrite). "
            f"The engine calls write() at F_out and nowhere else "
            f"(docs/recirculation-serving.md §1.1); leaving 'recombine' "
            f"in place would silently drop the gate.")
    if _shared_apply_write is not None:
        return _shared_apply_write(adaptation, h_out, payload)
    write = getattr(adaptation, "write", None)
    return payload if write is None else write(h_out, payload)


# ---------------------------------------------------------------------------
# The capability surface for W_phi
# ---------------------------------------------------------------------------
# The W labels this engine APPLIES at a write port.  This is a
# capability surface, not documentation: the library's serving builder
# probes it (adapters/serving.py::refuse_custom_write) and refuses, at
# checkpoint-load time and by name, a member whose W this engine does
# not serve — instead of the blanket "any non-default W" refusal it had
# to use while the engine hard-coded W = replace.
#
# Why a WHITELIST and not "whatever the leaf defines".  apply_write
# duck-types ``leaf.write(h_out, payload)``, so mechanically it would
# run any write at all; what the engine cannot check from inside that
# call is whether the ``payload`` it assembled is the one that write was
# written for.  The labels below are exactly the ones whose payload
# contract this engine satisfies and whose numerics the suite pins
# (tests/adaptation/test_inout_sites.py::TestWriteCapabilitySurface):
#
#   ``replace``      the default W — ``BaseAdapter.write`` returns the
#                    payload unchanged, which is every member that
#                    predates the W axis, and the bare pipe.
#   ``interpolate``  ``InterpolateWrite``'s gated mix,
#                    ``h + g*(payload - h)``: the re-executing pipe's
#                    gate, bit-exact identity at g = 0.
#
# ``add`` and ``norm_mix`` are deliberately ABSENT, and not because
# ``h_out + payload`` is hard.  Each is the write of a ROUTE this engine
# does not have.  ``AddWrite`` is a FORWARD pipe's write: it deposits a
# delta at a LATER port, and nothing here carries a payload from one
# layer's hook to another's.  ``NormMixWrite`` needs ``payload`` to be a
# unit direction produced at the source port of a CARRY pipe, which
# needs the per-request state slot the fork deliberately does not have
# (docs/recirculation-serving.md §5 in the adapters repo).  Running
# either against a payload assembled the way THIS engine assembles it
# would compute a different function, fluently and without an error —
# the exact failure the W axis exists to make loud.  A route that grows
# adds its label here in the same commit that implements it.
SUPPORTED_WRITE_LABELS = frozenset({"replace", "interpolate"})

# ---------------------------------------------------------------------
# Serving-requirement capability surface.
#
# A member may DECLARE what the engine must be configured as for its
# computation to be the one it was trained as.  The declaration is a
# property of the member (a class attribute on the leaf, written by the
# adapters library), and the engine's job is to honour it or to refuse.
#
# Two routes install a member and only one of them used to honour these:
#
#   * BAKED — ``adapter_config`` rides ``EngineArgs`` at construction,
#     so ``_enforce_unchunked_prefill_for_sequence_mixing`` sees the
#     member before the scheduler is configured and forces the setting.
#   * MULTISITE — ``LLM(...)`` is built FIRST and members arrive later
#     by ``collective_rpc("load_adapter", ...)``.  ``adapter_config`` is
#     None at engine-config time, so every policy gated on
#     ``if not self.adapter_config`` was INERT here, while
#     ``_set_default_args`` turns ``enable_chunked_prefill`` on
#     unconditionally for every v1 generate model.  A sequence-mixing
#     member mounted at, say, ``linear:self_attn.v_proj`` therefore
#     served with its scan silently reset at every chunk boundary, and
#     the output still read fluently.
#
# This flag is the ANSWER to "does this fork honour an unchunked-prefill
# declaration on the multisite route too?".  It is True from the commit
# that made it so: the declaration is carried to engine-config time in
# ``EngineArgs.adapter_serving_requirements`` (forced there), and
# re-checked against the FROZEN config at load time in
# ``WorkerBase.load_adapter``, which REFUSES by member and requirement
# when the engine cannot satisfy it.  A builder that cannot find this
# name is talking to a fork that predates the guarantee and must keep
# refusing such a member itself — the same "unknown is not yes" rule
# that governs SUPPORTED_WRITE_LABELS above.
MULTISITE_HONOURS_UNCHUNKED_PREFILL = True

# The leaf attributes that carry a declaration.  Named here because the
# fork READS them off a reconstructed member: they travel with the class,
# not with the manifest, so a member exported by any version of the
# adapters library declares the same way.
SERVING_REQUIREMENT_ATTRS = {
    "unchunked_prefill": "serving_requires_unchunked_prefill",
    "eager": "serving_requires_eager",
}


def declared_serving_requirements(adaptation: nn.Module) -> dict:
    """What *adaptation* declares it needs from the engine.

    Returns ``{"unchunked_prefill": bool, "eager": bool}``.  Composites
    (``adaptation.members``) are recursed: a declaration does not
    disappear by being wrapped, exactly as
    ``check_adaptation_supported`` treats a rejection.

    Only an EXPLICIT declaration counts.  Sequence mixing is a separate,
    inferred predicate (``needs_sequence_segmentation``) that the baked
    route uses; conflating the two here would newly constrain every
    mixer member on the multisite route, which is a policy change and
    not this surface's business.
    """
    out = {k: False for k in SERVING_REQUIREMENT_ATTRS}
    if adaptation is None:
        return out
    for key, attr in SERVING_REQUIREMENT_ATTRS.items():
        if bool(getattr(adaptation, attr, False)):
            out[key] = True
    for member in (getattr(adaptation, "members", None) or ()):
        for key, value in declared_serving_requirements(member).items():
            out[key] = out[key] or value
    return out


def readout_then_write(adaptation: nn.Module,
                       hidden: torch.Tensor) -> torch.Tensor:
    """The DIAGONAL (F_in == F_out) form of the two-step contract.

    One port, so the stream the member reads is the stream it writes:
    ``write(h, readout(h))``.  With the default W this is exactly
    ``readout(h)``, which is what this engine did before the W axis.
    """
    return apply_write(adaptation, hidden, adaptation.readout(hidden))


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
    ``lerp(h, W(h, R(h)), mask)`` — written as ``h + mask*(out-h)``.
    R is a MAP, not a correction: replacement and multiplicative
    readouts are first-class (mask=1 yields exactly the member's value,
    whatever R and W are).

    This site is DIAGONAL (F_in == F_out), so the member's W_phi is
    applied against the same stream R read — ``write(h, readout(h))``,
    which for the default W (a replacement) is exactly ``readout(h)``.

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
    h3d = hidden.unsqueeze(0)
    out = apply_write(adaptation, h3d, adaptation.readout(h3d)).squeeze(0)
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
