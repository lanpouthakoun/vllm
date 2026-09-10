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
    "CARRIER_DTYPE_ATTR",
    "CARRIER_DTYPE_STAMP",
    "CarrierDtypeUnknown",
    "declared_serving_requirements",
    "declared_carrier_dtype",
    "resolve_carrier_dtype",
    "stamp_carrier_dtype",
    "carrier_dtype_of",
    "readout_at_port",
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


# ---------------------------------------------------------------------------
# The member's CARRIER DTYPE — what the stream is, INSIDE the member
# ---------------------------------------------------------------------------
# A member is mounted in a host that runs at ONE dtype (bf16, here).  It
# does not follow that the member computes at that dtype.  The HF side
# has always allowed a member to run its own port in fp32 inside a bf16
# host — ``campaign/state_tracking_transfer/dichotomy/members.py``'s
# ``Fp32Stream`` is exactly that and nothing else::
#
#     def readout(self, fx, state=None, x=None):
#         return self.leaf.readout(fx.float(), state, x).to(fx.dtype)
#
# cast the stream UP at the port, run the leaf, cast the payload back
# DOWN at the port.  The dtype in the middle is the member's CARRIER
# DTYPE: a property of the member, not of the host.
#
# This engine used to have no name for it.  ``layer._prepare_adapter``
# cast every ``nn.Linear`` in a mounted member to the MODEL's dtype and
# called ``readout`` on the raw bf16 stream — so an fp32 member landed
# with bf16 weights and an fp32-casting body, and the first matmul died
# with ``expected mat1 and mat2 to have the same dtype, but got: float
# != c10::BFloat16`` (campaign/serving/parity_headline, job 312676, the
# S5 ``ext`` member: admitted, exported bit-for-bit, baked route, then
# this).  A member that happens to PIN its own parameters (the faithful
# native members' ``_apply``) survived that cast only by undoing it.
#
# So the carrier dtype is named, carried and applied:
#
#   * the serving record DECLARES it (``adapters.mounting.save_members``
#     writes ``carrier_dtype`` per record; ``adapters.serving`` derives
#     it from the checkpoint's own parameter dtype when the member's
#     class does not say), and it rides ``adapter_config`` to both
#     routes;
#   * a member's CLASS may declare it directly
#     (``serving_carrier_dtype``) — which is how a member whose
#     parameters are fp32 but whose PORT is bf16 (the faithful native
#     members: fp32 mixer math, bf16 carrier, cast back at the fusion)
#     says so;
#   * failing both, it is DERIVED from the member's own floating-point
#     parameters, which must be unanimous;
#   * and if it can be neither declared nor derived, the member is
#     REFUSED BY NAME rather than served through a guessed cast.
#
# Nothing else in this engine up- or down-casts a member: the weights
# are left exactly as the checkpoint had them, and the ONLY casts are
# the two at the port.
# The name is the LIBRARY's, imported for the same reason the site table
# is: a member declares its carrier once, and the fork must not keep a
# second copy of what that declaration is called.  The fallback is only
# for a library predating the axis, which by definition has no member
# that declares one.
try:
    from adapters._base import CARRIER_DTYPE_ATTR
except ImportError:  # pragma: no cover - depends on library age
    CARRIER_DTYPE_ATTR = "serving_carrier_dtype"

#: where the resolved dtype is stamped on a prepared member, so the
#: per-token hot path does not re-derive it (and so a refusal happens at
#: LOAD time, not inside a forward).
CARRIER_DTYPE_STAMP = "_adapter_carrier_dtype"

_CARRIER_UNSET = object()


class CarrierDtypeUnknown(RuntimeError):
    """A member's carrier dtype can be neither declared nor derived.

    Raised by name, with the member in it.  Serving on regardless would
    mean picking a cast for it — which is the silent numeric change this
    whole surface exists to make impossible."""


def _as_torch_dtype(value) -> Optional["torch.dtype"]:
    """``torch.float32`` / ``"float32"`` / ``"torch.float32"`` -> dtype."""
    if value is None:
        return None
    if isinstance(value, torch.dtype):
        return value
    name = str(value).split(".")[-1]
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise CarrierDtypeUnknown(
            f"{value!r} is not a torch dtype: a carrier dtype is declared "
            f"as a torch.dtype or its name (e.g. 'float32', 'bfloat16').")
    return dtype


def declared_carrier_dtype(adaptation: nn.Module) -> Optional["torch.dtype"]:
    """The carrier dtype *adaptation*'s CLASS declares, or ``None``.

    Composites (``adaptation.members``) are recursed and must AGREE: a
    declaration does not disappear by being wrapped, and two members
    that want different ports cannot share one.
    """
    if adaptation is None:
        return None
    found = set()
    own = getattr(adaptation, CARRIER_DTYPE_ATTR, None)
    if own is not None:
        found.add(_as_torch_dtype(own))
    for member in (getattr(adaptation, "members", None) or ()):
        sub = declared_carrier_dtype(member)
        if sub is not None:
            found.add(sub)
    if not found:
        return None
    if len(found) > 1:
        raise CarrierDtypeUnknown(
            f"REFUSING to serve {type(adaptation).__name__}: its members "
            f"declare DIFFERENT carrier dtypes "
            f"({sorted(str(d) for d in found)}) through "
            f"{CARRIER_DTYPE_ATTR!r}. One composite has one port; there "
            f"is no cast that is right for both.")
    return found.pop()


def _parameter_dtypes(adaptation: nn.Module) -> set:
    """The floating-point parameter dtypes *adaptation* itself holds.

    A shared view (``adapters.serving._SharedServingView``) keeps its
    core as a PLAIN attribute so per-layer views do not duplicate the
    parameters — ``nn.Module.parameters()`` never reaches it — so it is
    walked explicitly.  Buffers are deliberately NOT consulted: a
    bookkeeping buffer's dtype is not what the member's matmuls run in.

    A COMPOSITE's own parameters exclude its members'.  A composite is
    not one computation at one port: ``layer._blend_one_adaptation``
    applies each member under its own mask, so each is cast at the port
    in its own right and two members may legitimately carry different
    dtypes.  Folding theirs in here would refuse exactly that.
    """
    out = set()
    members = getattr(adaptation, "members", None) or ()
    owned_by_members = {id(p) for m in members for p in m.parameters()}
    for p in adaptation.parameters():
        if p.is_floating_point() and id(p) not in owned_by_members:
            out.add(p.dtype)
    core = getattr(adaptation, "core", None)
    if isinstance(core, nn.Module):
        for p in core.parameters():
            if p.is_floating_point():
                out.add(p.dtype)
    return out


def resolve_carrier_dtype(adaptation: nn.Module,
                          declared=None,
                          label: Optional[str] = None,
                          ) -> Optional["torch.dtype"]:
    """*adaptation*'s carrier dtype: declared, else derived, else refused.

    Args:
        adaptation: the member.
        declared: what the SERVING RECORD says (``adapter_config``'s
            ``carrier_dtype``), if anything.  The record wins, because
            it was written from the checkpoint that the weights came
            from.
        label: the member's name, for the refusal.

    Returns ``None`` for a member with no floating-point parameters at
    all — it has no carrier of its own, so it simply runs in the
    stream's dtype and NOTHING is cast.  That is the absence of a cast,
    not a guessed one.
    """
    if declared is not None:
        return _as_torch_dtype(declared)
    own = declared_carrier_dtype(adaptation)
    if own is not None:
        return own
    dtypes = _parameter_dtypes(adaptation)
    if not dtypes:
        return None
    if len(dtypes) == 1:
        return dtypes.pop()
    raise CarrierDtypeUnknown(
        f"REFUSING to serve {label or type(adaptation).__name__}: its "
        f"CARRIER DTYPE cannot be established. The member's class does "
        f"not declare {CARRIER_DTYPE_ATTR!r}, the serving record carries "
        f"no 'carrier_dtype', and its own floating-point parameters are "
        f"NOT unanimous ({sorted(str(d) for d in dtypes)}), so there is "
        f"no dtype the stream can be cast to at the port. The engine "
        f"casts the stream to the member's carrier at F_in and the "
        f"payload back at F_out and does not up- or down-cast anywhere "
        f"else; picking one of these dtypes would silently change what "
        f"this member computes. Declare "
        f"{CARRIER_DTYPE_ATTR} = '<dtype>' on the leaf class, or export "
        f"the member with a 'carrier_dtype' in its serving record "
        f"(adapters.mounting.save_members).")


def stamp_carrier_dtype(adaptation: nn.Module,
                        declared=None,
                        label: Optional[str] = None,
                        ) -> Optional["torch.dtype"]:
    """Resolve *adaptation*'s carrier dtype and record it ON the member.

    Called once, at LOAD time, so the refusal lands where a human is
    looking (``load_adapter`` / engine construction) instead of inside a
    forward, and so the per-token path is a plain attribute read.
    Composite members are stamped individually — each is applied at the
    port in its own right (``layer._blend_one_adaptation``).
    """
    for member in (getattr(adaptation, "members", None) or ()):
        stamp_carrier_dtype(member, declared=declared, label=label)
    carrier = resolve_carrier_dtype(adaptation, declared=declared,
                                    label=label)
    object.__setattr__(adaptation, CARRIER_DTYPE_STAMP, carrier)
    return carrier


def carrier_dtype_of(adaptation: nn.Module) -> Optional["torch.dtype"]:
    """The carrier dtype of a mounted member, as the hot path reads it.

    Uses the stamp left by ``stamp_carrier_dtype`` when there is one and
    resolves (and stamps) on first use otherwise — the legacy in-process
    registry path installs members without going through
    ``_prepare_adapter``.
    """
    stamped = getattr(adaptation, CARRIER_DTYPE_STAMP, _CARRIER_UNSET)
    if stamped is not _CARRIER_UNSET:
        return stamped
    return stamp_carrier_dtype(adaptation)


def readout_at_port(adaptation: nn.Module,
                    h_in: torch.Tensor) -> torch.Tensor:
    """R_phi at F_in, run in the member's carrier dtype.

    The two casts of ``Fp32Stream``, and only those two: the stream goes
    IN as the member's carrier and the payload comes back OUT in the
    stream's own dtype, so ``write`` (W_phi at F_out) is always applied
    with both of its arguments in the stream's dtype.
    """
    carrier = carrier_dtype_of(adaptation)
    stream_dtype = h_in.dtype
    if carrier is not None and carrier is not stream_dtype:
        h_in = h_in.to(carrier)
    payload = adaptation.readout(h_in)
    if payload.dtype is not stream_dtype:
        payload = payload.to(stream_dtype)
    return payload


def readout_then_write(adaptation: nn.Module,
                       hidden: torch.Tensor) -> torch.Tensor:
    """The DIAGONAL (F_in == F_out) form of the two-step contract.

    One port, so the stream the member reads is the stream it writes:
    ``write(h, readout(h))``.  With the default W this is exactly
    ``readout(h)``, which is what this engine did before the W axis.

    R runs at the member's CARRIER dtype and its payload comes back in
    the stream's dtype (``readout_at_port``), so W is applied in the
    stream's dtype on both arguments — bit-identical to the old path for
    a member whose carrier IS the stream's dtype, which is every member
    that ever served before the carrier axis existed.
    """
    return apply_write(adaptation, hidden, readout_at_port(adaptation, hidden))


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

    The member runs at its own CARRIER dtype: the stream is cast to it
    on the way in and the payload cast back on the way out
    (``readout_at_port``), so the blend, the write and the mask are all
    in the STREAM's dtype.  A member whose carrier is the stream's dtype
    — every member that served before this axis existed — takes exactly
    the same arithmetic it took before, with no cast at all.

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
        # A member that blends itself still reads and writes the stream
        # at the port, so the port casts are the same two.
        carrier = carrier_dtype_of(adaptation)
        if carrier is not None and carrier is not hidden.dtype:
            return apply_masked(hidden.to(carrier), mask).to(hidden.dtype)
        return apply_masked(hidden, mask)
    h3d = hidden.unsqueeze(0)
    out = readout_then_write(adaptation, h3d).squeeze(0)
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
