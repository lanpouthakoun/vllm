"""vllm.adaptation.specs – first-class adapter (Representation Fine-Tuning) support in vLLM.

This package owns everything required to run adapter-adapted models in vLLM:

  * ``VllmConfig.adapter_config`` — serializable blueprint dict that flows from
    ``LLM()`` → ``EngineArgs`` → ``VllmConfig`` → model constructors.
    Multiprocess-safe — no global state required.
  * ``spec_to_adapter_config()`` / ``adapter_config_to_spec()`` — convert between
    live adapter_spec (with nn.Module adapters) and serializable config dicts.
  * adapter-aware decoder layer factories (see layer.py).

Usage (preferred — via VllmConfig)::

    from vllm.adaptation.specs import spec_to_adapter_config

    adapter_spec = adapter_model.export_vllm_adapter_spec()
    llm = LLM(model=model_name, adapter_config=spec_to_adapter_config(adapter_spec))

Or simply::

    llm = adapter_model.build_vllm(model_name, **kwargs)

The deprecated ``set_adapter_spec()`` / ``clear_adapter_spec()`` global-state API
is still supported as a fallback (used by TRL training hooks) but should not
be used in new code.
"""

import logging
import os
import tempfile
import threading
from typing import Any, Optional

logger = logging.getLogger("vllm.adaptation")

from vllm.adaptation.manager import ServedAdapter, AdapterManager
from vllm.adaptation.request import AdapterRequest

__all__ = [
    "ServedAdapter",
    "AdapterManager",
    "AdapterRequest",
    "adapter_config_to_spec",
    "spec_to_adapter_config",
    # Deprecated global-state API (backward compat only):
    "set_adapter_spec",
    "get_adapter_spec",
    "clear_adapter_spec",
]

# ---------------------------------------------------------------------------
# Thread-local adapter spec storage (same-process / same-thread fast path)
# ---------------------------------------------------------------------------

_adapter_spec_local = threading.local()

# Environment variable that holds the path to the pickled spec file.
# Set by set_adapter_spec() in the parent process; inherited by spawned workers.
_SPEC_FILE_ENV_KEY = "_VLLM_ADAPTER_SPEC_FILE"


import warnings


def _serialize_state_dict(state_dict: dict) -> dict:
    """Convert a state_dict's tensors to a format safe for any serializer.

    Each tensor becomes ``{"__adapter_t": True, "data": <bytes>,
    "dtype": "bfloat16", "shape": [...]}``.  This survives both pickle
    (used by VllmConfig) and msgspec (used by collective_rpc).

    Uses ``numpy().tobytes()`` instead of ``.tolist()`` because a large
    nested Python list is ~100x bigger and ~50x slower to construct/pickle
    than the equivalent contiguous bytes blob.  At hidden=8192 × rank=8
    × 80 layers × 128 adapters, the .tolist() path turned into a
    multi-minute serialization stall + IPC backpressure that hung
    multi-adapter sweeps mid-loop.
    """
    import torch
    out = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            t = v.detach().cpu().contiguous()
            # bf16 has no numpy dtype; round-trip through fp32 bytes for
            # cross-process transport, then cast back on the receiving side.
            saved_dtype = str(v.dtype).split(".")[-1]  # "bfloat16"
            if t.dtype == torch.bfloat16:
                wire = t.float()
                wire_dtype_np = "float32"
            else:
                wire = t
                wire_dtype_np = str(t.dtype).split(".")[-1]
            out[k] = {
                "__adapter_t": True,
                "data": wire.numpy().tobytes(),
                "dtype": saved_dtype,
                "wire_dtype": wire_dtype_np,
                "shape": list(v.shape),
            }
        else:
            out[k] = v
    return out


def _deserialize_state_dict(state_dict: dict) -> dict:
    """Reconstruct tensors from the format produced by ``_serialize_state_dict``.

    Handles three encodings for backward compatibility:
      - raw torch tensors (no-op)
      - new bytes-encoded entries (``data`` is a bytes blob with
        ``wire_dtype`` describing the wire format)
      - legacy list-encoded entries (``data`` is a nested Python list)
    """
    import numpy as np
    import torch
    out = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            out[k] = v
        elif isinstance(v, dict) and v.get("__adapter_t"):
            target_dtype = getattr(torch, v["dtype"], torch.float32)
            data = v["data"]
            if isinstance(data, (bytes, bytearray, memoryview)):
                wire_np = getattr(np, v.get("wire_dtype", v["dtype"]), np.float32)
                arr = np.frombuffer(data, dtype=wire_np).reshape(v["shape"])
                # `arr` may share memory with the input bytes; copy so the
                # resulting tensor is writable and outlives the bytes object.
                out[k] = torch.from_numpy(arr.copy()).to(dtype=target_dtype)
            else:
                # Legacy list-encoded path.
                out[k] = torch.tensor(data, dtype=target_dtype).reshape(v["shape"])
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# VllmConfig-based API (preferred — multiprocess-safe, no global state)
# ---------------------------------------------------------------------------

def spec_to_adapter_config(adapter_spec: dict[str, Any]) -> dict[str, Any]:
    """Convert a live *adapter_spec* (with nn.Module adapters) to a serializable
    dict suitable for ``VllmConfig.adapter_config``.

    The result contains only plain Python types and CPU tensors, so it can be
    pickled across processes by vLLM's multiprocessing machinery.
    """
    config: dict[str, Any] = {}
    config["layer_indices"] = list(adapter_spec["layer_indices"])
    config["position"] = adapter_spec["position"]

    # Serialize sample_adapter → blueprint
    adapter = adapter_spec.get("sample_adapter")
    if adapter is not None and not isinstance(adapter, dict):
        config["sample_adapter"] = _adapter_to_blueprint(adapter)
    elif adapter is not None:
        config["sample_adapter"] = adapter  # already a blueprint

    # Serialize per-layer adapters → portable state_dicts
    adapters = adapter_spec.get("adapters")
    if adapters is not None:
        def _portable_state(a):
            sd = dict(a.state_dict())
            # shared views: layer_emb is a non-persistent buffer, and the
            # per-layer embedding is the ONLY state a view owns — carry it
            if getattr(a, "core", None) is not None and hasattr(a, "layer_emb"):
                sd["layer_emb"] = a.layer_emb
            return _serialize_state_dict(sd)

        config["adapter_states"] = {
            idx: _portable_state(a) for idx, a in adapters.items()
        }

    # Pass through any extra keys (e.g. debug_mask)
    for k in adapter_spec:
        if k not in ("layer_indices", "position", "sample_adapter", "adapters"):
            config[k] = adapter_spec[k]

    return config


def adapter_config_to_spec(adapter_config: Optional[dict[str, Any]],
                        ) -> Optional[dict[str, Any]]:
    """Convert a serialized ``VllmConfig.adapter_config`` back into a live
    *adapter_spec* with reconstructed nn.Module adapters.

    Returns ``None`` if *adapter_config* is ``None``.
    """
    if adapter_config is None:
        return None

    import copy

    spec: dict[str, Any] = dict(adapter_config)

    # Reconstruct sample_adapter from blueprint
    adapter = spec.get("sample_adapter")
    if isinstance(adapter, dict) and adapter.get("__type__") in (
            "AdapterBlueprint", "SharedViewBlueprint"):
        spec["sample_adapter"] = _blueprint_to_adapter(adapter)

    # Reconstruct per-layer adapters from saved state dicts
    adapter_states = spec.pop("adapter_states", None)
    sample = spec.get("sample_adapter")
    if adapter_states is not None and sample is not None:
        adapters: dict[int, Any] = {}
        shared_core = getattr(sample, "core", None)
        for idx, sd in adapter_states.items():
            state = _deserialize_state_dict(sd)
            if shared_core is not None and "layer_emb" in state:
                # rebuild the view around the ONE shared core with its
                # own per-layer embedding (deepcopy would fork the core
                # and stamp every layer with the sample's embedding)
                a = type(sample)(shared_core, int(idx),
                                 state["layer_emb"])
            else:
                a = copy.deepcopy(sample)
                a.load_state_dict(state, strict=False)
            if hasattr(a, "install_inference_caches"):
                a.install_inference_caches()
            adapters[int(idx)] = a
        spec["adapters"] = adapters

    return spec


# ---------------------------------------------------------------------------
# Deprecated global-state API (kept for backward compatibility)
# ---------------------------------------------------------------------------

def set_adapter_spec(spec: Optional[dict[str, Any]]) -> None:
    """**Deprecated.** Use ``LLM(model=..., adapter_config=spec_to_adapter_config(spec))``
    and let VllmConfig carry the config to model constructors instead.
    """
    _adapter_spec_local.spec = spec
    if spec is not None:
        _write_spec_file(spec)
    else:
        _remove_spec_file()


def get_adapter_spec() -> Optional[dict[str, Any]]:
    """Return the active adapter spec for this thread, or ``None``.

    Falls back to deserialising from the temp file when called from a spawned
    worker process that did not inherit the thread-local.
    """
    spec = getattr(_adapter_spec_local, "spec", None)
    if spec is not None:
        return spec
    # Cross-process fallback: spawned workers inherit env vars.
    return _read_spec_file()


def clear_adapter_spec() -> None:
    """**Deprecated.** No longer needed when using VllmConfig.adapter_config."""
    _adapter_spec_local.spec = None
    _remove_spec_file()


# ---------------------------------------------------------------------------
# Temp-file helpers for cross-process spec passing
# ---------------------------------------------------------------------------

_SENTINEL = object()


def _adapter_to_blueprint(adapter) -> dict:
    """Extract a serializable blueprint from any adapter instance.

    Stores the class path and constructor kwargs so the adapter can be
    re-instantiated fresh in spawned worker processes.  This avoids two
    problems with saving the module directly:

    1. ``torch.save`` cannot pickle modules that have
       ``torch.nn.utils.parametrizations.orthogonal`` (or ``weight_norm``)
       applied.
    2. De-parametrizing before saving causes a key-name mismatch when
       ``sync_adapter_state`` loads the HF state dict: the HF checkpoint stores
       ``rotate_layer.parametrizations.weight.original`` but a de-parametrized
       adapter only has ``rotate_layer.weight``, so the trained R matrix is
       silently skipped and the adapter runs with the wrong rotation.

    A fresh adapter instantiated from the blueprint has the full parametrized
    key tree, so the HF state dict loads without any remapping and all
    trained weights — including R — are correctly applied.

    Works for any adapter class; constructor kwargs are extracted by walking
    the MRO to find the first ``__init__`` with explicit named parameters, then
    matching them to stored instance attributes.
    """
    import inspect

    cls = type(adapter)

    # Shared-view adapters (a per-layer wrapper holding one shared core +
    # a depth embedding) have no ctor-kwarg identity of their own: blueprint
    # the core recursively and carry layer_idx + the embedding tensor.
    _core = getattr(adapter, "core", None)
    if _core is not None and hasattr(adapter, "layer_emb"):
        return {
            "__type__": "SharedViewBlueprint",
            "__module__": cls.__module__,
            "__qualname__": cls.__qualname__,
            "layer_idx": adapter.layer_idx,
            "layer_emb": _serialize_state_dict(
                {"layer_emb": adapter.layer_emb}),
            "core": _adapter_to_blueprint(_core),
        }

    # Walk the MRO and accumulate explicit named parameters from every
    # __init__ in the chain.  Subclasses like LoadapterRidgeAdapter define
    # ``def __init__(self, *args, lam=1e-3, **kwargs)`` — the old code
    # stopped at the first __init__ with *any* explicit param (``lam``)
    # and never reached W2Adapter.__init__ which defines ``hidden_size``,
    # ``low_rank_dim``, etc.  Now we collect params from all levels, with
    # subclass params taking priority on name collisions.
    init_params: dict = {}
    for klass in reversed(cls.__mro__):
        if klass is object:
            continue
        init_fn = klass.__dict__.get("__init__")
        if init_fn is None:
            continue
        sig = inspect.signature(init_fn)
        explicit = {
            k: v for k, v in sig.parameters.items()
            if k != "self" and v.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        }
        init_params.update(explicit)

    kwargs: dict = {}
    for name, param in init_params.items():
        if name == "device":
            kwargs["device"] = "cpu"
            continue

        val = getattr(adapter, name, _SENTINEL)
        if val is _SENTINEL:
            # Attribute not stored; use the default if one exists.
            if param.default is not inspect.Parameter.empty:
                kwargs[name] = param.default
        elif isinstance(val, (int, float, str, bool, type(None))):
            kwargs[name] = val
        elif name == "dtype":
            # torch.dtype → string, e.g. "torch.bfloat16"
            kwargs[name] = str(val)
        elif name == "mixer":
            # Stored as an nn.Module instance; reverse-lookup the string key.
            kwargs[name] = _mixer_instance_to_name(val)
        elif param.default is not inspect.Parameter.empty:
            # Non-serializable value; fall back to the default.
            kwargs[name] = param.default
        # else: omit — constructor must not require it or will use its default

    # dtype is not always stored as a direct attribute; derive from weights.
    # Check for None too — the MRO walk may pick up the default (None) when
    # the adapter doesn't store dtype as self.dtype.
    if kwargs.get("dtype") is None:
        ls = getattr(adapter, "learned_source", None)
        if ls is not None and hasattr(ls, "weight"):
            kwargs["dtype"] = str(ls.weight.dtype)

    # Save trained weights alongside the blueprint so that worker processes
    # start with correct weights *before* CUDA graph capture / torch.compile
    # warmup.  Without this, the blueprint adapter is constructed with random
    # init weights, and post-init sync_adapter_state() updates them in-place —
    # but compiled/captured graphs may not reflect the in-place updates.
    adapter_state = _serialize_state_dict(adapter.state_dict())

    blueprint = {
        "__type__": "AdapterBlueprint",
        "__module__": cls.__module__,
        "__qualname__": cls.__qualname__,
        "kwargs": kwargs,
        "state_dict": adapter_state,
    }
    logger.debug(
        "_adapter_to_blueprint: %s from %s | kwargs=%s | state_keys=%s",
        cls.__qualname__, cls.__module__,
        sorted(kwargs.keys()), sorted(adapter_state.keys()),
    )
    return blueprint


def _mixer_instance_to_name(mixer_instance) -> Optional[str]:
    """Return the MIXER_REGISTRY key for *mixer_instance*, or ``None``."""
    if mixer_instance is None:
        return None
    try:
        from pyreft.adapters._mixer import MIXER_REGISTRY
        for name, mixer_cls in MIXER_REGISTRY.items():
            if isinstance(mixer_instance, mixer_cls):
                return name
    except ImportError:
        pass
    return None


def _blueprint_to_adapter(blueprint: dict):
    """Re-instantiate a fresh adapter from a saved blueprint dict.

    Imports the adapter class by its module path, converts the stored dtype
    string back to a ``torch.dtype``, then calls the constructor.  If the
    blueprint contains a ``state_dict``, loads trained weights immediately
    so the adapter is ready *before* CUDA graph capture / torch.compile.
    """
    import importlib
    import torch

    if blueprint.get("__type__") == "SharedViewBlueprint":
        core = _blueprint_to_adapter(blueprint["core"])
        emb = _deserialize_state_dict(blueprint["layer_emb"])["layer_emb"]
        vmod = importlib.import_module(blueprint["__module__"])
        vcls = getattr(vmod, blueprint["__qualname__"])
        return vcls(core, blueprint["layer_idx"], emb)

    mod_name = blueprint["__module__"]
    # Backward compat: old blueprints stored "adaptors.*" module paths
    if mod_name.startswith("adaptors."):
        mod_name = mod_name.replace("adaptors.", "pyreft.adapters.", 1)
    qual_name = blueprint["__qualname__"]
    has_state = "state_dict" in blueprint and blueprint["state_dict"]
    logger.debug(
        "_blueprint_to_adapter: module=%s qualname=%s kwargs_keys=%s has_state_dict=%s",
        mod_name, qual_name, sorted(blueprint['kwargs'].keys()), has_state,
    )

    mod = importlib.import_module(mod_name)
    cls = getattr(mod, qual_name)

    kwargs = dict(blueprint["kwargs"])
    dtype_val = kwargs.get("dtype")
    if isinstance(dtype_val, str):
        # "torch.bfloat16" → torch.bfloat16
        kwargs["dtype"] = getattr(torch, dtype_val.split(".")[-1], None)

    adapter = cls(**kwargs)

    # Load trained weights if available in the blueprint.
    saved_state = blueprint.get("state_dict")
    if saved_state:
        adapter.load_state_dict(_deserialize_state_dict(saved_state), strict=False)
        if hasattr(adapter, "install_inference_caches"):
            adapter.install_inference_caches()

    return adapter


def _write_spec_file(spec: dict) -> None:
    """Serialise *spec* to a temp file and record the path in the environment.

    Delegates to :func:`spec_to_adapter_config` — the SAME serializer the
    ``VllmConfig.adapter_config`` (fork-mode) path uses — so spawned
    workers reading the file deserialize exactly what fork-mode workers
    receive.  In particular, shared-view adapters (one shared core plus
    per-layer embeddings) carry their ``layer_emb`` through
    ``adapter_states``; the old raw ``state_dict()`` save dropped the
    non-persistent buffer and every layer came back stamped with the
    sample's embedding.
    """
    import torch

    saveable_spec = spec_to_adapter_config(spec)

    fd, path = tempfile.mkstemp(suffix=".pt", prefix="vllm_adapter_spec_")
    os.close(fd)
    try:
        torch.save(saveable_spec, path)
    except Exception:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    os.environ[_SPEC_FILE_ENV_KEY] = path


def _read_spec_file() -> Optional[dict]:
    """Load and return the spec from the temp file, or ``None``.

    Delegates to :func:`adapter_config_to_spec` — the same deserializer
    the ``VllmConfig.adapter_config`` path uses — so spawn-mode workers
    reconstruct adapters identically to fork-mode workers (both
    ``AdapterBlueprint`` and ``SharedViewBlueprint`` samples, and the
    shared-core rebuild for per-layer ``adapter_states``).
    """
    import torch

    path = os.environ.get(_SPEC_FILE_ENV_KEY)
    if not path or not os.path.exists(path):
        return None
    try:
        spec = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None

    try:
        return adapter_config_to_spec(spec)
    except Exception as e:
        adapter = spec.get("sample_adapter") if isinstance(spec, dict) else None
        logger.error(
            "Failed to reconstruct adapter spec from %s: %s. "
            "sample_adapter type=%s",
            path, e,
            adapter.get("__type__") if isinstance(adapter, dict)
            else type(adapter).__name__,
        )
        return None


def _remove_spec_file() -> None:
    """Delete the temp file and clear the env var."""
    path = os.environ.pop(_SPEC_FILE_ENV_KEY, None)
    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except OSError:
            pass
