"""adapter model manager with LRU eviction.

Manages the lifecycle of adapters: registration (CPU cache),
activation (GPU layers), deactivation, and LRU eviction.  Mirrors the
``LRUCacheLoRAModelManager`` pattern from ``vllm.lora.models`` but is
much simpler because adapters are tiny (~16K params/layer).
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn

from vllm.adaptation.layer import (_add_adapter_to_layer, _prepare_adapter,
                             _remove_adapter_from_layer)
from vllm.utils import LRUCache

logger = logging.getLogger("vllm.adaptation.manager")


@dataclass
class ServedAdapter:
    """CPU-side representation of a registered adapter."""

    id: int
    """Unique integer ID (>= 1).  0 is reserved for base model."""
    position: str
    """Position mode: ``"first"``, ``"last"``, ``"prefill"``,
    ``"decode"``, or ``"all"``.  ``"decode"`` is the exact complement of
    ``"prefill"``."""
    adapter_config: dict
    """Serializable config dict (from ``spec_to_adapter_config``)."""
    layer_indices: frozenset[int] = field(default_factory=frozenset)
    """Which decoder layers this adapter applies to."""
    site: str = "block_output"
    """Mount point inside each decoder layer: ``"block_output"``
    (default), ``"block_input"``, ``"post_attn"``, ``"post_mlp"``, or
    ``"linear:<submodule.path>"``."""

    def __post_init__(self):
        if self.id < 1:
            raise ValueError(
                f"adapter id must be >= 1, got {self.id}")


class _AdapterLRUCache(LRUCache[int, ServedAdapter]):
    """LRU cache that calls *deactivate_fn* when an entry is evicted."""

    def __init__(self, capacity: int, deactivate_fn):
        super().__init__(capacity)
        self.deactivate_fn = deactivate_fn

    def _on_remove(self, key: int, value: Optional[ServedAdapter]):
        logger.debug("LRU evicting adapter id=%d", key)
        self.deactivate_fn(key)
        return super()._on_remove(key, value)


class AdapterManager:
    """Centralized lifecycle manager for adapters.

    Two-tier LRU caching:
      - **registered** (CPU, ``max_cpu_adapters``): holds ``ServedAdapter``
        descriptors.  Cheap to store — just metadata + serializable config.
      - **active** (GPU, ``max_adapters``): adapters whose weights are
        loaded into the decoder layer ``nn.ModuleDict``.  When full,
        the least-recently-used adapter is evicted from GPU.

    The manager reuses the existing per-layer functions from
    ``vllm.adaptation.layer``: ``_prepare_adapter`` and ``_add_adapter_to_layer``
    for loading, and the deletion logic from ``worker_base.py`` for
    unloading.
    """

    def __init__(
        self,
        adapter_layers: list[nn.Module],
        max_adapters: int = 256,
        max_cpu_adapters: int = 1024,
        device: torch.device = torch.device("cuda"),
        model_dtype: torch.dtype = torch.bfloat16,
        chunked_prefill_enabled: bool = False,
    ):
        self.adapter_layers = adapter_layers
        self.max_adapters = max_adapters
        self.device = device
        self.model_dtype = model_dtype
        # Engine scheduler setting, for the sequence-mixing serving
        # warning below (the dynamic route cannot be guarded at
        # engine-config time: members are only known at load).
        self.chunked_prefill_enabled = chunked_prefill_enabled

        # Slot map: index → adapter id (None = free).
        self.adapter_index_to_id: list[Optional[int]] = [None] * max_adapters

        # CPU cache — eviction here just logs, no GPU teardown needed.
        self._registered: _AdapterLRUCache = _AdapterLRUCache(
            max_cpu_adapters, self._on_cpu_evict)

        # GPU cache — eviction triggers _deactivate_adapter.
        self._active: _AdapterLRUCache = _AdapterLRUCache(
            max_adapters, self._deactivate_adapter)

    # ------------------------------------------------------------------
    # Registration (CPU cache)
    # ------------------------------------------------------------------

    def add_adapter(self, adapter_model: ServedAdapter) -> bool:
        """Register an adapter in the CPU cache.

        Returns True if the adapter was newly registered, False if it was
        already registered (LRU order is updated either way).
        """
        if adapter_model.id in self._registered:
            self._registered.touch(adapter_model.id)
            return False
        self._registered[adapter_model.id] = adapter_model
        logger.debug("Registered adapter id=%d (%d registered)",
                     adapter_model.id, len(self._registered))
        return True

    def remove_adapter(self, adapter_id: int) -> bool:
        """Remove an adapter from both CPU and GPU caches."""
        if adapter_id in self._active:
            del self._active[adapter_id]
        if adapter_id in self._registered:
            del self._registered[adapter_id]
            return True
        return False

    def list_adapters(self) -> dict[int, ServedAdapter]:
        """Return all registered adapters."""
        return dict(self._registered.cache)

    # ------------------------------------------------------------------
    # Activation (GPU)
    # ------------------------------------------------------------------

    def activate_adapter(self, adapter_id: int) -> bool:
        """Load an adapter onto GPU layers.

        If the GPU cache is full, the LRU adapter is automatically evicted.
        Returns True on success.
        """
        if adapter_id in self._active:
            self._active.touch(adapter_id)
            return True

        adapter_model = self._registered.get(adapter_id)
        if adapter_model is None:
            raise ValueError(
                f"adapter id={adapter_id} is not registered. "
                f"Call add_adapter() first.")

        # Find a free slot (LRU auto-evicts if full).
        if len(self._active) >= self.max_adapters:
            self._active.remove_oldest()

        slot = self._find_free_slot()
        if slot is None:
            raise RuntimeError("No free adapter slots after LRU eviction — "
                               "this should not happen.")

        self.adapter_index_to_id[slot] = adapter_id

        # Serving-contract warning: recurrent scan state does not carry
        # across prefill chunks (protocol.py check_adaptation_supported
        # contract).  The baked route forces unchunked prefill at
        # engine-config time; the dynamic route can only warn here —
        # the scheduler config is already frozen.
        if self.chunked_prefill_enabled:
            from vllm.adaptation.specs import adapter_config_needs_eager
            if adapter_config_needs_eager(adapter_model.adapter_config):
                logger.warning(
                    "sequence-mixing adaptation id=%d loaded while "
                    "chunked prefill is enabled: scan state resets at "
                    "every prefill chunk boundary, silently truncating "
                    "the member's effective window — both for prompts "
                    "longer than max_num_batched_tokens and for short "
                    "prompts split by batch packing.  Serve "
                    "sequence-mixing members with "
                    "enable_chunked_prefill=False and "
                    "max_num_batched_tokens >= max_model_len (the "
                    "baked adapter_config route enforces this "
                    "automatically).", adapter_id)

        # Load weights into decoder layers.
        count = self._load_adapter_to_layers(adapter_model)
        if count:
            from vllm.compilation.cuda_graph import (
                warn_if_dynamic_adaptation_under_cudagraphs)
            warn_if_dynamic_adaptation_under_cudagraphs("load")
        logger.debug("Activated adapter id=%d in slot %d "
                     "(%d layers)", adapter_id, slot, count)

        self._active[adapter_id] = adapter_model
        return True

    def _deactivate_adapter(self, adapter_id: int) -> None:
        """Remove adapter weights from GPU layers and free the slot."""
        # Free the slot.
        try:
            idx = self.adapter_index_to_id.index(adapter_id)
            self.adapter_index_to_id[idx] = None
        except ValueError:
            pass

        # Remove from all layers (also tears down site hooks).
        removed = 0
        for layer in self.adapter_layers:
            if _remove_adapter_from_layer(layer, adapter_id):
                removed += 1

        if removed:
            from vllm.compilation.cuda_graph import (
                warn_if_dynamic_adaptation_under_cudagraphs)
            warn_if_dynamic_adaptation_under_cudagraphs("unload")

        logger.debug("Deactivated adapter id=%d", adapter_id)

    def _on_cpu_evict(self, adapter_id: int) -> None:
        """Called when an adapter is evicted from the CPU cache."""
        # If still on GPU, deactivate first.
        if adapter_id in self._active:
            del self._active[adapter_id]
        logger.debug("CPU-evicted adapter id=%d", adapter_id)

    # ------------------------------------------------------------------
    # Batch-level API (called from model runner each step)
    # ------------------------------------------------------------------

    def ensure_active(self, batch_adapter_ids: set[int]) -> None:
        """Ensure all adapters needed for this batch are on GPU.

        Activates missing adapters (with LRU eviction if necessary) and
        bumps LRU order for already-active ones.  Adapters that exist on
        layers but were never registered with the manager (e.g. baked-in
        adapter id=1 from ``adapter_config``) are silently skipped.
        """
        for adapter_id in batch_adapter_ids:
            if adapter_id in self._active:
                self._active.touch(adapter_id)
            elif adapter_id in self._registered:
                self.activate_adapter(adapter_id)
            # else: adapter may have been loaded directly onto layers
            # (e.g. baked-in via adapter_config=), skip silently.

    def is_active(self, adapter_id: int) -> bool:
        return adapter_id in self._active

    def pin_adapter(self, adapter_id: int) -> bool:
        """Pin an adapter against LRU eviction (CPU and GPU caches).

        Used for adapters whose live layer weights are the source of
        truth (training-time weight sync): eviction would rebuild them
        from their stale blueprint and silently revert training.
        """
        if adapter_id not in self._registered:
            raise ValueError(
                f"adapter id={adapter_id} is not registered; cannot pin.")
        self._registered.pin(adapter_id)
        if adapter_id not in self._active:
            self.activate_adapter(adapter_id)
        self._active.pin(adapter_id)
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_free_slot(self) -> Optional[int]:
        """Return the index of the first free slot, or None."""
        for i, v in enumerate(self.adapter_index_to_id):
            if v is None:
                return i
        return None

    def _load_adapter_to_layers(self, adapter_model: ServedAdapter) -> int:
        """Instantiate adapter weights and add to relevant layers.

        Returns the number of layers loaded.
        """
        from vllm.adaptation.specs import adapter_config_to_spec

        spec = adapter_config_to_spec(adapter_model.adapter_config)
        if spec is None:
            return 0

        count = 0
        for layer in self.adapter_layers:
            # Determine layer index from the layer's prefix attribute.
            layer_idx = getattr(layer, "_adapter_layer_idx", None)
            if layer_idx is None or layer_idx < 0:
                continue
            if adapter_model.layer_indices and layer_idx not in adapter_model.layer_indices:
                continue

            source = spec.get("adapters", {}).get(
                layer_idx, spec.get("sample_adapter"))
            if source is None:
                continue

            dev = self.device
            # The member's CARRIER dtype rides its serving record; when
            # the record does not say, the member's own class /
            # parameters answer, and a member for which neither can is
            # refused BY NAME here rather than served through a guess
            # (vllm/adaptation/protocol.py::resolve_carrier_dtype).
            adapter_copy = _prepare_adapter(
                source, dev, self.model_dtype,
                carrier_dtype=(adapter_model.adapter_config or {}).get(
                    "carrier_dtype"),
                label=f"id={adapter_model.id} at site="
                      f"{adapter_model.site} (L{layer_idx})")
            _add_adapter_to_layer(layer, adapter_model.id, adapter_copy,
                                  adapter_model.position, dev,
                                  site=adapter_model.site)
            count += 1

        return count
