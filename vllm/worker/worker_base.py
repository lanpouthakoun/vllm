# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import (Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar,
                    Union)

import cloudpickle
import torch
import torch.nn as nn

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.sequence import ExecuteModelRequest
from vllm.utils import (enable_trace_function_call_for_thread,
                        resolve_obj_by_qualname, run_method,
                        update_environment_variables,
                        warn_for_unimplemented_methods)
from vllm.v1.outputs import SamplerOutput

logger = init_logger(__name__)

_R = TypeVar("_R")


@warn_for_unimplemented_methods
class WorkerBase:
    """Worker interface that allows vLLM to cleanly separate implementations for
    different hardware. Also abstracts control plane communication, e.g., to
    communicate request metadata to other workers.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config
        self.kv_transfer_config = vllm_config.kv_transfer_config
        self.compilation_config = vllm_config.compilation_config
        from vllm.platforms import current_platform
        self.current_platform = current_platform

    def init_device(self) -> None:
        """Initialize device state, such as loading the model or other on-device
        memory allocations.
        """
        raise NotImplementedError

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Initialize the KV cache with the given size in blocks.
        """
        raise NotImplementedError

    def get_model(self) -> nn.Module:
        raise NotImplementedError

    def apply_model(self, fn: Callable[[nn.Module], _R]) -> _R:
        """Apply a function on the model inside this worker."""
        return fn(self.get_model())

    def sync_lora_weights(self, state_dict: dict[str, torch.Tensor],
                          peft_config: dict, lora_int_id: int = 1) -> bool:
        """Sync LoRA adapter weights from in-memory tensors. No disk I/O.

        Called via ``collective_rpc("sync_lora_weights",
        args=(state_dict, peft_config, lora_int_id))``.

        Args:
            state_dict: PEFT-format state dict, e.g.
                ``{"base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight": tensor, ...}``
            peft_config: Serialized PEFT LoraConfig dict (r, lora_alpha,
                target_modules, etc.).
            lora_int_id: Adapter ID to register under.

        Returns:
            True if weights were synced successfully.
        """
        from vllm.lora.models import LoRAModel
        from vllm.lora.peft_helper import PEFTHelper

        model_runner = getattr(self, "model_runner", None)
        if model_runner is None or not hasattr(model_runner, "lora_manager"):
            logger.warning("sync_lora_weights: no lora_manager on model_runner")
            return False

        worker_mgr = model_runner.lora_manager

        # Build PEFTHelper from config dict
        peft_helper = PEFTHelper.from_dict(peft_config)

        # Build LoRAModel from tensors (handles name mapping internally)
        lora_model = LoRAModel.from_lora_tensors(
            lora_model_id=lora_int_id,
            tensors=state_dict,
            peft_helper=peft_helper,
            device="cpu",
            dtype=worker_mgr.lora_config.lora_dtype,
            target_embedding_padding=worker_mgr.vocab_size
            + worker_mgr.lora_config.lora_extra_vocab_size,
            embedding_modules=worker_mgr.embedding_modules,
            embedding_padding_modules=worker_mgr.embedding_padding_modules,
        )

        # Register + activate in-memory (no disk), replacing any stale
        # version, protecting against LRU eviction, and invalidating the
        # cached punica mapping.
        worker_mgr.register_synced_adapter(lora_model)
        logger.debug("sync_lora_weights: loaded adapter %d (%d modules)",
                      lora_int_id, len(lora_model.loras))
        return True

    def refresh_adapter_caches(self, adapter_int_id: Optional[int] = None) -> None:
        """Recompute derived adapter caches after adapter weights are updated.

        Called via ``collective_rpc("refresh_adapter_caches")`` after TRL's
        ``sync_weights()`` pushes new adapter parameters through
        ``load_weights()``.

        Args:
            adapter_int_id: If given, only refresh caches for this adapter.
                If None, refresh all adapters.
        """
        model = self.get_model()
        for layer in model.model.layers:
            if not hasattr(layer, "served_adapters"):
                continue
            for str_id, adapter in layer.served_adapters.items():
                if adapter_int_id is not None and int(str_id) != adapter_int_id:
                    continue
                if hasattr(adapter, "install_inference_caches"):
                    adapter.install_inference_caches()

    def sync_adapter_weights(self, weight_dict: dict[int, dict[str, bytes]],
                          refresh_caches: bool = True,
                          adapter_int_id: int = 1) -> int:
        """Load adapter weights for a specific adapter ID.

        Called via ``collective_rpc("sync_adapter_weights",
        args=(weight_dict,), kwargs=...)`` from pyreft's ``sync_to_vllm`` and
        from trl's vllm_generation. The 2nd positional arg is
        ``refresh_caches`` (so trl's ``args=(weight_dict, False)`` correctly
        disables cache refresh during step-time sync).

        Args:
            weight_dict: Mapping from layer index to serialized state_dict.
                Values are ``{param_name: tensor}`` dicts with tensors on CPU.
            adapter_int_id: Which adapter to update (default 1 for backward compat).
            refresh_caches: Whether to recompute inference caches after loading.

        Returns:
            Number of adapter layers synced.
        """
        model = self.get_model()
        key = str(adapter_int_id)
        count = 0
        for idx, state_dict in weight_dict.items():
            layer = model.model.layers[idx]
            if not hasattr(layer, "served_adapters"):
                continue
            adapter = layer.served_adapters[key] if key in layer.served_adapters else None
            if adapter is None:
                continue
            device = next(adapter.parameters()).device
            # spawn-mode RPC serializes tensors as lists: coerce
            # tensor-like values before .to; metadata entries
            # (strings etc.) are not loadable state — skip them.
            import torch as _t
            from vllm.adaptation.specs import _deserialize_state_dict
            if any(isinstance(v, dict) and v.get("__adapter_t")
                   for v in state_dict.values()):
                # canonical adaptation-serializer payload (bytes blobs)
                state_dict = _deserialize_state_dict(state_dict)
            sd = {}
            for k, v in state_dict.items():
                if _t.is_tensor(v):
                    sd[k] = v.to(device)
                elif isinstance(v, (int, float, bool)):
                    sd[k] = _t.tensor(v).to(device)
                elif isinstance(v, list):
                    try:
                        sd[k] = _t.tensor(v).to(device)
                    except (ValueError, TypeError):
                        pass  # non-numeric metadata list, not state
            adapter.load_state_dict(sd, strict=False)
            count += 1
        if refresh_caches and count:
            self.refresh_adapter_caches(adapter_int_id)
        if count:
            # The layer weights are now the source of truth; pin the
            # adapter so LRU eviction can't rebuild it from its stale
            # blueprint and silently revert training.  Adapters not
            # managed by the LRU (construction-baked) need no pin.
            get_mgr = getattr(self, "_get_adapter_manager", None)
            manager = get_mgr() if get_mgr is not None else None
            if manager is not None and adapter_int_id in manager.list_adapters():
                manager.pin_adapter(adapter_int_id)
        return count

    def _get_adapter_manager(self):
        """Try to get the AdapterManager from the model runner, if any."""
        model_runner = getattr(self, "model_runner", None)
        if model_runner is not None:
            return getattr(model_runner, "adapter_manager", None)
        return None

    def load_adapter(self, adapter_int_id: int, adapter_config: dict,
                          position: str = "prefill",
                          site: str = "block_output") -> int:
        """Load a new adapter into all relevant layers.

        Called via ``collective_rpc("load_adapter",
        args=(adapter_int_id, adapter_config, position, site))``.

        If a :class:`~vllm.adaptation.manager.AdapterManager` is available (the
        ``enable_adapters`` path), the adapter is registered and activated through
        the manager's LRU cache.  Otherwise falls back to direct per-layer
        loading.

        Args:
            adapter_int_id: Unique integer ID for this adapter (>= 1).
            adapter_config: Serializable config dict (from spec_to_adapter_config).
                May carry "site" / "position" keys, overridden by the
                explicit arguments when those are non-default.
            position: Position mode for this adapter (any registered
                position name; builtins: all/prefill/decode/first/last).
            site: Mount point inside each decoder layer (block_output,
                block_input, post_attn, post_mlp, linear:<path>).

        Returns:
            Number of layers the adapter was loaded into.
        """
        if site == "block_output":
            site = adapter_config.get("site", "block_output")
        # A schedule-rewiring member cannot arrive this way: the engine's
        # KV-cache spec (and therefore the per-pass caches its span
        # needs) was frozen before this RPC could run.  Refuse with the
        # reason rather than serve a span that overwrites the host's K/V.
        from vllm.adaptation.recirculation import refuse_rewired
        refuse_rewired(adapter_config, "multisite")
        # A member may DECLARE what the engine must be configured as
        # (unchunked prefill, eager) for its computation to be the one it
        # was trained as.  On this route the engine config was frozen
        # before the RPC could run, so nothing here can still be forced —
        # the declaration was supposed to reach
        # EngineArgs.adapter_serving_requirements before LLM(...).  This
        # is the re-check: it REFUSES, by member and requirement, when the
        # engine cannot satisfy what the member asks for, rather than
        # serving a scan that resets at every prefill chunk boundary and
        # still reads fluently.  Members that declare nothing (every leaf
        # in the zoo but the faithful native ones) reach exactly the
        # behaviour they had before.
        from vllm.adaptation.specs import (
            check_serving_requirements_honoured)
        check_serving_requirements_honoured(
            adapter_config, getattr(self, "vllm_config", None),
            label=f"id={adapter_int_id} at site={site}")
        # Prompt-anchored positions silently break under an external
        # KV prefix — warn at load time (undetectable at mask time).
        from vllm.adaptation.layer import warn_if_prefix_incompatible_position
        warn_if_prefix_incompatible_position(
            position, getattr(self, "vllm_config", None))
        # Delegate to centralized manager when available.
        manager = self._get_adapter_manager()
        if manager is not None:
            from vllm.adaptation.specs import adapter_config_to_spec
            from vllm.adaptation.manager import ServedAdapter
            spec = adapter_config_to_spec(adapter_config)
            if spec is None:
                return 0
            adapter_model = ServedAdapter(
                id=adapter_int_id,
                position=position,
                adapter_config=adapter_config,
                layer_indices=frozenset(spec.get("layer_indices", ())),
                site=site,
            )
            manager.add_adapter(adapter_model)
            manager.activate_adapter(adapter_int_id)
            return len(adapter_model.layer_indices)

        # Fallback: direct per-layer loading (backward compat).
        from vllm.adaptation.specs import adapter_config_to_spec
        from vllm.adaptation.layer import _prepare_adapter, _add_adapter_to_layer

        spec = adapter_config_to_spec(adapter_config)
        if spec is None:
            return 0

        model = self.get_model()
        count = 0
        for layer_idx in spec["layer_indices"]:
            layer = model.model.layers[layer_idx]
            if not hasattr(layer, "served_adapters"):
                continue
            try:
                dev = next(layer.parameters()).device
            except StopIteration:
                dev = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu")
            source = spec.get("adapters", {}).get(layer_idx,
                                                   spec["sample_adapter"])
            model_dtype = torch.bfloat16
            # Same carrier rule as every other route: the record's
            # declaration wins, the member's class / parameters answer
            # when it is silent, and a member whose carrier can be
            # neither declared nor derived is refused by name.
            adapter_copy = _prepare_adapter(
                source, dev, model_dtype,
                carrier_dtype=adapter_config.get("carrier_dtype"),
                label=f"id={adapter_int_id} at site={site} "
                      f"(L{layer_idx})")
            _add_adapter_to_layer(layer, adapter_int_id, adapter_copy,
                                  position, dev, site=site)
            count += 1
        if count:
            from vllm.compilation.cuda_graph import (
                warn_if_dynamic_adaptation_under_cudagraphs)
            warn_if_dynamic_adaptation_under_cudagraphs("load")
        return count

    def unload_adapter(self, adapter_int_id: int) -> int:
        """Remove a adapter from all layers.

        Delegates to the :class:`~vllm.adaptation.manager.AdapterManager` when
        available, otherwise falls back to direct per-layer removal.

        Returns:
            Number of layers the adapter was removed from.
        """
        # Delegate to centralized manager when available.
        manager = self._get_adapter_manager()
        if manager is not None:
            was_removed = manager.remove_adapter(adapter_int_id)
            return 1 if was_removed else 0

        # Fallback: direct per-layer removal (backward compat).
        from vllm.adaptation.layer import _remove_adapter_from_layer
        model = self.get_model()
        count = 0
        for layer in model.model.layers:
            if _remove_adapter_from_layer(layer, adapter_int_id):
                count += 1
        if count:
            from vllm.compilation.cuda_graph import (
                warn_if_dynamic_adaptation_under_cudagraphs)
            warn_if_dynamic_adaptation_under_cudagraphs("unload")
        return count

    def get_adapter_debug_stats(self) -> dict:
        """Return per-layer adapter debug stats from the model."""
        model = self.get_model()
        if hasattr(model, "get_adapter_debug_stats"):
            return model.get_adapter_debug_stats()
        return {}

    def get_adapter_weight_fingerprints(self, layer_indices=None) -> dict:
        """Return adapter param/buffer fingerprints for diagnostic comparison."""
        model = self.get_model()
        if hasattr(model, "get_adapter_weight_fingerprints"):
            return model.get_adapter_weight_fingerprints(layer_indices)
        return {}

    def load_model(self) -> None:
        """Load model onto target device."""
        raise NotImplementedError

    def execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> Optional[List[SamplerOutput]]:
        raise NotImplementedError

    def start_worker_execution_loop(self) -> None:
        """Execute model loop in parallel worker.

        You can stop the loop by executing a driver worker with an empty output.
        See `stop_remote_worker_execution_loop` for more details.
        """
        with self.current_platform.inference_mode():
            while True:
                output = self.execute_model(execute_model_req=None)
                if output is None:
                    return None

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available blocks for the GPU KV cache and
        swappable CPU KV cache.

        The implementation may run profiling or other heuristics to determine
        the size of caches.

        Returns a Tuple[num_gpu_blocks, num_cpu_blocks], where num_gpu_blocks
        are blocks that are "active" on the device and can be appended to.
        num_cpu_blocks refers to "swapped" blocks in CPU memory and cannot be
        appended to.
        """
        raise NotImplementedError

    def get_cache_block_size_bytes(self) -> int:
        """Return the size of a single cache block, in bytes. Used in
        speculative decoding.
        """
        raise NotImplementedError

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def pin_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def list_loras(self) -> Set[int]:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size from model configuration."""
        return self.model_config.get_vocab_size()

    def shutdown(self) -> None:
        """Clean up resources held by the worker."""
        return


class WorkerWrapperBase:
    """
    This class represents one process in an executor/engine. It is responsible
    for lazily initializing the worker and handling the worker's lifecycle.
    We first instantiate the WorkerWrapper, which remembers the worker module
    and class name. Then, when we call `update_environment_variables`, and the
    real initialization happens in `init_worker`.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        rpc_rank: int = 0,
    ) -> None:
        """
        Initialize the worker wrapper with the given vllm_config and rpc_rank.
        Note: rpc_rank is the rank of the worker in the executor. In most cases,
        it is also the rank of the worker in the distributed group. However,
        when multiple executors work together, they can be different.
        e.g. in the case of SPMD-style offline inference with TP=2,
        users can launch 2 engines/executors, each with only 1 worker.
        All workers have rpc_rank=0, but they have different ranks in the TP
        group.
        """
        self.rpc_rank = rpc_rank
        self.worker: Optional[WorkerBase] = None
        self.vllm_config: Optional[VllmConfig] = None
        # do not store this `vllm_config`, `init_worker` will set the final
        # one. TODO: investigate if we can remove this field in
        # `WorkerWrapperBase`, `init_cached_hf_modules` should be
        # unnecessary now.
        if vllm_config.model_config is not None:
            # it can be None in tests
            trust_remote_code = vllm_config.model_config.trust_remote_code
            if trust_remote_code:
                # note: lazy import to avoid importing torch before initializing
                from vllm.utils import init_cached_hf_modules
                init_cached_hf_modules()

    def shutdown(self) -> None:
        if self.worker is not None:
            self.worker.shutdown()

    def adjust_rank(self, rank_mapping: Dict[int, int]) -> None:
        """
        Adjust the rpc_rank based on the given mapping.
        It is only used during the initialization of the executor,
        to adjust the rpc_rank of workers after we create all workers.
        """
        if self.rpc_rank in rank_mapping:
            self.rpc_rank = rank_mapping[self.rpc_rank]

    def update_environment_variables(self, envs_list: List[Dict[str,
                                                                str]]) -> None:
        envs = envs_list[self.rpc_rank]
        key = 'CUDA_VISIBLE_DEVICES'
        if key in envs and key in os.environ:
            # overwriting CUDA_VISIBLE_DEVICES is desired behavior
            # suppress the warning in `update_environment_variables`
            del os.environ[key]
        update_environment_variables(envs)

    def init_worker(self, all_kwargs: List[Dict[str, Any]]) -> None:
        """
        Here we inject some common logic before initializing the worker.
        Arguments are passed to the worker class constructor.
        """
        kwargs = all_kwargs[self.rpc_rank]
        self.vllm_config = kwargs.get("vllm_config")
        assert self.vllm_config is not None, (
            "vllm_config is required to initialize the worker")
        enable_trace_function_call_for_thread(self.vllm_config)

        from vllm.plugins import load_general_plugins
        load_general_plugins()

        if isinstance(self.vllm_config.parallel_config.worker_cls, str):
            worker_class = resolve_obj_by_qualname(
                self.vllm_config.parallel_config.worker_cls)
        else:
            logger.warning(
                "passing worker_cls as a class object is strongly deprecated,"
                " as the serialization of class objects can be tricky and"
                " error-prone. To be safe, please keep the class in a separate"
                " module and pass the qualified name of the class as a string."
            )
            assert isinstance(self.vllm_config.parallel_config.worker_cls,
                              bytes)
            worker_class = cloudpickle.loads(
                self.vllm_config.parallel_config.worker_cls)
        if self.vllm_config.parallel_config.worker_extension_cls:
            worker_extension_cls = resolve_obj_by_qualname(
                self.vllm_config.parallel_config.worker_extension_cls)
            extended_calls = []
            if worker_extension_cls not in worker_class.__bases__:
                # check any conflicts between worker and worker_extension_cls
                for attr in dir(worker_extension_cls):
                    if attr.startswith("__"):
                        continue
                    assert not hasattr(worker_class, attr), (
                        f"Worker class {worker_class} already has an attribute"
                        f" {attr}, which conflicts with the worker"
                        f" extension class {worker_extension_cls}.")
                    if callable(getattr(worker_extension_cls, attr)):
                        extended_calls.append(attr)
                # dynamically inherit the worker extension class
                worker_class.__bases__ = worker_class.__bases__ + (
                    worker_extension_cls, )
                logger.info(
                    "Injected %s into %s for extended collective_rpc calls %s",
                    worker_extension_cls, worker_class, extended_calls)
        with set_current_vllm_config(self.vllm_config):
            # To make vLLM config available during worker initialization
            self.worker = worker_class(**kwargs)
            assert self.worker is not None

    def initialize_from_config(self, kv_cache_configs: List[Any]) -> None:
        kv_cache_config = kv_cache_configs[self.rpc_rank]
        with set_current_vllm_config(self.vllm_config):
            self.worker.initialize_from_config(kv_cache_config)  # type: ignore

    def init_device(self):
        with set_current_vllm_config(self.vllm_config):
            # To make vLLM config available during device initialization
            self.worker.init_device()  # type: ignore

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        try:
            # method resolution order:
            # if a method is defined in this class, it will be called directly.
            # otherwise, since we define `__getattr__` and redirect attribute
            # query to `self.worker`, the method will be called on the worker.
            return run_method(self, method, args, kwargs)
        except Exception as e:
            # if the driver worker also execute methods,
            # exceptions in the rest worker may cause deadlock in rpc like ray
            # see https://github.com/vllm-project/vllm/issues/3455
            # print the error and inform the user to solve the error
            msg = (f"Error executing method {method!r}. "
                   "This might cause deadlock in distributed execution.")
            logger.exception(msg)
            raise e

    def __getattr__(self, attr):
        return getattr(self.worker, attr)
