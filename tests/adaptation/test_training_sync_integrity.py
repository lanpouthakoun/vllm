# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Training-time weight sync must stay correct under CUDA graph serving.

Invariants exercised here:

1. Weight sync is an in-place update: every parameter and inference
   cache buffer keeps its storage address across syncs, so replayed
   CUDA graphs read the new weights without re-capture.
2. Weight sync must therefore NOT invalidate captured graphs (that
   would force a re-capture every training step); only adapter
   load/unload does.
3. The realistic adapter shape (params + non-persistent cache
   buffers + idempotent install_inference_caches) round-trips a
   training-side state dict (which carries no cache keys) and ends up
   generating with the refreshed caches.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn

from vllm.compilation import monitor
from vllm.compilation.cuda_graph import _cudagraph_wrappers
from vllm.adaptation.layer import _add_adapter_to_layer, _init_multi_adapter_state
from vllm.worker.worker_base import WorkerBase

from .test_cudagraph_invalidation import FakeWrapper

HIDDEN = 8
RANK = 4


class RealisticStreamAdapter(nn.Module):
    """Mimics the adapters library RotateAdapter cache protocol:

    - trainable params (rotate_layer.weight, learned_source),
    - non-persistent cache buffers derived from them,
    - idempotent install_inference_caches (registers once, refreshes
      in place afterwards) — matching adapters/_base.py.
    """

    def __init__(self):
        super().__init__()
        self.rotate_layer = nn.Linear(HIDDEN, RANK, bias=False)
        self.learned_source = nn.Linear(HIDDEN, RANK)

    def install_inference_caches(self, model_dtype=torch.float32):
        R = self.rotate_layer.weight.detach().to(model_dtype)
        if hasattr(self, "_R_cache"):
            self._R_cache.data.copy_(R)
        else:
            self.register_buffer("_R_cache", R.clone(), persistent=False)

    def readout(self, fx, state=None, x=None):
        source = self.learned_source(fx)
        return fx + source @ self._R_cache


def _make_model(num_layers=2):
    layers = []
    for i in range(num_layers):
        layer = nn.Module()
        layer._adapter_layer_idx = i
        _init_multi_adapter_state(layer, torch.device("cpu"), HIDDEN)
        _add_adapter_to_layer(layer, 1, RealisticStreamAdapter(), "prefill",
                              torch.device("cpu"))
        layer.served_adapters["1"].install_inference_caches()
        layers.append(layer)
    model = SimpleNamespace(model=SimpleNamespace(layers=layers))
    worker = SimpleNamespace(get_model=lambda: model)
    worker.refresh_adapter_caches = (
        lambda adapter_int_id=None: WorkerBase.refresh_adapter_caches(
            worker, adapter_int_id))
    return worker, layers


def _training_state_dict(fill):
    """What the trainer sends: params only — caches are non-persistent
    on the training side too, so they never appear in the dict."""
    return {
        "rotate_layer.weight": torch.full((RANK, HIDDEN), fill),
        "learned_source.weight": torch.full((RANK, HIDDEN), fill),
        "learned_source.bias": torch.full((RANK, ), fill),
    }


class TestInPlaceness:

    def test_param_and_cache_addresses_stable_across_sync(self):
        worker, layers = _make_model()
        adapter = layers[0].served_adapters["1"]
        ptrs_before = {
            name: p.data_ptr()
            for name, p in adapter.named_parameters()
        }
        cache_ptr_before = adapter._R_cache.data_ptr()

        WorkerBase.sync_adapter_weights(worker,
                                     {0: _training_state_dict(0.25),
                                      1: _training_state_dict(0.25)},
                                     refresh_caches=True,
                                     adapter_int_id=1)

        for name, p in adapter.named_parameters():
            assert p.data_ptr() == ptrs_before[name], \
                f"{name} was reallocated by sync — breaks CUDA graph replay"
        assert adapter._R_cache.data_ptr() == cache_ptr_before, \
            "_R_cache was reallocated by refresh — breaks CUDA graph replay"

    def test_caches_hold_new_weights_after_sync(self):
        worker, layers = _make_model()
        WorkerBase.sync_adapter_weights(worker,
                                     {0: _training_state_dict(0.5),
                                      1: _training_state_dict(0.5)},
                                     refresh_caches=True,
                                     adapter_int_id=1)
        for layer in layers:
            adapter = layer.served_adapters["1"]
            assert torch.all(adapter._R_cache == 0.5), \
                "inference cache is stale after weight sync"

    def test_stale_cache_without_refresh(self):
        # refresh_caches=False (TRL's step-time call) intentionally
        # leaves caches stale; the follow-up refresh_adapter_caches RPC
        # must fix them.
        worker, layers = _make_model()
        WorkerBase.sync_adapter_weights(worker,
                                     {0: _training_state_dict(0.5),
                                      1: _training_state_dict(0.5)},
                                     refresh_caches=False,
                                     adapter_int_id=1)
        adapter = layers[0].served_adapters["1"]
        assert not torch.all(adapter._R_cache == 0.5)
        WorkerBase.refresh_adapter_caches(worker)
        assert torch.all(adapter._R_cache == 0.5)

    def test_training_state_dict_without_cache_keys_loads(self):
        # Non-persistent caches must not make strict load fail.
        worker, layers = _make_model()
        count = WorkerBase.sync_adapter_weights(worker,
                                            {0: _training_state_dict(0.1)},
                                            refresh_caches=True,
                                            adapter_int_id=1)
        assert count == 1

    def test_delta_uses_new_weights(self):
        worker, layers = _make_model()
        WorkerBase.sync_adapter_weights(worker,
                                     {0: _training_state_dict(0.5),
                                      1: _training_state_dict(0.5)},
                                     refresh_caches=True,
                                     adapter_int_id=1)
        adapter = layers[0].served_adapters["1"]
        h = torch.ones(1, 3, HIDDEN)
        delta = adapter.readout(h) - h
        # source = 0.5*8 + 0.5 = 4.5 per rank dim; delta = source @ R
        # with R = 0.5 everywhere: 4.5 * 0.5 * RANK = 9.0.
        assert torch.allclose(delta, torch.full((1, 3, HIDDEN), 9.0))


class TestSyncDoesNotInvalidateGraphs:

    def test_adapter_sync_leaves_graphs_captured(self):
        worker, _ = _make_model()
        w = FakeWrapper(2)
        _cudagraph_wrappers.add(w)
        saved_flag = monitor.cudagraph_capturing_enabled
        try:
            monitor.cudagraph_capturing_enabled = False
            WorkerBase.sync_adapter_weights(worker,
                                         {0: _training_state_dict(0.5)},
                                         refresh_caches=True,
                                         adapter_int_id=1)
            # In-place update: graphs stay valid, no re-capture allowed.
            assert len(w.concrete_cudagraph_entries) == 2
            assert monitor.cudagraph_capturing_enabled is False
        finally:
            monitor.cudagraph_capturing_enabled = saved_flag
            _cudagraph_wrappers.discard(w)
