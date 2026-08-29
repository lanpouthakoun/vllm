# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA graph handling when the adapter set changes.

Captured graphs replay the compiled forward, which was traced with the
adapter set that existed at compile time — a adapter loaded
post-warmup cannot take effect there (vLLM's compile wrapper bypasses
Dynamo guards), and re-capturing during real steps bakes per-step
host-side conditionals (e.g. punica's no-LoRA skip), breaking other
adapters.  Structural changes therefore must NOT touch captured graphs:
they emit a loud warning instead
(warn_if_dynamic_adaptation_under_cudagraphs), and dynamic adapter serving
is supported in eager mode.  invalidate_all_cudagraphs() remains as an
explicitly-invoked mechanism (graphs are retired, never destroyed, to
protect the shared memory pool).
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.compilation import monitor
from vllm.compilation.cuda_graph import (
    _cudagraph_wrappers, invalidate_all_cudagraphs,
    warn_if_dynamic_adaptation_under_cudagraphs)
from vllm.adaptation.layer import _add_adapter_to_layer, _init_multi_adapter_state
from vllm.worker.worker_base import WorkerBase


class FakeWrapper:
    """Duck-typed stand-in for CUDAGraphWrapper (weakref-able)."""

    def __init__(self, num_entries: int):
        self.concrete_cudagraph_entries = {
            ("desc", i): object() for i in range(num_entries)
        }


@pytest.fixture
def capture_flag_guard():
    saved = monitor.cudagraph_capturing_enabled
    yield
    monitor.cudagraph_capturing_enabled = saved


class TestInvalidateAll:

    def test_clears_entries_and_reenables_capture(self, capture_flag_guard):
        w1, w2 = FakeWrapper(3), FakeWrapper(1)
        _cudagraph_wrappers.add(w1)
        _cudagraph_wrappers.add(w2)
        try:
            monitor.cudagraph_capturing_enabled = False
            cleared = invalidate_all_cudagraphs()
            assert cleared == 4
            assert not w1.concrete_cudagraph_entries
            assert not w2.concrete_cudagraph_entries
            # Lazy re-capture must be allowed again.
            assert monitor.cudagraph_capturing_enabled
        finally:
            _cudagraph_wrappers.discard(w1)
            _cudagraph_wrappers.discard(w2)

    def test_noop_when_nothing_captured(self, capture_flag_guard):
        w = FakeWrapper(0)
        _cudagraph_wrappers.add(w)
        try:
            monitor.cudagraph_capturing_enabled = False
            assert invalidate_all_cudagraphs() == 0
            # Nothing cleared -> no need to unfreeze capturing.
            assert not monitor.cudagraph_capturing_enabled
        finally:
            _cudagraph_wrappers.discard(w)

    def test_invalidated_graphs_are_retired_not_destroyed(
            self, capture_flag_guard):
        # Destroying graphs that share a memory pool while later
        # captures allocate from it trips the CUDA allocator's pool
        # bookkeeping; invalidation must hold references instead.
        from vllm.compilation.cuda_graph import _retired_cudagraphs
        marker = object()
        w = FakeWrapper(0)
        w.concrete_cudagraph_entries = {
            "d": SimpleNamespace(cudagraph=marker),
            "e": SimpleNamespace(cudagraph=None),  # never captured
        }
        _cudagraph_wrappers.add(w)
        before = len(_retired_cudagraphs)
        try:
            assert invalidate_all_cudagraphs() == 2
            assert not w.concrete_cudagraph_entries
            assert marker in _retired_cudagraphs
            assert len(_retired_cudagraphs) == before + 1
        finally:
            _cudagraph_wrappers.discard(w)
            if marker in _retired_cudagraphs:
                _retired_cudagraphs.remove(marker)

    def test_registry_is_weak(self):
        before = len(_cudagraph_wrappers)
        w = FakeWrapper(1)
        _cudagraph_wrappers.add(w)
        del w
        import gc
        gc.collect()
        assert len(_cudagraph_wrappers) == before


class _ConstAdapter(nn.Module):

    def __init__(self):
        super().__init__()
        self.marker = nn.Linear(1, 1)

    def readout(self, fx, state=None, x=None):
        return fx + torch.zeros_like(fx)


def _fake_worker_with_layers(num_layers=2):
    layers = []
    for _ in range(num_layers):
        layer = nn.Module()
        layer._adapter_layer_idx = 0
        _init_multi_adapter_state(layer, torch.device("cpu"), 8)
        layers.append(layer)
    model = SimpleNamespace(model=SimpleNamespace(layers=layers))
    worker = SimpleNamespace(get_model=lambda: model,
                             model_runner=SimpleNamespace())
    worker._get_adapter_manager = lambda: None
    return worker, layers


class TestWarnHelper:

    def test_counts_captured_graphs(self):
        w = FakeWrapper(3)
        _cudagraph_wrappers.add(w)
        try:
            assert warn_if_dynamic_adaptation_under_cudagraphs("load") == 3
            # Warning must not disturb the captured graphs.
            assert len(w.concrete_cudagraph_entries) == 3
        finally:
            _cudagraph_wrappers.discard(w)

    def test_silent_when_nothing_captured(self):
        assert warn_if_dynamic_adaptation_under_cudagraphs("load") == 0


class TestWorkerIntegration:
    """Structural adapter changes must warn but leave captured graphs
    (and the capture freeze) untouched — re-capture cannot bring a
    post-compile adapter into the frozen compiled forward, and taking
    new captures during real steps corrupts other adapters."""

    def test_load_adapter_preserves_graphs(self, monkeypatch,
                                           capture_flag_guard):
        import vllm.adaptation.specs as vllm_adapter
        monkeypatch.setattr(
            vllm_adapter, "adapter_config_to_spec", lambda cfg: {
                "layer_indices": [0, 1],
                "sample_adapter": _ConstAdapter(),
                "position": "all",
            })
        worker, layers = _fake_worker_with_layers()
        for i, layer in enumerate(layers):
            layer._adapter_layer_idx = i

        w = FakeWrapper(2)
        _cudagraph_wrappers.add(w)
        try:
            monitor.cudagraph_capturing_enabled = False
            count = WorkerBase.load_adapter(worker, 1, {},
                                                 position="decode")
            assert count == 2
            assert len(w.concrete_cudagraph_entries) == 2
            assert monitor.cudagraph_capturing_enabled is False
        finally:
            _cudagraph_wrappers.discard(w)

    def test_unload_adapter_preserves_graphs(self, capture_flag_guard):
        worker, layers = _fake_worker_with_layers()
        for layer in layers:
            _add_adapter_to_layer(layer, 1, _ConstAdapter(), "all",
                                  torch.device("cpu"))
        w = FakeWrapper(1)
        _cudagraph_wrappers.add(w)
        try:
            monitor.cudagraph_capturing_enabled = False
            removed = WorkerBase.unload_adapter(worker, 1)
            assert removed == 2
            assert len(w.concrete_cudagraph_entries) == 1
            assert monitor.cudagraph_capturing_enabled is False
        finally:
            _cudagraph_wrappers.discard(w)

    def test_manager_activate_preserves_graphs(self, monkeypatch,
                                               capture_flag_guard):
        import vllm.adaptation.specs as vllm_adapter
        from vllm.adaptation.manager import ServedAdapter, AdapterManager
        monkeypatch.setattr(
            vllm_adapter, "adapter_config_to_spec", lambda cfg: {
                "layer_indices": [0],
                "sample_adapter": _ConstAdapter(),
                "position": "all",
            })
        layer = nn.Module()
        layer._adapter_layer_idx = 0
        _init_multi_adapter_state(layer, torch.device("cpu"), 8)
        manager = AdapterManager([layer], max_adapters=2, max_cpu_adapters=2,
                                   device=torch.device("cpu"),
                                   model_dtype=torch.float32)
        manager.add_adapter(
            ServedAdapter(id=1, position="all", adapter_config={},
                      layer_indices=frozenset([0])))

        w = FakeWrapper(1)
        _cudagraph_wrappers.add(w)
        try:
            monitor.cudagraph_capturing_enabled = False
            manager.activate_adapter(1)
            assert len(w.concrete_cudagraph_entries) == 1
            assert monitor.cudagraph_capturing_enabled is False

            manager.remove_adapter(1)
            assert len(w.concrete_cudagraph_entries) == 1
            assert monitor.cudagraph_capturing_enabled is False
        finally:
            _cudagraph_wrappers.discard(w)
