# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Training-time adapter weight sync for multiple adapters.

Exercises WorkerBase.sync_adapter_weights / refresh_adapter_caches
against dummy decoder layers, covering the two-adapter (prefill +
decode pair) training loop: each adapter is synced by its own
adapter_int_id and only that adapter's weights/caches may change.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.adaptation.layer import (_add_adapter_to_layer,
                                   _init_multi_adapter_state)
from vllm.worker.worker_base import WorkerBase

HIDDEN = 4


class TrackingAdapter(nn.Module):
    """Adapter with real params and an install_inference_caches counter."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(HIDDEN, HIDDEN)
        self.cache_installs = 0

    def install_inference_caches(self, model_dtype=torch.float32):
        self.cache_installs += 1

    def readout(self, fx, state=None, x=None):
        return fx + self.proj(fx)


def _make_model(num_layers: int = 2, adapter_ids=(1, 2)):
    layers = []
    for _ in range(num_layers):
        layer = nn.Module()
        _init_multi_adapter_state(layer, torch.device("cpu"), HIDDEN)
        for adapter_id in adapter_ids:
            position = "decode" if adapter_id % 2 == 0 else "prefill"
            _add_adapter_to_layer(layer, adapter_id, TrackingAdapter(),
                                  position, torch.device("cpu"))
        layers.append(layer)
    model = SimpleNamespace(model=SimpleNamespace(layers=layers))
    return model, layers


def _fake_worker(model) -> SimpleNamespace:
    worker = SimpleNamespace(get_model=lambda: model)
    worker.refresh_adapter_caches = (
        lambda adapter_int_id=None: WorkerBase.refresh_adapter_caches(
            worker, adapter_int_id))
    return worker


def _state_dict(fill: float) -> dict[str, torch.Tensor]:
    return {
        "proj.weight": torch.full((HIDDEN, HIDDEN), fill),
        "proj.bias": torch.full((HIDDEN, ), fill),
    }


def _adapter(layer, adapter_id: int) -> TrackingAdapter:
    return layer.served_adapters[str(adapter_id)]


class TestSyncPinsAgainstEviction:
    """A synced (training) adapter's weights live only in the layer
    modules; LRU eviction would rebuild it from its stale blueprint and
    silently revert training.  Syncing must pin the adapter."""

    def _manager_setup(self, monkeypatch, max_adapters):
        import vllm.adaptation.specs as vllm_specs
        from vllm.adaptation.manager import ServedAdapter, AdapterManager
        monkeypatch.setattr(
            vllm_specs, "adapter_config_to_spec", lambda cfg: {
                "layer_indices": [0],
                "sample_adapter": TrackingAdapter(),
                "position": "prefill",
            })
        layer = nn.Module()
        layer._adapter_layer_idx = 0
        _init_multi_adapter_state(layer, torch.device("cpu"), HIDDEN)
        manager = AdapterManager([layer], max_adapters=max_adapters,
                                 max_cpu_adapters=8,
                                 device=torch.device("cpu"),
                                 model_dtype=torch.float32)
        model = SimpleNamespace(model=SimpleNamespace(layers=[layer]))
        worker = SimpleNamespace(get_model=lambda: model,
                                 _get_adapter_manager=lambda: manager)
        worker.refresh_adapter_caches = (
            lambda adapter_int_id=None: WorkerBase.refresh_adapter_caches(
                worker, adapter_int_id))

        def add(rid):
            manager.add_adapter(
                ServedAdapter(id=rid, position="prefill", adapter_config={},
                              layer_indices=frozenset([0])))
            manager.activate_adapter(rid)

        return manager, worker, layer, add

    def test_synced_adapter_survives_lru_eviction(self, monkeypatch):
        manager, worker, layer, add = self._manager_setup(monkeypatch,
                                                          max_adapters=2)
        add(1)
        add(2)
        WorkerBase.sync_adapter_weights(worker, {0: _state_dict(0.5)},
                                        refresh_caches=False,
                                        adapter_int_id=1)
        synced_weight = layer.served_adapters["1"].proj.weight.clone()

        # Slot pressure: id 3 must evict id 2, never the pinned id 1.
        add(3)
        assert manager.is_active(1)
        assert not manager.is_active(2)
        assert torch.equal(layer.served_adapters["1"].proj.weight,
                           synced_weight)

    def test_eviction_raises_when_only_pinned_left(self, monkeypatch):
        manager, worker, layer, add = self._manager_setup(monkeypatch,
                                                          max_adapters=1)
        add(1)
        WorkerBase.sync_adapter_weights(worker, {0: _state_dict(0.5)},
                                        refresh_caches=False,
                                        adapter_int_id=1)
        with pytest.raises(RuntimeError):
            add(2)

    def test_pin_unregistered_raises(self, monkeypatch):
        manager, _, _, _ = self._manager_setup(monkeypatch, max_adapters=2)
        with pytest.raises(ValueError, match="not registered"):
            manager.pin_adapter(99)

    def test_sync_without_manager_registration_is_fine(self, monkeypatch):
        # Construction-baked adapters (adapter_config=) are not in the
        # manager; syncing them must not fail on the pin step.
        manager, worker, layer, _ = self._manager_setup(monkeypatch,
                                                        max_adapters=2)
        _add_adapter_to_layer(layer, 7, TrackingAdapter(), "prefill",
                              torch.device("cpu"))
        count = WorkerBase.sync_adapter_weights(worker,
                                                {0: _state_dict(0.25)},
                                                refresh_caches=False,
                                                adapter_int_id=7)
        assert count == 1


class TestSyncAdapterWeights:

    def test_sync_updates_only_target_adapter(self):
        model, layers = _make_model()
        worker = _fake_worker(model)
        before_a1 = [
            _adapter(layer, 1).proj.weight.clone() for layer in layers
        ]

        count = WorkerBase.sync_adapter_weights(worker,
                                                {0: _state_dict(0.25),
                                                 1: _state_dict(0.25)},
                                                refresh_caches=False,
                                                adapter_int_id=2)
        assert count == 2
        for layer, before in zip(layers, before_a1):
            assert torch.all(_adapter(layer, 2).proj.weight == 0.25)
            assert torch.equal(_adapter(layer, 1).proj.weight, before)

    def test_sync_two_adapters_independently(self):
        model, layers = _make_model()
        worker = _fake_worker(model)

        assert WorkerBase.sync_adapter_weights(worker,
                                               {0: _state_dict(0.5),
                                                1: _state_dict(0.5)},
                                               refresh_caches=False,
                                               adapter_int_id=1) == 2
        assert WorkerBase.sync_adapter_weights(worker,
                                               {0: _state_dict(0.25),
                                                1: _state_dict(0.25)},
                                               refresh_caches=False,
                                               adapter_int_id=2) == 2
        for layer in layers:
            assert torch.all(_adapter(layer, 1).proj.weight == 0.5)
            assert torch.all(_adapter(layer, 2).proj.weight == 0.25)

    def test_sync_subset_of_layers(self):
        model, layers = _make_model()
        worker = _fake_worker(model)
        count = WorkerBase.sync_adapter_weights(worker,
                                                {1: _state_dict(0.75)},
                                                refresh_caches=False,
                                                adapter_int_id=1)
        assert count == 1
        assert not torch.all(_adapter(layers[0], 1).proj.weight == 0.75)
        assert torch.all(_adapter(layers[1], 1).proj.weight == 0.75)

    def test_sync_missing_adapter_id_is_noop(self):
        model, layers = _make_model(adapter_ids=(1, ))
        worker = _fake_worker(model)
        count = WorkerBase.sync_adapter_weights(worker,
                                                {0: _state_dict(0.9)},
                                                refresh_caches=True,
                                                adapter_int_id=7)
        assert count == 0
        assert _adapter(layers[0], 1).cache_installs == 0

    def test_refresh_caches_only_for_synced_id(self):
        model, layers = _make_model()
        worker = _fake_worker(model)
        WorkerBase.sync_adapter_weights(worker,
                                        {0: _state_dict(0.1),
                                         1: _state_dict(0.1)},
                                        refresh_caches=True,
                                        adapter_int_id=2)
        for layer in layers:
            assert _adapter(layer, 2).cache_installs == 1
            assert _adapter(layer, 1).cache_installs == 0

    def test_refresh_all_caches(self):
        model, layers = _make_model()
        worker = _fake_worker(model)
        WorkerBase.refresh_adapter_caches(worker)
        for layer in layers:
            assert _adapter(layer, 1).cache_installs == 1
            assert _adapter(layer, 2).cache_installs == 1

    def test_paired_training_loop_round_trips(self):
        """Simulate a training loop alternating syncs of a pair."""
        model, layers = _make_model()
        worker = _fake_worker(model)
        for step in range(3):
            prefill_fill = 0.1 * (step + 1)
            decode_fill = 0.01 * (step + 1)
            WorkerBase.sync_adapter_weights(worker,
                                            {0: _state_dict(prefill_fill),
                                             1: _state_dict(prefill_fill)},
                                            refresh_caches=True,
                                            adapter_int_id=1)
            WorkerBase.sync_adapter_weights(worker,
                                            {0: _state_dict(decode_fill),
                                             1: _state_dict(decode_fill)},
                                            refresh_caches=True,
                                            adapter_int_id=2)
        for layer in layers:
            assert torch.allclose(_adapter(layer, 1).proj.weight,
                                  torch.full((HIDDEN, HIDDEN), 0.3))
            assert torch.allclose(_adapter(layer, 2).proj.weight,
                                  torch.full((HIDDEN, HIDDEN), 0.03))
            assert _adapter(layer, 1).cache_installs == 3
            assert _adapter(layer, 2).cache_installs == 3
