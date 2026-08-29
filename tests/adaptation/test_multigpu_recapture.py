# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-GPU-safe CUDA graph re-capture for adaptations.

Two pieces:

1. Lazy captures (post-invalidation) must run inside the distributed
   ``graph_capture()`` coordination context — side stream + custom
   allreduce capture mode — exactly like warmup captures.  The wrapper
   enters it per capture when distributed is initialized and no outer
   context is active.

2. Captured graphs bake Python control flow.  In graph mode the
   adaptation forward must therefore be batch-agnostic: every loaded
   adapter computes unconditionally and per-batch selection lives only
   in the fixed-address mask buffers.  ``graph_safe=True`` on the mask
   updater disables the Python-state skips and zeroes stale buffers of
   adapters that have no tokens this step.
"""

from contextlib import nullcontext
from types import SimpleNamespace

import torch
import torch.nn as nn

from vllm.compilation.cuda_graph import _lazy_capture_context
from vllm.distributed import parallel_state
from vllm.adaptation.layer import (_add_adapter_to_layer, _init_multi_adapter_state,
                             _multi_adapter_forward,
                             update_adapter_position_masks)

HIDDEN = 8


def _meta(num_prefill_tokens, num_decodes, num_prefills):
    return SimpleNamespace(
        num_prefill_tokens=num_prefill_tokens,
        num_decodes=num_decodes,
        num_prefills=num_prefills,
        query_start_loc=None,
        seq_lens=None,
    )


class TestLazyCaptureContext:

    def test_nullcontext_when_distributed_not_initialized(self):
        assert not parallel_state.model_parallel_is_initialized()
        ctx = _lazy_capture_context()
        assert isinstance(ctx, nullcontext)

    def test_nullcontext_when_outer_context_active(self, monkeypatch):
        monkeypatch.setattr(parallel_state, "model_parallel_is_initialized",
                            lambda: True)
        monkeypatch.setattr(parallel_state, "_graph_capture_depth", 1)
        ctx = _lazy_capture_context()
        assert isinstance(ctx, nullcontext)

    def test_enters_graph_capture_when_distributed(self, monkeypatch):
        entered = []

        def fake_graph_capture(device):
            entered.append(device)
            return nullcontext("fake-ctx")

        monkeypatch.setattr(parallel_state, "model_parallel_is_initialized",
                            lambda: True)
        monkeypatch.setattr(parallel_state, "_graph_capture_depth", 0)
        monkeypatch.setattr(parallel_state, "graph_capture",
                            fake_graph_capture)
        ctx = _lazy_capture_context()
        with ctx as val:
            assert val == "fake-ctx"
        assert len(entered) == 1

    def test_depth_flag_default_and_accessor(self):
        assert parallel_state.is_graph_capture_context_active() is False
        parallel_state._graph_capture_depth += 1
        try:
            assert parallel_state.is_graph_capture_context_active() is True
        finally:
            parallel_state._graph_capture_depth -= 1


class ShapeRecordingAdapter(nn.Module):

    def __init__(self, value):
        super().__init__()
        self.value = value
        self.marker = nn.Linear(1, 1)
        self.seen_token_counts: list[int] = []

    def readout(self, fx, state=None, x=None):
        self.seen_token_counts.append(fx.shape[-2])
        return fx + torch.full_like(fx, self.value)


def _make_layer():
    layer = nn.Module()
    _init_multi_adapter_state(layer, torch.device("cpu"), HIDDEN)
    return layer


def _forward_stream(layer, num_tokens, positions=None):
    hidden = torch.ones(num_tokens, HIDDEN)
    residual = torch.ones(num_tokens, HIDDEN)
    if positions is None:
        positions = torch.arange(num_tokens)

    def super_forward(positions, hidden_states, residual):
        return hidden_states, residual

    h, r = _multi_adapter_forward(layer, positions, hidden, residual,
                               super_forward=super_forward)
    return h + r


class TestGraphSafeMaskUpdate:

    def test_pure_decode_does_not_python_skip(self):
        """Prefill-only adapter on a pure-decode batch: eager mode skips
        via Python state (bad for graphs); graph-safe mode computes with
        a zero mask instead — same numbers, batch-agnostic control flow.
        """
        layer = _make_layer()
        adapter = ShapeRecordingAdapter(0.5)
        _add_adapter_to_layer(layer, 1, adapter, "prefill",
                              torch.device("cpu"))
        token_ids = torch.ones(3, dtype=torch.int32)
        positions = torch.tensor([5, 6, 7])
        update_adapter_position_masks([layer], token_ids, positions,
                                         _meta(0, 3, 0), 3, graph_safe=True)
        assert layer._adapter_all_masks_zero is False
        assert layer._adapter_active_ids is None
        stream = _forward_stream(layer, 3, positions)
        # Adapter computed (control flow is batch-agnostic)...
        assert adapter.seen_token_counts == [3]
        # ...but the zero mask nullified it.
        assert torch.allclose(stream, torch.full((3, HIDDEN), 2.0))

    def test_stale_buffers_zeroed_for_inactive_adapters(self):
        layer = _make_layer()
        a1 = ShapeRecordingAdapter(0.5)
        a2 = ShapeRecordingAdapter(0.25)
        _add_adapter_to_layer(layer, 1, a1, "all", torch.device("cpu"))
        _add_adapter_to_layer(layer, 2, a2, "all", torch.device("cpu"))

        # Step 1: batch references adapter 1 -> its buffer is nonzero.
        token_ids = torch.ones(4, dtype=torch.int32)
        update_adapter_position_masks([layer], token_ids,
                                         torch.arange(4), _meta(4, 0, 1), 4,
                                         graph_safe=True)
        assert layer._adapter_combined_masks[1][:4].sum() > 0

        # Step 2: batch references adapter 2 only.  Adapter 1's stale
        # mask must be zeroed, else the (batch-agnostic) forward would
        # keep applying it.
        token_ids = torch.full((4, ), 2, dtype=torch.int32)
        update_adapter_position_masks([layer], token_ids,
                                         torch.arange(4), _meta(4, 0, 1), 4,
                                         graph_safe=True)
        assert layer._adapter_combined_masks[1][:4].sum() == 0
        stream = _forward_stream(layer, 4)
        assert torch.allclose(stream, torch.full((4, HIDDEN), 2.25))

    def test_graph_safe_equivalent_to_eager_path(self):
        def build():
            layer = _make_layer()
            _add_adapter_to_layer(layer, 1, ShapeRecordingAdapter(0.5),
                                  "prefill", torch.device("cpu"))
            _add_adapter_to_layer(layer, 2, ShapeRecordingAdapter(0.25),
                                  "decode", torch.device("cpu"))
            return layer

        token_ids = torch.tensor([1, 1, 1, 1], dtype=torch.int32)
        decode_slot = torch.tensor([2, 2, 2, 2], dtype=torch.int32)
        positions = torch.tensor([9, 4, 0, 1])
        meta = _meta(2, 2, 1)

        streams = []
        for graph_safe in (False, True):
            layer = build()
            update_adapter_position_masks(
                [layer], token_ids, positions, meta, 4,
                decode_token_adapter_ids=decode_slot, graph_safe=graph_safe)
            streams.append(_forward_stream(layer, 4, positions))
        assert torch.allclose(streams[0], streams[1])

    def test_graph_safe_pure_decode_only_prefill_adapters_zeroes_all(self):
        # The eager early-return path (skip everything) must not run in
        # graph-safe mode; buffers get zeroed instead.
        layer = _make_layer()
        _add_adapter_to_layer(layer, 1, ShapeRecordingAdapter(0.5),
                              "prefill", torch.device("cpu"))
        # Make the buffer dirty first.
        update_adapter_position_masks([layer],
                                         torch.ones(4, dtype=torch.int32),
                                         torch.arange(4), _meta(4, 0, 1), 4,
                                         graph_safe=True)
        assert layer._adapter_combined_masks[1][:4].sum() > 0
        update_adapter_position_masks([layer],
                                         torch.ones(2, dtype=torch.int32),
                                         torch.tensor([5, 6]), _meta(0, 2, 0),
                                         2, graph_safe=True)
        assert layer._adapter_all_masks_zero is False
        assert layer._adapter_combined_masks[1].sum() == 0
