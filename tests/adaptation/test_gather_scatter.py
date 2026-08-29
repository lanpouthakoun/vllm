# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gather -> compute -> scatter for adaptation member tokens.

Adapter compute used to run over the full flattened batch and get
masked afterwards, so one adapter-carrying request in a wide batch paid
adapter GEMMs over every token.  With ``use_gather=True`` (eager mode
only — the dynamic member count is not CUDA-graph-safe), member token
indices are computed once per step outside the forward, the adapter
runs only on member tokens, and results are scattered back.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

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


class ShapeRecordingAdapter(nn.Module):
    """Constant-delta adapter that records the token counts it sees."""

    def __init__(self, value):
        super().__init__()
        self.value = value
        self.marker = nn.Linear(1, 1)
        self.seen_token_counts: list[int] = []

    def readout(self, fx, state=None, x=None):
        self.seen_token_counts.append(fx.shape[-2])
        return fx + torch.full_like(fx, self.value)


class RecordingReplacer(nn.Module):
    """Replacement blend that records the token counts it sees."""

    def __init__(self):
        super().__init__()
        self.marker = nn.Linear(1, 1)
        self.seen_token_counts: list[int] = []

    def apply_masked(self, h, mask):
        self.seen_token_counts.append(h.shape[0])
        m = mask.unsqueeze(-1).to(h.dtype)
        return (1.0 - m) * h + m * (3.0 * h)


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


class TestGatherEquivalence:

    def _run(self, use_gather, adapter_factory):
        layer = _make_layer()
        adapter = adapter_factory()
        _add_adapter_to_layer(layer, 1, adapter, "all", torch.device("cpu"))
        # 10 tokens, only 3 belong to adapter 1.
        token_ids = torch.tensor([0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
                                 dtype=torch.int32)
        positions = torch.arange(10)
        update_adapter_position_masks([layer], token_ids, positions,
                                         _meta(10, 0, 1), 10,
                                         use_gather=use_gather)
        return _forward_stream(layer, 10), adapter

    def test_additive_equivalence(self):
        full, _ = self._run(False, lambda: ShapeRecordingAdapter(0.5))
        gathered, _ = self._run(True, lambda: ShapeRecordingAdapter(0.5))
        assert torch.allclose(full, gathered)

    def test_custom_blend_equivalence(self):
        full, _ = self._run(False, RecordingReplacer)
        gathered, _ = self._run(True, RecordingReplacer)
        assert torch.allclose(full, gathered)

    def test_gather_computes_only_member_tokens(self):
        _, adapter = self._run(True, lambda: ShapeRecordingAdapter(0.5))
        assert adapter.seen_token_counts == [3]

    def test_full_path_computes_all_tokens(self):
        _, adapter = self._run(False, lambda: ShapeRecordingAdapter(0.5))
        assert adapter.seen_token_counts == [10]

    def test_custom_blend_sees_only_members_when_gathered(self):
        _, adapter = self._run(True, RecordingReplacer)
        assert adapter.seen_token_counts == [3]


class TestGatherEdgeCases:

    def test_empty_membership_skips_adapter_entirely(self):
        layer = _make_layer()
        adapter = ShapeRecordingAdapter(0.5)
        _add_adapter_to_layer(layer, 1, adapter, "all", torch.device("cpu"))
        # Batch references adapter 2 only; adapter 1 loaded but no
        # member tokens.  (Both ids loaded so id 1 stays "active" only
        # if referenced — it isn't, so nothing should run.)
        _add_adapter_to_layer(layer, 2, ShapeRecordingAdapter(0.25), "all",
                              torch.device("cpu"))
        token_ids = torch.full((4, ), 2, dtype=torch.int32)
        update_adapter_position_masks([layer], token_ids,
                                         torch.arange(4), _meta(4, 0, 1), 4,
                                         use_gather=True)
        stream = _forward_stream(layer, 4)
        assert adapter.seen_token_counts == []
        assert torch.allclose(stream, torch.full((4, HIDDEN), 2.25))

    def test_phase_mask_empty_members_skips_adapter(self):
        # Decode-only adapter on a pure-prefill batch: membership exists
        # but the phase mask zeroes everything -> no compute at all.
        layer = _make_layer()
        adapter = ShapeRecordingAdapter(0.5)
        _add_adapter_to_layer(layer, 1, adapter, "decode",
                              torch.device("cpu"))
        token_ids = torch.ones(4, dtype=torch.int32)
        update_adapter_position_masks([layer], token_ids,
                                         torch.arange(4), _meta(4, 0, 1), 4,
                                         use_gather=True)
        stream = _forward_stream(layer, 4)
        assert adapter.seen_token_counts == []
        assert torch.allclose(stream, torch.full((4, HIDDEN), 2.0))

    def test_padded_batch_leaves_pad_region_untouched(self):
        # Forward tensors padded beyond the scheduled token count.
        layer = _make_layer()
        _add_adapter_to_layer(layer, 1, ShapeRecordingAdapter(0.5), "all",
                              torch.device("cpu"))
        token_ids = torch.ones(4, dtype=torch.int32)  # 4 real tokens
        update_adapter_position_masks([layer], token_ids,
                                         torch.arange(4), _meta(4, 0, 1), 4,
                                         use_gather=True)
        stream = _forward_stream(layer, 6)  # padded to 6
        assert torch.allclose(stream[:4], torch.full((4, HIDDEN), 2.5))
        assert torch.allclose(stream[4:], torch.full((2, HIDDEN), 2.0))

    def test_gather_disabled_by_default(self):
        layer = _make_layer()
        adapter = ShapeRecordingAdapter(0.5)
        _add_adapter_to_layer(layer, 1, adapter, "all", torch.device("cpu"))
        token_ids = torch.tensor([1, 0, 0, 0], dtype=torch.int32)
        update_adapter_position_masks([layer], token_ids,
                                         torch.arange(4), _meta(4, 0, 1), 4)
        _forward_stream(layer, 4)
        assert adapter.seen_token_counts == [4]

    def test_stale_indices_cleared_when_gather_turned_off(self):
        layer = _make_layer()
        adapter = ShapeRecordingAdapter(0.5)
        _add_adapter_to_layer(layer, 1, adapter, "all", torch.device("cpu"))
        token_ids = torch.tensor([1, 1, 0, 0], dtype=torch.int32)
        update_adapter_position_masks([layer], token_ids,
                                         torch.arange(4), _meta(4, 0, 1), 4,
                                         use_gather=True)
        _forward_stream(layer, 4)
        # Next step without gather: full path again, no stale indices.
        update_adapter_position_masks([layer], token_ids,
                                         torch.arange(4), _meta(4, 0, 1), 4,
                                         use_gather=False)
        _forward_stream(layer, 4)
        assert adapter.seen_token_counts == [2, 4]


class TestGatherAtHookedSites:

    def test_post_attn_site_gathers(self):
        from tests.adaptation.test_multi_site import TinyDecoderLayer
        layer = TinyDecoderLayer()
        adapter = ShapeRecordingAdapter(0.3)
        _add_adapter_to_layer(layer, 1, adapter, "all", torch.device("cpu"),
                              site="post_attn")
        token_ids = torch.tensor([1, 1, 0, 0, 0], dtype=torch.int32)
        positions = torch.arange(5)
        update_adapter_position_masks([layer], token_ids, positions,
                                         _meta(5, 0, 1), 5, use_gather=True)
        hidden = torch.ones(5, HIDDEN)
        h, r = layer(positions, hidden, None)
        stream = h + r
        assert adapter.seen_token_counts == [2]
        assert torch.allclose(stream[:2], torch.full((2, HIDDEN), 1.3))
        assert torch.allclose(stream[2:], torch.ones(3, HIDDEN))
