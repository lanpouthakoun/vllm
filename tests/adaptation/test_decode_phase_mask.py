# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the adapter "decode" position mask.

The "decode" position is the exact complement of "prefill": in a v1
batch (decode tokens first, prefill tokens last) it selects the decode
region.  A request running its final prompt chunk is still in the
prefill region, so its first sampled token is produced by the prefill
adapter — the decode adapter takes over from the next step.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.adaptation.layer import (_add_adapter_to_layer,
                                   _compute_position_mask,
                                   _init_multi_adapter_state,
                                   _multi_adapter_forward,
                                   update_adapter_position_masks)

HIDDEN = 8


def _meta(num_prefill_tokens, num_decodes, num_prefills,
          query_start_loc=None, seq_lens=None):
    return SimpleNamespace(
        num_prefill_tokens=num_prefill_tokens,
        num_decodes=num_decodes,
        num_prefills=num_prefills,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
    )


def _mask(position, positions, attn_metadata):
    return _compute_position_mask(positions, position, torch.float32,
                                  positions.numel(), attn_metadata)


class TestDecodePositionMask:

    def test_mixed_batch(self):
        # 4 decode tokens then 6 prefill tokens.
        positions = torch.tensor([5, 9, 3, 7, 0, 1, 2, 0, 1, 2])
        meta = _meta(6, 4, 2)
        mask = _mask("decode", positions, meta)
        assert mask.tolist() == [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

    def test_pure_decode_batch(self):
        positions = torch.tensor([5, 9, 3])
        meta = _meta(0, 3, 0)
        mask = _mask("decode", positions, meta)
        assert mask.tolist() == [1, 1, 1]

    def test_pure_prefill_batch(self):
        positions = torch.tensor([0, 1, 2, 3])
        meta = _meta(4, 0, 1)
        mask = _mask("decode", positions, meta)
        assert mask.tolist() == [0, 0, 0, 0]

    @pytest.mark.parametrize("num_prefill,num_decodes", [(6, 4), (0, 3),
                                                         (4, 0), (1, 1)])
    def test_decode_is_exact_complement_of_prefill(self, num_prefill,
                                                   num_decodes):
        n = num_prefill + num_decodes
        positions = torch.arange(n)
        meta = _meta(num_prefill, num_decodes, 1 if num_prefill else 0)
        prefill_mask = _mask("prefill", positions, meta)
        decode_mask = _mask("decode", positions, meta)
        assert torch.all(prefill_mask + decode_mask == 1.0)

    def test_fallback_no_metadata_prefill_batch(self):
        # Without metadata, a batch starting at position 0 is treated as
        # prefill: decode mask must be all zeros.
        positions = torch.tensor([0, 1, 2])
        mask = _mask("decode", positions, None)
        assert mask.tolist() == [0, 0, 0]

    def test_fallback_no_metadata_decode_batch(self):
        positions = torch.tensor([7, 12])
        mask = _mask("decode", positions, None)
        assert mask.tolist() == [1, 1]


class ConstAdapter(nn.Module):
    """Adapter returning a constant correction; has params for device
    probing."""

    def __init__(self, value: float):
        super().__init__()
        self.value = value
        self.marker = nn.Linear(1, 1)

    def readout(self, fx, state=None, x=None):
        return fx + torch.full_like(fx, self.value)


def _make_layer() -> nn.Module:
    layer = nn.Module()
    _init_multi_adapter_state(layer, torch.device("cpu"), HIDDEN)
    return layer


class TestMultiAdapterMaskUpdate:

    def test_pure_decode_skips_prefill_only_adapters(self):
        layer = _make_layer()
        _add_adapter_to_layer(layer, 1, ConstAdapter(0.5), "prefill",
                              torch.device("cpu"))
        token_ids = torch.tensor([1, 1, 1], dtype=torch.int32)
        positions = torch.tensor([5, 9, 3])
        update_adapter_position_masks([layer], token_ids, positions,
                                      _meta(0, 3, 0), 3)
        assert layer._adapter_all_masks_zero

    def test_pure_decode_does_not_skip_decode_adapters(self):
        layer = _make_layer()
        _add_adapter_to_layer(layer, 2, ConstAdapter(0.25), "decode",
                              torch.device("cpu"))
        token_ids = torch.tensor([2, 2, 2], dtype=torch.int32)
        positions = torch.tensor([5, 9, 3])
        update_adapter_position_masks([layer], token_ids, positions,
                                      _meta(0, 3, 0), 3)
        assert not layer._adapter_all_masks_zero
        assert 2 in layer._adapter_active_ids
        assert layer._adapter_combined_masks[2][:3].tolist() == [1, 1, 1]

    def test_mixed_batch_membership_and_position_combine(self):
        layer = _make_layer()
        _add_adapter_to_layer(layer, 1, ConstAdapter(0.5), "prefill",
                              torch.device("cpu"))
        _add_adapter_to_layer(layer, 2, ConstAdapter(0.25), "decode",
                              torch.device("cpu"))
        # Batch: 2 decode tokens (req A) then 4 prefill tokens (req B).
        # Req A uses adapter 2 (decode); req B uses adapter 1 (prefill).
        token_ids = torch.tensor([2, 2, 1, 1, 1, 1], dtype=torch.int32)
        positions = torch.tensor([8, 4, 0, 1, 2, 3])
        update_adapter_position_masks([layer], token_ids, positions,
                                      _meta(4, 2, 1), 6)
        assert layer._adapter_combined_masks[1][:6].tolist() == \
            [0, 0, 1, 1, 1, 1]
        assert layer._adapter_combined_masks[2][:6].tolist() == \
            [1, 1, 0, 0, 0, 0]

    def test_decode_adapter_excluded_from_final_prefill_chunk(self):
        # A request finishing its prompt is in the prefill region: the
        # decode adapter must contribute nothing on that step.
        layer = _make_layer()
        _add_adapter_to_layer(layer, 2, ConstAdapter(0.25), "decode",
                              torch.device("cpu"))
        token_ids = torch.tensor([2, 2, 2, 2], dtype=torch.int32)
        positions = torch.tensor([0, 1, 2, 3])  # pure prefill chunk
        update_adapter_position_masks([layer], token_ids, positions,
                                      _meta(4, 0, 1), 4)
        assert layer._adapter_combined_masks[2][:4].tolist() == [0, 0, 0, 0]


class TestMultiAdapterForward:

    def _run_forward(self, layer, num_tokens):
        hidden = torch.ones(num_tokens, HIDDEN)
        residual = torch.ones(num_tokens, HIDDEN)
        positions = torch.arange(num_tokens)

        def super_forward(positions, hidden_states, residual):
            return hidden_states, residual

        return _multi_adapter_forward(layer, positions, hidden, residual,
                                      super_forward=super_forward)

    def test_paired_adapters_apply_in_their_own_regions(self):
        layer = _make_layer()
        _add_adapter_to_layer(layer, 1, ConstAdapter(0.5), "prefill",
                              torch.device("cpu"))
        _add_adapter_to_layer(layer, 2, ConstAdapter(0.25), "decode",
                              torch.device("cpu"))
        # 2 decode tokens (adapter 2), 3 prefill tokens (adapter 1).
        token_ids = torch.tensor([2, 2, 1, 1, 1], dtype=torch.int32)
        positions = torch.tensor([9, 5, 0, 1, 2])
        update_adapter_position_masks([layer], token_ids, positions,
                                      _meta(3, 2, 1), 5)

        hidden_out, residual_out = self._run_forward(layer, 5)
        # h_full = hidden + residual = 2.0; output hidden = masked delta.
        assert torch.allclose(residual_out,
                              torch.full((5, HIDDEN), 2.0))
        expected_delta = torch.tensor([0.25, 0.25, 0.5, 0.5, 0.5])
        assert torch.allclose(hidden_out,
                              expected_delta.unsqueeze(-1).expand(5, HIDDEN))

    def test_all_masks_zero_short_circuits(self):
        layer = _make_layer()
        _add_adapter_to_layer(layer, 1, ConstAdapter(0.5), "prefill",
                              torch.device("cpu"))
        token_ids = torch.tensor([1, 1], dtype=torch.int32)
        positions = torch.tensor([9, 5])
        update_adapter_position_masks([layer], token_ids, positions,
                                      _meta(0, 2, 0), 2)
        hidden_out, residual_out = self._run_forward(layer, 2)
        # Skipped entirely: layer output passes through unchanged.
        assert torch.allclose(hidden_out, torch.ones(2, HIDDEN))
        assert torch.allclose(residual_out, torch.ones(2, HIDDEN))
