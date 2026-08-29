# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The parameterized "every_<k>" position family.

``position="every_k"`` applies an adaptation to every token whose
position id satisfies ``pos % k == 0``; ``every_k_offset_j`` shifts the
phase to ``pos % k == j``.  Masks are computed from position ids alone,
so they are exact under chunked prefill, batching, and during decode,
and need no attention metadata.  Names resolve on demand — no explicit
registration required.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.adaptation import (PhaseInfo, get_position_mask,
                             position_active_in_decode)
from vllm.adaptation.layer import (_add_adapter_to_layer, _init_multi_adapter_state,
                             _multi_adapter_forward,
                             update_adapter_position_masks)

HIDDEN = 8


def _phase(num_prefill_tokens, num_decodes=0, num_prefills=1):
    return PhaseInfo(num_prefill_tokens=num_prefill_tokens,
                     num_decodes=num_decodes,
                     num_prefills=num_prefills,
                     query_start_loc=None,
                     seq_lens=None)


def _meta(num_prefill_tokens, num_decodes, num_prefills):
    return SimpleNamespace(num_prefill_tokens=num_prefill_tokens,
                           num_decodes=num_decodes,
                           num_prefills=num_prefills,
                           query_start_loc=None,
                           seq_lens=None)


class TestEveryKMask:

    def test_basic_period(self):
        positions = torch.arange(10)
        mask = get_position_mask("every_3", positions, torch.float32, 10,
                                 _phase(10))
        assert mask.tolist() == [1, 0, 0, 1, 0, 0, 1, 0, 0, 1]

    def test_offset(self):
        positions = torch.arange(9)
        mask = get_position_mask("every_3_offset_1", positions,
                                 torch.float32, 9, _phase(9))
        assert mask.tolist() == [0, 1, 0, 0, 1, 0, 0, 1, 0]

    def test_period_one_is_all_tokens(self):
        positions = torch.arange(4)
        mask = get_position_mask("every_1", positions, torch.float32, 4,
                                 _phase(4))
        assert mask.tolist() == [1, 1, 1, 1]

    def test_chunked_prefill_uses_absolute_positions(self):
        # Mid-prompt chunk: positions 30..37.
        positions = torch.arange(30, 38)
        mask = get_position_mask("every_4", positions, torch.float32, 8,
                                 _phase(8))
        # Multiples of 4 in [30, 38): 32, 36.
        assert mask.tolist() == [0, 0, 1, 0, 0, 0, 1, 0]

    def test_decode_tokens_match_by_position(self):
        # Pure decode batch, no metadata needed: two requests at
        # positions 8 and 13 with k=4 -> only 8 matches.
        positions = torch.tensor([8, 13])
        mask = get_position_mask("every_4", positions, torch.float32, 2,
                                 PhaseInfo(None, 2, 0, None, None))
        assert mask.tolist() == [1, 0]

    def test_active_in_decode(self):
        assert position_active_in_decode("every_7")

    def test_resolves_repeatedly(self):
        positions = torch.arange(4)
        for _ in range(2):
            mask = get_position_mask("every_2", positions, torch.float32, 4,
                                     _phase(4))
            assert mask.tolist() == [1, 0, 1, 0]


class TestEveryKValidation:

    def test_zero_period_rejected(self):
        with pytest.raises(ValueError, match="every_"):
            get_position_mask("every_0", torch.arange(2), torch.float32, 2,
                              _phase(2))

    def test_offset_must_be_less_than_period(self):
        with pytest.raises(ValueError, match="offset"):
            get_position_mask("every_3_offset_3", torch.arange(2),
                              torch.float32, 2, _phase(2))

    def test_unrelated_unknown_names_still_rejected(self):
        with pytest.raises(ValueError, match="Registered positions"):
            get_position_mask("sometimes", torch.arange(2), torch.float32,
                              2, _phase(2))


class ConstAdapter(nn.Module):

    def __init__(self, value):
        super().__init__()
        self.value = value
        self.marker = nn.Linear(1, 1)

    def readout(self, fx, state=None, x=None):
        return fx + torch.full_like(fx, self.value)


class TestEveryKServing:

    def test_every_k_adapter_in_forward(self):
        layer = nn.Module()
        _init_multi_adapter_state(layer, torch.device("cpu"), HIDDEN)
        _add_adapter_to_layer(layer, 1, ConstAdapter(0.5), "every_2",
                              torch.device("cpu"))
        token_ids = torch.ones(6, dtype=torch.int32)
        positions = torch.arange(6)
        update_adapter_position_masks([layer], token_ids, positions,
                                         _meta(6, 0, 1), 6)
        hidden = torch.ones(6, HIDDEN)
        residual = torch.ones(6, HIDDEN)

        def super_forward(positions, hidden_states, residual):
            return hidden_states, residual

        h, r = _multi_adapter_forward(layer, positions, hidden, residual,
                                   super_forward=super_forward)
        stream = h + r
        expected = torch.tensor([2.5, 2.0, 2.5, 2.0, 2.5, 2.0])
        assert torch.allclose(stream,
                              expected.unsqueeze(-1).expand(6, HIDDEN))

    def test_every_k_not_skipped_on_pure_decode(self):
        layer = nn.Module()
        _init_multi_adapter_state(layer, torch.device("cpu"), HIDDEN)
        _add_adapter_to_layer(layer, 1, ConstAdapter(0.5), "every_4",
                              torch.device("cpu"))
        token_ids = torch.ones(2, dtype=torch.int32)
        positions = torch.tensor([8, 13])
        update_adapter_position_masks([layer], token_ids, positions,
                                         _meta(0, 2, 0), 2)
        assert not layer._adapter_all_masks_zero
        assert layer._adapter_combined_masks[1][:2].tolist() == [1, 0]

    def test_every_k_paired_with_prefill_adapter(self):
        # Prefill adapter on slot 0, every_3 adapter on the decode slot:
        # during decode, the every_3 adapter fires only on matching
        # positions.
        layer = nn.Module()
        _init_multi_adapter_state(layer, torch.device("cpu"), HIDDEN)
        _add_adapter_to_layer(layer, 1, ConstAdapter(0.5), "prefill",
                              torch.device("cpu"))
        _add_adapter_to_layer(layer, 2, ConstAdapter(0.25), "every_3",
                              torch.device("cpu"))
        primary = torch.tensor([1, 1], dtype=torch.int32)
        decode_slot = torch.tensor([2, 2], dtype=torch.int32)
        positions = torch.tensor([6, 7])  # decode steps; 6 % 3 == 0
        update_adapter_position_masks([layer], primary, positions,
                                         _meta(0, 2, 0), 2,
                                         decode_token_adapter_ids=decode_slot)
        assert layer._adapter_combined_masks[2][:2].tolist() == [1, 0]
        assert 1 not in layer._adapter_active_ids
