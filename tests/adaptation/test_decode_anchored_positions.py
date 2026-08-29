# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generation-anchored position families.

Unlike ``every_k`` (anchored to prompt position 0), these anchor to the
start of *generation*: a token's generation index is
``position_id - prompt_len`` of its request (0 = the first sampled
token).  Requires per-request prompt lengths (``PhaseInfo.prompt_lens``,
supplied by the model runner); without them the masks are inert (all
zeros) rather than misanchored.

Families:
  - ``every_<k>_decode[_offset_<j>]``: gen_idx % k == j
  - ``first_<k>_decode``:              0 <= gen_idx < k
  - ``decode_range_<a>_<b>``:          a <= gen_idx < b
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.adaptation import PhaseInfo, get_position_mask
from vllm.adaptation.positions import position_active_in_decode
from vllm.adaptation.layer import (_add_adapter_to_layer, _init_multi_adapter_state,
                             update_adapter_position_masks)

HIDDEN = 8


def _phase(num_prefill_tokens, num_decodes, num_prefills,
           prompt_lens=None, query_start_loc=None, seq_lens=None):
    return PhaseInfo(num_prefill_tokens=num_prefill_tokens,
                     num_decodes=num_decodes,
                     num_prefills=num_prefills,
                     query_start_loc=query_start_loc,
                     seq_lens=seq_lens,
                     prompt_lens=prompt_lens)


def _mask(name, positions, phase):
    return get_position_mask(name, positions, torch.float32,
                             positions.numel(), phase)


class TestGenerationIndexMasks:

    def test_every_k_decode_pure_decode(self):
        # 3 decoding requests with prompt lengths [5, 9, 4] at
        # positions [7, 9, 4] -> generation indices [2, 0, 0].
        positions = torch.tensor([7, 9, 4])
        phase = _phase(0, 3, 0, prompt_lens=torch.tensor([5, 9, 4]))
        assert _mask("every_2_decode", positions,
                     phase).tolist() == [1, 1, 1]
        assert _mask("every_2_decode_offset_1", positions,
                     phase).tolist() == [0, 0, 0]
        assert _mask("every_3_decode", positions,
                     phase).tolist() == [0, 1, 1]

    def test_first_k_decode(self):
        positions = torch.tensor([7, 9, 4])  # gen_idx [2, 0, 0]
        phase = _phase(0, 3, 0, prompt_lens=torch.tensor([5, 9, 4]))
        assert _mask("first_1_decode", positions,
                     phase).tolist() == [0, 1, 1]
        assert _mask("first_3_decode", positions,
                     phase).tolist() == [1, 1, 1]

    def test_decode_range(self):
        positions = torch.tensor([7, 9, 4])  # gen_idx [2, 0, 0]
        phase = _phase(0, 3, 0, prompt_lens=torch.tensor([5, 9, 4]))
        assert _mask("decode_range_1_3", positions,
                     phase).tolist() == [1, 0, 0]
        assert _mask("decode_range_0_1", positions,
                     phase).tolist() == [0, 1, 1]

    def test_prefill_tokens_never_match(self):
        # Mixed batch: 1 decode token (req0, prompt 5, position 6 ->
        # gen 1) then a prefill request (prompt 5, positions 0..4 ->
        # gen negative).
        positions = torch.tensor([6, 0, 1, 2, 3, 4])
        qsl = torch.tensor([0, 1, 6])
        phase = _phase(5, 1, 1, prompt_lens=torch.tensor([5, 5]),
                       query_start_loc=qsl)
        assert _mask("every_1_decode", positions,
                     phase).tolist() == [1, 0, 0, 0, 0, 0]
        assert _mask("first_9_decode", positions,
                     phase).tolist() == [1, 0, 0, 0, 0, 0]

    def test_inert_without_prompt_lens(self):
        positions = torch.tensor([7, 9])
        phase = _phase(0, 2, 0, prompt_lens=None)
        assert _mask("every_2_decode", positions,
                     phase).tolist() == [0, 0]
        assert _mask("first_2_decode", positions,
                     phase).tolist() == [0, 0]

    def test_active_in_decode(self):
        assert position_active_in_decode("every_4_decode")
        assert position_active_in_decode("first_10_decode")
        assert position_active_in_decode("decode_range_5_10")


class TestValidation:

    def test_every_zero_decode_rejected(self):
        with pytest.raises(ValueError, match="k >= 1"):
            _mask("every_0_decode", torch.arange(2),
                  _phase(2, 0, 1, prompt_lens=torch.tensor([2])))

    def test_bad_offset_rejected(self):
        with pytest.raises(ValueError, match="offset"):
            _mask("every_2_decode_offset_2", torch.arange(2),
                  _phase(2, 0, 1, prompt_lens=torch.tensor([2])))

    def test_empty_range_rejected(self):
        with pytest.raises(ValueError, match="range"):
            _mask("decode_range_5_5", torch.arange(2),
                  _phase(2, 0, 1, prompt_lens=torch.tensor([2])))

    def test_first_zero_rejected(self):
        with pytest.raises(ValueError, match="k >= 1"):
            _mask("first_0_decode", torch.arange(2),
                  _phase(2, 0, 1, prompt_lens=torch.tensor([2])))


class ConstAdapter(nn.Module):

    def __init__(self, value):
        super().__init__()
        self.value = value
        self.marker = nn.Linear(1, 1)

    def readout(self, fx, state=None, x=None):
        return fx + torch.full_like(fx, self.value)


class TestMaskUpdateIntegration:

    def test_prompt_lens_flow_through_mask_update(self):
        layer = nn.Module()
        _init_multi_adapter_state(layer, torch.device("cpu"), HIDDEN)
        _add_adapter_to_layer(layer, 1, ConstAdapter(0.5), "first_1_decode",
                              torch.device("cpu"))
        # Two decoding requests: gen indices [0, 3].
        token_ids = torch.ones(2, dtype=torch.int32)
        positions = torch.tensor([4, 9])
        meta = SimpleNamespace(num_prefill_tokens=0, num_decodes=2,
                               num_prefills=0, query_start_loc=None,
                               seq_lens=None)
        update_adapter_position_masks(
            [layer], token_ids, positions, meta, 2,
            prompt_lens=torch.tensor([4, 6]))
        assert layer._adapter_combined_masks[1][:2].tolist() == [1, 0]

    def test_without_prompt_lens_mask_is_zero(self):
        layer = nn.Module()
        _init_multi_adapter_state(layer, torch.device("cpu"), HIDDEN)
        _add_adapter_to_layer(layer, 1, ConstAdapter(0.5), "first_1_decode",
                              torch.device("cpu"))
        token_ids = torch.ones(2, dtype=torch.int32)
        positions = torch.tensor([4, 9])
        meta = SimpleNamespace(num_prefill_tokens=0, num_decodes=2,
                               num_prefills=0, query_start_loc=None,
                               seq_lens=None)
        update_adapter_position_masks([layer], token_ids, positions,
                                         meta, 2)
        assert layer._adapter_combined_masks[1][:2].tolist() == [0, 0]
