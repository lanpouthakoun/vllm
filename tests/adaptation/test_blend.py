# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pluggable blend: adaptations may override how their computation is
combined with the hidden stream.

Default blend is additive (`h + mask * delta`, matching all existing
adapters).  An adaptation can instead define
``apply_masked(h, mask) -> h'`` for replacement/gating semantics; the
phase machinery still supplies the same per-token mask.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn

from vllm.adaptation import apply_adaptation
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


class ConstDelta(nn.Module):

    def __init__(self, value):
        super().__init__()
        self.value = value
        self.marker = nn.Linear(1, 1)

    def readout(self, fx, state=None, x=None):
        return fx + torch.full_like(fx, self.value)


class Replacer(nn.Module):
    """Replacement blend: h' = (1-m)*h + m*f(h) with f(h) = 2*h."""

    def __init__(self):
        super().__init__()
        self.marker = nn.Linear(1, 1)

    def apply_masked(self, h, mask):
        m = mask.unsqueeze(-1).to(h.dtype)
        return (1.0 - m) * h + m * (2.0 * h)


class TestApplyAdaptation:

    def test_default_blend_is_additive(self):
        h = torch.ones(4, HIDDEN)
        mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
        out = apply_adaptation(ConstDelta(0.5), h, mask)
        expected = h.clone()
        expected[0] += 0.5
        expected[2] += 0.5
        assert torch.allclose(out, expected)

    def test_apply_masked_override_wins(self):
        h = torch.ones(3, HIDDEN)
        mask = torch.tensor([0.0, 1.0, 1.0])
        out = apply_adaptation(Replacer(), h, mask)
        expected = torch.ones(3, HIDDEN)
        expected[1] = 2.0
        expected[2] = 2.0
        assert torch.allclose(out, expected)


def _make_layer():
    layer = nn.Module()
    _init_multi_adapter_state(layer, torch.device("cpu"), HIDDEN)
    return layer


def _forward(layer, num_tokens):
    hidden = torch.ones(num_tokens, HIDDEN)
    residual = torch.ones(num_tokens, HIDDEN)
    positions = torch.arange(num_tokens)

    def super_forward(positions, hidden_states, residual):
        return hidden_states, residual

    h, r = _multi_adapter_forward(layer, positions, hidden, residual,
                               super_forward=super_forward)
    return h + r  # the value of the residual stream


class TestForwardBlend:

    def test_custom_blend_in_layer_forward(self):
        layer = _make_layer()
        _add_adapter_to_layer(layer, 1, Replacer(), "decode",
                              torch.device("cpu"))
        # 2 decode tokens, 2 prefill tokens.
        token_ids = torch.tensor([1, 1, 1, 1], dtype=torch.int32)
        positions = torch.tensor([5, 8, 0, 1])
        update_adapter_position_masks([layer], token_ids, positions,
                                         _meta(2, 2, 1), 4)
        stream = _forward(layer, 4)
        # h_full = 2.0 everywhere; decode tokens replaced by 2*h = 4.0.
        assert torch.allclose(stream[:2], torch.full((2, HIDDEN), 4.0))
        assert torch.allclose(stream[2:], torch.full((2, HIDDEN), 2.0))

    def test_additive_adapters_unchanged_by_refactor(self):
        layer = _make_layer()
        _add_adapter_to_layer(layer, 1, ConstDelta(0.5), "prefill",
                              torch.device("cpu"))
        _add_adapter_to_layer(layer, 2, ConstDelta(0.25), "decode",
                              torch.device("cpu"))
        token_ids = torch.tensor([2, 1, 1], dtype=torch.int32)
        positions = torch.tensor([6, 0, 1])
        update_adapter_position_masks([layer], token_ids, positions,
                                         _meta(2, 1, 1), 3)
        stream = _forward(layer, 3)
        assert torch.allclose(stream[0], torch.full((HIDDEN, ), 2.25))
        assert torch.allclose(stream[1], torch.full((HIDDEN, ), 2.5))
        assert torch.allclose(stream[2], torch.full((HIDDEN, ), 2.5))
