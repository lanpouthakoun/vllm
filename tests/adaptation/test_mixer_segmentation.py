# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request segmentation for sequence-mixing adaptations.

Stateless sequence mixers (cnn/bigram) and chunked adapters mix along
the sequence axis.  vLLM's flattened batch concatenates unrelated
requests, so without segmentation such an adaptation would leak
information across request boundaries.  Adaptations that declare (or
imply, via a ``mixer`` attribute) sequence mixing now run once per
request span in eager mode.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn

from vllm.adaptation import needs_sequence_segmentation
from vllm.adaptation.layer import (_add_adapter_to_layer, _init_multi_adapter_state,
                             _multi_adapter_forward,
                             update_adapter_position_masks)

HIDDEN = 4


def _meta(num_prefill_tokens, num_decodes, num_prefills,
          query_start_loc=None):
    return SimpleNamespace(num_prefill_tokens=num_prefill_tokens,
                           num_decodes=num_decodes,
                           num_prefills=num_prefills,
                           query_start_loc=query_start_loc,
                           seq_lens=None)


class _PlainMixer:
    stateful = False
    needs_kv = False


class CumsumMixingAdapter(nn.Module):
    """delta[t] = sum_{i<=t} h[i] along the sequence — any cross-request
    leakage changes the numbers."""

    def __init__(self):
        super().__init__()
        self.marker = nn.Linear(1, 1)
        self.mixer = _PlainMixer()
        self.seen_token_counts: list[int] = []

    def readout(self, fx, state=None, x=None):
        self.seen_token_counts.append(fx.shape[-2])
        return fx + torch.cumsum(fx, dim=-2)


class TokenLocalAdapter(nn.Module):

    def __init__(self):
        super().__init__()
        self.marker = nn.Linear(1, 1)
        self.seen_token_counts: list[int] = []

    def readout(self, fx, state=None, x=None):
        self.seen_token_counts.append(fx.shape[-2])
        return fx + fx * 0.5


class TestDetection:

    def test_mixer_implies_segmentation(self):
        assert needs_sequence_segmentation(CumsumMixingAdapter())

    def test_token_local_does_not(self):
        assert not needs_sequence_segmentation(TokenLocalAdapter())

    def test_explicit_flag_wins(self):
        a = CumsumMixingAdapter()
        a.sequence_mixing = False
        assert not needs_sequence_segmentation(a)
        b = TokenLocalAdapter()
        b.sequence_mixing = True
        assert needs_sequence_segmentation(b)


def _run(layer, num_tokens, meta, token_ids=None, positions=None,
         graph_safe=False):
    if token_ids is None:
        token_ids = torch.ones(num_tokens, dtype=torch.int32)
    if positions is None:
        positions = torch.arange(num_tokens)
    update_adapter_position_masks([layer], token_ids, positions, meta,
                                     num_tokens, graph_safe=graph_safe)
    hidden = torch.ones(num_tokens, HIDDEN)
    residual = torch.ones(num_tokens, HIDDEN)

    def super_forward(positions, hidden_states, residual):
        return hidden_states, residual

    h, r = _multi_adapter_forward(layer, positions, hidden, residual,
                               super_forward=super_forward)
    return h + r


class TestSegmentedApplication:

    def _mixed_batch_meta(self):
        # 2 decode tokens, then prefill requests of 3 and 4 tokens.
        qsl = torch.tensor([0, 1, 2, 5, 9])
        return _meta(7, 2, 2, query_start_loc=qsl)

    def test_no_leakage_across_request_boundaries(self):
        layer = nn.Module()
        _init_multi_adapter_state(layer, torch.device("cpu"), HIDDEN)
        adapter = CumsumMixingAdapter()
        _add_adapter_to_layer(layer, 1, adapter, "all", torch.device("cpu"))
        positions = torch.tensor([5, 8, 0, 1, 2, 0, 1, 2, 3])
        stream = _run(layer, 9, self._mixed_batch_meta(),
                      positions=positions)
        # h_full = 2.0 everywhere.  Within each request span, delta is
        # a cumsum restarting at the boundary: token j of a span gets
        # 2.0 * (j + 1).
        expected = 2.0 + 2.0 * torch.tensor(
            [1, 1, 1, 2, 3, 1, 2, 3, 4], dtype=torch.float32)
        assert torch.allclose(stream[:, 0], expected)
        # Ran once per span, not once over the flattened batch.
        assert adapter.seen_token_counts == [1, 1, 3, 4]

    def test_token_local_adapter_unsegmented(self):
        layer = nn.Module()
        _init_multi_adapter_state(layer, torch.device("cpu"), HIDDEN)
        adapter = TokenLocalAdapter()
        _add_adapter_to_layer(layer, 1, adapter, "all", torch.device("cpu"))
        _run(layer, 9, self._mixed_batch_meta(),
             positions=torch.tensor([5, 8, 0, 1, 2, 0, 1, 2, 3]))
        assert adapter.seen_token_counts == [9]

    def test_membership_mask_still_applies_per_segment(self):
        layer = nn.Module()
        _init_multi_adapter_state(layer, torch.device("cpu"), HIDDEN)
        adapter = CumsumMixingAdapter()
        _add_adapter_to_layer(layer, 1, adapter, "all", torch.device("cpu"))
        # Only the second prefill request belongs to adapter 1.
        token_ids = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1],
                                 dtype=torch.int32)
        positions = torch.tensor([5, 8, 0, 1, 2, 0, 1, 2, 3])
        stream = _run(layer, 9, self._mixed_batch_meta(),
                      token_ids=token_ids, positions=positions)
        expected = 2.0 + 2.0 * torch.tensor(
            [0, 0, 0, 0, 0, 1, 2, 3, 4], dtype=torch.float32)
        assert torch.allclose(stream[:, 0], expected)

    def test_single_request_batch_no_segmentation_needed(self):
        layer = nn.Module()
        _init_multi_adapter_state(layer, torch.device("cpu"), HIDDEN)
        adapter = CumsumMixingAdapter()
        _add_adapter_to_layer(layer, 1, adapter, "all", torch.device("cpu"))
        qsl = torch.tensor([0, 4])
        stream = _run(layer, 4, _meta(4, 0, 1, query_start_loc=qsl))
        expected = 2.0 + 2.0 * torch.tensor([1, 2, 3, 4],
                                            dtype=torch.float32)
        assert torch.allclose(stream[:, 0], expected)

    def test_graph_safe_mode_does_not_segment(self):
        # Dynamic per-request segment counts are not graph-representable;
        # graph mode keeps the flattened application (and dynamic adapter
        # is eager-only anyway).
        layer = nn.Module()
        _init_multi_adapter_state(layer, torch.device("cpu"), HIDDEN)
        adapter = CumsumMixingAdapter()
        _add_adapter_to_layer(layer, 1, adapter, "all", torch.device("cpu"))
        _run(layer, 9, self._mixed_batch_meta(),
             positions=torch.tensor([5, 8, 0, 1, 2, 0, 1, 2, 3]),
             graph_safe=True)
        assert adapter.seen_token_counts == [9]
