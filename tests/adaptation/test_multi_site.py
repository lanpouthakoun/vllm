# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-site mounts: adaptations at named points of the decoder layer.

Sites: block_output (default, residual stream after the block),
block_input (stream before the block), post_attn / post_mlp (submodule
outputs before their residual adds, via forward hooks), and
linear:<path> (any named submodule output, tuple-aware for vLLM's
parallel linears).

The TinyDecoderLayer here mirrors vLLM's decoder-layer contract:
``forward(positions, hidden_states, residual) -> (hidden, residual)``
with the residual stream value being ``hidden + residual``.  Its attn
and MLP submodules return zeros so every observed change comes from an
adaptation.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.adaptation import (resolve_site_submodule_path, validate_site)
from vllm.adaptation.layer import (_add_adapter_to_layer, _init_multi_adapter_state,
                             _multi_adapter_forward, _remove_adapter_from_layer,
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


class TupleLinear(nn.Module):
    """Returns (output, bias) like vLLM's parallel linear layers."""

    def forward(self, h):
        return torch.zeros_like(h), None


class TinyAttn(nn.Module):

    def forward(self, h):
        return torch.zeros_like(h)


class TinyMLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.down_proj = TupleLinear()

    def forward(self, h):
        out, _ = self.down_proj(h)
        return out


class TinyDecoderLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.self_attn = TinyAttn()
        self.mlp = TinyMLP()
        _init_multi_adapter_state(self, torch.device("cpu"), HIDDEN)

    def _inner_forward(self, positions, hidden_states, residual):
        if residual is None:
            residual = hidden_states
        else:
            hidden_states = hidden_states + residual
            residual = hidden_states
        hidden_states = self.self_attn(hidden_states)
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def forward(self, positions, hidden_states, residual):
        return _multi_adapter_forward(self, positions, hidden_states, residual,
                                   super_forward=self._inner_forward)


def _run(layer, num_tokens=4, meta=None, token_ids=None,
         positions_override=None):
    positions = (positions_override if positions_override is not None else
                 torch.arange(num_tokens))
    if meta is None:
        meta = _meta(num_tokens, 0, 1)
    if token_ids is None:
        token_ids = torch.ones(num_tokens, dtype=torch.int32)
    update_adapter_position_masks([layer], token_ids, positions, meta,
                                     num_tokens)
    hidden = torch.ones(num_tokens, HIDDEN)
    h, r = layer(positions, hidden, None)
    return h + r  # value of the residual stream


class TestSiteValidation:

    @pytest.mark.parametrize("site", [
        "block_output", "block_input", "post_attn", "post_mlp",
        "linear:mlp.down_proj"
    ])
    def test_valid_sites(self, site):
        validate_site(site)

    @pytest.mark.parametrize("site", ["attn", "linear:", "", "output"])
    def test_invalid_sites(self, site):
        with pytest.raises(ValueError):
            validate_site(site)

    def test_submodule_paths(self):
        assert resolve_site_submodule_path("block_output") is None
        assert resolve_site_submodule_path("block_input") is None
        assert resolve_site_submodule_path("post_attn") == "self_attn"
        assert resolve_site_submodule_path("post_mlp") == "mlp"
        assert resolve_site_submodule_path(
            "linear:mlp.down_proj") == "mlp.down_proj"


class TestSiteApplication:
    """Baseline: with zero attn/mlp, stream out == stream in (1.0)."""

    def test_no_adapters_passthrough(self):
        layer = TinyDecoderLayer()
        stream = _run(layer)
        assert torch.allclose(stream, torch.ones(4, HIDDEN))

    def test_block_output_site_default(self):
        layer = TinyDecoderLayer()
        _add_adapter_to_layer(layer, 1, ConstDelta(0.5), "all",
                              torch.device("cpu"))
        stream = _run(layer)
        assert torch.allclose(stream, torch.full((4, HIDDEN), 1.5))

    def test_block_input_site(self):
        layer = TinyDecoderLayer()
        _add_adapter_to_layer(layer, 1, ConstDelta(0.5), "all",
                              torch.device("cpu"), site="block_input")
        stream = _run(layer)
        # Input stream 1.0 -> 1.5 before the (identity) block.
        assert torch.allclose(stream, torch.full((4, HIDDEN), 1.5))

    def test_post_attn_site(self):
        layer = TinyDecoderLayer()
        _add_adapter_to_layer(layer, 1, ConstDelta(0.3), "all",
                              torch.device("cpu"), site="post_attn")
        stream = _run(layer)
        # attn output 0 -> 0.3, added into the stream.
        assert torch.allclose(stream, torch.full((4, HIDDEN), 1.3))

    def test_post_mlp_site(self):
        layer = TinyDecoderLayer()
        _add_adapter_to_layer(layer, 1, ConstDelta(0.2), "all",
                              torch.device("cpu"), site="post_mlp")
        stream = _run(layer)
        assert torch.allclose(stream, torch.full((4, HIDDEN), 1.2))

    def test_linear_site_tuple_output(self):
        layer = TinyDecoderLayer()
        _add_adapter_to_layer(layer, 1, ConstDelta(0.25), "all",
                              torch.device("cpu"),
                              site="linear:mlp.down_proj")
        stream = _run(layer)
        assert torch.allclose(stream, torch.full((4, HIDDEN), 1.25))

    def test_two_sites_same_request_via_dual_slots(self):
        layer = TinyDecoderLayer()
        _add_adapter_to_layer(layer, 1, ConstDelta(0.3), "prefill",
                              torch.device("cpu"), site="post_attn")
        _add_adapter_to_layer(layer, 2, ConstDelta(0.2), "decode",
                              torch.device("cpu"), site="block_output")
        # Mixed batch: 2 decode tokens then 3 prefill tokens, all of the
        # same request (slot0 -> adapter 1, slot1 -> adapter 2).
        num_tokens = 5
        positions = torch.tensor([7, 8, 0, 1, 2])
        primary = torch.ones(num_tokens, dtype=torch.int32)
        decode_slot = torch.full((num_tokens, ), 2, dtype=torch.int32)
        update_adapter_position_masks([layer], primary, positions,
                                         _meta(3, 2, 1), num_tokens,
                                         decode_token_adapter_ids=decode_slot)
        hidden = torch.ones(num_tokens, HIDDEN)
        h, r = layer(positions, hidden, None)
        stream = h + r
        # Decode tokens: only adapter 2 (block_output) fires -> 1.2.
        assert torch.allclose(stream[:2], torch.full((2, HIDDEN), 1.2))
        # Prefill tokens: only adapter 1 (post_attn) fires -> 1.3.
        assert torch.allclose(stream[2:], torch.full((3, HIDDEN), 1.3))


class TestPhaseGatingAtSites:

    def test_decode_only_post_attn_masked_during_prefill(self):
        layer = TinyDecoderLayer()
        _add_adapter_to_layer(layer, 1, ConstDelta(0.3), "decode",
                              torch.device("cpu"), site="post_attn")
        stream = _run(layer, meta=_meta(4, 0, 1))  # pure prefill
        assert torch.allclose(stream, torch.ones(4, HIDDEN))

    def test_decode_only_post_attn_fires_during_decode(self):
        layer = TinyDecoderLayer()
        _add_adapter_to_layer(layer, 1, ConstDelta(0.3), "decode",
                              torch.device("cpu"), site="post_attn")
        stream = _run(layer,
                      num_tokens=2,
                      meta=_meta(0, 2, 0),
                      positions_override=torch.tensor([5, 9]))
        assert torch.allclose(stream, torch.full((2, HIDDEN), 1.3))

    def test_membership_gating_at_hooked_site(self):
        layer = TinyDecoderLayer()
        _add_adapter_to_layer(layer, 1, ConstDelta(0.3), "all",
                              torch.device("cpu"), site="post_mlp")
        # Only the first two tokens belong to adapter 1.
        token_ids = torch.tensor([1, 1, 0, 0], dtype=torch.int32)
        stream = _run(layer, token_ids=token_ids)
        assert torch.allclose(stream[:2], torch.full((2, HIDDEN), 1.3))
        assert torch.allclose(stream[2:], torch.ones(2, HIDDEN))


class TestManagerSiteFlow:

    def test_site_flows_through_manager(self, monkeypatch):
        import vllm.adaptation.specs as vllm_adapter
        from vllm.adaptation.manager import ServedAdapter, AdapterManager

        layer = TinyDecoderLayer()
        layer._adapter_layer_idx = 0
        monkeypatch.setattr(
            vllm_adapter, "adapter_config_to_spec", lambda cfg: {
                "layer_indices": [0],
                "sample_adapter": ConstDelta(0.3),
                "position": "all",
            })
        manager = AdapterManager([layer], max_adapters=4, max_cpu_adapters=4,
                                   device=torch.device("cpu"),
                                   model_dtype=torch.float32)
        manager.add_adapter(
            ServedAdapter(id=1, position="all", adapter_config={},
                      layer_indices=frozenset([0]), site="post_attn"))
        manager.activate_adapter(1)
        assert layer._adapter_adapter_sites[1] == "post_attn"
        assert len(layer.self_attn._forward_hooks) == 1

        manager.remove_adapter(1)
        assert 1 not in layer._adapter_adapter_sites
        assert len(layer.self_attn._forward_hooks) == 0


class TestHookLifecycle:

    def test_unload_removes_hook(self):
        layer = TinyDecoderLayer()
        _add_adapter_to_layer(layer, 1, ConstDelta(0.3), "all",
                              torch.device("cpu"), site="post_attn")
        assert len(layer.self_attn._forward_hooks) == 1
        _remove_adapter_from_layer(layer, 1)
        assert len(layer.self_attn._forward_hooks) == 0
        stream = _run(layer)
        assert torch.allclose(stream, torch.ones(4, HIDDEN))

    def test_unload_keeps_hook_while_site_still_used(self):
        layer = TinyDecoderLayer()
        _add_adapter_to_layer(layer, 1, ConstDelta(0.3), "all",
                              torch.device("cpu"), site="post_attn")
        _add_adapter_to_layer(layer, 2, ConstDelta(0.1), "all",
                              torch.device("cpu"), site="post_attn")
        _remove_adapter_from_layer(layer, 1)
        assert len(layer.self_attn._forward_hooks) == 1
        token_ids = torch.full((4, ), 2, dtype=torch.int32)
        stream = _run(layer, token_ids=token_ids)
        assert torch.allclose(stream, torch.full((4, HIDDEN), 1.1))

    def test_shared_hook_for_multiple_adapters(self):
        layer = TinyDecoderLayer()
        _add_adapter_to_layer(layer, 1, ConstDelta(0.3), "all",
                              torch.device("cpu"), site="post_attn")
        _add_adapter_to_layer(layer, 2, ConstDelta(0.1), "all",
                              torch.device("cpu"), site="post_attn")
        # One hook serves both adapters.
        assert len(layer.self_attn._forward_hooks) == 1

    def test_remove_adapter_clears_bookkeeping(self):
        layer = TinyDecoderLayer()
        _add_adapter_to_layer(layer, 1, ConstDelta(0.3), "all",
                              torch.device("cpu"),
                              site="linear:mlp.down_proj")
        _remove_adapter_from_layer(layer, 1)
        assert "1" not in layer.served_adapters
        assert 1 not in layer._adapter_adapter_sites
        assert 1 not in layer._adapter_adapter_positions
        assert 1 not in layer._adapter_combined_masks
