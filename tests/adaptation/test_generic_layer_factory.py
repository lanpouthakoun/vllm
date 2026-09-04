# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The architecture-generic adapter decoder-layer factory.

``make_adapter_decoder_layer(base_cls, adapter_spec)`` wraps ANY decoder
layer class following vLLM's conventions — constructor args are passed
through verbatim, the forward contract is
``(positions, hidden_states, residual, **kwargs) -> (hidden, residual)``
or the residual-free ``(positions, hidden_states) -> hidden`` used by
norm-after-block architectures (olmo2/olmo3) — and installs the
multi-adapter runtime on it.  The per-architecture factories
(qwen2/llama/qwen3moe/olmo2) become thin wrappers, and new
architectures (qwen3, gemma2, gemma3, ...) hook in with one line via
``maybe_adapter_layer_type``.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.adaptation.layer import (_add_adapter_to_layer, make_adapter_decoder_layer,
                             maybe_adapter_layer_type,
                             update_adapter_position_masks)

HIDDEN = 8


def _meta(num_prefill_tokens, num_decodes, num_prefills):
    return SimpleNamespace(num_prefill_tokens=num_prefill_tokens,
                           num_decodes=num_decodes,
                           num_prefills=num_prefills,
                           query_start_loc=None,
                           seq_lens=None)


class _Cfg:
    hidden_size = HIDDEN
    torch_dtype = torch.float32


class LegacyStyleLayer(nn.Module):
    """(config, cache_config, quant_config, prefix) ctor — qwen2/qwen3/
    gemma2/gemma3 style."""

    def __init__(self, config, cache_config=None, quant_config=None,
                 prefix: str = ""):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = nn.Identity()
        self.mlp = nn.Identity()
        self.lin = nn.Linear(2, 2)

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            residual = torch.zeros_like(hidden_states)
        return hidden_states, residual


class ModernStyleLayer(nn.Module):
    """(*, vllm_config, prefix) ctor — llama/qwen3moe style."""

    def __init__(self, *, vllm_config, prefix: str = "", config=None):
        super().__init__()
        self.hidden_size = vllm_config.model_config.hf_config.hidden_size
        self.lin = nn.Linear(2, 2)

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            residual = torch.zeros_like(hidden_states)
        return hidden_states, residual


class KwargsForwardLayer(LegacyStyleLayer):
    """Forward takes **kwargs — gemma3 style."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen_kwargs = []

    def forward(self, positions, hidden_states, residual, **kwargs):
        self.seen_kwargs.append(dict(kwargs))
        return super().forward(positions, hidden_states, residual)


class PlainStreamLayer(nn.Module):
    """Residual-free forward ``(positions, hidden_states) -> hidden`` —
    olmo2/olmo3 style (norms applied to block outputs, no deferred
    residual add).  Adds +1 so pass-through bugs are visible."""

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__()
        self.hidden_size = vllm_config.model_config.hf_config.hidden_size
        self.lin = nn.Linear(2, 2)

    def forward(self, positions, hidden_states):
        return hidden_states + 1.0


class ConstAdapter(nn.Module):

    def __init__(self, value=0.5):
        super().__init__()
        self.value = value
        self.marker = nn.Linear(1, 1)

    def readout(self, fx, state=None, x=None):
        return fx + torch.full_like(fx, self.value)


class ChunkScanAdapter(nn.Module):
    """Sequence-mixing adapter with gated_delta-style internal chunk
    padding: pads the token axis to a multiple of ``chunk``, computes a
    causal per-chunk cumulative sum, truncates back to S.  Exercises the
    non-multiple-of-chunk sequence lengths that chunked scans must
    handle (the x43d '[8, 64] vs [8, 63]' failure shape)."""

    sequence_mixing = True

    def __init__(self, chunk=4, value=1.0):
        super().__init__()
        self.chunk = chunk
        self.value = value
        self.marker = nn.Linear(1, 1)

    def readout(self, fx, state=None, x=None):
        B, S, D = fx.shape
        C = self.chunk
        pad = (C - S % C) % C
        h = torch.nn.functional.pad(fx, (0, 0, 0, pad))
        n = (S + pad) // C
        h = h.reshape(B, n, C, D).cumsum(2).reshape(B, n * C, D)
        return fx + self.value * h[:, :S]


def _run_forward(layer, num_tokens=4):
    token_ids = torch.ones(num_tokens, dtype=torch.int32)
    positions = torch.arange(num_tokens)
    update_adapter_position_masks([layer], token_ids, positions,
                                     _meta(num_tokens, 0, 1), num_tokens)
    h, r = layer(positions, torch.ones(num_tokens, HIDDEN),
                 torch.ones(num_tokens, HIDDEN))
    return h + r


class TestGenericFactory:

    def test_legacy_ctor_style(self):
        cls = make_adapter_decoder_layer(LegacyStyleLayer)
        layer = cls(_Cfg(), None, None, prefix="model.layers.3")
        assert layer._adapter_layer_idx == 3
        assert hasattr(layer, "served_adapters")
        assert len(layer.served_adapters) == 0

    def test_modern_ctor_style(self):
        cls = make_adapter_decoder_layer(ModernStyleLayer)
        vllm_config = SimpleNamespace(model_config=SimpleNamespace(
            hf_config=_Cfg()))
        layer = cls(vllm_config=vllm_config, prefix="model.layers.7")
        assert layer._adapter_layer_idx == 7
        assert hasattr(layer, "served_adapters")

    def test_positional_prefix_extracted(self):
        cls = make_adapter_decoder_layer(LegacyStyleLayer)
        layer = cls(_Cfg(), None, None, "model.layers.5")
        assert layer._adapter_layer_idx == 5

    def test_dynamic_adapter_applies(self):
        cls = make_adapter_decoder_layer(LegacyStyleLayer)
        layer = cls(_Cfg(), None, None, prefix="model.layers.0")
        _add_adapter_to_layer(layer, 1, ConstAdapter(0.5), "all",
                              torch.device("cpu"))
        stream = _run_forward(layer)
        assert torch.allclose(stream, torch.full((4, HIDDEN), 2.5))

    def test_forward_kwargs_passthrough(self):
        cls = make_adapter_decoder_layer(KwargsForwardLayer)
        layer = cls(_Cfg(), None, None, prefix="model.layers.0")
        _add_adapter_to_layer(layer, 1, ConstAdapter(0.5), "all",
                              torch.device("cpu"))
        token_ids = torch.ones(4, dtype=torch.int32)
        positions = torch.arange(4)
        update_adapter_position_masks([layer], token_ids, positions,
                                         _meta(4, 0, 1), 4)
        h, r = layer(positions, torch.ones(4, HIDDEN),
                     torch.ones(4, HIDDEN), custom_flag=17)
        assert layer.seen_kwargs == [{"custom_flag": 17}]
        assert torch.allclose(h + r, torch.full((4, HIDDEN), 2.5))

    def test_baked_spec_installs_adapter_id_1(self):
        spec = {
            "layer_indices": [2],
            "position": "prefill",
            "sample_adapter": ConstAdapter(0.25),
        }
        cls = make_adapter_decoder_layer(LegacyStyleLayer, spec)
        in_scope = cls(_Cfg(), None, None, prefix="model.layers.2")
        out_of_scope = cls(_Cfg(), None, None, prefix="model.layers.4")
        assert "1" in in_scope.served_adapters
        assert len(out_of_scope.served_adapters) == 0
        stream = _run_forward(in_scope)
        assert torch.allclose(stream, torch.full((4, HIDDEN), 2.25))

    def test_class_name_reflects_base(self):
        cls = make_adapter_decoder_layer(LegacyStyleLayer)
        assert cls.__name__ == "adapterLegacyStyleLayer"
        assert issubclass(cls, LegacyStyleLayer)


class TestPlainStreamContract:
    """Residual-free (olmo2-style) layers run the same multi-adapter
    path; only the calling convention is adapted."""

    def _make_layer(self, spec=None, prefix="model.layers.0"):
        cls = make_adapter_decoder_layer(PlainStreamLayer, spec)
        vllm_config = SimpleNamespace(model_config=SimpleNamespace(
            hf_config=_Cfg()))
        return cls(vllm_config=vllm_config, prefix=prefix)

    def test_wraps_and_installs_state(self):
        layer = self._make_layer(prefix="model.layers.11")
        assert layer._adapter_layer_idx == 11
        assert hasattr(layer, "served_adapters")
        assert len(layer.served_adapters) == 0

    def test_no_adapter_passthrough_is_identical(self):
        # Zero mounted adapters must reproduce the base forward exactly
        # (the zeroinit/base bit-identity contract at the layer level).
        layer = self._make_layer()
        positions = torch.arange(4)
        x = torch.randn(4, HIDDEN)
        out = layer(positions, x)
        assert torch.equal(out, x + 1.0)

    def test_dynamic_adapter_applies_to_stream(self):
        layer = self._make_layer()
        _add_adapter_to_layer(layer, 1, ConstAdapter(0.5), "all",
                              torch.device("cpu"))
        token_ids = torch.ones(4, dtype=torch.int32)
        positions = torch.arange(4)
        update_adapter_position_masks([layer], token_ids, positions,
                                         _meta(4, 0, 1), 4)
        out = layer(positions, torch.ones(4, HIDDEN))
        # stream = base_forward(1) + adapter delta = (1+1) + 0.5
        assert torch.allclose(out, torch.full((4, HIDDEN), 2.5))

    def test_baked_spec_installs_adapter_id_1(self):
        spec = {
            "layer_indices": [2],
            "position": "prefill",
            "sample_adapter": ConstAdapter(0.25),
        }
        cls = make_adapter_decoder_layer(PlainStreamLayer, spec)
        vllm_config = SimpleNamespace(model_config=SimpleNamespace(
            hf_config=_Cfg()))
        in_scope = cls(vllm_config=vllm_config, prefix="model.layers.2")
        out_of_scope = cls(vllm_config=vllm_config, prefix="model.layers.4")
        assert "1" in in_scope.served_adapters
        assert len(out_of_scope.served_adapters) == 0
        token_ids = torch.ones(4, dtype=torch.int32)
        positions = torch.arange(4)
        update_adapter_position_masks([in_scope], token_ids, positions,
                                         _meta(4, 0, 1), 4)
        out = in_scope(positions, torch.ones(4, HIDDEN))
        assert torch.allclose(out, torch.full((4, HIDDEN), 2.25))

    def test_maybe_adapter_layer_type_wraps_plain_contract(self):
        vllm_config = SimpleNamespace(enable_adapters=True,
                                      adapter_config=None,
                                      model_config=SimpleNamespace(
                                          hf_config=_Cfg()))
        cls = maybe_adapter_layer_type(vllm_config, PlainStreamLayer,
                                    arch="olmo2")
        assert cls is not PlainStreamLayer
        assert issubclass(cls, PlainStreamLayer)

    @pytest.mark.parametrize("num_tokens", [3, 5, 7, 9])
    def test_chunked_scan_at_non_multiple_lengths(self, num_tokens):
        """A chunked sequence-mixing member must produce the reference
        result at sequence lengths that are NOT chunk multiples (the
        x43d warmup failure was a chunk-64 scan at 504 tokens), through
        the residual-free contract, in eager mode."""
        layer = self._make_layer()
        _add_adapter_to_layer(layer, 1, ChunkScanAdapter(chunk=4, value=1.0),
                              "all", torch.device("cpu"))
        token_ids = torch.ones(num_tokens, dtype=torch.int32)
        positions = torch.arange(num_tokens)
        update_adapter_position_masks([layer], token_ids, positions,
                                         _meta(num_tokens, 0, 1), num_tokens)
        x = torch.randn(num_tokens, HIDDEN)
        out = layer(positions, x)
        base = x + 1.0                       # PlainStreamLayer forward
        # reference: chunked causal cumsum on the block output
        C = 4
        pad = (C - num_tokens % C) % C
        h = torch.nn.functional.pad(base.unsqueeze(0), (0, 0, 0, pad))
        n = (num_tokens + pad) // C
        ref_corr = h.reshape(1, n, C, HIDDEN).cumsum(2) \
            .reshape(1, n * C, HIDDEN)[0, :num_tokens]
        assert torch.allclose(out, base + ref_corr, atol=1e-6)

    def test_chunked_scan_parity_across_contracts(self):
        """The two forward contracts blend a chunked sequence-mixing
        member identically at a non-multiple-of-chunk length."""
        S = 7
        plain = self._make_layer()
        _add_adapter_to_layer(plain, 1, ChunkScanAdapter(chunk=4, value=1.0),
                              "all", torch.device("cpu"))
        tuple_cls = make_adapter_decoder_layer(LegacyStyleLayer)
        tup = tuple_cls(_Cfg(), None, None, prefix="model.layers.0")
        _add_adapter_to_layer(tup, 1, ChunkScanAdapter(chunk=4, value=1.0),
                              "all", torch.device("cpu"))
        token_ids = torch.ones(S, dtype=torch.int32)
        positions = torch.arange(S)
        for layer in (plain, tup):
            update_adapter_position_masks([layer], token_ids, positions,
                                             _meta(S, 0, 1), S)
        x = torch.randn(S, HIDDEN)
        out_plain = plain(positions, x - 1.0)   # PlainStreamLayer adds +1
        h, r = tup(positions, x, torch.zeros(S, HIDDEN))
        assert torch.allclose(out_plain, h + r, atol=1e-6)


class TestSequenceMixingEagerPolicy:
    """Baked sequence-mixing adaptations force eager at engine-config
    time (chunked scans are eager-only: segmentation + symbolic-shape
    padding are incompatible with general-shape compile/CUDA graphs)."""

    def _config_for(self, adapter):
        from vllm.adaptation.specs import spec_to_adapter_config
        return spec_to_adapter_config({
            "layer_indices": [0],
            "position": "all",
            "sample_adapter": adapter,
            "adapters": {0: adapter},
        })

    def test_sequence_mixing_config_needs_eager(self):
        from vllm.adaptation.specs import adapter_config_needs_eager
        cfg = self._config_for(ChunkScanAdapter(chunk=4, value=1.0))
        assert adapter_config_needs_eager(cfg) is True

    def test_elementwise_config_does_not_need_eager(self):
        from vllm.adaptation.specs import adapter_config_needs_eager
        cfg = self._config_for(ConstAdapter(0.5))
        assert adapter_config_needs_eager(cfg) is False

    def test_none_config_does_not_need_eager(self):
        from vllm.adaptation.specs import adapter_config_needs_eager
        assert adapter_config_needs_eager(None) is False


class TestMaybeAdapterLayerType:

    def _vllm_config(self, enable_adapters, adapter_config=None):
        return SimpleNamespace(enable_adapters=enable_adapters,
                               adapter_config=adapter_config,
                               model_config=SimpleNamespace(
                                   hf_config=_Cfg()))

    def test_disabled_returns_default(self):
        cls = maybe_adapter_layer_type(self._vllm_config(False),
                                    LegacyStyleLayer)
        assert cls is LegacyStyleLayer

    def test_enable_adapters_wraps(self):
        cls = maybe_adapter_layer_type(self._vllm_config(True),
                                    LegacyStyleLayer)
        assert cls is not LegacyStyleLayer
        assert issubclass(cls, LegacyStyleLayer)


class TestBackwardCompatFactories:

    @pytest.mark.parametrize("factory_name,base_path", [
        ("make_adapter_qwen2_layer",
         "vllm.model_executor.models.qwen2.Qwen2DecoderLayer"),
        ("make_adapter_llama_layer",
         "vllm.model_executor.models.llama.LlamaDecoderLayer"),
        ("make_adapter_qwen3_moe_layer",
         "vllm.model_executor.models.qwen3_moe.Qwen3MoeDecoderLayer"),
        ("make_adapter_olmo2_layer",
         "vllm.model_executor.models.olmo2.Olmo2DecoderLayer"),
    ])
    def test_wrappers_subclass_their_base(self, factory_name, base_path):
        import importlib

        import vllm.adaptation.layer as adapter_layer
        module_path, cls_name = base_path.rsplit(".", 1)
        base = getattr(importlib.import_module(module_path), cls_name)
        cls = getattr(adapter_layer, factory_name)(None)
        assert issubclass(cls, base)
