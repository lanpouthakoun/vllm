# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Architecture-compatibility matrix for phase-restricted adapter serving.

Kept in its own file (and run as its own pytest session) so each
model's engine gets the whole GPU — the main e2e module keeps several
module-scoped engines alive simultaneously.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="requires GPU")

from tests.adaptation.test_gpu_e2e import (BigDeltaAdapter,  # noqa: E402
                                           _gen_ids, _adapter_req)


# Architectures beyond Qwen2: qwen3 (new hook), llama3 (existing hook on
# a real Llama), phi3 (inherits the llama hook via subclassing), gemma2
# (new hook; gated on HF — skipped when inaccessible), olmo2 (residual-
# free forward contract; also serves Olmo3ForCausalLM via registry
# alias — skipped when not downloadable).
MODEL_MATRIX = [
    "Qwen/Qwen3-0.6B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "google/gemma-2-2b-it",
    "allenai/OLMo-2-0425-1B",
]


class TestModelMatrixE2E:

    @pytest.mark.parametrize("model_id", MODEL_MATRIX)
    def test_adapter_phase_serving(self, model_id):
        from transformers import AutoConfig

        from vllm import LLM
        try:
            hidden = AutoConfig.from_pretrained(model_id).hidden_size
            llm = LLM(model=model_id,
                      enforce_eager=True,
                      enable_adapters=True,
                      max_adapters=8,
                      # Small models at short context: keep the ask low
                      # so sequential engines never hit residual-memory
                      # pressure from earlier tests in the session.
                      gpu_memory_utilization=0.15,
                      max_model_len=256)
        except Exception as e:  # noqa: BLE001 — gated/unavailable models
            msg = str(e)
            if any(s in msg for s in ("gated", "401", "403", "Access",
                                      "restricted", "not found")):
                pytest.skip(f"{model_id} unavailable: {msg[:120]}")
            raise
        try:
            from vllm.adaptation.specs import spec_to_adapter_config
            base = _gen_ids(llm)

            llm.collective_rpc(
                "load_adapter",
                args=(31, spec_to_adapter_config({
                    "layer_indices": [0, 1],
                    "position": "decode",
                    "sample_adapter": BigDeltaAdapter(hidden_size=hidden,
                                                      value=-5.0),
                }), "decode"))
            decode_out = _gen_ids(llm, adapter_request=_adapter_req(31))
            # Exact-complement boundary holds on every architecture.
            assert decode_out[0] == base[0]
            assert decode_out != base

            llm.collective_rpc(
                "load_adapter",
                args=(32, spec_to_adapter_config({
                    "layer_indices": [0, 1],
                    "position": "prefill",
                    "sample_adapter": BigDeltaAdapter(hidden_size=hidden,
                                                      value=5.0),
                }), "prefill"))
            prefill_out = _gen_ids(llm, adapter_request=_adapter_req(32))
            assert prefill_out != base
        finally:
            import gc
            del llm
            gc.collect()
            torch.cuda.empty_cache()


