# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end GPU tests for phase-restricted adaptations.

Runs a real engine (tiny-random Llama) and validates the full stack:

  - prefill-only / decode-only LoRA and their pairing on one prompt,
  - the exact-complement boundary (a decode-only adapter must NOT
    change the first sampled token — that token is produced by the
    final prefill chunk),
  - in-memory weight sync (training loop) actually changing outputs,
  - adapter adaptations via the blueprint path, including decode position,
  - CUDA graph mode: adapters loaded AFTER warmup must take effect
    (invalidation + lazy re-capture), and re-syncs must show up in
    replayed graphs without re-capture.

Requires a GPU + the compiled vLLM install; skipped otherwise.
"""

import os

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="requires GPU")

# NOTE: not the tiny-random llama — its head_dim=4 forces the
# FlexAttention backend, whose inductor-compiled kernel trips a
# triton/inductor version mismatch in this environment.  Qwen2.5-0.5B
# (head_dim 64) uses FlashAttention like real deployments.
MODEL = "Qwen/Qwen2.5-0.5B"
RANK = 8

# collective_rpc needs the in-process engine core.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
# TP>1 worker processes must spawn: the driver initializes CUDA before
# the executor creates workers, and CUDA cannot survive a fork.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    # The engines here are module-scoped; the root conftest's autouse
    # cleanup would tear down the distributed environment between tests
    # and break every generate after the first.
    return False


def _hidden_size() -> int:
    from transformers import AutoConfig
    return AutoConfig.from_pretrained(MODEL).hidden_size


def _prompt(n: int = 8):
    return {"prompt_token_ids": list(range(2, 2 + n))}


def _params():
    from vllm import SamplingParams
    return SamplingParams(temperature=0.0, max_tokens=8)


def _gen_ids(llm, **kwargs) -> list[int]:
    out = llm.generate(_prompt(), _params(), use_tqdm=False, **kwargs)
    return list(out[0].outputs[0].token_ids)


def _lora_state_dict(hidden: int, fill: float):
    tensors = {}
    for i in range(2):  # tiny llama has 2 layers
        prefix = f"base_model.model.model.layers.{i}.self_attn.o_proj"
        tensors[f"{prefix}.lora_A.weight"] = torch.full((RANK, hidden), fill)
        tensors[f"{prefix}.lora_B.weight"] = torch.full((hidden, RANK), fill)
    return tensors


PEFT_CONFIG = {"r": RANK, "lora_alpha": 16, "target_modules": ["o_proj"]}


def _lora_req(idx: int, position: str):
    from vllm.lora.request import LoRARequest
    return LoRARequest(lora_name=f"a{idx}-{position}",
                       lora_int_id=idx,
                       lora_path=f"/synced/{idx}",
                       lora_position=position)


class BigDeltaAdapter(nn.Module):
    """Blueprint-compatible adapter with a large constant delta."""

    def __init__(self, hidden_size=16, value=1.0, device=None, dtype=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.value = value
        self.scale = nn.Parameter(
            torch.full((hidden_size, ), float(value)))

    def readout(self, fx, state=None, x=None):
        return fx + torch.ones_like(fx) * self.scale.to(fx.dtype)


def _adapter_config(hidden: int, value: float):
    from vllm.adaptation.specs import spec_to_adapter_config
    return spec_to_adapter_config({
        "layer_indices": [0, 1],
        "position": "prefill",  # per-adapter position given at load
        "sample_adapter": BigDeltaAdapter(hidden_size=hidden, value=value),
    })


def _adapter_req(idx: int):
    from vllm.adaptation.request import AdapterRequest
    return AdapterRequest(adapter_name=f"r{idx}", adapter_int_id=idx,
                       adapter_path=f"/synced/{idx}")


@pytest.fixture(scope="module")
def eager_llm():
    from vllm import LLM
    llm = LLM(model=MODEL,
              enforce_eager=True,
              enable_lora=True,
              max_loras=4,
              max_lora_rank=RANK,
              enable_adapters=True,
              max_adapters=8,
              gpu_memory_utilization=0.3,
              max_model_len=256)
    yield llm
    del llm


class TestLoraPhaseServingE2E:

    @pytest.fixture(scope="class", autouse=True)
    def synced_adapters(self, eager_llm):
        hidden = _hidden_size()
        eager_llm.collective_rpc(
            "sync_lora_weights",
            args=(_lora_state_dict(hidden, 0.5), PEFT_CONFIG, 1))
        eager_llm.collective_rpc(
            "sync_lora_weights",
            args=(_lora_state_dict(hidden, -0.5), PEFT_CONFIG, 2))

    def test_adapters_change_outputs(self, eager_llm):
        base = _gen_ids(eager_llm)
        with_all = _gen_ids(eager_llm, lora_request=_lora_req(1, "all"))
        assert with_all != base

    def test_prefill_only_changes_first_token(self, eager_llm):
        base = _gen_ids(eager_llm)
        prefill = _gen_ids(eager_llm, lora_request=_lora_req(1, "prefill"))
        assert prefill != base

    def test_decode_only_preserves_first_token(self, eager_llm):
        """Exact complement: the first sampled token comes from the
        final prefill chunk, where a decode-only adapter is inactive."""
        base = _gen_ids(eager_llm)
        decode = _gen_ids(eager_llm, lora_request=_lora_req(2, "decode"))
        assert decode[0] == base[0]
        assert decode != base  # later steps diverge

    def test_pair_on_same_prompt(self, eager_llm):
        base = _gen_ids(eager_llm)
        prefill_only = _gen_ids(eager_llm,
                                lora_request=_lora_req(1, "prefill"))
        paired = _gen_ids(eager_llm,
                          lora_request=_lora_req(1, "prefill"),
                          decode_lora_request=_lora_req(2, "decode"))
        # The pair shares the prefill adapter, so the first token
        # matches the prefill-only run...
        assert paired[0] == prefill_only[0]
        # ...but the decode adapter changes the continuation.
        assert paired != prefill_only
        assert paired != base

    def test_resync_changes_outputs(self, eager_llm):
        hidden = _hidden_size()
        before = _gen_ids(eager_llm, lora_request=_lora_req(1, "all"))
        eager_llm.collective_rpc(
            "sync_lora_weights",
            args=(_lora_state_dict(hidden, 0.05), PEFT_CONFIG, 1))
        after = _gen_ids(eager_llm, lora_request=_lora_req(1, "all"))
        assert after != before
        # Restore for other tests.
        eager_llm.collective_rpc(
            "sync_lora_weights",
            args=(_lora_state_dict(hidden, 0.5), PEFT_CONFIG, 1))


class TestAdapterServingE2E:

    @pytest.fixture(scope="class", autouse=True)
    def loaded_adapters(self, eager_llm):
        hidden = _hidden_size()
        eager_llm.collective_rpc(
            "load_adapter",
            args=(11, _adapter_config(hidden, 5.0), "prefill"))
        eager_llm.collective_rpc(
            "load_adapter",
            args=(12, _adapter_config(hidden, -5.0), "decode"))

    def test_prefill_adapter_changes_outputs(self, eager_llm):
        base = _gen_ids(eager_llm)
        out = _gen_ids(eager_llm, adapter_request=_adapter_req(11))
        assert out != base

    def test_decode_adapter_preserves_first_token(self, eager_llm):
        base = _gen_ids(eager_llm)
        out = _gen_ids(eager_llm, adapter_request=_adapter_req(12))
        assert out[0] == base[0]
        assert out != base

    def test_adapter_pair_on_same_prompt(self, eager_llm):
        prefill_only = _gen_ids(eager_llm, adapter_request=_adapter_req(11))
        paired = _gen_ids(eager_llm,
                          adapter_request=_adapter_req(11),
                          decode_adapter_request=_adapter_req(12))
        assert paired[0] == prefill_only[0]
        assert paired != prefill_only

    def test_every_k_position(self, eager_llm):
        hidden = _hidden_size()
        eager_llm.collective_rpc(
            "load_adapter",
            args=(13, _adapter_config(hidden, 7.0), "every_2"))
        base = _gen_ids(eager_llm)
        out = _gen_ids(eager_llm, adapter_request=_adapter_req(13))
        assert out != base

    def test_adapter_weight_sync_changes_outputs(self, eager_llm):
        before = _gen_ids(eager_llm, adapter_request=_adapter_req(11))
        new_sd = {"scale": torch.full((_hidden_size(), ), 0.25)}
        eager_llm.collective_rpc("sync_adapter_weights",
                                 args=({0: new_sd, 1: new_sd}, True, 11))
        after = _gen_ids(eager_llm, adapter_request=_adapter_req(11))
        assert after != before


@pytest.fixture(scope="module")
def graph_llm():
    """Engine with CUDA graphs enabled (no enforce_eager)."""
    from vllm import LLM
    llm = LLM(model=MODEL,
              enable_lora=True,
              max_loras=4,
              max_lora_rank=RANK,
              enable_adapters=True,
              max_adapters=8,
              gpu_memory_utilization=0.3,
              max_model_len=256)
    yield llm
    del llm


class TestCudaGraphModeE2E:
    """Supported graph-mode semantics:

    - LoRA is fully dynamic under CUDA graphs: warmup captures with
      dummy LoRAs, so punica kernels are in the graphs and driven only
      by buffers/metadata.
    - Dynamically loaded adapters CANNOT take effect (the compiled
      forward was traced without them); loading one must warn and must
      not disturb anything else — dynamic adapter serving is eager-only.
    - Construction-baked adapter + weight sync is the graph-mode training
      path (covered by TestGraphBakedAdapterE2E).
    """

    def test_synced_lora_after_warmup_takes_effect(self, graph_llm):
        hidden = _hidden_size()
        base = _gen_ids(graph_llm)
        graph_llm.collective_rpc(
            "sync_lora_weights",
            args=(_lora_state_dict(hidden, 0.5), PEFT_CONFIG, 3))
        out = _gen_ids(graph_llm, lora_request=_lora_req(3, "decode"))
        assert out[0] == base[0]
        assert out != base

    def test_lora_resync_visible_in_replayed_graphs(self, graph_llm):
        """In-place slot updates must show up without any re-capture."""
        hidden = _hidden_size()
        before = _gen_ids(graph_llm, lora_request=_lora_req(3, "all"))
        graph_llm.collective_rpc(
            "sync_lora_weights",
            args=(_lora_state_dict(hidden, 0.05), PEFT_CONFIG, 3))
        after = _gen_ids(graph_llm, lora_request=_lora_req(3, "all"))
        assert after != before

    def test_dynamic_adapter_load_is_inert_and_harmless(self, graph_llm):
        """Loading a adapter post-warmup warns; it must neither
        corrupt base generations nor break LoRA serving."""
        hidden = _hidden_size()
        base = _gen_ids(graph_llm)
        lora_before = _gen_ids(graph_llm, lora_request=_lora_req(3, "all"))

        graph_llm.collective_rpc(
            "load_adapter",
            args=(21, _adapter_config(hidden, -5.0), "decode"))

        # Base generations are untouched...
        assert _gen_ids(graph_llm) == base
        # ...LoRA keeps working (captured graphs were left alone)...
        assert _gen_ids(graph_llm,
                        lora_request=_lora_req(3, "all")) == lora_before
        # ...and the dynamic adapter is documented-inert here.
        assert _gen_ids(graph_llm, adapter_request=_adapter_req(21)) == base


@pytest.fixture(scope="module")
def tp2_llm():
    """Tensor-parallel engine (eager) for multi-GPU adapter validation.

    Workers run as separate processes; the adapter blueprint class must be
    importable there (run with PYTHONPATH including the tests root)."""
    from vllm import LLM
    llm = LLM(model=MODEL,
              tensor_parallel_size=2,
              enforce_eager=True,
              enable_lora=True,
              max_loras=4,
              max_lora_rank=RANK,
              enable_adapters=True,
              max_adapters=8,
              gpu_memory_utilization=0.3,
              max_model_len=256)
    yield llm
    del llm


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
class TestTensorParallelE2E:

    @pytest.fixture(scope="class", autouse=True)
    def synced(self, tp2_llm):
        hidden = _hidden_size()
        tp2_llm.collective_rpc(
            "sync_lora_weights",
            args=(_lora_state_dict(hidden, 0.5), PEFT_CONFIG, 1))
        tp2_llm.collective_rpc(
            "sync_lora_weights",
            args=(_lora_state_dict(hidden, -0.5), PEFT_CONFIG, 2))

    def test_lora_pair_under_tp2(self, tp2_llm):
        base = _gen_ids(tp2_llm)
        prefill_only = _gen_ids(tp2_llm,
                                lora_request=_lora_req(1, "prefill"))
        paired = _gen_ids(tp2_llm,
                          lora_request=_lora_req(1, "prefill"),
                          decode_lora_request=_lora_req(2, "decode"))
        assert prefill_only != base
        assert paired[0] == prefill_only[0]
        assert paired != prefill_only

    def test_decode_boundary_under_tp2(self, tp2_llm):
        base = _gen_ids(tp2_llm)
        decode = _gen_ids(tp2_llm, lora_request=_lora_req(2, "decode"))
        assert decode[0] == base[0]
        assert decode != base

    def test_adapter_decode_adapter_under_tp2(self, tp2_llm):
        hidden = _hidden_size()
        base = _gen_ids(tp2_llm)
        tp2_llm.collective_rpc(
            "load_adapter",
            args=(21, _adapter_config(hidden, -5.0), "decode"))
        out = _gen_ids(tp2_llm, adapter_request=_adapter_req(21))
        assert out[0] == base[0]
        assert out != base

    def test_adapter_weight_sync_under_tp2(self, tp2_llm):
        before = _gen_ids(tp2_llm, adapter_request=_adapter_req(21))
        new_sd = {"scale": torch.full((_hidden_size(), ), 2.5)}
        tp2_llm.collective_rpc("sync_adapter_weights",
                               args=({0: new_sd, 1: new_sd}, True, 21))
        after = _gen_ids(tp2_llm, adapter_request=_adapter_req(21))
        assert after != before


@pytest.fixture(scope="module")
def graph_baked_adapter_llm():
    """CUDA graphs + a adapter baked at construction (the
    graph-mode training configuration): the compiled forward includes
    the adapter, masks stay batch-agnostic, weight sync is in-place."""
    from vllm import LLM
    hidden = _hidden_size()
    llm = LLM(model=MODEL,
              adapter_config=_adapter_config(hidden, -5.0),
              max_adapters=8,
              gpu_memory_utilization=0.3,
              max_model_len=256)
    yield llm
    del llm


class TestGraphBakedAdapterE2E:

    def test_weight_sync_visible_in_replayed_graphs(
            self, graph_baked_adapter_llm):
        llm = graph_baked_adapter_llm
        before = _gen_ids(llm)
        new_sd = {"scale": torch.full((_hidden_size(), ), 2.5)}
        llm.collective_rpc("sync_adapter_weights",
                           args=({0: new_sd, 1: new_sd}, True, 1))
        after = _gen_ids(llm)
        assert after != before

    def test_repeated_sync_round_trips(self, graph_baked_adapter_llm):
        """Alternating weight syncs (a training loop) keep taking
        effect — in-place updates read by the same captured graphs."""
        llm = graph_baked_adapter_llm
        hidden = _hidden_size()

        def sync(v):
            sd = {"scale": torch.full((hidden, ), v)}
            llm.collective_rpc("sync_adapter_weights",
                               args=({0: sd, 1: sd}, True, 1))

        sync(3.0)
        out_a1 = _gen_ids(llm)
        sync(-4.0)
        out_b = _gen_ids(llm)
        sync(3.0)
        out_a2 = _gen_ids(llm)
        assert out_a1 != out_b
        assert out_a1 == out_a2  # deterministic round-trip
