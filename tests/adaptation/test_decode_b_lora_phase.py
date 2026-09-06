# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""decode_b through the LoRA-view serving route (phase-audit H4).

Members trained with decode_b semantics (HF: pos >= loc — the final
prompt token plus every generated token) must serve with EXACTLY that
phase.  The LoRA route previously could not express it: LoRARequest
rejected "decode_b" outright, and the decode slot's step granularity
cannot fire on the boundary token's prefill step, so routing decode_b
through it would silently drop the trained boundary application.

Now lora_position="decode_b" is first-class: token-level overlay in
InputBatch.make_lora_inputs applies the adapter to the final prompt
token of the final prefill chunk plus all decode steps — mirroring the
stream route's decode_b mask (decode ∪ last-prompt-token,
vllm/adaptation/positions.py).  All tests run on CPU.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_utils import generate_block_hash_extra_keys
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

VOCAB_SIZE = 128
MAX_NUM_REQS = 16
MAX_MODEL_LEN = 512


def _make_batch() -> InputBatch:
    return InputBatch(
        max_num_reqs=MAX_NUM_REQS,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=1024,
        device=torch.device("cpu"),
        pin_memory=False,
        vocab_size=VOCAB_SIZE,
        block_sizes=[16],
    )


def _lora(idx: int, position: str = "all") -> LoRARequest:
    return LoRARequest(
        lora_name=f"adapter-{idx}-{position}",
        lora_int_id=idx,
        lora_path=f"/fake/{idx}",
        lora_position=position,
    )


_REQ_COUNTER = [0]


def _req(
    num_prompt_tokens: int,
    num_computed_tokens: int,
    num_output_tokens: int = 0,
    lora_request: LoRARequest = None,
    decode_lora_request: LoRARequest = None,
) -> CachedRequestState:
    _REQ_COUNTER[0] += 1
    return CachedRequestState(
        req_id=f"req-db-{_REQ_COUNTER[0]}",
        prompt_token_ids=list(range(num_prompt_tokens)),
        mm_features=[],
        sampling_params=SamplingParams(temperature=0.0),
        pooling_params=None,
        generator=None,
        block_ids=([0], ),
        num_computed_tokens=num_computed_tokens,
        output_token_ids=[1] * num_output_tokens,
        lora_request=lora_request,
        decode_lora_request=decode_lora_request,
    )


def _mappings(batch: InputBatch, num_scheduled: list[int]):
    ns = np.array(num_scheduled, dtype=np.int32)
    prompt_mapping, token_mapping, active = batch.make_lora_inputs(ns)
    return prompt_mapping, token_mapping, active


class TestDecodeBRequest:

    def test_lora_request_accepts_decode_b(self):
        # Regression: this raised ValueError before the fix, making
        # decode_b weight members unservable on the LoRA route.
        r = _lora(1, "decode_b")
        assert r.lora_position == "decode_b"

    def test_invalid_position_still_rejected(self):
        with pytest.raises(ValueError, match="lora_position"):
            _lora(1, "boundary")


class TestDecodeBMapping:

    def test_masked_on_early_prefill_chunk(self):
        batch = _make_batch()
        # 6 of 10 prompt tokens scheduled: boundary not reached.
        batch.add_request(_req(10, 0, lora_request=_lora(3, "decode_b")))
        prompt, token, _ = _mappings(batch, [6])
        assert prompt == (0, )
        assert token == tuple([0] * 6)

    def test_boundary_token_fires_in_single_chunk_prompt(self):
        batch = _make_batch()
        # Whole prompt in one chunk: only the LAST prompt token (whose
        # forward samples the first output token) gets the adapter.
        batch.add_request(_req(10, 0, lora_request=_lora(3, "decode_b")))
        prompt, token, _ = _mappings(batch, [10])
        assert prompt == (3, )
        assert token == tuple([0] * 9 + [3])

    def test_boundary_token_fires_in_final_chunk(self):
        batch = _make_batch()
        # Chunked prefill, final chunk covers tokens 6..9: only token 9
        # (the boundary) fires.
        batch.add_request(_req(10, 6, lora_request=_lora(3, "decode_b")))
        prompt, token, _ = _mappings(batch, [4])
        assert prompt == (3, )
        assert token == (0, 0, 0, 3)

    def test_active_on_decode_steps(self):
        batch = _make_batch()
        batch.add_request(
            _req(10, 10, num_output_tokens=1,
                 lora_request=_lora(3, "decode_b")))
        prompt, token, _ = _mappings(batch, [1])
        assert prompt == (3, )
        assert token == (3, )

    def test_exact_decode_b_semantics_across_whole_request(self):
        # decode fires everywhere decode_b does EXCEPT the boundary:
        # union check against the plain-decode adapter on a twin
        # request, step by step.
        for computed, scheduled, out_toks in [(0, 6, 0), (6, 4, 0),
                                              (10, 1, 1), (11, 1, 2)]:
            batch = _make_batch()
            batch.add_request(
                _req(10, computed, num_output_tokens=out_toks,
                     lora_request=_lora(3, "decode_b")))
            batch.add_request(
                _req(10, computed, num_output_tokens=out_toks,
                     lora_request=_lora(4, "decode")))
            _, token, _ = _mappings(batch, [scheduled, scheduled])
            db = np.array(token[:scheduled]) != 0
            dec = np.array(token[scheduled:]) != 0
            is_boundary = np.zeros(scheduled, dtype=bool)
            if computed < 10 <= computed + scheduled:
                is_boundary[10 - computed - 1] = True
            assert (db == (dec | is_boundary)).all(), (computed, token)

    def test_multi_request_batch_indexes_correct_tokens(self):
        batch = _make_batch()
        # req A: mid-prefill (no fire), req B: final chunk (boundary
        # fires), req C: decode (fires) — offsets must respect the
        # flattened batch layout.
        batch.add_request(_req(10, 0, lora_request=_lora(3, "decode_b")))
        batch.add_request(_req(8, 5, lora_request=_lora(5, "decode_b")))
        batch.add_request(
            _req(6, 6, num_output_tokens=1,
                 lora_request=_lora(7, "decode_b")))
        prompt, token, _ = _mappings(batch, [4, 3, 1])
        assert prompt == (0, 5, 7)
        #        req A (4 tok)   req B (3 tok, boundary last)  req C
        assert token == (0, 0, 0, 0, 0, 0, 5, 7)

    def test_pairing_with_decode_slot_rejected(self):
        # decode_b overlaps the decode phase — pairing it with a
        # decode-slot adapter must fail loudly, not double-apply.
        batch = _make_batch()
        with pytest.raises(ValueError, match="prefill"):
            batch.add_request(
                _req(10, 0, lora_request=_lora(3, "decode_b"),
                     decode_lora_request=_lora(4, "decode")))


class TestDecodeBPrefixCacheKeys:

    def test_decode_b_keys_the_block_hash(self):
        # decode_b touches the final prompt token's forward, hence its
        # KV: it must NOT share cached prefills with the base model
        # (unlike plain decode, which is exempt).
        req = SimpleNamespace(mm_features=[],
                              lora_request=_lora(9, "decode_b"),
                              decode_lora_request=None,
                              adapter_request=None,
                              decode_adapter_request=None,
                              cache_salt=None)
        keys, _ = generate_block_hash_extra_keys(req, 0, 16, 0)
        assert keys == (9, )

    def test_plain_decode_still_exempt(self):
        req = SimpleNamespace(mm_features=[],
                              lora_request=_lora(9, "decode"),
                              decode_lora_request=None,
                              adapter_request=None,
                              decode_adapter_request=None,
                              cache_salt=None)
        keys, _ = generate_block_hash_extra_keys(req, 0, 16, 0)
        assert keys is None
