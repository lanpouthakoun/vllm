# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plumbing tests: decode_lora_request / decode_adapter_request must survive
the trip EngineCoreRequest -> Request -> NewRequestData, be validated by
the Processor, and be counted by the scheduler's adapter budget."""

from types import SimpleNamespace

import pytest

from vllm.lora.request import LoRARequest
from vllm.adaptation.request import AdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.core.sched.scheduler import (_request_lora_ids,
                                          _request_adapter_ids)
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.processor import Processor
from vllm.v1.request import Request


def _lora(idx: int, position: str = "all") -> LoRARequest:
    return LoRARequest(lora_name=f"a{idx}-{position}",
                       lora_int_id=idx,
                       lora_path=f"/fake/{idx}",
                       lora_position=position)


def _adapter(idx: int) -> AdapterRequest:
    return AdapterRequest(adapter_name=f"r{idx}", adapter_int_id=idx,
                       adapter_path=f"/fake/{idx}")


def _engine_core_request(**overrides) -> EngineCoreRequest:
    fields = dict(
        request_id="req-1",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=4),
        pooling_params=None,
        eos_token_id=None,
        arrival_time=0.0,
        lora_request=None,
        adapter_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    fields.update(overrides)
    return EngineCoreRequest(**fields)


class TestRequestPropagation:

    def test_engine_core_request_defaults(self):
        req = _engine_core_request()
        assert req.decode_lora_request is None
        assert req.decode_adapter_request is None

    def test_fields_survive_to_request_and_scheduler_output(self):
        prefill = _lora(1, "prefill")
        decode = _lora(2, "decode")
        adapter_decode = _adapter(4)
        core_req = _engine_core_request(
            lora_request=prefill,
            decode_lora_request=decode,
            adapter_request=_adapter(3),
            decode_adapter_request=adapter_decode,
        )
        req = Request.from_engine_core_request(core_req, block_hasher=None)
        assert req.lora_request is prefill
        assert req.decode_lora_request is decode
        assert req.decode_adapter_request is adapter_decode

        new_req_data = NewRequestData.from_request(req, block_ids=([0], ))
        assert new_req_data.lora_request is prefill
        assert new_req_data.decode_lora_request is decode
        assert new_req_data.decode_adapter_request is adapter_decode


class TestSchedulerBudgetHelpers:

    def test_no_adapters(self):
        req = SimpleNamespace(lora_request=None,
                              decode_lora_request=None,
                              adapter_request=None,
                              decode_adapter_request=None)
        assert _request_lora_ids(req) == set()
        assert _request_adapter_ids(req) == set()

    def test_pair_counts_two_lora_ids(self):
        req = SimpleNamespace(lora_request=_lora(1, "prefill"),
                              decode_lora_request=_lora(2, "decode"),
                              adapter_request=None,
                              decode_adapter_request=None)
        assert _request_lora_ids(req) == {1, 2}

    def test_pair_counts_two_adapter_ids(self):
        req = SimpleNamespace(lora_request=None,
                              decode_lora_request=None,
                              adapter_request=_adapter(5),
                              decode_adapter_request=_adapter(6))
        assert _request_adapter_ids(req) == {5, 6}


class TestProcessorValidation:

    def _validate(self, lora_request, decode_lora_request):
        # _validate_lora / _validate_decode_lora only touch lora_config
        # and tokenizer, so a stub self is sufficient.
        fake = SimpleNamespace(lora_config=SimpleNamespace(max_loras=4),
                               tokenizer=None)
        fake._validate_lora = (
            lambda lr: Processor._validate_lora(fake, lr))
        Processor._validate_decode_lora(fake, lora_request,
                                        decode_lora_request)

    def test_valid_pair_accepted(self):
        self._validate(_lora(1, "prefill"), _lora(2, "decode"))

    def test_decode_alone_accepted(self):
        self._validate(None, _lora(2, "decode"))

    def test_decode_slot_wrong_position_rejected(self):
        with pytest.raises(ValueError, match="decode"):
            self._validate(_lora(1, "prefill"), _lora(2, "prefill"))

    def test_primary_all_position_rejected_when_paired(self):
        with pytest.raises(ValueError, match="prefill"):
            self._validate(_lora(1, "all"), _lora(2, "decode"))

    def test_same_id_rejected(self):
        with pytest.raises(ValueError, match="distinct"):
            self._validate(_lora(7, "prefill"), _lora(7, "decode"))

    def test_lora_disabled_rejected(self):
        fake = SimpleNamespace(lora_config=None, tokenizer=None)
        fake._validate_lora = (
            lambda lr: Processor._validate_lora(fake, lr))
        with pytest.raises(ValueError, match="not enabled"):
            Processor._validate_decode_lora(fake, None, _lora(2, "decode"))
