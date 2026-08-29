# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request prefill+decode adapter pairing.

A request may carry a primary `adapter_request` (an adapter loaded with
a prefill-flavored position) plus a `decode_adapter_request` (an adapter
loaded with position="decode").  InputBatch tracks members in N slots;
`make_adapter_inputs` returns one per-token mapping per populated slot;
the position masks inside the layers then restrict each adapter to its
own phase.  (Slots are positionless — phase restriction lives entirely
in the adapters' position masks, not in which slot an id occupies.)
"""

import numpy as np
import torch

from vllm.adaptation.layer import (_add_adapter_to_layer,
                                   _init_multi_adapter_state,
                                   update_adapter_position_masks)
from vllm.adaptation.request import AdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

from .test_decode_phase_mask import ConstAdapter, _make_layer, _meta


def _make_batch() -> InputBatch:
    return InputBatch(
        max_num_reqs=8,
        max_model_len=256,
        max_num_batched_tokens=512,
        device=torch.device("cpu"),
        pin_memory=False,
        vocab_size=128,
        block_sizes=[16],
    )


def _adapter(idx: int) -> AdapterRequest:
    return AdapterRequest(adapter_name=f"adapter-{idx}", adapter_int_id=idx,
                          adapter_path=f"/fake/{idx}")


_COUNTER = [0]


def _req(num_prompt: int,
         num_computed: int,
         num_output: int = 0,
         adapter_request=None,
         decode_adapter_request=None) -> CachedRequestState:
    _COUNTER[0] += 1
    return CachedRequestState(
        req_id=f"req-{_COUNTER[0]}",
        prompt_token_ids=list(range(num_prompt)),
        mm_features=[],
        sampling_params=SamplingParams(temperature=0.0),
        pooling_params=None,
        generator=None,
        block_ids=([0], ),
        num_computed_tokens=num_computed,
        output_token_ids=[1] * num_output,
        adapter_request=adapter_request,
        decode_adapter_request=decode_adapter_request,
    )


class TestMakeAdapterInputs:

    def test_single_slot_backward_compat(self):
        batch = _make_batch()
        batch.add_request(_req(4, 0, adapter_request=_adapter(1)))
        slots = batch.make_adapter_inputs(np.array([4]))
        assert len(slots) == 1
        assert slots[0].tolist() == [1, 1, 1, 1]

    def test_pair_produces_both_mappings(self):
        batch = _make_batch()
        batch.add_request(
            _req(4, 0, adapter_request=_adapter(1),
                 decode_adapter_request=_adapter(2)))
        primary, decode = batch.make_adapter_inputs(np.array([4]))
        assert primary.tolist() == [1, 1, 1, 1]
        assert decode.tolist() == [2, 2, 2, 2]

    def test_mixed_batch_mappings(self):
        batch = _make_batch()
        batch.add_request(_req(4, 0, adapter_request=_adapter(1)))
        batch.add_request(
            _req(3, 3, num_output=1, adapter_request=_adapter(1),
                 decode_adapter_request=_adapter(2)))
        batch.add_request(_req(2, 0))  # no adapters
        primary, decode = batch.make_adapter_inputs(np.array([4, 1, 2]))
        assert primary.tolist() == [1, 1, 1, 1, 1, 0, 0]
        assert decode.tolist() == [0, 0, 0, 0, 2, 0, 0]

    def test_decode_slot_alone_compacts_to_first_slot(self):
        # Members fill slots contiguously: a lone decode adapter lands
        # in slot 0.  Phase restriction comes from its position mask,
        # not from its slot, so this is equivalent.
        batch = _make_batch()
        batch.add_request(
            _req(4, 4, num_output=1, decode_adapter_request=_adapter(2)))
        slots = batch.make_adapter_inputs(np.array([1]))
        assert len(slots) == 1
        assert slots[0].tolist() == [2]

    def test_remove_request_clears_both_members(self):
        batch = _make_batch()
        req = _req(4, 0, adapter_request=_adapter(1),
                   decode_adapter_request=_adapter(2))
        batch.add_request(req)
        batch.remove_request(req.req_id)
        assert 1 not in batch.adapter_id_to_request_ids
        assert 2 not in batch.adapter_id_to_request_ids

    def test_same_id_in_both_slots_removes_cleanly(self):
        # Odd but legal (positions live on the adapter, not the
        # request): the same id in both slots must not crash removal.
        batch = _make_batch()
        req = _req(4, 0, adapter_request=_adapter(1),
                   decode_adapter_request=_adapter(1))
        batch.add_request(req)
        batch.remove_request(req.req_id)
        assert 1 not in batch.adapter_id_to_request_ids

    def test_condense_moves_member_slots(self):
        batch = _make_batch()
        r1 = _req(4, 4, num_output=1, adapter_request=_adapter(1))
        r2 = _req(4, 4, num_output=1, adapter_request=_adapter(3),
                  decode_adapter_request=_adapter(4))
        batch.add_request(r1)
        batch.add_request(r2)
        batch.remove_request(r1.req_id)
        batch.condense()
        primary, decode = batch.make_adapter_inputs(np.array([1]))
        assert primary.tolist() == [3]
        assert decode.tolist() == [4]


class TestPairedMaskComputation:

    def test_pair_on_same_request_across_phases(self):
        """One request served by adapter 1 (prefill) + adapter 2 (decode).

        During its prefill chunk only adapter 1 fires; during its decode
        steps only adapter 2 fires.
        """
        layer = _make_layer()
        _add_adapter_to_layer(layer, 1, ConstAdapter(0.5), "prefill",
                              torch.device("cpu"))
        _add_adapter_to_layer(layer, 2, ConstAdapter(0.25), "decode",
                              torch.device("cpu"))

        # Step 1: request in prefill (4 prompt tokens).
        primary = torch.tensor([1, 1, 1, 1], dtype=torch.int32)
        decode_slot = torch.tensor([2, 2, 2, 2], dtype=torch.int32)
        positions = torch.tensor([0, 1, 2, 3])
        update_adapter_position_masks([layer], primary, positions,
                                      _meta(4, 0, 1), 4,
                                      decode_token_adapter_ids=decode_slot)
        assert layer._adapter_combined_masks[1][:4].tolist() == [1, 1, 1, 1]
        assert layer._adapter_combined_masks[2][:4].tolist() == [0, 0, 0, 0]

        # Step 2: request decoding (1 token).
        primary = torch.tensor([1], dtype=torch.int32)
        decode_slot = torch.tensor([2], dtype=torch.int32)
        positions = torch.tensor([4])
        update_adapter_position_masks([layer], primary, positions,
                                      _meta(0, 1, 0), 1,
                                      decode_token_adapter_ids=decode_slot)
        assert not layer._adapter_all_masks_zero
        assert layer._adapter_combined_masks[2][:1].tolist() == [1]
        # Adapter 1 (prefill) must not be active in a pure-decode batch.
        assert 1 not in layer._adapter_active_ids

    def test_two_requests_different_pairs(self):
        layer = _make_layer()
        _add_adapter_to_layer(layer, 1, ConstAdapter(0.5), "prefill",
                              torch.device("cpu"))
        _add_adapter_to_layer(layer, 2, ConstAdapter(0.25), "decode",
                              torch.device("cpu"))
        _add_adapter_to_layer(layer, 3, ConstAdapter(0.125), "prefill",
                              torch.device("cpu"))
        _add_adapter_to_layer(layer, 4, ConstAdapter(0.0625), "decode",
                              torch.device("cpu"))

        # Req A decoding (pair 1+2), req B prefilling 3 tokens (pair 3+4).
        primary = torch.tensor([1, 3, 3, 3], dtype=torch.int32)
        decode_slot = torch.tensor([2, 4, 4, 4], dtype=torch.int32)
        positions = torch.tensor([6, 0, 1, 2])
        update_adapter_position_masks([layer], primary, positions,
                                      _meta(3, 1, 1), 4,
                                      decode_token_adapter_ids=decode_slot)
        assert layer._adapter_combined_masks[2][:4].tolist() == [1, 0, 0, 0]
        assert layer._adapter_combined_masks[3][:4].tolist() == [0, 1, 1, 1]
        # Adapter 1 (prefill) has membership only on the decode token of
        # req A -> masked out.  Adapter 4 (decode) has membership only on
        # prefill tokens of req B -> masked out.
        assert (1 not in layer._adapter_active_ids
                or layer._adapter_combined_masks[1][:4].tolist()
                == [0, 0, 0, 0])
        assert (4 not in layer._adapter_active_ids
                or layer._adapter_combined_masks[4][:4].tolist()
                == [0, 0, 0, 0])
