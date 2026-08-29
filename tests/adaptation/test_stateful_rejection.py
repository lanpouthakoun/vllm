# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stateful mixers (recurrent state across decode steps) and needs_kv
adapters cannot run correctly under vLLM's batched, reordered decode
loop.  Loading one must fail loudly, not corrupt generations silently.
"""

import pytest
import torch
import torch.nn as nn

from vllm.adaptation import check_adaptation_supported
from vllm.adaptation.layer import _prepare_adapter


class _StatefulMixer(nn.Module):
    stateful = True
    needs_kv = False


class _KVMixer(nn.Module):
    stateful = False
    needs_kv = True


class _PlainMixer(nn.Module):
    stateful = False
    needs_kv = False


def _adapter_with(mixer):
    adapter = nn.Module()
    adapter.lin = nn.Linear(4, 4)
    adapter.mixer = mixer
    adapter.readout = lambda fx, state=None, x=None: fx
    return adapter


class TestStatefulRejection:

    def test_stateful_mixer_rejected(self):
        with pytest.raises(ValueError, match="stateful"):
            check_adaptation_supported(_adapter_with(_StatefulMixer()))

    def test_needs_kv_rejected(self):
        with pytest.raises(ValueError, match="needs_kv"):
            check_adaptation_supported(_adapter_with(_KVMixer()))

    def test_plain_mixer_accepted(self):
        check_adaptation_supported(_adapter_with(_PlainMixer()))

    def test_no_mixer_accepted(self):
        check_adaptation_supported(_adapter_with(None))

    def test_prepare_adapter_rejects_stateful(self):
        adapter = _adapter_with(_StatefulMixer())
        with pytest.raises(ValueError, match="stateful"):
            _prepare_adapter(adapter, torch.device("cpu"), torch.float32)

    def test_composite_member_with_stateful_mixer_rejected(self):
        # Wrapping a rejected mechanism inside a composite must not
        # smuggle it past the capability check.
        composite = nn.Module()
        composite.members = nn.ModuleList(
            [_adapter_with(_PlainMixer()),
             _adapter_with(_StatefulMixer())])
        with pytest.raises(ValueError, match="stateful"):
            check_adaptation_supported(composite)

    def test_composite_with_supported_members_accepted(self):
        composite = nn.Module()
        composite.members = nn.ModuleList(
            [_adapter_with(_PlainMixer()), _adapter_with(None)])
        check_adaptation_supported(composite)
