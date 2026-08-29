# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the named position-mask registry.

Positions ("all", "prefill", "decode", "first", "last") become named
entries in a registry so custom adaptations can define their own
per-token phase semantics without touching vLLM internals.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.adaptation import (PhaseInfo, get_position_mask,
                             position_active_in_decode,
                             register_position_mask, registered_positions)
from vllm.adaptation.layer import _compute_position_mask


def _meta(num_prefill_tokens, num_decodes, num_prefills):
    return SimpleNamespace(
        num_prefill_tokens=num_prefill_tokens,
        num_decodes=num_decodes,
        num_prefills=num_prefills,
        query_start_loc=None,
        seq_lens=None,
    )


class TestBuiltins:

    def test_builtins_registered(self):
        names = registered_positions()
        for name in ("all", "all_tokens", "prefill", "decode", "first",
                     "last"):
            assert name in names, f"builtin {name!r} missing"

    def test_all_returns_none_mask(self):
        phase = PhaseInfo(num_prefill_tokens=4, num_decodes=0, num_prefills=1,
                          query_start_loc=None, seq_lens=None)
        mask = get_position_mask("all", torch.arange(4), torch.float32, 4,
                                 phase)
        assert mask is None

    def test_prefill_and_decode_via_registry(self):
        positions = torch.tensor([7, 3, 0, 1, 2])
        phase = PhaseInfo(num_prefill_tokens=3, num_decodes=2, num_prefills=1,
                          query_start_loc=None, seq_lens=None)
        prefill = get_position_mask("prefill", positions, torch.float32, 5,
                                    phase)
        decode = get_position_mask("decode", positions, torch.float32, 5,
                                   phase)
        assert prefill.tolist() == [0, 0, 1, 1, 1]
        assert decode.tolist() == [1, 1, 0, 0, 0]

    def test_decode_active_metadata(self):
        assert position_active_in_decode("all")
        assert position_active_in_decode("decode")
        assert not position_active_in_decode("prefill")
        assert not position_active_in_decode("first")
        assert not position_active_in_decode("last")


class TestCustomPositions:

    def test_register_and_dispatch(self):

        def every_other(positions, dtype, num_tokens, phase):
            idx = torch.arange(num_tokens, device=positions.device)
            return (idx % 2 == 0).to(dtype)

        register_position_mask("test_every_other", every_other)
        try:
            positions = torch.arange(6)
            phase = PhaseInfo(num_prefill_tokens=6, num_decodes=0,
                              num_prefills=1, query_start_loc=None,
                              seq_lens=None)
            mask = get_position_mask("test_every_other", positions,
                                     torch.float32, 6, phase)
            assert mask.tolist() == [1, 0, 1, 0, 1, 0]
        finally:
            registered_positions().pop("test_every_other", None)

    def test_custom_position_flows_through_layer_dispatch(self):
        """_compute_position_mask (used by the mask updater) must dispatch
        to registered custom positions."""

        def first_two_decode(positions, dtype, num_tokens, phase):
            if phase.num_prefill_tokens is None:
                return torch.zeros(num_tokens, dtype=dtype,
                                   device=positions.device)
            num_decode = num_tokens - phase.num_prefill_tokens
            idx = torch.arange(num_tokens, device=positions.device)
            return ((idx < num_decode) & (idx < 2)).to(dtype)

        register_position_mask("test_first_two_decode", first_two_decode,
                               active_in_decode=True)
        try:
            positions = torch.tensor([5, 6, 7, 0, 1])
            mask = _compute_position_mask(positions, "test_first_two_decode",
                                          torch.float32, 5, _meta(2, 3, 1))
            assert mask.tolist() == [1, 1, 0, 0, 0]
        finally:
            registered_positions().pop("test_first_two_decode", None)

    def test_custom_position_defaults_to_decode_active(self):
        register_position_mask("test_custom_default",
                               lambda p, d, n, ph: None)
        try:
            assert position_active_in_decode("test_custom_default")
        finally:
            registered_positions().pop("test_custom_default", None)

    def test_unknown_position_raises_with_known_names(self):
        with pytest.raises(ValueError, match="prefill"):
            get_position_mask("no_such_position", torch.arange(2),
                              torch.float32, 2,
                              PhaseInfo(None, 0, 0, None, None))

    def test_duplicate_registration_rejected_without_override(self):
        register_position_mask("test_dup", lambda p, d, n, ph: None)
        try:
            with pytest.raises(ValueError, match="registered"):
                register_position_mask("test_dup", lambda p, d, n, ph: None)
            # Explicit override is allowed.
            register_position_mask("test_dup", lambda p, d, n, ph: None,
                                   override=True)
        finally:
            registered_positions().pop("test_dup", None)


class TestBackwardCompatDispatch:

    @pytest.mark.parametrize("position,expected", [
        ("prefill", [0, 1, 1]),
        ("decode", [1, 0, 0]),
    ])
    def test_layer_compute_position_mask_uses_registry(
            self, position, expected):
        positions = torch.tensor([9, 0, 1])
        mask = _compute_position_mask(positions, position, torch.float32, 3,
                                      _meta(2, 1, 1))
        assert mask.tolist() == expected

    def test_unknown_position_in_layer_dispatch_raises(self):
        with pytest.raises(ValueError):
            _compute_position_mask(torch.arange(2), "bogus", torch.float32,
                                   2, _meta(2, 0, 1))


class TestPromptAnchoredClassification:

    @pytest.mark.parametrize("name", [
        "first", "every_2", "every_16", "every_4_offset_1"])
    def test_prompt_anchored(self, name):
        from vllm.adaptation import position_prompt_anchored
        assert position_prompt_anchored(name)

    @pytest.mark.parametrize("name", [
        "all", "all_tokens", "prefill", "decode", "decode_b", "last",
        "every_2_decode", "every_4_decode_offset_1", "first_8_decode",
        "decode_range_0_16", "custom_thing"])
    def test_not_prompt_anchored(self, name):
        from vllm.adaptation import position_prompt_anchored
        assert not position_prompt_anchored(name)


class TestPrefixIncompatibilityWarning:

    def _config(self, connector, prefix_len):
        extra = {"prefix_len": prefix_len}
        ktc = SimpleNamespace(
            kv_connector=connector,
            get_from_extra_config=lambda key, default: extra.get(key,
                                                                 default))
        return SimpleNamespace(kv_transfer_config=ktc)

    @staticmethod
    def _capture_warnings():
        """Handler-based capture (the vllm logger does not propagate)."""
        import contextlib
        import logging

        @contextlib.contextmanager
        def _cm():
            records = []

            class _H(logging.Handler):

                def emit(self, record):
                    records.append(record.getMessage())

            log = logging.getLogger("vllm.adaptation.layer")
            handler = _H(level=logging.WARNING)
            log.addHandler(handler)
            try:
                yield records
            finally:
                log.removeHandler(handler)

        return _cm()

    def test_warns_for_first_under_prefix(self):
        from vllm.adaptation.layer import (
            _prefix_position_warned, warn_if_prefix_incompatible_position)
        _prefix_position_warned.discard("first")
        with self._capture_warnings() as records:
            warn_if_prefix_incompatible_position(
                "first", self._config("PrefixInjectionConnector", 64))
        assert any("anchored to absolute position 0" in m for m in records)
        # Once per position: repeated loads stay quiet.
        with self._capture_warnings() as records:
            warn_if_prefix_incompatible_position(
                "first", self._config("PrefixInjectionConnector", 64))
        assert not records

    def test_silent_for_decode_anchored_position(self):
        from vllm.adaptation.layer import warn_if_prefix_incompatible_position
        with self._capture_warnings() as records:
            warn_if_prefix_incompatible_position(
                "every_4_decode", self._config("PrefixInjectionConnector", 64))
            warn_if_prefix_incompatible_position(
                "last", self._config("PrefixInjectionConnector", 64))
        assert not records

    def test_silent_without_prefix_connector(self):
        from vllm.adaptation.layer import (
            _prefix_position_warned, warn_if_prefix_incompatible_position)
        _prefix_position_warned.discard("first")
        with self._capture_warnings() as records:
            warn_if_prefix_incompatible_position(
                "first", self._config("SharedStorageConnector", 64))
            warn_if_prefix_incompatible_position(
                "first", self._config("PrefixInjectionConnector", 0))
            warn_if_prefix_incompatible_position("first", None)
            warn_if_prefix_incompatible_position(
                "first", SimpleNamespace(kv_transfer_config=None))
        assert not records
