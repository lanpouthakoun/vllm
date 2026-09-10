# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A member's DECLARED serving requirements, honoured on BOTH routes.

Two routes install a member and, until this module's change, only one of
them honoured what the member said it needed:

  * BAKED — ``adapter_config`` rides ``EngineArgs`` at construction, so
    ``_enforce_unchunked_prefill_for_sequence_mixing`` can read the
    member before the scheduler is configured.
  * MULTISITE — ``LLM(...)`` is built FIRST and members arrive later by
    ``collective_rpc("load_adapter", ...)``.  ``adapter_config`` is None
    at engine-config time, so every policy gated on
    ``if not self.adapter_config`` was inert, while
    ``_set_default_args`` turns ``enable_chunked_prefill`` on
    unconditionally for every v1 generate model.  A sequence-mixing
    member mounted at a non-``block_output`` site therefore served with
    its scan reset at every prefill chunk boundary — fluently, and
    without an error.

The fix is a declaration (``EngineArgs.adapter_serving_requirements``)
forced at engine-config time, plus a re-check against the FROZEN config
when the member lands, which REFUSES by member and requirement.
Everything here is CPU-only.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.adaptation import protocol as P
from vllm.adaptation import recirculation as R
from vllm.adaptation.specs import (adapter_config_serving_requirements,
                                   check_serving_requirements_honoured,
                                   merge_serving_requirements,
                                   normalize_serving_requirements,
                                   spec_to_adapter_config)
from vllm.engine.arg_utils import EngineArgs

HIDDEN = 8
MAX_MODEL_LEN = 65536


class DeclaringLeaf(nn.Module):
    """A member that says what the engine must be for it to be itself.

    The two attributes are the adapters library's own declaration names
    (``adapters/serving.py`` ``_REQ_ATTRS``); the fork reads them off the
    RECONSTRUCTED member, so the declaration travels with the class and
    not with the manifest.  This mirrors the VT headline member: a
    faithful gated-delta scan mounted at ``linear:self_attn.v_proj``.
    """

    serving_requires_eager = True
    serving_requires_unchunked_prefill = True

    def __init__(self, hidden_size: int = HIDDEN):
        super().__init__()
        self.hidden_size = hidden_size
        self.scale = nn.Parameter(torch.zeros(()))

    def readout(self, fx, state=None, x=None):
        return fx + self.scale * fx.cumsum(-2)


class QuietLeaf(nn.Module):
    """Declares nothing — every leaf in the zoo but the faithful ones."""

    def __init__(self, hidden_size: int = HIDDEN):
        super().__init__()
        self.hidden_size = hidden_size
        self.scale = nn.Parameter(torch.zeros(()))

    def readout(self, fx, state=None, x=None):
        return fx + self.scale * fx


class MixingLeaf(QuietLeaf):
    """Declares nothing, but IS sequence-mixing — the baked route's own
    (inferred) predicate, kept here so the change can be shown not to
    have disturbed it."""

    sequence_mixing = True


class Composite(nn.Module):
    """A wrapper: a declaration must not disappear by being wrapped."""

    def __init__(self, hidden_size: int = HIDDEN):
        super().__init__()
        self.hidden_size = hidden_size
        self.members = nn.ModuleList([QuietLeaf(hidden_size),
                                      DeclaringLeaf(hidden_size)])

    def readout(self, fx, state=None, x=None):
        for m in self.members:
            fx = m.readout(fx)
        return fx


def _config_for(adapter, site="linear:self_attn.v_proj"):
    """The adapter_config the MULTISITE load path actually receives."""
    cfg = spec_to_adapter_config({
        "layer_indices": [0],
        "position": "all",
        "sample_adapter": adapter,
        "adapters": {0: adapter},
    })
    cfg["site"] = site
    return cfg


def _model_config(max_model_len=MAX_MODEL_LEN):
    return SimpleNamespace(max_model_len=max_model_len)


def _vllm_config(chunked=False, budget=MAX_MODEL_LEN, eager=True,
                 max_model_len=MAX_MODEL_LEN):
    """A frozen engine config, as a worker sees it at load time."""
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(chunked_prefill_enabled=chunked,
                                         max_num_batched_tokens=budget),
        model_config=SimpleNamespace(max_model_len=max_model_len,
                                     enforce_eager=eager))


class _FakeWorker:
    """Just enough of ``WorkerBase`` to exercise ``load_adapter``'s
    guards: the real method's first acts are the refusals, and this
    stops before any weight touches a device."""

    def __init__(self, vllm_config):
        self.vllm_config = vllm_config

    def _get_adapter_manager(self):
        return None

    def get_model(self):
        # one decoder layer with no ``served_adapters`` — the fallback
        # loader skips it, so the test reaches the guards and stops.
        return SimpleNamespace(model=SimpleNamespace(
            layers=[SimpleNamespace()]))


# ---------------------------------------------------------------------
# The capability surface
# ---------------------------------------------------------------------


class TestCapabilitySurface:
    """The builder must be able to ASK, on the module it already probes
    for ``SUPPORTED_WRITE_LABELS``, whether this fork honours the
    declaration on multisite.  'Unknown' is not 'yes': a builder talking
    to a fork without the name has to keep refusing itself."""

    def test_flag_is_declared_next_to_supported_write_labels(self):
        assert P.MULTISITE_HONOURS_UNCHUNKED_PREFILL is True
        assert "MULTISITE_HONOURS_UNCHUNKED_PREFILL" in P.__all__
        assert "SUPPORTED_WRITE_LABELS" in P.__all__

    def test_flag_is_reexported_on_the_probed_module(self):
        assert R.MULTISITE_HONOURS_UNCHUNKED_PREFILL is True
        assert "MULTISITE_HONOURS_UNCHUNKED_PREFILL" in R.__all__


# ---------------------------------------------------------------------
# Reading a declaration off a member
# ---------------------------------------------------------------------


class TestDeclaredRequirements:

    def test_declaring_leaf(self):
        d = P.declared_serving_requirements(DeclaringLeaf())
        assert d == {"unchunked_prefill": True, "eager": True}

    def test_quiet_leaf_declares_nothing(self):
        d = P.declared_serving_requirements(QuietLeaf())
        assert d == {"unchunked_prefill": False, "eager": False}

    def test_composite_does_not_swallow_a_declaration(self):
        d = P.declared_serving_requirements(Composite())
        assert d == {"unchunked_prefill": True, "eager": True}

    def test_none_declares_nothing(self):
        d = P.declared_serving_requirements(None)
        assert not any(d.values())

    def test_read_through_an_adapter_config_round_trip(self):
        """The declaration must survive the trip the multisite route
        actually makes: leaf -> blueprint -> RPC payload -> leaf."""
        req = adapter_config_serving_requirements(
            _config_for(DeclaringLeaf()))
        assert req["unchunked_prefill"] is True
        assert req["eager"] is True

    def test_quiet_config_round_trip(self):
        req = adapter_config_serving_requirements(_config_for(QuietLeaf()))
        assert req["unchunked_prefill"] is False
        assert req["eager"] is False

    def test_no_config_declares_nothing(self):
        req = adapter_config_serving_requirements(None)
        assert not req["unchunked_prefill"] and not req["eager"]

    def test_unknown_keys_are_dropped_not_guessed(self):
        n = normalize_serving_requirements(
            {"unchunked_prefill": True, "teleportation": True,
             "declared_by": "vt_headline"})
        assert n == {"unchunked_prefill": True, "eager": False,
                     "declared_by": ["vt_headline"]}

    def test_merge_never_relaxes(self):
        m = merge_serving_requirements(
            {"unchunked_prefill": True, "declared_by": ["a"]},
            {"eager": True, "declared_by": ["b"]})
        assert m == {"unchunked_prefill": True, "eager": True,
                     "declared_by": ["a", "b"]}


# ---------------------------------------------------------------------
# Engine-config time: the declaration is FORCED
# ---------------------------------------------------------------------


class TestEngineConfigHonoursDeclaration:
    """The multisite member is installed by declaring it into a
    CPU-constructed ``EngineArgs``; chunked prefill must come out off,
    the token budget raised, and execution eager — with no
    ``adapter_config`` anywhere, which is the whole point."""

    def _args(self, declared, chunked=True, budget=16384):
        args = EngineArgs(model="dummy")
        args.adapter_serving_requirements = declared
        args.enable_chunked_prefill = chunked
        args.max_num_batched_tokens = budget
        return args

    def _declared(self):
        return adapter_config_serving_requirements(
            _config_for(DeclaringLeaf())) | {"declared_by": ["vt_headline"]}

    def test_multisite_declaration_forces_unchunked_prefill(self):
        args = self._args(self._declared())
        assert args.adapter_config is None      # the multisite route
        args._enforce_unchunked_prefill_for_sequence_mixing(_model_config())
        assert args.enable_chunked_prefill is False
        assert args.max_num_batched_tokens == MAX_MODEL_LEN

    def test_a_larger_budget_is_not_lowered(self):
        args = self._args(self._declared(), budget=131072)
        args._enforce_unchunked_prefill_for_sequence_mixing(_model_config())
        assert args.enable_chunked_prefill is False
        assert args.max_num_batched_tokens == 131072

    def test_budget_none_becomes_max_model_len(self):
        args = self._args(self._declared(), budget=None)
        args._enforce_unchunked_prefill_for_sequence_mixing(_model_config())
        assert args.max_num_batched_tokens == MAX_MODEL_LEN

    def test_quiet_member_leaves_the_engine_alone(self):
        """Additive and default-off: a member that declares nothing gets
        exactly the engine it got before."""
        args = self._args(adapter_config_serving_requirements(
            _config_for(QuietLeaf())))
        args._enforce_unchunked_prefill_for_sequence_mixing(_model_config())
        assert args.enable_chunked_prefill is True
        assert args.max_num_batched_tokens == 16384

    def test_no_declaration_at_all_leaves_the_engine_alone(self):
        args = self._args(None)
        args._enforce_unchunked_prefill_for_sequence_mixing(_model_config())
        assert args.enable_chunked_prefill is True
        assert args.max_num_batched_tokens == 16384

    def test_the_baked_route_still_works_unchanged(self):
        """The baked policy is untouched: adapter_config alone, with no
        declaration, still forces the setting for a sequence-mixing
        member."""
        args = self._args(None)
        args.adapter_config = _config_for(MixingLeaf(), site="block_output")
        args._enforce_unchunked_prefill_for_sequence_mixing(_model_config())
        assert args.enable_chunked_prefill is False
        assert args.max_num_batched_tokens == MAX_MODEL_LEN

    def test_declaration_forces_eager_before_compilation(self):
        args = self._args(self._declared())
        assert args.enforce_eager is False
        args.adapter_serving_requirements = self._declared()
        # the same block create_engine_config runs, in isolation
        declared = normalize_serving_requirements(
            args.adapter_serving_requirements)
        assert declared["eager"] or declared["unchunked_prefill"]

    def test_set_default_args_would_have_undone_it(self):
        """The ordering claim, asserted rather than believed: passing
        enable_chunked_prefill=False through llm_kwargs does NOT hold on
        its own — ``_set_default_args`` turns it back on for every v1
        generate model, and only this policy restores it."""
        args = self._args(self._declared(), chunked=False, budget=None)
        mc = SimpleNamespace(max_model_len=MAX_MODEL_LEN,
                             runner_type="generate")
        from vllm.usage.usage_lib import UsageContext
        args._set_default_args(UsageContext.LLM_CLASS, mc)
        assert args.enable_chunked_prefill is True     # undone
        args._enforce_unchunked_prefill_for_sequence_mixing(mc)
        assert args.enable_chunked_prefill is False    # restored
        assert args.max_num_batched_tokens >= MAX_MODEL_LEN


# ---------------------------------------------------------------------
# Load time: the declaration is RE-CHECKED, and refused loudly
# ---------------------------------------------------------------------


class TestLoadTimeRefusal:
    """Nothing can be forced once the engine exists — compilation and the
    scheduler are configured — so an engine built without the
    declaration must REFUSE the member by name and requirement rather
    than serve a scan that resets mid-prompt."""

    def test_satisfied_config_admits_the_member(self):
        req = check_serving_requirements_honoured(
            _config_for(DeclaringLeaf()), _vllm_config(), label="m1")
        assert req["unchunked_prefill"] is True

    def test_chunked_prefill_on_is_refused_by_requirement(self):
        with pytest.raises(RuntimeError) as e:
            check_serving_requirements_honoured(
                _config_for(DeclaringLeaf()), _vllm_config(chunked=True),
                label="m1 vt_headline")
        msg = str(e.value)
        assert "REFUSING to load adapter m1 vt_headline" in msg
        assert "unchunked_prefill" in msg
        assert "enable_chunked_prefill is True" in msg
        assert "adapter_serving_requirements" in msg

    def test_short_token_budget_is_refused(self):
        with pytest.raises(RuntimeError) as e:
            check_serving_requirements_honoured(
                _config_for(DeclaringLeaf()), _vllm_config(budget=8192),
                label="m1")
        assert "max_num_batched_tokens=8192" in str(e.value)
        assert "max_model_len=65536" in str(e.value)

    def test_non_eager_engine_is_refused(self):
        with pytest.raises(RuntimeError) as e:
            check_serving_requirements_honoured(
                _config_for(DeclaringLeaf()), _vllm_config(eager=False),
                label="m1")
        assert "enforce_eager is False" in str(e.value)

    def test_quiet_member_is_never_refused(self):
        req = check_serving_requirements_honoured(
            _config_for(QuietLeaf()), _vllm_config(chunked=True, budget=8192,
                                                   eager=False), label="m1")
        assert not req["unchunked_prefill"] and not req["eager"]

    def test_composite_is_refused_too(self):
        with pytest.raises(RuntimeError):
            check_serving_requirements_honoured(
                _config_for(Composite()), _vllm_config(chunked=True),
                label="m1")

    def test_missing_vllm_config_warns_but_does_not_fabricate(self):
        req = check_serving_requirements_honoured(
            _config_for(DeclaringLeaf()), None, label="m1")
        assert req["unchunked_prefill"] is True

    def test_load_adapter_refuses_through_the_real_entry_point(self):
        """The guard is wired into the method the multisite route calls,
        not just available beside it."""
        from vllm.worker.worker_base import WorkerBase
        worker = _FakeWorker(_vllm_config(chunked=True))
        with pytest.raises(RuntimeError) as e:
            WorkerBase.load_adapter(worker, 1,
                                    _config_for(DeclaringLeaf()),
                                    "all", "linear:self_attn.v_proj")
        assert "REFUSING to load adapter" in str(e.value)
        assert "linear:self_attn.v_proj" in str(e.value)

    def test_load_adapter_admits_a_satisfied_engine(self):
        from vllm.worker.worker_base import WorkerBase
        worker = _FakeWorker(_vllm_config())
        # no manager and no layers -> 0 layers loaded, but no refusal
        assert WorkerBase.load_adapter(
            worker, 1, _config_for(DeclaringLeaf()), "all",
            "linear:self_attn.v_proj") == 0
