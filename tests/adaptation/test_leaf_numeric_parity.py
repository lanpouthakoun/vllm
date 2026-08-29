# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Training<->serving numeric parity, at the leaf level, BITWISE.

History: numerics drifted on the previous fork because the serving-side
adapter was a re-implementation of the training-side one. The current
contract is stronger: serving reconstructs the SAME class from a
blueprint (ctor kwargs + state_dict), so the delta computation is the
same code on the same tensors — parity must be exact equality, not
allclose. These tests pin that contract for one leaf per mechanism
family, plus the refresh path (a synced state_dict must change the
served delta bitwise-identically to the training module).

CPU-only (deterministic; no kernels involved — the leaf math is plain
torch either side).
"""

import copy

import pytest
import torch

adapters_registry = pytest.importorskip(
    "adapters.registry",
    reason="requires the adapters library (shared-contract branch)")
from adapters.registry import LEAF_REGISTRY  # noqa: E402

from vllm.adaptation.specs import (_adapter_to_blueprint,  # noqa: E402
                       _blueprint_to_adapter)

HIDDEN = 64
# one leaf per mechanism family (steering / storage / recurrent)
FAMILIES = ["direft_gelusq", "pkmem_open_k8", "gated_delta_cap",
            "gated_delta_fast", "gated_delta_wmfast"]


def _make(name):
    torch.manual_seed(0)
    a = LEAF_REGISTRY[name](HIDDEN, low_rank_dim=8, layer_idx=0,
                            device="cpu", dtype=torch.float32)
    a.eval()
    return a


def _delta(a, x):
    # v3 T/R contract: leaves define readout (R); the serving-path
    # correction is R(x) - x.
    with torch.no_grad():
        return a.readout(x) - x


@pytest.fixture(scope="module")
def x():
    torch.manual_seed(1234)
    return torch.randn(2, 6, HIDDEN, dtype=torch.float32)


@pytest.mark.parametrize("name", FAMILIES)
def test_blueprint_round_trip_bitwise(name, x):
    a = _make(name)
    b = _blueprint_to_adapter(_adapter_to_blueprint(a))
    b.eval()
    sd_a, sd_b = a.state_dict(), b.state_dict()
    assert set(sd_a) == set(sd_b)
    for k in sd_a:
        assert torch.equal(sd_a[k], sd_b[k]), f"{name}.{k} not bitwise"
    assert torch.equal(_delta(a, x), _delta(b, x)), name


@pytest.mark.parametrize("name", FAMILIES)
def test_zero_init_contract(name, x):
    """Fresh leaves must be exact no-ops (delta == 0), so loading an
    untrained adapter can never perturb the base model."""
    d = _delta(_make(name), x)
    assert torch.count_nonzero(d) == 0, name


@pytest.mark.parametrize("name", FAMILIES)
def test_refresh_reflects_new_weights_bitwise(name, x):
    """The RL refresh path is load_state_dict on the serving instance:
    after a simulated optimizer step on the training module, syncing
    its state_dict must reproduce the training delta exactly."""
    train = _make(name)
    serve = _blueprint_to_adapter(_adapter_to_blueprint(train))
    serve.eval()
    with torch.no_grad():
        for p in train.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    serve.load_state_dict(
        {k: v.clone() for k, v in train.state_dict().items()},
        strict=False)
    if hasattr(serve, "refresh_inference_caches"):
        serve.refresh_inference_caches()
    d_train, d_serve = _delta(train, x), _delta(serve, x)
    assert torch.count_nonzero(d_train) > 0, f"{name}: probe inert"
    assert torch.equal(d_train, d_serve), name


@pytest.mark.parametrize("name", FAMILIES)
def test_determinism_two_calls(name, x):
    """Eval-mode leaves must be run-to-run deterministic (train-time
    stochasticity like key_jitter must be gated on .training)."""
    a = _make(name)
    with torch.no_grad():
        for p in a.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    assert torch.equal(_delta(a, x), _delta(a, x)), name


def test_sequence_mixing_flag_survives_round_trip():
    """The recurrent leaf's sequence_mixing flag drives eager-mode +
    per-request segmentation at serving; losing it in the blueprint
    would silently mix state across requests."""
    a = _make("gated_delta_cap")
    assert getattr(a, "sequence_mixing", False)
    b = _blueprint_to_adapter(_adapter_to_blueprint(a))
    assert getattr(b, "sequence_mixing", False)
