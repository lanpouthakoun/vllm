# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spawn-mode spec-file transport must deserialize identically to the
fork-mode ``VllmConfig.adapter_config`` path.

The temp-file path (``set_adapter_spec`` -> ``_write_spec_file`` ->
spawned worker ``get_adapter_spec`` -> ``_read_spec_file``) is the
transport for TRL training hooks.  Both writer and reader now delegate
to ``spec_to_adapter_config`` / ``adapter_config_to_spec``, so what a
spawned worker reconstructs is exactly what a fork-mode worker gets.

Regression coverage for two spawn-only breakages:
  * a ``SharedViewBlueprint`` sample was not recognised by the old
    reader (only ``AdapterBlueprint``), so spawn workers got a raw dict;
  * per-layer shared-view adapters were saved as bare ``state_dict()``
    (dropping the non-persistent ``layer_emb`` buffer) and rebuilt via
    ``deepcopy(sample)``, forking the shared core and stamping every
    layer with the sample's embedding.
"""

import os

import pytest
import torch

pytest.importorskip(
    "adapters.registry",
    reason="requires the adapters library (shared-contract branch)")
from adapters.registry import LEAF_REGISTRY  # noqa: E402
from adapters.serving import _SharedServingView  # noqa: E402

from vllm.adaptation.specs import (_SPEC_FILE_ENV_KEY,  # noqa: E402
                                   _read_spec_file, _write_spec_file,
                                   adapter_config_to_spec,
                                   spec_to_adapter_config)

HIDDEN = 32


@pytest.fixture(autouse=True)
def _clean_spec_file():
    yield
    path = os.environ.pop(_SPEC_FILE_ENV_KEY, None)
    if path and os.path.exists(path):
        os.unlink(path)


def _leaf():
    torch.manual_seed(0)
    a = LEAF_REGISTRY["direft_gelusq"](HIDDEN, low_rank_dim=4, layer_idx=0,
                                       device="cpu", dtype=torch.float32)
    a.eval()
    return a


def _assert_same_state(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    assert sorted(sa) == sorted(sb)
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k


class TestSpecFileParity:

    def test_plain_leaf_round_trip_matches_fork_path(self):
        spec = {"layer_indices": [0, 1], "position": "prefill",
                "sample_adapter": _leaf()}
        fork_side = adapter_config_to_spec(spec_to_adapter_config(spec))

        _write_spec_file(spec)
        spawn_side = _read_spec_file()

        assert spawn_side is not None
        assert spawn_side["layer_indices"] == [0, 1]
        assert spawn_side["position"] == "prefill"
        assert type(spawn_side["sample_adapter"]) is type(
            fork_side["sample_adapter"])
        _assert_same_state(spawn_side["sample_adapter"],
                           fork_side["sample_adapter"])

    def test_shared_view_round_trip_keeps_per_layer_embeddings(self):
        core = _leaf()
        views = {}
        for idx in (0, 1):
            emb = torch.full((HIDDEN, ), 0.5 * (idx + 1))
            views[idx] = _SharedServingView(core, idx, emb)
        spec = {"layer_indices": [0, 1], "position": "prefill",
                "sample_adapter": views[0], "adapters": views}

        _write_spec_file(spec)
        spawn_side = _read_spec_file()

        assert spawn_side is not None
        sample = spawn_side["sample_adapter"]
        # The old reader only recognised AdapterBlueprint and returned
        # the SharedViewBlueprint as a raw dict.
        assert not isinstance(sample, dict)
        rebuilt = spawn_side["adapters"]
        for idx in (0, 1):
            assert torch.equal(rebuilt[idx].layer_emb, views[idx].layer_emb), \
                f"layer {idx} embedding lost in spawn transport"
        # One shared core, not per-layer forks.
        assert rebuilt[0].core is rebuilt[1].core

    def test_read_returns_none_without_file(self):
        os.environ.pop(_SPEC_FILE_ENV_KEY, None)
        assert _read_spec_file() is None
