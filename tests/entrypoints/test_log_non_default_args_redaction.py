# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The "non-default args" log record must never contain a raw tensor payload.

Adapter blueprints reach the engine through ``EngineArgs.adapter_config`` as a
serialisable dict whose ``state_dict`` entries carry raw tensor bytes
(``{"__adapter_t": True, "data": b"..."}``). Python renders bytes as a ``\\xNN``
escape repr — four characters per byte — so logging that dict verbatim wrote
~620 MB per job for a 56 M-parameter readout, and 3.7 GB across one nine-cell
wave, onto a filesystem at 98 % use.

These tests pin the redaction and, just as importantly, pin that nothing else
about the record changes: the adapter is still fully identified by the log line.
"""
import hashlib

from vllm.entrypoints.utils import _loggable, _MAX_LOGGED_BYTES


def _blueprint(payload: bytes) -> dict:
    """The shape a real adapter_config has at the log site."""
    return {
        "gpu_memory_utilization": 0.5,
        "enforce_eager": True,
        "adapter_config": {
            "layer_indices": list(range(28)),
            "position": "all",
            "sample_adapter": {
                "__type__": "AdapterBlueprint",
                "__module__": "adapters.types.direft_dynamic_gelusq",
                "kwargs": {"hidden_size": 3072, "low_rank_dim": 128},
                "state_dict": {
                    "head_U": {"__adapter_t": True, "shape": [3072, 128],
                               "data": payload},
                    "head_V": {"__adapter_t": True, "shape": [128, 3072],
                               "data": payload},
                },
            },
        },
    }


def test_large_payload_is_replaced_by_length_and_digest():
    payload = bytes(range(256)) * 4096          # 1 MiB, well over the cap
    out = _loggable(_blueprint(payload))
    sd = out["adapter_config"]["sample_adapter"]["state_dict"]

    for name in ("head_U", "head_V"):
        got = sd[name]["data"]
        assert isinstance(got, str), f"{name} payload was not redacted"
        assert got == (f"<bytes len={len(payload)} "
                       f"sha256={hashlib.sha256(payload).hexdigest()[:16]}>")


def test_rendered_record_is_small_and_holds_no_escape_dump():
    payload = bytes(range(256)) * 4096          # 1 MiB
    raw_len = len(str(_blueprint(payload)))
    red_len = len(str(_loggable(_blueprint(payload))))

    # the unredacted record is enormous: >= 4 chars per byte, twice over
    assert raw_len > 4_000_000, raw_len
    # the redacted one is a log line, not a file
    assert red_len < 2_000, red_len
    assert "\\x" not in str(_loggable(_blueprint(payload)))


def test_everything_that_identifies_the_adapter_survives():
    out = _loggable(_blueprint(b"\x00" * 4096))

    assert out["gpu_memory_utilization"] == 0.5
    assert out["enforce_eager"] is True
    ac = out["adapter_config"]
    assert ac["layer_indices"] == list(range(28))
    assert ac["position"] == "all"
    sa = ac["sample_adapter"]
    assert sa["__module__"] == "adapters.types.direft_dynamic_gelusq"
    assert sa["kwargs"] == {"hidden_size": 3072, "low_rank_dim": 128}
    # shape is the thing you actually diagnose from, and it is untouched
    assert sa["state_dict"]["head_U"]["shape"] == [3072, 128]
    assert sa["state_dict"]["head_U"]["__adapter_t"] is True


def test_small_payloads_are_left_exactly_alone():
    small = b"\x01\x02\x03"
    assert _loggable({"data": small})["data"] == small
    edge = b"\xab" * _MAX_LOGGED_BYTES
    assert _loggable({"data": edge})["data"] == edge
    assert isinstance(_loggable({"data": edge + b"\x00"})["data"], str)


def test_payloads_nested_in_lists_and_tuples_are_caught():
    payload = b"\xcd" * 8192
    out = _loggable({"a": [{"data": payload}], "b": ({"data": payload},)})
    assert isinstance(out["a"][0]["data"], str)
    assert isinstance(out["b"][0]["data"], str)
    assert isinstance(out["b"], tuple)


def test_non_bytes_values_and_odd_types_pass_through_unchanged():
    sentinel = object()
    out = _loggable({"x": sentinel, "y": None, "z": 3, "s": "text" * 10_000})
    assert out["x"] is sentinel
    assert out["y"] is None
    assert out["z"] == 3
    assert out["s"] == "text" * 10_000      # only BYTES are redacted


def test_deep_recursion_terminates():
    node: dict = {"data": b"\xff" * 8192}
    for _ in range(50):
        node = {"n": node}
    _loggable(node)          # must not raise RecursionError
