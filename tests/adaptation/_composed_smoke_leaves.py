# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Leaves for the COMPOSED-member GPU smoke, as an importable module.

``spec_to_adapter_config`` blueprints an adapter by ``__module__`` /
``__qualname__`` and the worker re-imports it by that path, so these
cannot live in the smoke script (whose module is ``__main__``).

Two shapes, standing in for the two kinds of record the composed member
of ``campaign/cartridge_readout/longhealth_raw`` sub-lane B carries
alongside its pipe:

  * :class:`CountingScan` — a SEQUENCE-MIXING leaf (the `stack_k1` /
    delta-rule `ext` shape).  Mounted INSIDE the re-executed span, which
    is the whole reason the loop is there: the span is what re-develops
    its scan.  It counts its readouts into :data:`FIRINGS`, so the smoke
    can assert firings == 1 + passes on a real engine and not only on a
    CPU mock.
  * :class:`CountingReader` — a stateless per-token head (the
    `direft_gelusq` shape), mounted at EVERY layer, so it is also at the
    span's INPUT port: the co-mount that used to be refused.

Both are exact no-ops at their init (``scale = 0``), which is what makes
the smoke's identity check a claim about the WHOLE composed member rather
than about the pipe alone.

Neither carries engine state.  That is deliberate and it is the finding:
the fork declares ``CARRIES_ADAPTER_DECODE_STATE = False``, so a member
whose trained function read a carried ``State`` is refused by name, and
the composed member's P axis is expressed with phase masks plus the
span's per-pass KV instead.
"""

from collections import defaultdict
from typing import Optional

import torch
import torch.nn as nn

#: label -> readout calls.  A plain module-level dict: the smoke runs the
#: EngineCore IN-PROCESS (``VLLM_ENABLE_V1_MULTIPROCESSING=0``), so the
#: worker's leaves and the script share this interpreter and therefore
#: this dict.  Reset it with :func:`reset_firings` between engines.
FIRINGS: dict = defaultdict(int)


def reset_firings() -> None:
    FIRINGS.clear()


class CountingScan(nn.Module):
    """Sequence-mixing readout, zero-init, counted.

    ``sequence_mixing = True`` is the declaration the engine reads
    (``protocol.needs_sequence_segmentation``) to run the member per
    request span rather than across the flattened batch.  The serving
    requirements are declared too, so the engine is forced to unchunked
    prefill and eager — a chunked prefill would reset the scan mid-prompt
    and still read fluently.
    """

    sequence_mixing = True
    serving_requires_eager = True
    serving_requires_unchunked_prefill = True

    def __init__(self, hidden_size: int = 2048, scale: float = 0.0,
                 label: str = "scan", dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.label = label
        self.scale = nn.Parameter(
            torch.full((), float(scale), dtype=dtype or torch.float32))

    def readout(self, fx, state=None, x=None):
        FIRINGS[self.label] += 1
        n = fx.shape[-2]
        running = fx.cumsum(dim=-2) / torch.arange(
            1, n + 1, device=fx.device, dtype=fx.dtype).unsqueeze(-1)
        return fx + self.scale.to(fx.dtype) * running


class CountingReader(nn.Module):
    """Stateless per-token head, zero-init, counted."""

    def __init__(self, hidden_size: int = 2048, scale: float = 0.0,
                 label: str = "reader", dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.label = label
        self.scale = nn.Parameter(
            torch.full((), float(scale), dtype=dtype or torch.float32))

    def readout(self, fx, state=None, x=None):
        FIRINGS[self.label] += 1
        return fx + self.scale.to(fx.dtype) * torch.tanh(fx)


def member_specs(hidden_size: int, span_start: int, span_end: int,
                 passes: int, n_layers: int, *, gate: float = 0.0,
                 scan_scale: float = 0.0, reader_scale: float = 0.0,
                 scan_layers=(0,), scan_phase: str = "all",
                 reader_layers=None):
    """The composed member as a LIST of adapter specs, pipe LAST.

    The order is load-bearing: the fork blends a layer's members in
    insertion order and runs the span after every same-port blend
    (``protocol.SPAN_COMOUNT_ORDER == "diagonal_then_span"``), so the pipe
    has to be the last ``load_adapter`` — which is also the HF engine's
    record order, and what ``adapters/serving.py::_rewired_last``
    enforces.
    """
    from tests.adaptation._inout_smoke_leaf import GatedPipe
    if reader_layers is None:
        reader_layers = tuple(range(n_layers))
    specs = []
    readers = {li: CountingReader(hidden_size, reader_scale,
                                  f"reader.L{li}") for li in reader_layers}
    specs.append({"layer_indices": sorted(readers), "position": "all",
                  "sample_adapter": next(iter(readers.values())),
                  "adapters": readers, "site": "block_output",
                  "label": "R.reader"})
    for n, li in enumerate(scan_layers):
        leaf = CountingScan(hidden_size, scan_scale, f"scan{n}.L{li}")
        specs.append({"layer_indices": [li], "position": scan_phase,
                      "sample_adapter": leaf, "adapters": {li: leaf},
                      "site": "block_output", "label": f"T.scan{n}"})
    pipe = GatedPipe(hidden_size=hidden_size, loop_start=span_start,
                     loop_end=span_end, loop_passes=passes, gated=True,
                     gate_init=gate)
    specs.append({"layer_indices": [span_end], "position": "all",
                  "sample_adapter": pipe, "adapters": {span_end: pipe},
                  "site": "block_output", "output_site": "block_input",
                  "output_layer": span_start, "passes": passes,
                  "label": "F.loop+W.gate"})
    return specs


def span_declaration(span_start: int, span_end: int, passes: int) -> dict:
    """``EngineArgs.adapter_recirc_span``, as
    ``adapters/serving.py::recirc_span_declaration`` builds it."""
    return {"site": "block_output", "layer_indices": [span_end],
            "output_site": "block_input", "output_layer": span_start,
            "passes": passes, "declared_by": ["F.loop+W.gate"]}
