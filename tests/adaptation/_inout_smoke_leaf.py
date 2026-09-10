# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The recirculation pipe leaf, as an importable module.

``spec_to_adapter_config`` blueprints an adapter by ``__module__`` /
``__qualname__`` and the worker re-imports it by that path, so the leaf
used by the GPU smoke script cannot live in the script itself (its
module would be ``__main__``).  It lives here.

This is a SMOKE-TEST leaf, deliberately not registered anywhere.  The
real one is ``adapters.types.recirculation.RecirculationAdapter`` in the
adapters library, which owns the member zoo; the smoke script prefers it
when the installed library has it and falls back to this.  Both satisfy
the same tiny engine contract:

    readout(fx)              -> the payload handed INTO the span (identity)
    write(h_out, payload)    -> the stream handed on after it (W_phi)
"""

from typing import Optional

import torch
import torch.nn as nn


class GatedPipe(nn.Module):
    """Identity transition, identity readout, gated W_phi.

    ``out = fx + g*(piped - fx)``.  At ``g == 0`` this is ``fx`` bit for
    bit — the zero-init contract — while ``dout/dg|_0 = piped - fx`` is
    non-zero, so the member starts at an exact no-op without starting at
    a saddle.

    ``loop_start`` / ``loop_end`` / ``loop_passes`` are carried as ctor
    kwargs so the blueprint round-trip reproduces them; the ENGINE reads
    the span from the mount's ``output_layer`` / ``layer_indices`` /
    ``passes``, not from the leaf.
    """

    # adapters/serving.py reads these off the leaf (_REQ_ATTRS).
    serving_requires_eager = True
    serving_requires_unchunked_prefill = True

    def __init__(self,
                 hidden_size: int = 2048,
                 loop_start: int = 0,
                 loop_end: int = 3,
                 loop_passes: int = 2,
                 gated: bool = True,
                 gate_init: float = 0.0,
                 dtype: Optional[torch.dtype] = None):
        super().__init__()
        if loop_start < 0 or loop_end <= loop_start:
            raise ValueError(
                f"need 0 <= loop_start < loop_end, got "
                f"{loop_start}..{loop_end}")
        if loop_passes < 1:
            raise ValueError(f"loop_passes must be >= 1, got {loop_passes}")
        self.hidden_size = hidden_size
        self.loop_start = loop_start
        self.loop_end = loop_end
        self.loop_passes = loop_passes
        self.gated = gated
        self.gate_init = gate_init
        self.gate = nn.Parameter(
            torch.full((), float(gate_init),
                       dtype=dtype or torch.float32))

    def readout(self, fx, state=None, x=None):
        """R is the identity: the pipe hands the stream to the span
        unchanged.  The engine supplies the re-execution."""
        return fx

    def write(self, h_out, payload):
        """W_phi at the write port, after the span has been re-executed.

        The gate is the member's WRITE, not an engine callback: the
        engine calls ``write(h_out, payload)`` at F_out and nowhere else
        (docs/recirculation-serving.md §1.1)."""
        if not self.gated:
            return payload
        return h_out + self.gate.to(h_out.dtype) * (payload - h_out)


def make_pipe_spec(hidden_size: int, loop_start: int, loop_end: int,
                   passes: int, gate: float, gated: bool = True,
                   position: str = "all") -> dict:
    """A baked adapter_spec for the pipe, shaped like serving.py's."""
    leaf = GatedPipe(hidden_size=hidden_size, loop_start=loop_start,
                     loop_end=loop_end, loop_passes=passes, gated=gated,
                     gate_init=gate)
    return {
        "layer_indices": [loop_end],
        "position": position,
        "sample_adapter": leaf,
        "adapters": {loop_end: leaf},
        "site": "block_output",
        "output_site": "block_input",
        "output_layer": loop_start,
        "passes": passes,
    }
