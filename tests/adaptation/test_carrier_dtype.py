# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The member's CARRIER DTYPE: what the stream is INSIDE the member.

``campaign/serving/parity_headline`` (job 312676) put the S5 ``ext``
member — an extended-range DeltaNet leaf at ``block_output`` L0 — through
``adapters.serving.build_vllm``.  It was admitted, it exported
bit-for-bit, it took the baked route, and then it died at the serve
stage with::

    expected mat1 and mat2 to have the same dtype, but got:
    float != c10::BFloat16

The member's carrier dtype is fp32 and the host's is bf16.  On the HF
side that is the whole of ``Fp32Stream``
(``campaign/state_tracking_transfer/dichotomy/members.py``): cast the
stream up at the port, run the leaf, cast the payload back.  The fork
had no name for that, so ``_prepare_adapter`` cast the member's linears
to the MODEL's dtype and ``apply_adaptation`` handed the leaf the raw
bf16 stream — and the leaf's own ``fx.float()`` then met bf16 weights.

These tests pin the fix as a numeric claim, on CPU:

  * an fp32 member inside a bf16 stream computes what an explicit
    ``Fp32Stream``-style reference computes, within bf16 tolerance;
  * a bf16 member inside a bf16 stream is BIT-EXACT to running it
    directly — the carrier axis costs nothing where it does not apply;
  * the port casts are exactly two: what the member's ``readout`` sees
    is the carrier, and what ``write`` sees is the stream;
  * the carrier is DECLARED by the record first, by the class second,
    by the parameters third, and REFUSED BY NAME when it is none of
    those.
"""

import copy

import pytest
import torch
import torch.nn as nn

from vllm.adaptation import protocol as P
from vllm.adaptation.layer import _prepare_adapter

HIDDEN = 16
STREAM_DTYPE = torch.bfloat16


# ---------------------------------------------------------------------------
# The member under test: the S5 `ext` leaf's SHAPE, at a tiny width.
# ---------------------------------------------------------------------------

class _Fp32PortLeaf(nn.Module):
    """A leaf that casts its own input to fp32 and fuses back at the end.

    This is exactly the shape of the campaign's ``RecurrentLeaf``::

        def readout(self, fx, state=None, x=None):
            delta = self.up(self.post_core(self.core(self.down(fx.float()))))
            return (fx.float() + delta).to(fx.dtype)

    with the mixer replaced by a plain nonlinearity, so the test depends
    on the dtype discipline and not on any particular operator.  Its
    parameters are fp32 and it declares nothing: the carrier has to be
    derived from the weights, which is the case the fault was in.
    """

    def __init__(self, hidden=HIDDEN, width=8, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.hidden_size = hidden
        self.down = nn.Linear(hidden, width, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.norm = nn.LayerNorm(width)
        with torch.no_grad():
            self.up.weight.normal_(0, 0.05)

    def readout(self, fx, state=None, x=None):
        h = self.down(fx.float())
        delta = self.up(torch.tanh(self.norm(h)))
        return (fx.float() + delta).to(fx.dtype)


class _Bf16Leaf(nn.Module):
    """The same computation with NO internal casting, in bf16 — a member
    whose carrier IS the stream, i.e. every member that served before
    this axis existed."""

    def __init__(self, hidden=HIDDEN, width=8, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.hidden_size = hidden
        self.down = nn.Linear(hidden, width, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.norm = nn.LayerNorm(width)
        with torch.no_grad():
            self.up.weight.normal_(0, 0.05)
        self.to(STREAM_DTYPE)

    def readout(self, fx, state=None, x=None):
        return fx + self.up(torch.tanh(self.norm(self.down(fx))))


class _DeclaredBf16CarrierLeaf(_Fp32PortLeaf):
    """fp32 PARAMETERS, bf16 PORT — the faithful native members' shape.

    Their mixer math is certified fp32 and their parameters are pinned
    to it, but the stream they read and write is the host's bf16, which
    they cast themselves at the fusion.  Deriving the carrier from the
    parameters would give the wrong answer, so the class declares it."""

    serving_carrier_dtype = "bfloat16"


class _MixedDtypeLeaf(nn.Module):
    """Half fp32, half bf16, declaring nothing: there is no dtype the
    stream can be cast to at this member's port."""

    def __init__(self, hidden=HIDDEN, width=8):
        super().__init__()
        self.hidden_size = hidden
        self.down = nn.Linear(hidden, width, bias=False).to(torch.float32)
        self.up = nn.Linear(width, hidden, bias=False).to(torch.bfloat16)

    def readout(self, fx, state=None, x=None):
        return fx


class _NoParameterLeaf(nn.Module):
    """A member holding no floating-point parameter at all: it has no
    carrier of its own and runs in the stream's dtype, casting nothing."""

    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.hidden_size = hidden

    def readout(self, fx, state=None, x=None):
        return fx * 2


# ---------------------------------------------------------------------------
# The reference: Fp32Stream, written out, with nothing else in it.
# ---------------------------------------------------------------------------

class _Fp32StreamReference(nn.Module):
    """``campaign/state_tracking_transfer/dichotomy/members.py``'s
    ``Fp32Stream``, transcribed: fp32 compute, fuse back at the port."""

    def __init__(self, leaf):
        super().__init__()
        self.leaf = leaf
        self.hidden_size = leaf.hidden_size

    def readout(self, fx, state=None, x=None):
        return self.leaf.readout(fx.float(), state, x).to(fx.dtype)


def _stream(n=6, seed=11):
    torch.manual_seed(seed)
    return torch.randn(n, HIDDEN).to(STREAM_DTYPE)


def _served(leaf, **kw):
    """The member as the engine mounts it (both routes go through here)."""
    return _prepare_adapter(leaf, torch.device("cpu"), STREAM_DTYPE, **kw)


# ---------------------------------------------------------------------------
# 1. The fault, as a numeric claim
# ---------------------------------------------------------------------------

class TestFp32MemberInABf16Stream:

    def test_it_serves_at_all(self):
        """The regression itself: before the carrier axis this raised
        ``expected mat1 and mat2 to have the same dtype, but got:
        float != c10::BFloat16`` from the member's first matmul."""
        served = _served(_Fp32PortLeaf())
        out = P.apply_adaptation(served, _stream(), None)
        assert out.dtype is STREAM_DTYPE
        assert torch.isfinite(out.float()).all()

    def test_it_matches_the_fp32stream_reference(self):
        """Within bf16 tolerance of the HF path's own expression."""
        leaf = _Fp32PortLeaf()
        reference = _Fp32StreamReference(copy.deepcopy(leaf))
        h = _stream()

        served = P.apply_adaptation(_served(leaf), h, None)
        with torch.no_grad():
            expected = reference.readout(h.unsqueeze(0)).squeeze(0)

        assert served.dtype is expected.dtype is STREAM_DTYPE
        assert torch.allclose(served.float(), expected.float(), atol=1e-3,
                              rtol=1e-3), (served - expected).abs().max()

    def test_the_weights_are_not_touched(self):
        """No silent up- or down-cast anywhere but the port: the member
        keeps exactly the tensors the checkpoint had."""
        leaf = _Fp32PortLeaf()
        served = _served(leaf)
        for (name, before), (_, after) in zip(leaf.named_parameters(),
                                              served.named_parameters()):
            assert after.dtype is before.dtype is torch.float32, name
            assert torch.equal(after, before), name

    def test_the_port_casts_are_exactly_two(self):
        """R sees the CARRIER; W sees the STREAM, on both arguments."""
        leaf = _Fp32PortLeaf()
        seen = {}
        real_readout = leaf.readout

        def spy_readout(fx, state=None, x=None):
            seen["readout_in"] = fx.dtype
            out = real_readout(fx, state, x)
            seen["readout_out"] = out.dtype
            return out

        def spy_write(h_out, payload):
            seen["write_h_out"] = h_out.dtype
            seen["write_payload"] = payload.dtype
            return payload

        leaf.readout = spy_readout
        leaf.write = spy_write
        P.apply_adaptation(_served(leaf), _stream(), None)

        assert seen["readout_in"] is torch.float32
        assert seen["write_h_out"] is STREAM_DTYPE
        assert seen["write_payload"] is STREAM_DTYPE


# ---------------------------------------------------------------------------
# 2. Where the axis does not apply, it costs nothing
# ---------------------------------------------------------------------------

class TestCarrierEqualsStream:

    def test_a_bf16_member_is_bit_exact(self):
        leaf = _Bf16Leaf()
        h = _stream()
        served = P.apply_adaptation(_served(leaf), h, None)
        with torch.no_grad():
            direct = leaf.readout(h.unsqueeze(0)).squeeze(0)
        assert served.dtype is direct.dtype is STREAM_DTYPE
        assert torch.equal(served, direct)

    def test_a_bf16_member_is_bit_exact_under_a_mask(self):
        leaf = _Bf16Leaf()
        h = _stream()
        mask = torch.tensor([1., 0., 1., 1., 0., 1.])
        served = P.apply_adaptation(_served(leaf), h, mask)
        with torch.no_grad():
            direct = leaf.readout(h.unsqueeze(0)).squeeze(0)
        blended = h + (direct - h) * mask.unsqueeze(-1).to(direct.dtype)
        assert torch.equal(served, blended)

    def test_a_member_with_no_parameters_is_left_alone(self):
        served = _served(_NoParameterLeaf())
        assert P.carrier_dtype_of(served) is None
        h = _stream()
        assert torch.equal(P.apply_adaptation(served, h, None), h * 2)


# ---------------------------------------------------------------------------
# 3. Declared beats derived, and unknown is refused
# ---------------------------------------------------------------------------

class TestResolution:

    def test_parameters_answer_when_nothing_declares(self):
        assert P.resolve_carrier_dtype(_Fp32PortLeaf()) is torch.float32
        assert P.resolve_carrier_dtype(_Bf16Leaf()) is torch.bfloat16

    def test_the_class_beats_the_parameters(self):
        """The faithful native members' case: fp32 weights, bf16 port."""
        leaf = _DeclaredBf16CarrierLeaf()
        assert next(leaf.parameters()).dtype is torch.float32
        assert P.resolve_carrier_dtype(leaf) is torch.bfloat16

    def test_the_record_beats_the_class(self):
        """``adapter_config``'s ``carrier_dtype`` is written from the
        checkpoint the weights came from, so it wins."""
        leaf = _DeclaredBf16CarrierLeaf()
        assert P.resolve_carrier_dtype(
            leaf, declared="float32") is torch.float32
        assert P.carrier_dtype_of(
            _served(leaf, carrier_dtype="float32")) is torch.float32

    def test_a_declared_carrier_is_honoured_at_the_port(self):
        """A member that declares bf16 is handed the stream as it
        stands, even though its own parameters are fp32."""
        leaf = _DeclaredBf16CarrierLeaf()
        seen = {}
        real = leaf.readout

        def spy(fx, state=None, x=None):
            seen["in"] = fx.dtype
            return real(fx, state, x)

        leaf.readout = spy
        P.apply_adaptation(_served(leaf), _stream(), None)
        assert seen["in"] is STREAM_DTYPE

    def test_an_unestablishable_carrier_is_refused_by_name(self):
        with pytest.raises(P.CarrierDtypeUnknown) as ei:
            _served(_MixedDtypeLeaf(), label="m0:the_mixed_member")
        text = str(ei.value)
        assert "REFUSING to serve m0:the_mixed_member" in text
        assert "CARRIER DTYPE" in text
        assert "float32" in text and "bfloat16" in text
        assert P.CARRIER_DTYPE_ATTR in text

    def test_the_refusal_happens_at_load_not_in_a_forward(self):
        """A member is refused when it is mounted, where a human is
        looking — never mid-generation."""
        with pytest.raises(P.CarrierDtypeUnknown):
            _served(_MixedDtypeLeaf())

    def test_the_resolution_is_stamped_once(self):
        served = _served(_Fp32PortLeaf())
        assert getattr(served, P.CARRIER_DTYPE_STAMP) is torch.float32


# ---------------------------------------------------------------------------
# 4. Composites: one member, one port, each in its own right
# ---------------------------------------------------------------------------

class _Composite(nn.Module):

    def __init__(self):
        super().__init__()
        self.hidden_size = HIDDEN
        self.members = nn.ModuleList([_Fp32PortLeaf(), _Bf16Leaf()])

    def readout(self, fx, state=None, x=None):
        return fx


class TestComposites:

    def test_each_member_keeps_its_own_carrier(self):
        served = _prepare_adapter(_Composite(), torch.device("cpu"),
                                  STREAM_DTYPE)
        assert P.carrier_dtype_of(served.members[0]) is torch.float32
        assert P.carrier_dtype_of(served.members[1]) is torch.bfloat16

    def test_a_composite_member_serves_through_the_port(self):
        composite = _Composite()
        served = _prepare_adapter(composite, torch.device("cpu"),
                                  STREAM_DTYPE)
        h = _stream()
        out = P.apply_adaptation(served.members[0], h, None)
        reference = _Fp32StreamReference(copy.deepcopy(composite.members[0]))
        with torch.no_grad():
            expected = reference.readout(h.unsqueeze(0)).squeeze(0)
        assert torch.allclose(out.float(), expected.float(), atol=1e-3,
                              rtol=1e-3)
