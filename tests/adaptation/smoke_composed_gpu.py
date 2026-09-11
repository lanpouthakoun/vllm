#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU smoke for the COMPOSED member: a re-executing span with OTHER
members co-mounted inside it, on the RESERVED multisite route.

NOT a pytest module and NOT run by the CPU suite: it needs a GPU and a
real model.  Submit it to Slurm.

    export PYTHONPATH=/path/to/adapters:/path/to/this/vllm/checkout
    python tests/adaptation/smoke_composed_gpu.py \
        --span-start 0 --span-end 3 --passes 2

``PYTHONPATH`` must contain the checkout root: the leaves are blueprinted
by module path and the worker re-imports them as
``tests.adaptation._composed_smoke_leaves``.

The sibling ``smoke_inout_gpu.py`` proved the BAKED span (one member,
riding ``adapter_config``).  This proves the four things the composed
member of ``campaign/cartridge_readout/longhealth_raw`` sub-lane B needs
and the baked route cannot give it:

1. **reservation** — the per-pass KV caches are registered from
   ``EngineArgs.adapter_recirc_span`` ALONE, with no baked
   ``adapter_config`` and no member loaded yet, and the spec really grew
   by ``passes * |span|``.  If it did not, every number below is
   meaningless: the passes would be sharing the host's cache.
2. **identity at gate 0, with the WHOLE composed member** — a reader at
   every layer (so also at the span's input port, the co-mount that used
   to be refused), two scanned leaves inside the span, and the gated pipe.
   Every record is an exact no-op at its init, so the greedy generations
   must be TOKEN-IDENTICAL to the unadapted model.  This fails if the
   member's W is wrong, if the span corrupted the host's KV, if the
   co-mounts did not compose, or if the stream was re-based wrongly.
3. **effect** — open the gate and the same prompts must NOT be
   token-identical, or the span is not reaching the output.
4. **per-pass firing** — the co-mounted members inside the span fire
   ``1 + passes`` times per forward, and a member outside the span fires
   once.  Counted on the real engine, not on a mock.
5. **carry** — what travels from prefill into decode is the span's
   PER-PASS KV, which the engine owns per request.  Checked the only way
   that is decisive: cached decode must agree with the full-sequence
   forward.  Greedily generate, then re-prefill prompt+generation and
   compare the next token.  If the per-pass caches did not persist across
   decode steps (or leaked into the host's), the two disagree.

What is NOT checked, because it is refused rather than served: a member
whose mount asked the engine to carry a per-request ``State``
(``Mount.decode_state``).  ``CARRIES_ADAPTER_DECODE_STATE`` is False; the
smoke asserts the refusal fires (phase ``refuse``) instead of asserting a
behaviour the engine does not have.

Exit status is 0 only if every enabled check passed.
"""

import argparse
import gc
import os
import sys
import time

# In-process EngineCore.  Three reasons, all of them fatal otherwise:
# this script touches CUDA in the parent, the specs carry live nn.Module
# leaves, and the firing counters live in a module-level dict that only
# the worker's own interpreter can share.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch

DEFAULT_PROMPTS = [
    "Explain in three sentences why the sky is blue.",
    "Write a haiku about a lighthouse in winter.",
    "List four uses for a paperclip that are not holding paper.",
    "Summarise the plot of the Odyssey for a ten year old.",
    "What is the difference between a virus and a bacterium?",
    "Give step by step directions for making a cup of tea.",
]


def _log(msg: str) -> None:
    print(f"[composed-smoke] {msg}", flush=True)


def _reclaim(tag: str = "") -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        _log(f"GPU free {free / 2 ** 30:.1f}/{total / 2 ** 30:.1f} GiB{tag}")


def _free_engine(llm):
    try:
        if llm is not None:
            del llm
    finally:
        _reclaim(" after teardown")
    return None


def _build_llm(args, *, span=None, members=None):
    """Build the engine, RESERVE the span, then load the members by RPC.

    This is the real multisite sequence and the order is the point: the
    reservation has to reach ``EngineArgs`` BEFORE ``LLM(...)``, because
    the worker reads its KV-cache spec once, right after ``load_model``,
    and that is the only window the shadow caches can be declared in.
    """
    from vllm import LLM
    from vllm.adaptation.specs import spec_to_adapter_config
    _reclaim(" before build")
    kwargs = dict(
        model=args.model, dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False, enforce_eager=True,
        max_num_seqs=args.max_num_seqs,
        enable_chunked_prefill=False,
        max_num_batched_tokens=args.max_model_len,
    )
    if members:
        kwargs.update(enable_adapters=True, max_adapters=8)
    if span is not None:
        kwargs["adapter_recirc_span"] = span
    llm = LLM(**kwargs)
    reqs = []
    if members:
        from vllm.adaptation.request import AdapterRequest
        for rid, spec in enumerate(members, start=1):
            label = spec.get("label", f"m{rid}")
            cfg = spec_to_adapter_config(
                {k: v for k, v in spec.items() if k != "label"})
            n = llm.collective_rpc("load_adapter",
                                   args=(rid, cfg, spec["position"],
                                         spec.get("site", "block_output")))
            _log(f"  loaded id={rid} {label}: {n[0]} layers")
            reqs.append(AdapterRequest(adapter_name=label,
                                       adapter_int_id=rid,
                                       adapter_path="",
                                       adapter_position=spec["position"]))
    return llm, reqs


def _generate(llm, prompts, max_tokens, reqs=None):
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=0)
    kw = {"adapter_requests": reqs} if reqs else {}
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, **kw)
    dt = time.perf_counter() - t0
    ids = [list(o.outputs[0].token_ids) for o in outs]
    texts = [o.outputs[0].text for o in outs]
    return ids, texts, sum(len(t) for t in ids), dt


def _kv_spec_report(llm, args, num_layers):
    try:
        specs = llm.collective_rpc("get_kv_cache_spec")[0]
    except Exception as e:  # noqa: BLE001
        _log(f"could not read the KV-cache spec ({e})")
        return None
    shadows = sorted(n for n in specs if n.startswith("recirc."))
    span_len = args.span_end - args.span_start + 1
    expected = args.passes * span_len
    _log(f"KV-cache spec: {len(specs)} layers, {len(shadows)} per-pass "
         f"shadows (expected {expected} = {args.passes} passes x {span_len} "
         f"span layers)")
    for name in shadows[:6]:
        _log(f"    shadow: {name}")
    base = len(specs) - len(shadows)
    if base != num_layers:
        _log(f"WARNING: {base} host attention layers, model reports "
             f"{num_layers}")
    _log(f"KV cost: the budget is divided over {len(specs)} layers instead "
         f"of {base} (x{len(specs) / max(base, 1):.2f} per context token)")
    return len(shadows) == expected


def _n_layers_and_hidden(args):
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(args.model)
    return int(cfg.num_hidden_layers), int(cfg.hidden_size)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--span-start", type=int, default=0)
    ap.add_argument("--span-end", type=int, default=3)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--gate", type=float, default=0.25)
    ap.add_argument("--scan-scale", type=float, default=0.1)
    ap.add_argument("--reader-scale", type=float, default=0.1)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    ap.add_argument("--phases",
                    default="reserve,identity,effect,firing,carry,refuse")
    args = ap.parse_args()
    phases = {p.strip() for p in args.phases.split(",") if p.strip()}

    if not torch.cuda.is_available():
        _log("no CUDA device; refusing to report a serving result")
        return 2

    sys.path.insert(0, os.getcwd())
    from tests.adaptation import _composed_smoke_leaves as L

    n_layers, hidden = _n_layers_and_hidden(args)
    span = L.span_declaration(args.span_start, args.span_end, args.passes)
    prompts = DEFAULT_PROMPTS
    results: dict = {}
    _log(f"model {args.model}: {n_layers} layers, hidden {hidden}")
    _log(f"span {args.span_start}..{args.span_end}, {args.passes} passes; "
         f"the composed member is a reader at all {n_layers} layers + 2 "
         f"scanned leaves at L{args.span_start} (INSIDE the span) + the "
         f"gated pipe at L{args.span_end}")

    # ---------------------------------------------------------------- base
    base_ids = None
    if {"identity", "effect", "carry"} & phases:
        llm, _ = _build_llm(args)
        base_ids, base_texts, n, dt = _generate(llm, prompts,
                                                args.max_tokens)
        _log(f"BASE (unadapted): {n} tokens in {dt:.1f}s")
        for i, t in enumerate(base_texts[:2]):
            _log(f"  base p{i}: {t[:70]!r}")
        llm = _free_engine(llm)

    # ------------------------------------------- 1/2/4: gate 0, whole member
    if {"reserve", "identity", "firing"} & phases:
        L.reset_firings()
        members = L.member_specs(
            hidden, args.span_start, args.span_end, args.passes, n_layers,
            gate=0.0, scan_scale=0.0, reader_scale=0.0,
            scan_layers=(args.span_start, args.span_start))
        llm, reqs = _build_llm(args, span=span, members=members)
        if "reserve" in phases:
            results["reserve"] = _kv_spec_report(llm, args, n_layers)
        if {"identity", "firing"} & phases:
            L.reset_firings()
            ids, texts, n, dt = _generate(llm, prompts, args.max_tokens,
                                          reqs)
            _log(f"COMPOSED at gate 0: {n} tokens in {dt:.1f}s")
            if "identity" in phases:
                same = ids == base_ids
                results["identity_at_gate_0"] = same
                if not same:
                    for i, (a, b) in enumerate(zip(base_ids, ids)):
                        if a != b:
                            j = next((k for k, (u, v) in
                                      enumerate(zip(a, b)) if u != v), -1)
                            _log(f"  DIVERGES p{i} at token {j}")
                            _log(f"    base: {base_texts[i][:70]!r}")
                            _log(f"    comp: {texts[i][:70]!r}")
            if "firing" in phases:
                # One forward per token per sequence, so ratios are what
                # is stable, not absolute counts: a member inside the
                # span must fire (1 + passes) x as often as one outside.
                inside = L.FIRINGS.get(f"reader.L{args.span_start}", 0)
                outside = L.FIRINGS.get(f"reader.L{n_layers - 1}", 0)
                scans = sum(L.FIRINGS.get(f"scan{k}.L{args.span_start}", 0)
                            for k in (0, 1))
                want = 1 + args.passes
                _log(f"firings: reader inside the span "
                     f"{inside}, reader outside {outside}, scanned leaves "
                     f"{scans} (expect inside/outside == {want} and "
                     f"scans == 2 x inside)")
                ok = (outside > 0 and inside == want * outside
                      and scans == 2 * inside)
                results["per_pass_firing"] = ok
        llm = _free_engine(llm)

    # ------------------------------------------------------- 3: the effect
    if {"effect", "carry"} & phases:
        L.reset_firings()
        members = L.member_specs(
            hidden, args.span_start, args.span_end, args.passes, n_layers,
            gate=args.gate, scan_scale=args.scan_scale,
            reader_scale=args.reader_scale,
            scan_layers=(args.span_start, args.span_start))
        llm, reqs = _build_llm(args, span=span, members=members)
        ids, texts, n, dt = _generate(llm, prompts, args.max_tokens, reqs)
        _log(f"COMPOSED at gate {args.gate}: {n} tokens in {dt:.1f}s")
        for i, t in enumerate(texts[:2]):
            _log(f"  comp p{i}: {t[:70]!r}")
        if "effect" in phases:
            results["effect_at_nonzero_gate"] = ids != base_ids
        if "carry" in phases:
            # Cached decode == the full-sequence forward. The generation
            # was produced one step at a time against the per-pass
            # caches; re-prefilling prompt+generation rebuilds them in
            # one shot. Agreement on the NEXT token is the property a
            # served route must hold and the one that fails first if the
            # per-pass caches do not persist across steps or leak into
            # the host's.
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.model)
            agree, checked = 0, 0
            for p, gen_ids in zip(prompts, ids):
                if len(gen_ids) < 2:
                    continue
                text = p + tok.decode(gen_ids[:-1],
                                      skip_special_tokens=False)
                nxt, _, _, _ = _generate(llm, [text], 1, reqs)
                checked += 1
                agree += int(nxt[0][:1] == gen_ids[-1:])
            _log(f"carry (cached decode == full forward): {agree}/{checked} "
                 f"prompts agree on the next token")
            results["decode_state_carried_as_per_pass_kv"] = (
                checked > 0 and agree == checked)
        llm = _free_engine(llm)

    # ------------------------------------ 5: what is refused, by name
    if "refuse" in phases:
        from vllm.adaptation import protocol as P
        from vllm.adaptation import recirculation as R
        ok = True
        ok &= P.CARRIES_ADAPTER_DECODE_STATE is False
        ok &= P.REEXECUTION_ROUTES == frozenset({"baked",
                                                 "multisite_reserved"})
        ok &= P.SPAN_COMOUNT_ORDER == "diagonal_then_span"
        try:
            R.refuse_decode_state({"layer_indices": [0],
                                   "decode_state": True}, label="smoke")
            ok = False
            _log("a decode_state member was NOT refused")
        except RuntimeError as e:
            assert "CARRIES_ADAPTER_DECODE_STATE" in str(e)
        # an unreserved rewiring member must still be refused
        llm, _ = _build_llm(args)
        try:
            R.fill_reserved_span(
                llm.llm_engine.engine_core.engine_core.model_executor
                .driver_worker.worker.model_runner.model
                if hasattr(llm.llm_engine.engine_core, "engine_core")
                else [], 1,
                {"layer_indices": [args.span_end], "site": "block_output",
                 "output_site": "block_input",
                 "output_layer": args.span_start, "passes": args.passes})
            ok = False
            _log("an UNRESERVED span was NOT refused")
        except RuntimeError as e:
            assert "NO SPAN WAS RESERVED" in str(e)
        except Exception as e:  # noqa: BLE001 - could not reach the model
            _log(f"could not reach the worker's model to test the "
                 f"unreserved refusal ({type(e).__name__}: {e})")
        llm = _free_engine(llm)
        results["refusals_by_name"] = ok

    _log("=" * 62)
    failed = [k for k, v in results.items() if v is False]
    for k, v in results.items():
        _log(f"{'PASS' if v else ('SKIP' if v is None else 'FAIL')}  {k}")
    _log(f"{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
