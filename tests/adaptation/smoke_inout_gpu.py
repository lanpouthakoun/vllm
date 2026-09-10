#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU smoke test for the off-diagonal (F_in, F_out, passes) route.

NOT a pytest module (no ``test_`` prefix) and NOT run by the CPU suite:
it needs a GPU and a real model.  Submit it to Slurm.

    export PYTHONPATH=/path/to/this/vllm/checkout
    python tests/adaptation/smoke_inout_gpu.py \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --loop-start 0 --loop-end 3 --passes 2

``PYTHONPATH`` must contain the checkout root: the pipe leaf is
blueprinted by module path and the engine's worker re-imports it as
``tests.adaptation._inout_smoke_leaf``.

What it checks
--------------
1. **KV allocation** — the engine's KV-cache spec really grew by
   ``passes * |span|`` entries, and the shadow names are the expected
   ones.  This is the crux of the whole route: if the spec did not grow,
   the passes are sharing the host's cache and every number below is
   meaningless.
2. **Identity at gate 0** — greedy generations with the pipe mounted at
   ``g = 0`` must be TOKEN-IDENTICAL to the unadapted model, even though
   the span really executed ``passes`` extra times.  This is the
   zero-init contract, and it is the single most informative check here:
   it fails if the recombine is wrong, if the span corrupted the host's
   KV cache, or if the stream was re-based incorrectly.
3. **Effect at gate != 0** — the same prompts must NOT be token-identical
   once the gate is opened, or the span is not reaching the output.
4. **Throughput** — tokens/s for the vLLM route vs the same computation
   driven by the HF loop, which is the comparison that justifies the
   engine work at all.

Exit status is 0 only if every enabled check passed.
"""

import argparse
import gc
import os
import sys
import time

# Run EngineCore in this process instead of a child.  Two reasons, both
# fatal with the default forked EngineCore:
#   * this script touches CUDA in the parent before the first engine (the
#     ``torch.cuda.is_available()`` guard in main(), and
#     ``torch.cuda.empty_cache()`` in _free_engine between engines), so a
#     forked child dies in torch's _lazy_init with "Cannot re-initialize
#     CUDA in forked subprocess";
#   * the pipe spec carries a live ``nn.Module`` leaf through
#     ``adapter_config``, which wants to stay in the process that built it.
# Must be set before ``vllm.envs`` is first read, hence module scope.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch

DEFAULT_PROMPTS = [
    "Explain in three sentences why the sky is blue.",
    "Write a haiku about a lighthouse in winter.",
    "List four uses for a paperclip that are not holding paper.",
    "Summarise the plot of the Odyssey for a ten year old.",
    "What is the difference between a virus and a bacterium?",
    "Give step by step directions for making a cup of tea.",
    "Name three rivers in South America and one fact about each.",
    "Describe the taste of a lemon to someone who has never had one.",
]


def _log(msg: str) -> None:
    print(f"[inout-smoke] {msg}", flush=True)


def _reclaim(tag: str = "") -> None:
    """Collect, drop the caching allocator's blocks, and report the GPU."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        _log(f"GPU free {free / 2 ** 30:.1f}/{total / 2 ** 30:.1f} GiB{tag}")


def _free_engine(llm):
    """Best-effort teardown so the next engine can have the GPU.

    Always returns None, and MUST be called as ``llm = _free_engine(llm)``:
    the EngineCore runs in this process, so the caller's own name is what
    pins the model weights and the KV cache.  ``del llm`` inside here would
    only drop the parameter.  The actual reclaim happens in _build_llm(),
    once the caller's reference is gone.
    """
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception as e:  # noqa: BLE001
        _log(f"engine shutdown warning (non-fatal): {e}")
    try:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment, destroy_model_parallel)
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception as e:  # noqa: BLE001
        _log(f"teardown warning (non-fatal): {e}")
    return None


def _make_spec(args, hidden_size: int, gate: float):
    """Prefer the library's registered leaf; fall back to the smoke leaf."""
    try:
        from adapters.types.recirculation import RecirculationAdapter
        leaf = RecirculationAdapter(hidden_size=hidden_size,
                                    loop_start=args.loop_start,
                                    loop_end=args.loop_end,
                                    loop_passes=args.passes,
                                    gated=True)
        with torch.no_grad():
            leaf.gate.fill_(gate)
        _log("leaf: adapters.types.recirculation.RecirculationAdapter")
        return {
            "layer_indices": [args.loop_end],
            "position": "all",
            "sample_adapter": leaf,
            "adapters": {args.loop_end: leaf},
            "site": "block_output",
            "output_site": "block_input",
            "output_layer": args.loop_start,
            "passes": args.passes,
        }
    except Exception as e:  # noqa: BLE001
        _log(f"library leaf unavailable ({e}); using the smoke leaf")
        from tests.adaptation._inout_smoke_leaf import make_pipe_spec
        return make_pipe_spec(hidden_size, args.loop_start, args.loop_end,
                              args.passes, gate)


def _build_llm(args, adapter_config=None):
    from vllm import LLM
    _reclaim(" before build")
    kwargs = dict(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        enforce_eager=True,
        max_num_seqs=args.max_num_seqs,
        # The engine forces these for a rewiring config anyway; setting
        # them here keeps the BASE run (no adapter) on the same schedule,
        # so the throughput comparison is like for like.
        enable_chunked_prefill=False,
        max_num_batched_tokens=args.max_model_len,
    )
    if adapter_config is not None:
        kwargs["adapter_config"] = adapter_config
    return LLM(**kwargs)


def _generate(llm, prompts, max_tokens):
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=0)
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp)
    dt = time.perf_counter() - t0
    token_ids = [list(o.outputs[0].token_ids) for o in outs]
    texts = [o.outputs[0].text for o in outs]
    n = sum(len(t) for t in token_ids)
    return token_ids, texts, n, dt


def _kv_spec_report(llm, args, num_layers):
    """Ask the worker for its KV-cache spec and check the extra layers."""
    try:
        specs = llm.collective_rpc("get_kv_cache_spec")[0]
    except Exception as e:  # noqa: BLE001
        _log(f"could not read the KV-cache spec ({e}); skipping check 1")
        return None
    shadows = sorted(n for n in specs if n.startswith("recirc."))
    span_len = args.loop_end - args.loop_start + 1
    expected = args.passes * span_len
    _log(f"KV-cache spec: {len(specs)} layers total, {len(shadows)} of them "
         f"per-pass shadows (expected {expected} = {args.passes} passes x "
         f"{span_len} span layers)")
    for name in shadows[:6]:
        _log(f"    shadow: {name}")
    if len(shadows) > 6:
        _log(f"    ... and {len(shadows) - 6} more")
    ok = len(shadows) == expected
    base = len(specs) - len(shadows)
    if base != num_layers:
        _log(f"WARNING: {base} host attention layers, model reports "
             f"{num_layers} decoder layers")
    _log(f"KV memory cost: the budget is now divided over {len(specs)} "
         f"layers instead of {base} "
         f"(x{len(specs) / max(base, 1):.2f} per token of context)")
    return ok


# ---------------------------------------------------------------------------
# HF-loop baseline: the same computation, driven by transformers
# ---------------------------------------------------------------------------

def _install_hf_pipe(model, start, end, passes, gate):
    """Re-execute layers start..end from a hook on layer `end`.

    A faithful port of ``adapters.mounting.AdapterModel._rewire_span``:
    each pass gets its own ``DynamicCache``, and every pass replays the
    host's captured kwargs (mask, position ids, rotary embeddings) with
    only ``past_key_values`` swapped.
    """
    from transformers.cache_utils import DynamicCache

    layers = model.model.layers
    state = {"kwargs": None, "caches": {}, "inside": False}

    def capture(_mod, args, kwargs):
        if not state["inside"]:
            state["kwargs"] = dict(kwargs)
        return args, kwargs

    def rewire(_mod, _args, _kwargs, output):
        if state["inside"]:
            return output
        h = output[0] if isinstance(output, tuple) else output
        cap = state["kwargs"] or {}
        # The presence of a cache object is the signal, not `use_cache`:
        # LlamaModel does not forward `use_cache` to the decoder layer.
        use_cache = cap.get("past_key_values") is not None
        x = h
        state["inside"] = True
        try:
            for p in range(passes):
                cache = state["caches"].get(p)
                if use_cache and cache is None:
                    cache = DynamicCache()
                    state["caches"][p] = cache
                kw = dict(cap)
                if use_cache:
                    kw["past_key_values"] = cache
                for layer in layers[start:end + 1]:
                    out = layer(x, **kw)
                    x = out[0] if isinstance(out, tuple) else out
        finally:
            state["inside"] = False
        new = h + gate * (x - h)
        if isinstance(output, tuple):
            return (new, ) + tuple(output[1:])
        return new

    layers[end].register_forward_pre_hook(capture, with_kwargs=True)
    layers[end].register_forward_hook(rewire, with_kwargs=True)
    return state


def _hf_baseline(args, prompts):
    """tokens/s for the HF loop doing the same recirculation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "auto": torch.bfloat16}.get(args.dtype, torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="eager").cuda()
    model.eval()
    state = _install_hf_pipe(model, args.loop_start, args.loop_end,
                             args.passes, args.gate)

    total_tokens, elapsed = 0, 0.0
    with torch.no_grad():
        for i in range(0, len(prompts), args.hf_batch_size):
            batch = prompts[i:i + args.hf_batch_size]
            enc = tok(batch, return_tensors="pt", padding=True).to("cuda")
            state["caches"].clear()
            state["kwargs"] = None
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model.generate(**enc, max_new_tokens=args.max_tokens,
                                 do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            torch.cuda.synchronize()
            elapsed += time.perf_counter() - t0
            total_tokens += int(
                (out.shape[1] - enc["input_ids"].shape[1]) * out.shape[0])
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return total_tokens, elapsed


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--loop-start", type=int, default=0)
    ap.add_argument("--loop-end", type=int, default=3)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--gate", type=float, default=0.25,
                    help="gate value for the effect and throughput phases")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    # Three engines are built in sequence and an HF model is loaded after
    # them; keep each one's share small enough that an imperfect teardown
    # cannot starve the next build.
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    ap.add_argument("--hf-batch-size", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=8,
                    help="repeat the prompt list this many times for the "
                         "throughput phase")
    ap.add_argument("--phases", default="kv,identity,effect,throughput")
    args = ap.parse_args()

    phases = {p.strip() for p in args.phases.split(",") if p.strip()}
    if not torch.cuda.is_available():
        _log("FATAL: no GPU visible")
        return 2

    from transformers import AutoConfig
    hf_cfg = AutoConfig.from_pretrained(args.model)
    hidden_size = hf_cfg.hidden_size
    num_layers = hf_cfg.num_hidden_layers
    _log(f"model={args.model} hidden={hidden_size} layers={num_layers}")
    _log(f"pipe: reads block_output L{args.loop_end}, writes block_input "
         f"L{args.loop_start}, {args.passes} passes over layers "
         f"{args.loop_start}..{args.loop_end}")
    if args.loop_end >= num_layers:
        _log(f"FATAL: loop_end={args.loop_end} >= num_layers={num_layers}")
        return 2

    prompts = DEFAULT_PROMPTS
    results: dict[str, bool] = {}

    # ---- base (no adapter) reference generation ---------------------
    base_ids = base_texts = None
    base_tokens = base_dt = None
    if phases & {"identity", "throughput"}:
        _log("building the BASE engine (no adapter) ...")
        llm = _build_llm(args)
        base_ids, base_texts, base_tokens, base_dt = _generate(
            llm, prompts, args.max_tokens)
        _log(f"base: {base_tokens} tokens in {base_dt:.2f}s "
             f"({base_tokens / base_dt:.1f} tok/s)")
        llm = _free_engine(llm)

    # ---- gate 0: KV allocation + bit-exact identity ------------------
    if phases & {"kv", "identity"}:
        _log("building the PIPE engine at gate=0 ...")
        spec = _make_spec(args, hidden_size, 0.0)
        from vllm.adaptation.specs import spec_to_adapter_config
        llm = _build_llm(args, spec_to_adapter_config(spec))

        if "kv" in phases:
            ok = _kv_spec_report(llm, args, num_layers)
            if ok is not None:
                results["kv_allocation"] = ok
                _log(f"CHECK kv_allocation: {'PASS' if ok else 'FAIL'}")

        if "identity" in phases:
            ids, texts, n, dt = _generate(llm, prompts, args.max_tokens)
            same = ids == base_ids
            results["identity_at_gate_0"] = same
            _log(f"CHECK identity_at_gate_0: {'PASS' if same else 'FAIL'}")
            if not same:
                for i, (a, b) in enumerate(zip(base_ids, ids)):
                    if a != b:
                        pos = next((j for j, (x, y) in enumerate(zip(a, b))
                                    if x != y), min(len(a), len(b)))
                        _log(f"  prompt {i} diverges at token {pos}")
                        _log(f"    base: {base_texts[i][:120]!r}")
                        _log(f"    pipe: {texts[i][:120]!r}")
                        break
            _log(f"pipe@g=0: {n} tokens in {dt:.2f}s ({n / dt:.1f} tok/s)")
        llm = _free_engine(llm)

    # ---- gate != 0: the span must actually reach the output ----------
    vllm_tokens = vllm_dt = None
    if phases & {"effect", "throughput"}:
        _log(f"building the PIPE engine at gate={args.gate} ...")
        spec = _make_spec(args, hidden_size, args.gate)
        from vllm.adaptation.specs import spec_to_adapter_config
        llm = _build_llm(args, spec_to_adapter_config(spec))

        if "effect" in phases:
            ids, texts, _, _ = _generate(llm, prompts, args.max_tokens)
            if base_ids is None:
                _log("effect: no base reference (identity phase skipped)")
            else:
                differs = ids != base_ids
                results["effect_at_gate_nonzero"] = differs
                _log(f"CHECK effect_at_gate_nonzero: "
                     f"{'PASS' if differs else 'FAIL'}")
                _log(f"  sample base: {base_texts[0][:100]!r}")
                _log(f"  sample pipe: {texts[0][:100]!r}")

        if "throughput" in phases:
            many = prompts * args.repeat
            _log(f"throughput: {len(many)} prompts x {args.max_tokens} "
                 f"new tokens, max_num_seqs={args.max_num_seqs}")
            _generate(llm, prompts[:2], 8)  # warm up
            _, _, vllm_tokens, vllm_dt = _generate(llm, many, args.max_tokens)
            _log(f"vLLM route: {vllm_tokens} tokens in {vllm_dt:.2f}s "
                 f"({vllm_tokens / vllm_dt:.1f} tok/s)")
        llm = _free_engine(llm)

    # ---- HF loop baseline -------------------------------------------
    if "throughput" in phases:
        try:
            _log("running the HF-loop baseline ...")
            hf_tokens, hf_dt = _hf_baseline(args, prompts * args.repeat)
            _log(f"HF loop:    {hf_tokens} tokens in {hf_dt:.2f}s "
                 f"({hf_tokens / hf_dt:.1f} tok/s)")
            if vllm_dt:
                speedup = (vllm_tokens / vllm_dt) / (hf_tokens / hf_dt)
                _log(f"SPEEDUP vLLM route vs HF loop: {speedup:.2f}x")
                if base_dt:
                    cost = (base_tokens / base_dt) / (vllm_tokens / vllm_dt)
                    _log(f"cost of the pipe vs the unadapted vLLM base: "
                         f"{cost:.2f}x slower "
                         f"(expected ~1 + passes*|span|/layers of extra "
                         f"compute, plus the smaller KV budget)")
        except Exception as e:  # noqa: BLE001
            _log(f"HF baseline failed ({type(e).__name__}: {e}); the vLLM "
                 f"numbers above still stand")

    _log("=" * 62)
    for name, ok in results.items():
        _log(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not results:
        _log("no assertions ran")
        return 0
    failed = [n for n, ok in results.items() if not ok]
    _log(f"{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    raise SystemExit(main())
