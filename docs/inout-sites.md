# Off-diagonal mount sites: `F = (F_in, F_out)` with host re-execution

This fork's adaptation layer mounts a member at a **site**: a named port
on a decoder layer's residual stream. Until now the site axis was a
single port — the member read the stream somewhere and wrote it back at
the *same* place. The axis is now a **pair**: an input port and an output
port. When the output port **precedes** the input port in the host's
forward order, the mount stops being a computation at a point and becomes
a **schedule rewiring**: the engine re-executes the enclosed decoder
layers.

Everything below describes the fork/serving side. The library side
(`adapters.mounting.Mount`, `adapters.sites`,
`adapters.types.recirculation`) owns the vocabulary and the training
implementation; this document is about what the engine does with it.

---

## 1. The contract

### Manifest / `adapter_config` fields

Three keys, mirroring `adapters.mounting.Mount`'s serialization exactly:

| key | type | default | meaning |
|---|---|---|---|
| `site` | `str` | `"block_output"` | the **input** port: where the member reads |
| `output_site` | `str \| None` | `None` → same as `site` | the **output** port: where its result is written |
| `output_layer` | `int \| None` | `None` → the mount's own layer | which layer that output port belongs to |
| `passes` | `int` | `1` | how many times the engine re-executes the span |

`layer_indices` continues to name the layer(s) the member is mounted on;
for a rewiring member it must be exactly one layer, the input layer `j`.

**A mount rewires iff `output_site` or `output_layer` is non-`None`.**
This mirrors `Mount.rewires()`: naming the *same kind* of port at an
*earlier layer* is still off-diagonal. `passes` alone does not rewire —
a diagonal member may loop internally without touching the host schedule
(that is `iterate_k` / `piped_iterate`, a different mechanism).

### Byte-compatibility with every existing manifest

`adapters.mounting.save_members` writes the three keys **only when the
mount actually rewires**, so a diagonal member's manifest record is the
same four keys it always was (`site`, `phase`, `layers`, `shared`) and is
byte-identical to what it was before the pair axis existed. On the fork
side their absence reconstructs the diagonal and the member takes exactly
the code path it took before. `output_site=None` written explicitly
(which `adapters/serving.py` does emit on the v2 path) is also read as
diagonal.

This is pinned by tests, not asserted: `tests/adaptation/test_inout_sites.py::TestManifestRoundTrip`
checks that a diagonal config carries none of the pair keys, that
explicit `None`s stay diagonal, and
`TestDiagonalRegression` checks that a plain `block_output` member's
output is unchanged with the new code path present. The whole existing
suite still passes: `tests/adaptation` is **310 passed / 24 skipped**
with the pair axis present, from 237 / 24 before it existed.

### Semantics

With input port `(j, block_output)` and output port `(i, out_site)`:

```
start  = i         if out_site == "block_input"    # resume AT layer i
       = i + 1     if out_site == "block_output"   # resume after layer i
span   = layers[start .. j]                        # inclusive both ends
piped  = (L_j ∘ … ∘ L_start)^k ( R(h_j) )
h_j'   = write(h_j, piped)
```

and the host continues at layer `j+1` with `h_j'`. `R` is the member's
`readout` (the identity for a pipe) and `write` is its own W_phi
(`adapters/_base.py`; this was a private `recombine` callback until
2026-09-10 and a leaf still carrying that hook is now refused):

- default `write` → `piped` (the bare pipe replaces the stream);
- `InterpolateWrite` → `fx + g·(piped − fx)`, which is **`fx` bit for bit
  at `g = 0`** while `dout/dg|₀ = piped − fx ≠ 0`. An exact no-op at init
  that is not a saddle.

The span's total execution count per token is therefore `1 + k`: the
host's own pass plus `k` re-executions. Members mounted *inside* the span
fire on every re-executed pass; the rewiring member itself is a
pass-through inside its own span (a re-entrancy depth guard), or the
recursion would not terminate.

**Positions are not shifted.** Every pass is handed the host's
`positions` tensor unchanged, so rotary phase and the causal mask are
identical to the host's own pass. The recirculated stream is a different
*value* at the same *position*, not a longer sequence.

### Phase masks

The span always runs on the whole batch — it is a schedule, not a
per-token computation — and the member's ordinary combined phase ∧
membership mask then selects which tokens keep the recirculated value:
`h + mask·(write(h, piped) − h)`. This reuses the fork's existing
mask machinery unchanged.

---

## 2. The KV-cache scheme, and why it is the crux

Each re-executed pass visits the span's attention layers at the **same
token positions** as the host's own pass. If the passes shared one cache,
pass 2's queries would attend to pass 1's keys at the same slots. So
**each pass needs its own KV cache for every layer in the span.**

The obstacle is timing. vLLM v1 collects the KV-cache spec **once**, at
engine start, by walking `compilation_config.static_forward_context` for
`Attention` modules (`gpu_model_runner.get_kv_cache_spec`). After that
the tensors are allocated and the spec is frozen.

So the extra caches are declared as extra **entries in that dict**, in
the only window that exists: `install_recirculation()` runs at the end of
`gpu_model_runner.load_model()`, immediately before the worker asks for
the spec. For every span layer and every pass it registers a **shadow
`Attention`** — a `copy.copy` of the real one, sharing `impl`,
parameters, buffers and KV-scales (so it yields a byte-identical
`KVCacheSpec` and identical numerics), with two fresh plain attributes:
its own `layer_name` and its own `kv_cache` slot.

Everything downstream then follows from vLLM's own machinery, with no
further engine changes:

- **budgeting** — `get_kv_cache_spec` emits one `FullAttentionSpec` per
  shadow, so the memory profiler divides the KV budget over
  `L + k·|span|` layers instead of `L`. The extra caches are *budgeted*,
  not stolen from the host's.
- **block table sharing** — a shadow's spec is identical to its
  original's, so it lands in the same KV-cache group. Attention metadata
  is built per group and fanned out per layer name
  (`attn_metadata[layer_name] = attn_metadata_i`), so a shadow gets the
  same `block_table` and the same `slot_mapping` as its original. Pass
  `p` therefore writes token `t`'s K/V **at the same slot in its own
  tensor**, and a later decode step reading that block table sees
  exactly pass `p`'s history. This is what makes cached decode agree
  with a full-prefix forward.
- **binding** — `bind_kv_cache` binds each shadow's tensor by layer name.

At pass time only `layer_name` and `kv_cache` are swapped on the *real*
modules, which is enough for both attention call paths: the
`use_direct_call` path reads the swapped `self.kv_cache` and
`attn_metadata[self.layer_name]`, and the custom-op path passes
`self.layer_name` to `torch.ops.vllm.unified_attention*`, which
re-resolves the module out of `forward_context.no_compile_layers` — the
same dict the shadows live in — and so reads the shadow's cache. Both are
restored in a `finally`, so a mid-span exception cannot leave the host's
layers pointing at a pass cache.

### Shadow layer names

```
recirc.pass{p}.layers.{virtual_idx}.self_attn.attn{n}
```

Two constraints, both load-bearing and both tested:

1. `extract_layer_index` (used by `bind_kv_cache`) asserts the name
   contains **exactly one integer**. `pass{p}` and the suffix contribute
   none.
2. That integer must not collide with a real decoder layer's, or
   `bind_kv_cache` maps two names to one index and silently drops a cache
   from the runner's list. `virtual_idx` is allocated **above** the
   host's layer count for exactly that reason.

### Memory and compute cost

Let `L` = decoder layers, `S = |span| = j − start + 1`, `k = passes`.

- **KV memory per token of context**: `× (L + k·S) / L`.
- **Decoder FLOPs per token**: `× (L + k·S) / L`.
- **Max concurrency / max context** at a fixed `gpu_memory_utilization`:
  `× L / (L + k·S)`, because the same KV budget is divided over more
  layers.

Worked example — Llama-3.2-1B (`L = 16`, 8 KV heads, head dim 64, bf16),
span `0..3` (`S = 4`), `k = 2`:

| | base | with the pipe |
|---|---|---|
| cache-bearing layers | 16 | 24 |
| KV per token | 32 KiB | 48 KiB (× 1.5) |
| decoder layer executions per token | 16 | 24 (× 1.5) |
| relative max concurrency | 1.00 | 0.67 |

The engine logs the ratio at install time, so the cost is visible in the
run's own log rather than inferred.

---

## 3. Why this is faster than the HF loop

The recirculation itself is the same arithmetic on both sides — the
speedup is not in the pipe, it is in everything around it, and it is the
same reason the fork exists for diagonal members:

1. **Paged attention over a shared block table.** The per-pass caches are
   paged blocks drawn from one pool, allocated and freed by the same
   block manager as the host's. The HF path holds `k` `DynamicCache`
   objects that `torch.cat` on every decode step — an O(T²) reallocation
   over a generation of length T, on top of being dense per sequence
   rather than paged across the batch.
2. **Batched continuous-batching decode.** vLLM decodes many sequences in
   one flattened step, so the span's `k·S` extra layer executions are
   *batched* across every running request: the extra work per step is
   `k·S` GEMMs over the whole batch, not per sequence. The HF loop pays
   `k·S` layer calls per sequence per token, with a Python-level loop
   between them, and a left-padded batch pays for padding in every pass.
3. **One scheduler, one cache.** Because the shadows join the host's
   KV-cache group, the pipe adds no scheduling, no separate allocator and
   no host↔device round trips per pass; the only per-pass Python work is
   swapping two attributes per attention module.

The expected shape of the result, therefore: the pipe costs about
`(L + k·S)/L` against the *unadapted vLLM base* (1.5× in the example
above), and that penalty is applied to a throughput that is already an
order of magnitude above the HF loop's — so the route is far faster than
the HF path while being meaningfully slower than the unadapted engine.

**The argument above is a mechanism argument.** The measurement is
`tests/adaptation/smoke_inout_gpu.py`, which reports vLLM tok/s, HF-loop
tok/s, their ratio, and the pipe's cost against the unadapted base on
the **same** workload with both engines warmed up.

Measured, Llama-3.2-1B-Instruct on one H100-80GB, span `0..3`, `k = 2`,
64 prompts × 64 new tokens, `max_num_seqs=64`, `gpu_memory_utilization
0.25`, eager, no prefix caching (Slurm job 310307):

| | tok/s |
|---|---|
| unadapted vLLM base | 9187 |
| the pipe (gate 0.25) | 6454 |
| HF loop driving the same recirculation | 456 |

So the pipe costs **1.42×** against the unadapted engine — against a
predicted 1.50× from `(L + k·S)/L = (16 + 2·4)/16` — and is **14.2×**
faster than the HF loop. The shape §3 predicted is what the run shows:
a modest constant factor against the engine's own baseline, on top of a
baseline an order of magnitude above the HF path.

---

## 4. What is implemented

- The pair axis on the baked route: `site` / `output_site` /
  `output_layer` / `passes` through `spec_to_adapter_config` →
  `VllmConfig.adapter_config` → `adapter_config_to_spec`, round-trip
  stable, byte-compatible for diagonal members.
- Static validation at engine construction
  (`plan_from_adapter_config`) using the **library's** port order — the
  fork keeps no copy.
- Per-pass KV registration for the span's attention layers, one extra set
  per pass, declared before the spec is frozen.
- Span re-execution in the input layer's own forward, `passes` times,
  with the member's `readout` on the way in and its own `write` (W_phi)
  on the way out, for **prefill and decode alike** — the per-pass caches
  persist across steps exactly as the host's do.
- **A capability surface for W_phi.**
  `vllm.adaptation.protocol.SUPPORTED_WRITE_LABELS` (re-exported as
  `vllm.adaptation.recirculation.SUPPORTED_WRITE_LABELS`) names the
  writes this engine applies — `replace` and `interpolate` — in the
  library's own vocabulary (`adapters._base.write_label_of`). The
  library's serving builder probes it and refuses an unserved W *by
  name* at checkpoint-load time (`adapters/serving.py::
  refuse_custom_write`) instead of refusing every non-default W. It is a
  whitelist on purpose: `apply_write` would duck-type any leaf's
  `write`, but `add` and `norm_mix` are the FORWARD and CARRY pipes'
  writes and neither route exists here, so a payload assembled the way
  this engine assembles it is not the payload they expect. A route that
  grows adds its label in the same commit.
- Both residual contracts: llama-style
  `(positions, hidden, residual) -> (hidden, residual)` and the
  residual-free olmo2-style `(positions, hidden) -> hidden`. The host's
  extra per-layer kwargs are replayed verbatim on every pass.
- Re-entrancy safety: depth guard, and `layer_name`/`kv_cache` restored on
  exception.
- **The span gets a private copy of the stream.** A real decoder layer
  keeps no promise about the tensor it is handed: llama's aliases it as
  `residual` when called with `residual=None` (which is how each pass
  starts) and every `RMSNorm(x, residual)` after that is
  `ops.fused_add_rms_norm`, which writes the running residual sum back
  into that very tensor. The stream handed to the span is the host
  layer's own `h_full` — both the write's `h_out` and the value the layer
  re-bases its deferred residual on — so `_run_span_once` copies before
  the first layer touches it. Without the copy the gate-0 pipe silently
  replaced the host's block output with the span's first intermediate:
  bit-exact identity on an out-of-place mock, wrong tokens on a GPU.
  Pinned by `TestGateZeroIsANoOpOnADestructiveStack`, which runs the
  identity assertion over **both** an out-of-place and an in-place
  decoder mock, across prefill and explicit decode steps.
- Engine-config policy for a rewiring config, reusing the guards the fork
  already had for sequence-mixing members: `enforce_eager=True`,
  `enable_chunked_prefill=False` with `max_num_batched_tokens ≥
  max_model_len`, and `enable_prefix_caching=False`.

Tests: 73 CPU tests in `tests/adaptation/test_inout_sites.py`, including
cached-decode-equals-full-forward for prefill ∈ {1, 3, 6} × passes ∈
{1, 2} **with a negative control** that collapsing the per-pass caches
onto one name breaks it — so the assertion tests the separation and not
the mock — and gate-0 identity across prefill + decode steps on both
residual contracts, with a negative control that the in-place mock
really does clobber the tensor it is handed.

---

## 5. What is refused, and why

Each refusal names its reason at the point of failure. None of them
degrades silently.

| refused | reason |
|---|---|
| **multisite route** | Its members load through `collective_rpc("load_adapter")` **after** the KV-cache spec is frozen and the tensors allocated, so the per-pass caches can no longer be created. Re-executing without them would make every pass overwrite the host's own K/V at the same slots and corrupt the generation silently. |
| **lora_view route** | Same timing problem, and a LoRA view has no port pair at all — it is a weight-space member. |
| **chunked prefill** | A second prefill chunk re-enters the span with only that chunk's queries while the per-pass caches hold the earlier chunks' keys, so pass `p`'s chunk boundary sees a history assembled from a different pass's recirculated stream than the one that produced its queries. Forced off at engine-config time. |
| **CUDA graphs / compilation** | The span loop is Python control flow that re-enters the decoder stack, and it swaps each attention module's `layer_name` between passes — exactly the control flow graph capture and the general-shape compile strip. `enforce_eager=True` is forced. |
| **prefix caching** | A prefix-cache block is keyed by a hash of the token prefix alone, which does not distinguish the host's pass from the `k` extra ones that share the block table. Forced off. |
| **pipeline parallelism > 1** | The span is re-executed from inside layer `j`'s forward, which can only reach layers resident on the same rank; a span crossing a stage boundary would need the re-entry to travel back over the PP send/recv path. Tensor parallelism is fine — every rank holds every layer. |
| **KV-transfer connectors** | The connector is driven by layer name (`wait_for_kv_layer_from_connector` / `maybe_save_kv_layer_to_connector`) and knows nothing of the shadow names, so the per-pass caches would be neither waited on nor saved. |
| **input port ≠ `block_output`** | The baked route applies its member at the block's output; a read tap inside the block (`post_attn` / `post_mlp` / `linear:*`) would have to resume layer `j` mid-forward. |
| **output port not a write port** | Only `block_input` and `block_output` are legal write ports (the library's `WRITE_PORTS`): the engine resumes the host's layer loop there, and a decoder layer cannot be resumed from the middle of its own forward. |
| **output port at or after the input port** | Writing forward past the read point would *skip* the host's computation of the layers in between rather than re-run them. |
| **more than one input layer** | Each span would need its own per-pass KV sets and a defined nesting order. |
| **another member co-mounted at the input port** | The span re-execution replaces the whole stream at that port, so the composition order with a co-mounted member is undefined. |
| **a span layer with no attention** | Nothing to allocate a per-pass cache for; Mamba/linear-attention layers carry engine state this route does not duplicate. |
| **`passes` not an int ≥ 1** | Including `bool`, which would otherwise sneak through as 1. |
| **library without the pair-port API** | `adapters.sites` must export `WRITE_PORTS`, `PORT_ORDER`, `validate_write_port`, `port_order`. The fork imports them **optionally** (so every diagonal adapter keeps working against an older library) but never substitutes a private copy — that is the silent-drift bug class the hard `adapters.sites` import exists to kill. A rewiring config against an old library raises. |

---

## 6. The three costs this answers

The earlier assessment on `campaign/loop-member-20260909` (§10.7 of that
branch's `campaign/loop_member/RESULT.md`) declined host-loop
recirculation and costed it as three things. Explicitly, which of them
this change is:

1. **"The formalism does not index it."** — *Not this change.* The pair
   axis is the library's answer: `Mount` carries `(site, output_site,
   output_layer, passes)` and `MemberRecord.output_site_kinds` /
   `rewires_schedule` place it in the formalism. The fork consumes that
   vocabulary and, per the shared-table contract, defines none of it.
2. **"A rewrite of the mount engine's execution model" (training side).**
   — *Not this change.* `adapters.mounting.AdapterModel._rewire_span`
   does it on the HF side, driven from the input port's hook.
3. **"KV-cache semantics for re-executed layers" (serving side).** —
   **This change**, and it was the honest crux. The answer is that
   per-pass caches can be obtained without touching the KV-cache
   machinery at all, by declaring shadow attention layers into
   `static_forward_context` before the spec is frozen: the profiler then
   budgets them, the group logic gives them the host's block table and
   slot mapping, and `bind_kv_cache` binds them. The cost is stated in
   §2, not hidden: `(L + k·S)/L` on both KV memory and decoder FLOPs.

The same document's §10.6 also identified the non-obvious hazard for
cross-layer members — that a layer's adapter is skipped entirely when its
mask is empty (`_adapter_all_masks_zero`). That hazard does **not** apply
here, and deliberately: when the mask is all-zero the recirculated value
would be discarded token-by-token anyway, so skipping is the correct
behaviour rather than a missing tap. Nothing on that branch was fork
code — it was campaign scratch plus a library leaf — so nothing there is
superseded, and `piped_iterate`'s internal-iteration mechanism is
untouched by this change.

---

## 7. What remains for a full route

- **Multisite.** Needs the KV-cache spec to accept late additions, or the
  member set to be declared at engine construction so the spec can
  account for it. The second is much cheaper and is probably the right
  shape: a `max_recirc_passes` / span declaration on `EngineArgs` that
  reserves the shadow layers up front, with `load_adapter` then only
  allowed to fill a reserved span.
- **Chunked prefill.** Needs the pipe's own stream to be checkpointed at
  chunk boundaries — i.e. the span's input for chunk `c` must be the
  recirculated stream, not the host's, which means storing one hidden
  vector per pass per request at the boundary. That is per-request
  adapter state, which the fork deliberately does not have.
- **CUDA graphs.** Needs the span unrolled into the traced graph with the
  attention layer identity chosen statically per pass rather than by
  attribute swapping — plausible, since `passes` is static, but it means
  `k` distinct attention call sites per span layer in the compiled
  region.
- **Prefix caching.** Needs the block hash to include the pipe's identity
  and gate, so blocks are only reused across requests that recirculate
  identically.
- **Pipeline parallelism.** Needs the span confined to one stage, or a
  re-entry path over PP send/recv.
- **N independent spans, and nested spans.** Needs a defined nesting
  order and per-span shadow sets.
- **Non-attention span layers** (Mamba, linear attention): needs per-pass
  duplication of those layers' engine state, which is a different spec
  type (`MambaSpec`) with different reset semantics.
- **Input ports other than `block_output`.** Refused today (§5). Note the
  library's `block_input` is currently meaningful only as an *output*
  port anyway: `resolve_site_submodule_path("block_input")` returns
  `None` and `mounting._install` registers a *post*-forward hook, so a
  `block_input` read tap does not exist on the training side either.
  Making it a real read port is a library change first.
