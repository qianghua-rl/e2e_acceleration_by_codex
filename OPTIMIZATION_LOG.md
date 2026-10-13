# Optimization log

Each entry: profile evidence → Amdahl estimate → change → measured result → decision.

## Summary
| stage | phase | change | decode ms/tok (warm, ctx 1024) | makespan (s) | speedup vs base | accuracy | notes |
|---|---|---|---|---|---|---|---|
| stage00_baseline | - | HF eager, cuDNN SDPA, torch fallback for Gated-DeltaNet, DynamicCache | 63.0 (cold 438) | 1673.6 | 1.00 | 0.930 | GPU busy 21%; decode 87% / prefill 12% / tool 0.6% |
| p1_s01_no_cudnn_sdpa | p1 | `enable_cudnn_sdp(False)` | 59.9 (cold 59.4) | – | – | – | cold-shape cliff gone; prefill cold 1.3–2.2 s → 0.4–0.9 s |
| p1_s02_fla_fastpath | p1 | + fla / causal-conv1d kernels | 59.0 | – | – | – | prefill warm 412 → 110 ms @1024 (3.8×); decode unchanged; first-call Triton JIT 43 s |
| p1_s03_static_cache | p1 | + StaticCache (4096 slots) + static decode buffers/mask | 65.4 | – | – | – | 10% slower alone (masked attention over 4096 slots); prerequisite for graphs |
| p1_s04_cuda_graph | p1 | + CUDA-graph decode step | **31.5** | **660.8** | **2.53** | 0.9375 | 51% of roofline; GPU busy 0.83; 7/8 greedy outputs identical to p1_s02 (1 near-tie) |
| p1_s05_async_readback | p1 | + pipelined token readback | 31.1 | – | – | – | −1.3%; launch cost was profiler-inflated; 8/8 identical to p1_s04 |
| p1_s06_compile | p1 | + torch.compile decode (inside our graph) | 25.6 | – | – | – | 1.22×; elementwise 6.48 → 1.14 ms/step; 7/8 identical (near-tie) |
| p1_s07_gqa_decode_attn | p1 | + GQA-folded GEMM decode attention | **21.4** | **474.8** | **3.52** | 0.9375 | 75% of roofline; GEMM 92% of step; decode 92% of e2e |
| p2_s01_continuous_batching | p2 | slot cache + batch-32 decode graph + continuous runner | 30.4 ms/step (1.2 ms/tok amortized) | **60.8** | **27.5** | 0.9375 | ~25.5 live rows; prefill 46% / decode 40% / tool 13% |
| p2_s02_prefix_reuse | p2 | + cross-turn extend + shared-prefix snapshot | same | 60.7 | 27.6 | 0.9375 | prefill tokens −89% but prefill time unchanged (~107 ms/call fixed) |
| p2_s03_forced_prefill | p2 | + short suffixes fed as forced tokens through the decode graph | 30.1 ms/step | **46.7** | 35.8 | 0.9375 | prefill calls 256 → 7; 1222 steps; tool 17.7% |
| p2_s04_async_tools | p2 | + tool calls in a thread pool; suspend/resume slot states | 30.1 ms/step | **38.7** | 43.2 | 0.9375 | tool fully overlapped; decode 95% |
| p2_s05_inplace_state | p2 | + fla decode kernel writes recurrent state in place | 26.7 ms/step | **35.7** | 46.9 | 0.9375 | −3.2 ms/step state copy; GEMM 78% of step |
| p2_s06_fp8 | p2 | + FP8 linears (torchao, per-row scales) | 23.05 ms/step | 32.2 | 52.0 | 0.9375 | CUTLASS rowwise GEMM slow at M=32 |
| p2_s06t_fp8_tensor | p2 | FP8 with per-tensor scales (cuBLASLt) | **20.86 ms/step** | **28.6** | **58.5** | 0.9375 | GEMM 20.7 → 12.9 ms; kept |
| p2_s07_bucket_graphs | p2 | + 64 slots, one graph per batch bucket, live rows compacted | 26.9 ms/step @64, 16.2 @8 | **24.8** | 67.4 | 0.9375 | 990 steps ≈ longest chain → critical-path bound |
| p2_s08_bf16_state | p2 | + recurrent states stored in bf16 | 26.1 ms/step @64 | 23.6 | 71.0 | 0.9375 | recurrent kernel only −8% (launch-bound) |
| p2_s09_recurrent_launch | p2 | + tuned (BV, warps) launch of fla decode kernel | 23.1 ms/step @64, 15.7 @8 | **21.5** | 77.7 | 0.9297 | recurrent 7.7 → 4.1 ms @64; 1 borderline task flips |
| p2_s10_force_all | p2 | + all suffixes forced (force_max 512) | same | **21.3** | **78.5** | 0.9375 | eager prefill 1.9 → 0.6 s; chain +68 steps (noise) |

## stage00_baseline
- Profile: GPU busy 0.21; `_cudnn_attention_forward` 31% of CPU time (2.4 ms CPU / 13 µs GPU per call);
  ~5.9k kernels per generated token. Micro decode cold 438 vs warm 63 ms/tok.
- e2e: 1673.6 s, acc 0.930, decode 71.8 ms/tok (much closer to warm than the 4-task nsys run's 164 ms/tok).

## p1_s01_no_cudnn_sdpa — remove per-shape cuDNN SDPA plan building
- Evidence: cuDNN SDPA host time per new (q_len, kv_len); decode KV length is new at every step.
- Estimate: from the 4-task profile p≈0.57 of decode → ~2.2×. From the full e2e run (decode 71.8 vs warm ~60):
  p≈0.14 of e2e → ~1.17×. **The short profile over-weighted cold effects** — size Amdahl's p on the real workload.
- Result (micro): decode cold 438 → 59.4 ms/tok, warm 63.0 → 59.9. Kept.

## p1_s02_fla_fastpath — fla + causal-conv1d kernels for the 48 Gated-DeltaNet layers
- Evidence: baseline GPU time is thousands of tiny elementwise kernels from the torch fallback.
- Result (micro): prefill warm 3.8× faster (chunked delta rule in one Triton kernel instead of a Python loop
  over chunks); **decode unchanged (59.0)**. In decode the fallback was only a few ops per layer; the cost is the
  per-op host dispatch of the whole 64-layer forward, not the fallback math → decode is host/launch-bound.
- Caveat: first-call Triton JIT/autotune (43 s at the first shape, 11.8 s at a new bucket). One-time per process,
  but must be warmed before measuring (the e2e warmup task covers it).
- Kept (prefill is 12% of e2e: ~0.12 × (1 − 1/3.8) ≈ 9% e2e expected).

## p1_s03 / p1_s04 — StaticCache + CUDA-graph decode
- Evidence: decode ≈ 59 ms/tok vs 16 ms floor while GPU busy ≈ 21–27% → host-bound; a CUDA graph replays the
  whole step's kernels with one launch. StaticCache is the prerequisite (stable tensor addresses).
- Estimate: decode is ~87% of e2e; if decode → ~20 ms/tok (s≈3), overall ≈ 1/(0.13 + 0.87/3) ≈ 2.4×.
- Correctness: HF skips the attention mask for q_len = 1, which with a StaticCache would attend to unfilled
  slots → decode passes an explicit static 4D mask extended in-graph. `check_equiv`: p1_s03 and p1_s04 are
  identical to each other and 7/8 identical to p1_s02 (one divergence at token 42 inside a code comment).
- Result (micro, ctx 1024): p1_s03 65.4 ms/tok (slower: mem-efficient attention over 4096 masked slots, nothing
  else saved); **p1_s04 31.5 ms/tok** (1.87× vs p1_s02, 51% of roofline), flat across ctx 512–2048.
- e2e: **660.8 s (2.53× vs baseline)**, accuracy 0.9375; predicted ~700 s. Decode now 94% of e2e.
- Node-level nsys: GPU time per replay 30.3 ms ≈ the step → GPU-bound. GEMM 18.95 ms (85% of HBM BW),
  elementwise/copy 6.48 ms in 2648 tiny kernels, attention 3.92 ms (245 µs/layer over 4096 masked slots).
- Profile: GPU busy 0.29 → 0.83. `cudaGraphLaunch` costs ~4 ms CPU per replay (~3.6k nodes) and is serialized
  with the per-token blocking read → candidate p1_s05 (pipelined readback). The remaining GPU time (31.5 vs 16 ms
  floor) needs node-level graph tracing (`nsys --cuda-graph-trace=node`, re-profile submitted).

## p1_s05_async_readback — overlap graph launch with the previous step
- Evidence: torch.profiler showed `cudaGraphLaunch` ≈ 4 ms CPU per step, serialized with the blocking read.
- Estimate: up to 4/31.5 ≈ 13% of decode.
- Result: 31.5 → 31.1 ms/tok (−1.3%), 8/8 identical. **The 4 ms was profiler overhead**; host-side numbers from an
  instrumented run must be cross-checked with an uninstrumented A/B (the micro).

## p1_s06_compile — torch.compile (inductor) the decode forward, still inside our CUDA graph
- Evidence: node-level graph trace — 2648 elementwise/copy kernels per step, 6.48 ms (21% of the step).
- Estimate: fusion to a few hundred kernels → ~1.5 ms → step 30.3 → ~25 ms (≈1.2×).
- Result: **25.6 ms/tok (1.22×)**, matches the estimate; profile: elementwise 1.14 ms in 644 fused kernels,
  1485 kernels/step. 7/8 identical (near-tie flip from different bf16 rounding).
- Pitfall: inductor sets `TRITON_CACHE_DIR` to node-local /tmp when unset → fla kernels JIT-ed later missed the
  cache (cold prefill at a new length bucket 13 s). Pinned cache dirs in the engine.

## p1_s07_gqa_decode_attn — single-token attention as two GQA-folded GEMMs
- Evidence (p1_s06 profile): attention 4.45 ms/step (18%): HF `repeat_kv` 6× K/V copy (needed because a mask is
  passed) + `fmha_cutlassF` with one block per head over all 4096 slots, 237 us/layer.
- Estimate: → <1 ms, step ~21.5 ms.
- Result: **21.4 ms/tok** (−16%), 75% of the 16.1 ms floor, 8/8 identical. GEMM (~18.8 ms, 85% of HBM BW) is
  now ~88% of the step → phase-1 exit criterion met; the remaining levers are lower-precision weights (FP8,
  phase 2) or kernel work (out of scope).

## Phase 1 summary
1673.6 s → 474.8 s (3.52×), accuracy 0.930 → 0.9375; decode 63 (cold 438) → 21.4 ms/tok (25% → 75% of roofline).
What mattered, in order: CUDA graph (host-bound → GPU-bound, 1.87×), torch.compile fusion (1.22×), decode
attention rewrite (1.19×), cuDNN-SDPA off (cold cliff), fla kernels (prefill 3.8×). Two estimates were badly off
(p1_s01 from a too-short profile, p1_s05 from profiler-inflated host time) — both corrected by uninstrumented A/B.

## p2_s01_continuous_batching — 32 agents in flight, one decode graph
- Evidence: phase-1 end: decode 92% of e2e, and 88% of a step is the weight read → batch independent agents.
- Change: duck-typed slot cache (per-row KV write position + mask row), eager batch-1 prefill into a slot,
  CUDA-graph replay of the compiled step at batch 32, continuous admission.
- Result: **60.8 s (7.8× vs p1_s07, 27.5× vs baseline)**, acc 0.9375; 797 steps × 30.4 ms; check_batch 8/8.
  Task latency p50 3.2 → 11.5 s (throughput/latency trade).

## p2_s02_prefix_reuse — reuse previous turn + shared system prefix
- Evidence: 55% of prompt tokens re-prefill the previous turn; 352-token common prefix.
- Hybrid model: recurrent state only at the saved position → reuse at slot end (extend) or a snapshot.
- Estimate: ~−20 s (prefill tokens −89%). Result: **0 s** — prefill time 27.9 → 27.9 s, per-call median ~107 ms
  both ways. **Eager prefill is per-call (host) bound, not token bound**; the micro had shown it (105 ms @512 vs
  124 ms @1024). Measure p on the cost driver.
- Next: short suffixes as forced tokens through the decode graph (sim: ~47.7 s).

## p2_s03_forced_prefill — suffixes through the decode graph
- Evidence: prefill 46%, ~107 ms per eager call regardless of length; suffixes are short (mean 70 / 26 tokens).
- Change: `submit` queues the suffix; the graph feeds the queue head instead of the row's own output.
- Estimate (step simulator): ~47.7 s. Result: **46.7 s**, prefill calls 256 → 7, steps 797 → 1222.
- Next: tool 17.7% (inline tools block all rows).

## p2_s04_async_tools — tool calls off the decode loop
- Every replay updates all rows' linear states, so a waiting slot saves/restores its states (KV is safe).
- Result: **38.7 s** (predicted 39.6), check_batch suspend/resume 8/8 identical; decode 95%.

## p2_s05_inplace_state — no copy-back of recurrent states
- Evidence: `triton_poi_fused_copy_` 3.2 ms (10.7%) of a 29.8 ms step. Change: launch the same fla kernel with
  ht = h0 = cache buffer. Result: step 26.7 ms, **35.7 s**.

## p2_s06 / p2_s06t — FP8 weights
- Evidence: GEMM 78% of the step, bound by 54 GB of bf16 weights per step.
- Result: per-row 32.2 s, per-tensor **28.6 s**, accuracy unchanged both. GEMM fell 1.6× (not 2×): at M=32 the
  scale granularity picks the kernel (CUTLASS rowwise vs cuBLASLt), plus dynamic activation-quant kernels.
- Next: makespan is now set as much by step *count* as step time — 16.9 live rows per step in a 32-row graph
  (tail) and a ~1000-step longest chain → per-bucket graphs + compaction + 64 slots (sim ~22 s).

## p2_s07_bucket_graphs — tail and queueing
- Evidence: 16.9 live rows per step in a 32-row graph; the longest agent (~990 token-steps) can start late.
- Change: graphs for B ∈ {8,16,24,32,48,64} over [:B] views; logical slot → physical row map keeps live rows
  in [0, #live) via row swaps; S = 64. dynamo bug on the way (property reading cache.B not guarded).
- Result: **24.8 s**, 990 steps (chain 989), computed/live rows 1.89 → 1.12. check_batch with one bucket +
  per-row FP8: 20/20 identical; default 8/20 (bucket-dependent GEMMs + per-tensor FP8 couples rows).
- Lesson: per-tensor activation scales trade batch invariance for speed.

## p2_s08 / p2_s09 — recurrent state bytes, then the launch that can use them
- bf16 storage halved the bytes but the kernel went from bandwidth-bound to launch-bound (BV=8, 1 warp):
  −8% only (23.6 s). Sweep of (BV, warps, stages) in a micro → bf16 B=64 148 → 91 us/layer; **21.5 s**.
- Ablation: tuned launch on fp32 22.5 s. Each half alone gives part; together 3.3 s.

## p2_s10_force_all — last eager prefills
- 4 suffixes > 128 tokens → eager FP8 prefill 0.33 s each, stalling all rows. force_max 512 → **21.3 s**;
  same-turn simulation 21.5 → 20.2 s. Run-to-run chain variation (±70 steps) is now as big as one change.

## Phase 2 summary
474.8 s (p1_s07) → 21.3 s (22×), 1673.6 → 21.3 s overall (78.5×), accuracy 0.93–0.94 throughout.
Order of impact: continuous batching (7.8×), forced-token prefill (1.30×), FP8 (1.25×), async tools (1.21×),
bucket graphs + 64 slots (1.15×), state launch/bf16 (1.15×), in-place state (1.08×). Prefix reuse alone gave
0× (it enabled forced prefill). The run is now bound by its longest agent: ~1000 steps × 15–23 ms.

## Reference: vLLM 0.30 (same workload; `results/ref_vllm_fp8/README.md`)
- 3 repeats each. fp8: ours (p2_s10) 20.3 s median vs vLLM 20.6 s. bf16: ours (p2_s10_bf16) 28.2 s vs vLLM 28.9 s.
- Both are bound by task 423's chain. With the same 751-token chain, ours costs 26.5 vs 27.4 ms per chain token.
- vLLM admits all 128 agents, so its task p50 is 10.4 s vs ours 5.8 s (64 slots).
- vLLM loads faster: 223 s vs 580 s. Final write-up: `results/FINAL_REPORT.md`.

# Phase 3 — systems optimization toward the physical limit
Fixed: FP8 per-tensor weights, bf16 lm_head / KV / recurrent state, greedy, no speculation (optimize the system,
not the model). Task 423 (a 750-token repetition outlier) excluded from both workloads. Two workloads:
gsm8k 127 tasks, and `gsm8k_longctx_128` (1k–7k-token prompts, no shared context). Rebaseline p2_s10: 18.19 s.
Dev numbers (`results/_dev_*`, dev job, ±0.1 s / ±1 s) in order; official runs via `slurm/submit.sh` at the end.

| step (commit) | 127-task | longctx |
|---|---|---|
| v3 engine: own forward, async token-budget scheduler, chunked prefill mixed with decode, paged KV, every step a CUDA graph (3252fc5) | 15.58 → 10.95 s | 51.5 s |
| kernels: 128-row prefill attention tiles, single-pass silu_mul_quant, split-KV decode attention (d375bad) | 9.95 s | 47.4 s |
| merged small-M FP8 GEMMs (gate\|up, qkv\|z\|b\|a, q\|k\|v) in Triton (802ce97) | 9.51 s | 45.0 s |
| runner: no device sync per timed section, incremental tokenization (624421f) | 8.96 s | 44.6 s |
| programmatic dependent launch on every kernel (b7bdbf8; off: 9.30 s) | 8.95 s | 42.2 s |
| fused GDN chunk recurrence + output, conv fast path (78bf72d) | 8.85 s | 41.65 s |
| small-M GEMM retune, graph-loop timing (2defad5) | 8.77 s | — |
| **official p3_s01_kernels / _longctx (2defad5)** | **8.86 s** (acc 0.945) | **40.64 s** (acc 0.945) |
| GDN chunk kernel BV 64 / 4 warps / 3 stages (e8ea498; official p3_s02_gdnchunk) | 8.77 s | 39.61 s |
| gdn_decode tiers tuned in-model under the power cap (3eaf6fe; official p3_s03_inmodel) | 8.67 s (A/B) | 39.8 s (A/B, noisy) |
| GDN prefill: recompute_w_u fused into the chunk recurrence (0abefc7; official p3_s04_gdnfuse 8.47 / 40.83 s) | 8.59 s (A/B) | mixed step −1.8 ms |
| GDN kkt + inverse on tensor cores, cumsum fused, no memsets (0b442cc; official p3_s05_kkt 8.99 s) | **broken**: longctx acc 0.417 (p3_s05_kkt_longctx 77.8 s) | unstable tf32 inverse, see below |
| kkt diagonal blocks by fp32 forward substitution (097cd7f; official p3_s06_kktfix) | 9.45 s (acc 0.945; task 428 now 581 tokens, see replay below) | 40.46 s (acc 0.953) |
| packed bf16 rounding (_rbf16), bit-exact (1e84d12) | mixed step 114.25 → 113.80 ms | |
| mixed steps: prefill-row GDN / attention kernels on a side stream, concurrent with decode rows (6b5d5f5) | replay A/B: 8.486 → 8.429 s | replay A/B: 39.87 → 39.32 s |
| replay A/B, 0abefc7 → 6b5d5f5 (same GPU, same trajectories, 2 rounds each) | 8.515 → 8.429 s (−1.0%) | 40.13 → 39.32 s (−2.0%) |
| decode-step causal conv1d fused into the gdn_in GEMM epilogue, q/k l2-norm into gdn_decode (321c65f) | replay A/B: 8.414 → 8.380 s | |
| prefill attention by FA3 (built from source, paged, varlen) on the side stream (321c65f) | | mixed step 116.2 → 114.3 ms; both: replay A/B 39.17 → 38.90 s |
| official p3_s07_fa3conv / _longctx (321c65f) | 8.458 s (acc 0.945) | 39.00 s (acc 0.937, see below) |
| no cuBLASLt workspace: static nvjet kernels, no memset node breaking PDL (b692a7c) | replay A/B 8.468 → 8.422 s | 39.29 → 38.87 s |
| gdn gates in kkt, conv state update off the chain, gdn ordering (6892413) | 8.417 → 8.410 s | 38.86 → 38.68 s |
| conv scalar test, intra-chunk attention off the chunk chain (c357466 → c6ee482) | 8.425 → 8.415 s | 38.80 → 38.70 s |
| gdn_decode next-head register prefetch (15d508e; dev GPU h100-2) | 8.420 → 8.365 s | 38.73 → 38.67 s |
| **official p3_s08_final / _longctx (49b693b = 15d508e + scratch)** | 9.98 s (acc 0.945; task 428 9.86 s, a trajectory change) / replay of p3_s07: **8.36 s** | 38.21 s (acc 0.937) / replay: **38.17 s** |
| vLLM 0.30 fp8, same workloads | 17.54 s | 91.96 s |

## Where the time is vs physics (2defad5 + 01fe367)
- Mixed step (2×1024 prefill + 48 decode rows @ ctx 4000): 114 ms. GEMMs 76 ms at ~1.31 PFLOPS = cuBLAS FP8
  practical peak on this (power-limited) GPU; silu/rmsnorm at 88–90% of practical HBM bandwidth; GDN prefill
  (fla intra kernels + fused chunk kernel) ~11 ms vs ~2–3 ms of bytes/FLOPs — the main remaining gap.
- Decode B=128 @ ctx 1000: 24.3 ms vs ~18.3 ms of bytes (weights 9.1 + state r/w 6.5 + KV 2.8 ms at 2.97 TB/s).
  gdn_decode 2.66 TB/s = the ceiling of its own access pattern (pure tile read-modify-write: 2.64–2.86 TB/s;
  flat in-place RMW 3.0 TB/s); attn_decode 2.8–3.0 TB/s; cuBLAS GEMMs 2.5–2.9 TB/s.
- Decode B=1: 11.1 ms vs 8.5–9.1 ms weight read; the rest is ~1.5 ms of latency-bound small kernels.

## Tried, no gain (reverted)
- L2 prefetch of the next weights (`cp.async.bulk.prefetch.L2` before gdc_wait): per-element slower, bulk neutral.
- Early `griddepcontrol.launch_dependents`: slower at B ≥ 64.
- Two streams in mixed steps (decode rows ∥ prefill rows): GDN kernels never overlapped, attention 0.9 of 4.5 ms;
  step time unchanged — both sides already fill every SM. (Phase 3 retry with the side stream at high priority
  and all kernels on PDL: kept, see 6b5d5f5 / "Streams" below.)
- Triton GEMMs for down/out projections or large-M wide GEMMs: slower than cuBLAS.
- gdn_decode with both reductions on the old state in one pass: same speed (bytes-bound, not reduction-bound).
- Lesson (benchmarks): Triton `do_bench` flushes L2 and leaves dirty lines that skew 30–60 µs kernels; time in
  CUDA graphs cycling over ≥ 2 weight copies instead. Warm up on a side stream only after `wait_stream`
  (otherwise the first launch races the input initialization — it looked like a kernel OOB bug).

## Measurement: official runs vs same-GPU A/B (p3_s03 / p3_s04)
- Official runs land on different nodes: p3_s03 8.98 s (h100-5) vs p3_s02 8.77 s (h100-2) although same-GPU A/B
  (alternating revisions, 2 rounds, dev job) gives e8ea498 8.74 / 3eaf6fe 8.67 / 0abefc7 8.59 s. p3_s04 official
  8.47 s. Node-to-node spread ~2–3% > one change: compare revisions only on one GPU (`/mnt/data/u/feic/tmp_ab/ab.sh`).
- longctx makespan depends on the scheduling trajectory (620–756 steps across runs of nearby revisions; one run
  even had a different accuracy 0.953 from a different batch composition): judge kernels by per-step GPU time
  (`scratch/prof_mixed.py`) instead.

## Measurement: trajectory replay (`run_bench.py --replay <results dir>`)
- The 127-task makespan is the latency of the slowest task: p3_s06_kktfix 9.45 s vs p3_s04 8.47 s is task 428
  generating 581 instead of 391 tokens (7.15 → 9.31 s end to end; total generated tokens +1.7%, steps 515 → 593,
  GPU time per step 16.3 → 15.7 ms). Any rounding change moves it by ±1 s, more than any one optimization.
- `--replay` forces a recorded run's generated tokens (turns.jsonl `gen_ids`, else its transcripts re-tokenized):
  the host sees the recorded tokens (stops, tool calls, next prompts all follow the recording), the device still
  feeds its own samples (same work per step). Same-GPU A/B with replay is reproducible to ±0.005 s (127-task) /
  ±0.05 s (longctx), and isolates system speed from numerics. Accuracy still comes from normal runs.

## Streams (6b5d5f5)
- In a mixed step, the prefill-row GDN kernels (kkt + chunk) and attention prefill run on a high-priority side
  stream, concurrent with gdn_decode / attn_decode (disjoint rows and state slots; parallel branches of the CUDA
  graph; bit-exact). Big steps (2k tokens): the sections shorten (gdn 246 → 227 us, attn 821 → 731 us per layer)
  but the step does not — power-capped, energy per step is the bound. Small steps (B=40, 40 prefill rows):
  20.6 → 20.0 ms. End to end (replay A/B): 127-task −0.7%, longctx −1.4%.

## The power wall (longctx)
- Mixed steps run at ~690 of 700 W (throttle reason SW power cap), SM clock ~1.55–1.6 GHz instead of 1.98.
  cuBLAS FP8 there: ~1.35 PFLOPS. Longctx compute floor ≈ 549k prompt tokens × 48.7 GFLOP / 1.35 PF ≈ 20 s +
  ~1.2 s attention; mixed steps now take ~33 s of the run (≈ 64% of floor efficiency, GEMMs themselves ≈ 95%).
- Consequence: memory-bound kernels must be tuned inside the model. gdn_decode: an isolated micro showed a tie
  between configs, in-model (lower clocks) the software-pipelined head loop (HPP 4, STAGES 3) wins: 128 rows
  7.31 → 6.87 ms per decode step, 56 decode rows of a mixed step 4.05 → 3.41 ms.
- attn_decode: num_warps / num_stages / nsplit all within ±2% in-model (KV-bandwidth bound); unchanged.

## GDN prefill: no w / u in HBM (0abefc7)
- fla computes u = A (beta v), w = A (beta e^g k) in a separate kernel, and the recurrence uses v_new = u − w h.
  Same thing: v_new = A (beta (v − e^g k h)) — one 64×64 dot per chunk inside the recurrence instead of two
  materialized [T, 48, 128] tensors (write + read ~74 MB per 2k-token layer) and a kernel. Same bf16 MMA inputs
  (state diff vs before: 1 bf16 ulp), accuracy unchanged in both A/B workloads.
- Per layer incl. kkt_solve: 1992 tokens 189 → 160 us, 4000: 350 → 289, 4×512: 177 → 142. The bit-identical
  variant (u, w computed per chunk in registers) is 20% slower than before: the extra dots sit on the serial chain.
- Masking A's upper triangle in-kernel instead of zero-filling (saves a memset) is slower (masked load stalls).
- The recurrence is latency-bound (~4.6 us per 64-token chunk at 96 programs ≈ 15% of SM MMA peak); more
  programs (BV 32) or warps do not help in Triton.

## Tried in phase 3 kernel work, no gain
- PDL + early `launch_dependents` + L2 bulk prefetch of the next GEMM's weights: all CTAs resident ~19 us early
  (globaltimer), but ≤ 2.5 us per GEMM gain and none end-to-end.
- Split-K for the N = 5120 linears at small M: ≤ 0.9 us. Rowwise-scaled merged GEMMs (bit-exact merging of
  linears with different per-tensor scales): 15–30% slower than per-linear cuBLAS. gdn state contiguous layout:
  same speed.
- Prefill attention (ours ~473–487 TFLOPS): TMA descriptors +3%, warp specialization (Triton 3.6 crashes at
  4 warps), lazy rescaling, manual QK^T pipelining: no gain. Reference FA3 (vLLM build, hdim 256 GQA 6, page 64):
  653 TFLOPS at 1992@3000, 626 at 2×1024@2000, 520 at 1992@0 — worth ~2 ms per mixed step (~1.4% longctx),
  needs a hand-written warp-specialized kernel.

## GDN kkt on tensor cores (0b442cc) — and the accuracy bug it shipped (fixed in 097cd7f)
- fla's fused kkt + solve_tril: forward substitution on 16×16 diagonal blocks (14 serial cross-lane reductions),
  ~48 us per layer at 2k tokens vs ~10 us for loading k and computing k k^T alone. 0b442cc inverted the diagonal
  blocks by (I+Y)^-1 = (I−Y)(I+Y²)(I+Y⁴)(I+Y⁸) (Y strictly lower ⇒ nilpotent) in tf32 on tensor cores, then
  tf32 block merges like fla; gate cumsum fused, no zero-fill. On random (nearly orthogonal) keys its error vs an
  fp64 inverse equalled fla's, and the 127-task accuracy stayed 0.945.
- **It was wrong.** Official p3_s05_kkt_longctx: accuracy **0.417** (0.945 before), 77.8 s (garbled answers, long
  generations). Real keys inside a chunk are correlated: with k_i·k_j ≈ 1 the powers Y², Y⁴, Y⁸ have entries up
  to ~C(15, 8) ≈ 6000 times the result, so tf32's 1e-3 relative error leaves errors of 2–15 on entries ≤ 1
  (scratch/kkt_adv.py; fla stays at 3e-3). The product formula is numerically unstable, not just imprecise.
  The 127-task run (short prompts) only showed it as a "trajectory change": task 428 grew from 391 to 517–564
  tokens, which I wrote off as rounding noise. Lessons: (1) random-input kernel tests miss conditioning problems —
  test_v3_kernels now has a correlated-keys gdn_prefill case, which fails on 0b442cc; (2) a numerics change that
  moves trajectories needs the long-context accuracy run before it is believed, not after.
- 097cd7f: diagonal blocks by fp32 forward substitution again (the four blocks interleaved in one loop), merges
  and k k^T blocks stay as in 0b442cc. Same error as fla on every case of kkt_adv. gdn_prefill per layer
  (fla kkt → ours): 2 chunks 27.2 → 25.5 us, 1992 tokens 163 → 143 us, 4×512 144 → 129 us (0b442cc's broken
  version: 20.7 / 132 / 116).


## Ledger and the last GEMM / decode sweeps (before 321c65f)
- GPU ledger (127-task, replay): 8.36 s busy, 0.07 s idle — the host side is done. longctx: 80% of GPU time is
  mixed 2048-token steps (~1992 prefill + ~55 decode rows), ~116 ms each, of which cuBLAS FP8 GEMMs ~73 ms at
  ~1.33 PFLOPS (the power-capped practical peak).
- Medium-M GEMMs (M 64–512), Triton with TMA / warp specialization / swap-AB vs the current dispatch: ≤ 2% better
  on a few shapes, worse on most; kept (Triton mm_fp8 for gdn_in at small M, cuBLAS for down / out). Distance to
  the weight-read floor at M=128: gate_up 68.7 vs 57.5 us, down 33.6–35.3 vs 28.8, out 14.2–14.9 vs 10.1.
- The two N=48 b / a GEMMs at large M merged into one call: ~3 us per layer. Not done.
- Decode B=1: 11.07 ms vs 8.41 ms speed-of-light; GEMMs ~93% of their read floor, ~1.5 ms of latency-bound
  small kernels (add_rmsnorm, gnorm, silu ~2.5–4 us per layer each). Early trigger + L2 prefetch: other −0.32 ms,
  GEMMs +0.27 ms, net −0.05 ms. Closing the rest needs a persistent megakernel.

## FA3 prefill attention + decode conv fusion (321c65f)
- FA3 (hopper, built from source with only hdim 256 bf16 forward, paged KV, varlen; `/mnt/data/u/feic/pjs/src/
  build_fa3.sh`): 1992@3000 301 vs 411 us (ours), 1992@6000 501 vs 738, 4×512@1000 113 vs 171. Called directly
  on our paged cache (page table rows of the prefill slots, `p_seqk` = lengths after the step, 0 for padding
  entries so the fixed-size graph batch works). Falls back to the Triton kernel if flash_attn_3 is missing.
- Conv fusion: in decode steps the causal conv1d (window of 3 + new token, silu) runs in the gdn_in merged GEMM
  epilogue on the qkv columns; one program owns each (slot, channel) window, so it is race-free in place. The q/k
  l2-norm moved into gdn_decode (NORM flag) — the only non-bit-exact part (reduction order, rare last-bit).
- Accuracy: 127-task 0.945 (unchanged); longctx 0.937 vs 0.953 for p3_s06 — tasks 304 (also wrong in p3_s04)
  and 428 now run out of tokens in a verbose first turn. Historical longctx range 0.9375–0.953.
- Not caused by either change: real (non-replay) longctx runs of the same tree on the dev GPU
  (`scratch/accenv.sh`): both on 0.953 (the official run of the same commit: 0.937), FA3 off 0.953, conv fusion
  off 0.945, both off 0.961. Run-to-run spread of the same code is ±2 tasks (batch composition differs between
  nodes and runs), so longctx accuracy differences under ~0.02 are noise; only collapses (0b442cc) are signals.

## Mixed-step critical path: cuBLASLt workspace, gdn ordering (b692a7c, 6892413)
- Mixed 2048-bucket step (1992 prefill rows @3000 + 56 decode rows, `scratch/tl_mixed.py`), after both: span
  ~110.8 ms = cuBLAS GEMMs ~75.3, FA3 6.1, silu 4.3, rmsnorm 4.0, gdn chunk 3.8, gdn_decode 3.1, attn_decode
  2.9, conv_rows 2.7, kkt 2.35, gnorm 1.6 ms; the rest of the visible "gap" is profiler host time, not in-graph idle.
- b692a7c: with the default workspace cuBLASLt picks dynamic-scheduler (`coopA`) nvjet kernels at large M and
  puts a memset node before each GEMM in the graph; the memset breaks the PDL chain, ~7 us per GEMM.
  `CUBLASLT_WORKSPACE_SIZE=0` gives static-schedule kernels, same or faster, bit-exact (`scratch/ws_exact.py`,
  `memset*.py`). Replay A/B (2 rounds): 127-task 8.468 → 8.422 s, longctx 39.29 → 38.87 s (−1.1%).
- 6892413: gdn gates (softplus/sigmoid of the a/b GEMM outputs) computed inside the kkt kernel's cumsum loop
  (one launch less per layer); the conv state update split out of conv1d_rows and issued after the prefill
  chain; for buckets ≥ 1024 gdn_decode on the main stream waits for the side stream's kkt (so it overlaps the
  chunk kernel, not kkt + chunk). Bit-exact. Replay A/B vs b692a7c: longctx 38.86 → 38.68 s (−0.5%), 127-task
  8.417 vs 8.410 (neutral). The decode-after-kkt ordering for all buckets costs +0.8% on the 127-task run
  (small buckets: decode is the long branch there), hence the threshold.

## GDN chunk kernel: chain floor and a producer/consumer split (negative)
- The chunk kernel is a serial chain over 64-token chunks (h_{i+1} = decay·h_i + k^T v_new), 96 programs
  (48 v-heads × 2 value blocks of BV 64) for one 1992-token sequence — latency-bound, not
  bandwidth- or FLOP-bound. Isolated 1992 tokens: full kernel 106.8 us; chain only (no output o) 77.6 us
  (BV 64) / 59.6 us (BV 32) (`scratch/gdnchain.py`). So at most ~30–45% of it is the off-chain output work.
- Tried moving the output off the chain (`scratch/gdnpc.py`): 2×NP programs, roles by atomic ticket; producers
  run the chain and publish h_i (bf16) + v_new with a release flag per chunk, consumers spin (acquire) and
  compute o = q h_i + (qk^T ⊙ mask) v_new. Bit-exact at BV 64, but slower everywhere that matters:
  1992 tokens 107.9 → 155.7 us, 4×512 88.2 → 133.7, 4000 198 → 294; publishing h one chunk late (so the store
  is off the critical path) 149.5. Small batches equal. The h round trip through L2 (~50 MB per 1992 tokens)
  and the co-resident consumers stealing issue slots from the producer cost more than the o work it removes.
  Not adopted.

## Conv scalar test, GDN intra-chunk attention off the chain (c357466, 4f33408, 2bda9a1, c6ee482)
- c357466: conv1d_rows decided "whole block in one sequence, ≥ 3 rows in" by a block min/max reduction; two
  scalar loads (row_start of the first and last row) do the same. −10% on that kernel, bit-exact.
- ncu of the chunk kernel (1992 tokens, `scratch/gdnncu.py`): 165 KB smem at num_stages 3 → 1 CTA per SM, 96
  CTAs; issue/latency-bound (3.55 cycles per issued instruction, ~1620 instructions per warp per chunk), wgmma
  waits 27% of stall samples. The same 64×64 dots by mma.sync (1/2/8 warps) or wgmma give identical bits.
- The q k^T intra-chunk attention (masked, gated, bf16) does not depend on h, so it can leave the serial chain:
  Aq [T, HV, 64] computed elsewhere, the chunk kernel only does o = q h + Aq v_new. Bit-exact. kkt + chunk for
  1992 tokens: 149 us → own kernel 130 us (4f33408) → extra programs [0, H) of the kkt launch with their own gate
  cumsum, no wait for kkt 119 us (2bda9a1; GATES mode 153 → 121). Computing Aq inside the 1-warp kkt programs:
  170 us (rejected). `AGENTOPT_GDN_AQ=0` restores the old path.
- In the model the gain mostly disappears: in the mixed step chunk (~107 us raw) runs concurrently with
  gdn_decode (~103 us raw) on the other stream; alone they are ~73 and ~60 us (decode's 176 MB read floor ~59).
  Replay A/B c357466 → c6ee482 (3 rounds): longctx 38.80 / 38.81 / 38.78 → 38.71 / 38.71 / 38.68 s (−0.26%);
  127-task (2 rounds) 8.425 / 8.425 → 8.416 / 8.413 s (−0.13%). Accuracy unchanged (bit-exact).
- Power is a wall: in the longctx run the busy GPU averages ~631 W (limit 700), SM clock mean ~1620 MHz, min
  1350 (max 1980). Shortening a low-power latency-bound phase speeds it up but leaves less of the energy budget
  for the GEMMs; the floor is energy per step, not only time per kernel.

## gdn_decode register prefetch (15d508e) and the floors around it
- ncu of gdn_decode (56 rows, `scratch/decncu.py`): 27% of the stall samples were the state store waiting on the
  plain load of v (the a / b / v loads of a head are issued after its state tile). v, a, b of the next head now
  load one head early into registers (`PF`, bit-exact): 56 rows 69.6 → 66.9 us, 96 rows 114 → 110 us
  (`scratch/decpf.py`). `AGENTOPT_GDN_DECODE_PF=0` restores the old order.
- Floor: a copy kernel with the same grid / tiles that only reads each state tile and writes it back
  (`scratch/decfloor.py`) reaches ~2.9 TB/s; gdn_decode is 4–7% above it.
- Replay A/B c6ee482 → 15d508e (dev GPU h100-2, 2 rounds): longctx 38.68 / 38.78 → 38.65 / 38.68 s (−0.15%),
  127-task 8.416 / 8.424 → 8.375 / 8.355 s (−0.65%; small decode steps are where gdn_decode is on the critical
  path). Accuracy identical (bit-exact).

## Mixed-step GDN branch: why prefill ∥ decode is 161 us, not ~120 (`scratch/ovl.py`, `ovlprof.py`)
- Per layer of a 1992 + 56-row step: kkt 44 us alone (forward substitution ~14, block merges ~3.5, Aq programs
  2–7, the rest loads / k k^T), chunk 73 us alone, gdn_decode 67 us alone; concurrently 161 us vs ~120 ideal.
- Cause: the chunk kernel's 189 KB of smem on 96 SMs leaves room for one decode CTA per SM there; decode slows
  68 → ~106 us, chunk 73 → 88 us. Chunk configs with less smem (stages, BV) lose more on the chain than they
  give back; decode config sweeps, kkt warps / HPP / maxnreg: no gain (`scratch/kktt.py`, `smem.py`).
- Left: ~40 us per GDN layer (~1.9 ms per mixed step, ~1.7%) and ~9 us of kkt forward substitution — both need
  a different kernel structure (one persistent kernel scheduling chain and decode tiles), not tuning.

## Fused epilogues for the large GEMMs: no GEMM to fuse into (negative)
- Target: silu_mul_quant (4.3 ms per mixed step) and add_rmsnorm (4.0 ms), both ~90% of HBM bandwidth; fused
  into the gate_up / down / out epilogues they would save ~1.5% each — if the fused GEMM matched cuBLAS.
- gate_up at M = 2048 (N 34816, K 5120), relative max error vs fp32 on 256 rows:
  cuBLAS 504 us (1448 TFLOPS, err 2.2e-3). Triton persistent TMA (`scratch/mmbig.py`): default (fast) fp8
  accumulation err 6.6e-3; with fp32 promotion every 128 products (`max_num_imprecise_acc=128`, = cuBLAS
  precision) 684 us. CUTLASS 4.3 sm90 warp-specialized (`scratch/cutmm/`, 12 schedules): precise accumulation
  best 567 us (128×128 cooperative, cluster 1×1); 128×256 tiles spill (two accumulators) and run at ~1500 us;
  FP8FastAccum 483 us but err 6.3e-3 (down: 1.1e-2). cuBLAS has a precise-accumulation mainloop we do not.
  Losing ≥ 63 us per GEMM to save ~27 us of silu: not done. Fast accumulation is a precision change: not done.
- Small M (decode, B ≤ 16): Triton down / out GEMMs (N 5120) are 5–8% slower than cuBLAS (down 34.1 vs 32.4 us,
  out 14.6 vs 13.7, `scratch/mmn5120.py`), which eats most of what fusing silu / gnorm into them would save.

## Nano-batching (negative, `scratch/nano.py`)
- Idea: split a 2048-row mixed step so bandwidth-bound kernels of one half hide under the other half's GEMM.
  gate_up at M 1024 runs 261.5 us (2 × = 523 vs 504 at M 2048); silu hides only 24% under it. At M 512 silu hides
  89%, but the GEMM is 24% less efficient. Net loss either way under the power cap.

## Small-batch decode vs its floor (127-task)
- Half of the 127-task GPU time is decode steps with ≤ 16 rows at ~11 ms per step (`scratch/stepcat.py`); the
  weight-read floor is ~8.4 ms. B = 1 per GDN layer (`scratch/tl_dec.py`): GEMMs 137.6 us (gdn_in 31, out 13.2,
  gate_up 60.5, down 32.4; floor at 3.0 TB/s ~128), small latency-bound kernels ~20 us (add_rmsnorm 3.6 × 2,
  gdn_decode 4.9, gated_norm 3.4, silu 4.9). Kernels start within ~0.5 us of their predecessor (PDL): the
  20 us is execution latency, so only fewer kernels (fusion / a persistent megakernel) close it.
# Ascend 910B migration log

## 2026-10-08: stage 0 NPU smoke

- Evidence: `npu-smi` reports an idle Ascend 910B3; a bf16 matrix test succeeds through `torch_npu`.
- Compatibility setup: Python 3.12 environment `ascend-e2e`, PyTorch 2.9.1, torch-npu 2.9.1, Transformers 5.5.4, NumPy 1.26.4. No custom kernels were added.
- Model: local Qwen3.5 checkpoint, 27,356,728,560 parameters, loaded in 34.7 s on one NPU.
- Smoke measurement: 2 tasks, 13.45 s makespan; prefill 14.3%, decode 85.6%, 185.7 ms/token.
- Amdahl estimate: reducing decode overhead is the first candidate because it dominates 85.6% of measured time; the estimate is provisional until a valid full-turn baseline is recorded.
- Limitation: max-new-tokens=32 produced no tool calls and no answers, so this run is connectivity evidence only and is excluded from accuracy comparisons.

## 2026-10-08: stage 0 eight-task probe

- Measurement: serial runner, 8 tasks, max 128 new tokens, max 6 turns.
- Result: 197.86 s makespan, 0.500 accuracy, 1,010 generated tokens, 186.33 ms/decode token.
- Breakdown: decode 94.0%, prefill 5.9%, tools 0.0%.
- Amdahl estimate: with decode fraction 0.940, a 1.5x decode improvement could give at most 1.415x overall. This selects decode launch/dispatch and cache behavior for Phase 1 profiling.
- Interpretation: the workload is valid (4 tool calls and 4 answered tasks), but the 128-token cap makes this a probe rather than the final 512-token baseline.

## 2026-10-08: stage 0 representative baseline

- Measurement: 16 tasks (task 423 excluded), serial runner, max 256 new tokens, max 6 turns.
- Result: 473.66 s makespan, 0.9375 accuracy, 2,329 generated tokens, 191.84 ms/decode token.
- Breakdown: decode 93.1%, prefill 6.8%, tools 0.1%; 15 tool calls and 31 turns.
- Amdahl estimate: decode fraction 0.931 gives a theoretical 1.399x end-to-end gain for a 1.5x decode improvement. This confirms the Phase 1 target selected by the 8-task probe.
- Decision: retain this configuration as the representative baseline for iterative optimization; run the full 127-task confirmation after optimization stabilizes.

## 2026-10-08: Phase 1 compile attempt (failed)

- Evidence: `torch.compile(model, backend="torch_npu")` fails immediately with `InvalidBackend: Invalid backend: 'torch_npu'` under PyTorch 2.9.1 / torch-npu 2.9.1.
- Amdahl estimate before implementation: decode is 93.1%; a 1.5x compiled decode path would have a theoretical 1.399x end-to-end gain.
- Measured result: no runtime comparison was possible because the backend is unavailable.
- Decision: reject this candidate without changing the measured baseline. The next candidate is existing framework batching/scheduling, which belongs to Phase 2.

## 2026-10-08: Phase 2 NPU eager continuous batching

- Implementation: added an NPU eager path to the existing slot-cache batch engine. CUDA Graphs, CUDA Events, Triton, and custom kernels are bypassed; HF/torch-npu operators and the existing continuous runner remain unchanged.
- Evidence: 4 concurrent slots completed 4 tasks in 38.69 s; 3 tool calls; 167 decode steps; 514 live-row steps and 668 computed-row steps.
- Breakdown: decode 81.5%, prefill 18.4%; decode 62.09 ms/token versus 185.7 ms/token in the serial probe.
- Amdahl estimate: assuming a 3x reduction in weight-read cost, the 94% decode fraction predicts 2.54x overall speedup. The observed per-token reduction is 2.99x; makespan comparison is affected by different task trajectories.
- Correctness: 3/4 tasks answered correctly under the 128-token cap; no output divergence or runtime failure. Keep as a Phase 2 candidate and rerun with the representative workload.

## 2026-10-08: Phase 2 NPU eager batching validation

- Same 16-task workload and token limits as the serial baseline, with four continuous-batching slots.
- Result: 152.12 s makespan versus 473.66 s serial (**3.11x**), accuracy 0.9375 unchanged.
- Decode fell from 191.84 to 53.38 ms/token (**3.59x**); prefill rose to 19.3% of makespan because prefill remains eager and batch-1.
- Amdahl interpretation: batching removes repeated weight reads from decode; the remaining prefill share is now the dominant next target. Select prefix/state reuse or batched prefill for the next Phase 2 step.

## 2026-10-08: Phase 2 prefix reuse attempt (failed)

- Candidate: reuse the existing slot recurrent/KV state across turns and shared prefixes.
- Evidence: the attempt fails at the first prefill with `TypeError: 'NoneType' object is not callable` in the wrapper around `causal_conv1d_fn`.
- Cause: optional flash-linear-attention/causal-conv1d is unavailable on this NPU environment; Transformers uses a pure PyTorch fallback with no callable fast-path function to wrap.
- Decision: reject and record the failure. The validated NPU eager batching path remains the current best implementation.

## 2026-10-08: Phase 3 NPU scheduler scaling

- Change: increased eager continuous-batching slots from 4 to 8, using the same existing operators and asynchronous runner.
- Result: 121.65 s versus 152.12 s at 4 slots (**1.25x**), accuracy remained 0.9375.
- Decode improved from 53.38 to 40.64 ms/token; prefill is now 23.2% of makespan.
- Amdahl estimate before implementation: with decode fraction 0.806 and an expected 1.2x decode improvement, overall gain was `1/(0.194+0.806/1.2)=1.145x`; measured gain was 1.25x, exceeding the estimate.
- Exit signal: further slot increases are likely to yield diminishing returns because prefill and memory pressure now dominate; the next step is a final full-workload confirmation.

## 2026-10-08: final 127-task confirmation

- Configuration: 8-slot NPU eager continuous batching, 127 tasks with task 423 excluded, max 512 new tokens, max 6 turns.
- Result: 907.54 s makespan, accuracy 0.9370, 19,812 generated tokens, 252 turns, 125 tool calls.
- Breakdown: decode 66.9% (607.04 s, 31.03 ms/token); prefill 33.0% (299.09 s, 418.41 token/s).
- Correctness: 124 tasks finished with an answer and 3 reached the token/turn limit; answer accuracy remained at the expected ~0.94 level.
- Exit decision: the three-phase no-custom-kernel pipeline is complete for the fixed 127-task workload. Further gains require reducing eager prefill or using NPU graph/compiler features not available through the tested public backend; those are recorded as future candidates rather than silently implemented.

## 2026-10-09: H100 vs Ascend comparison

- Ascend final (127 tasks, 8 slots, bf16 eager): 907.54 s, accuracy 0.937, decode 31.0 ms/token.
- H100 final (same workload, FP8 + CUDA graphs + custom kernels): 8.36 s, accuracy 0.945.
- The Ascend run is about 108x slower in makespan. The gap is explained by the excluded techniques (CUDA graph, FP8, custom kernels, PDL, paged KV, fla fast paths), all of which failed or were unavailable on this stack.
- Ascend's own optimization pipeline still delivered 3.89x on the representative 16-task workload (473.66 s to 121.65 s) with unchanged accuracy.
- Full comparison table: `results/H100_VS_ASCEND_COMPARISON.md`.

## 2026-10-09: Phase 1 Ascend backend matrix

- A small bf16 MLP was used to avoid spending 10+ minutes compiling the full 27B model.
- `torch.compile(backend="npu")`: reached the CANN compiler, then failed because `torchair` is unavailable.
- `torch.compile(backend="npugraphs")`: failed because the environment lacks `triton`.
- `torch.compile(backend="npugraph_ex")`: failed because the driver/runtime lacks `aclmdlRICaptureBegin`.
- These are environment/runtime blockers, not evidence that 910B cannot support the methods. The results are recorded in `results/p1_npu_backend_matrix/`; the next action is to install the matched framework components before model-level graph tests.

## 2026-10-09: dependency follow-up

- Installed `triton==3.6.0` from the Huawei PyPI mirror. This is available for the PyTorch 2.9.1 environment, but `npugraphs` still fails during the small-module test with `RuntimeError: 0 active drivers ([]). There should only be one.`
- Built and installed the public TorchAir master Python package in editable mode from `Ascend/torchair`; no public PyPI wheel exists. The official compatibility table does not list the current combination (PyTorch/torch_npu 2.9.1 with CANN 8.0.RC3). The `npu` backend still fails in the TorchAir import/compiler path.
- `npugraph_ex` continues to fail with `Failed to find function aclmdlRICaptureBegin`; symbol lookup in the installed toolkit/driver shared libraries finds no exported symbol. This indicates the installed runtime/driver predates that capture API.
- Conclusion: graph/compiler dependencies cannot be made version-correct from public packages for this exact CANN stack. Preserve the validated 8-slot eager result and proceed to Phase 3 general optimization; do not claim Phase 1 graph speedups without a successful small-module correctness test.

## 2026-10-09: strict Ascend/H100 baseline comparison

- Completed the requested full Ascend serial run: 127 tasks, task 423 excluded, max 512 new tokens, max 6 turns.
- Ascend 910B3: **3466.21 s**, accuracy 0.9449, decode 165.94 ms/token.
- H100 adjusted baseline: **1470.6 s**, accuracy 0.937, decode 71.84 ms/token.
- Direct ratio: Ascend is **2.36x slower** in makespan and **2.31x slower** per decode token. This is a measured comparison on the same adjusted workload; no extrapolation is used.

## 2026-10-09: Phase 1 Ascend L2/L3 profile

- Fixed the L1 timer to synchronize `torch.npu` (the previous implementation only recognized CUDA).
- Added `bench/profile_npu.py` using `torch_npu.profiler` NPU activity and CANN `msprof` trace export.
- L3 operator evidence: MatMulV2 6.595 s/67,095 calls; convolution layout transform 1.559 s/6,480; Mul 0.998 s/160,302; ReduceSum 0.348 s/34,704; Cast/copy/slice about 0.9 s combined; FlashAttention only 0.091 s.
- Amdahl choice: FlashAttention is not the first candidate. GEMM plus GDN/conv layout and elementwise launch overhead dominate. A 1.2x MatMul improvement would yield only about 1.08x local speedup at the measured `p`≈0.65, so low-effort existing fast paths and static/cache launch changes must be tested before any Phase 3 kernel work.

## 2026-10-09: Phase 1 StaticCache

- One change: replaced per-request DynamicCache with the existing Transformers `StaticCache` (bf16, eager NPU, batch=1).
- 8-task validation: 197.86 s → 174.52 s (**11.8% faster**), decode 186.33 → 166.69 ms/token, accuracy unchanged at 0.500 under the 128-token probe cap.
- Amdahl estimate: 1.1x local decode improvement with p=0.94 predicted 1.089x overall; measured 1.134x. The extra gain is consistent with removing cache allocation/copy overhead in prefill.
- Decision: keep StaticCache and run the next fixed 16-task validation before applying further changes.

## 2026-10-09: StaticCache 16-task validation

- Same fixed 16-task configuration as the serial baseline.
- Result: 473.66 s → 422.44 s (**10.8% faster**), accuracy unchanged at 0.9375; decode 191.84 → 175.14 ms/token.
- This confirms the 8-task result and keeps StaticCache as the Phase 1 base for the next candidate. The gain is below the H100 CUDA-graph gains but is achieved without precision changes or custom kernels.

## 2026-10-09: asynchronous token readback attempt (failed)

- Candidate: with StaticCache, read token outputs in chunks of 8 to reduce per-token host synchronization.
- Same 16-task validation: 1214.45 s, accuracy 0.0, 61 turns, 6,957 generated tokens.
- The candidate changed cache/token feedback semantics and diverged badly from the 31-turn reference. It was reverted; no performance result is retained.
- Decision: do not trade semantic correctness for host-sync reduction. Future async readback must use a graph-safe fixed-token feedback mechanism and pass token-level equivalence before timing.

## 2026-10-09: Phase 1 internal-format and MatMul layout attempts

- `torch.npu.config.allow_internal_format=True`: rejected because GDN's Conv2D fallback fails CANN 8.0.RC3 op selection (`Conv2D is not found in this op store`).
- `set_mm_bmm_format_nd(False)` (NZ layout): same 8-task trajectory as StaticCache, but 200.00 s vs 174.52 s (14.6% slower); rejected.
- `set_mm_bmm_format_nd(True)` matched the installed default and is not a valid code change; it is excluded from speedup claims.
- Phase 1 status: StaticCache remains +10.8% on the 16-task validation. Phase 1 is not at a hardware ceiling; current remaining options depend on an Ascend-compatible GDN fast path or a matched graph compiler/runtime stack. No custom kernels have been introduced.

## 2026-10-09: Phase 2 live-row compaction attempt (failed)

- Profile evidence: the 8-slot trace had 17.4% long gaps and 31.1% computed-row padding; average live rows were 5.51/8. Amdahl selected bucketed compaction as the next candidate.
- Implementation: existing slot-cache bucket/row compaction, no custom kernel.
- Runtime validation failed in the GDN fallback conv update because the conv state batch stayed 8 while the compact input batch was 2 (`aclnnCat` shape mismatch).
- Decision: reject this implementation without a speed claim; preserve the 8-slot baseline and fix cache state view semantics before retrying.

## 2026-10-09: Phase 2 compaction retry

- Fix: synchronize `SlotCache.set_rows(B)` with each active bucket before the model step, correcting the prior GDN state shape mismatch.
- Correctness: 16-task accuracy 0.9375, 31 turns, 2,327 generated tokens, matching the fixed 8-slot trajectory.
- Performance: fixed 8 rows 121.65 s; bucket compaction 126.35 s (3.9% slower). Computed rows dropped 3376 -> 2486, but 25 row swaps and bucket/prefill overhead dominated.
- Amdahl decision: reject compaction as an end-to-end optimization despite lower padding; retain only the state-view correctness fix.

## 2026-10-09: Phase 2 16-slot scan

- Candidate: increase fixed eager batch from 8 to 16 slots after compaction showed row padding but excessive swap overhead.
- Result: 121.65 s -> 87.05 s (**1.40x**), accuracy unchanged at 0.9375; decode 40.64 -> 25.44 ms/token.
- Amdahl estimate: 1.3x local decode improvement with p=0.767 predicted 1.19x; measured gain was 1.40x due to better critical-path overlap.
- Decision: retain 16 slots and test 32 slots next, with HBM usage as a guardrail.

## 2026-10-09: Phase 2 32-slot scan

- 32-slot fixed eager batching initialized an 8.6 GiB cache but failed at the first workload step with NPU OOM: 59.66 GiB allocated of 60.96 GiB, only 5.30 MiB free.
- Decision: reject 32 slots for bf16; 16 slots is the largest tested stable concurrency. Further concurrency requires memory reduction (Phase 3 FP8 per-tensor weights or KV/state changes), not a larger bf16 slot count.

## 2026-10-09: Phase 2 full 127-task confirmation at 16 slots

- Full workload result: 656.37 s makespan, accuracy 0.9449, 253 turns and 126 tool calls.
- Compared with 8-slot final (907.54 s), this is **1.38x faster**; compared with strict serial baseline (3466.21 s), **5.29x faster**.
- Decode is 17.58 ms/token and only 52.3% of makespan; eager prefill is now 47.5% and is the next Amdahl target.
- 32-slot bf16 was rejected by OOM, establishing the memory boundary. Phase 2 next candidate is prefill reuse/forced-token scheduling, subject to cache-state correctness.

## 2026-10-09: Phase 3 FP8 capability probe

- Attempted the H100-style per-tensor FP8 path using torch-npu tensors and `torch._scaled_mm`.
- The initial bf16 -> `float8_e4m3fn` cast fails in CANN 8.0.RC3: dtype 36/UNDEFINED is not in the supported dtype list. No performance or accuracy result is claimed.
- Decision: defer FP8 until an Ascend quantization API or a general Phase 3 kernel path is available; retain the measured 16-slot bf16 result.

## 2026-10-09: Phase 3 quantized matmul API probe

- Tested `torch_npu.npu_quant_matmul` with int8 activations/weights and bf16 output.
- CANN 8.0.RC3 rejects A8W8: `only aclnnQuantMatmulV5 support A8W4, please upgrade CANN`.
- This confirms the installed runtime cannot provide the required per-tensor linear quantization path. Retain the 16-slot bf16 result; any further Phase 3 progress requires a general Ascend C kernel or a CANN upgrade.

## 2026-10-09: Phase 3 weight-only quantization capability

- Tested the existing Ascend `npu_weight_quant_batchmatmul` operator with bf16 activations, int8 weights, per-output-channel bf16 antiquant scales, and bf16 output.
- Probe succeeded for `(4,512) x (512,1024)`, returning bf16 output. This is the first usable quantization path on the installed CANN stack and may reduce weight memory enough to test >16 slots.
- No model-wide replacement or speed claim yet; next step requires deterministic weight conversion, layer-level reference error checks, then e2e accuracy and profile.

## 2026-10-09: weight-only quant layer reference

- Per-output-channel symmetric int8 quantization was compared against bf16 Linear using `npu_weight_quant_batchmatmul`.
- M=8 result: max abs error 0.75, mean abs 0.134, mean relative 6.35%; quant 0.062 ms versus bf16 0.024 ms.
- Amdahl decision: do not replace model layers for batch-1 latency. Continue only as a memory/concurrency experiment at larger M, with full accuracy validation required.
- M=32 follow-up: quant 0.079 ms versus bf16 0.029 ms (2.7x slower), mean abs error 0.135. Reject as a latency optimization; do not modify model weights without a separate memory/accuracy study.

## 2026-10-09: Ascend C kernel build probe

- Obtained the official `Ascend/samples/operator` AddCustom sample and invoked its `install.sh -v Ascend910B3` against CANN 8.0.RC3.
- Build failed before compilation: `/usr/local/python3.9.2/bin/python3.9` cannot load `libpython3.9.so.1.0`; CANN `opp/*/custom` directories also deny user access, so no `CustomOp` package was generated.
- Decision: do not claim a Phase 3 kernel result. A root-admin Python/runtime and CANN custom-op permission fix is required before compiling a general Ascend C kernel.

## 2026-10-09: Ascend C build retry

- Added `/usr/local/python3.9.2/lib` to `LD_LIBRARY_PATH`; msOpGen generated the CustomOp project.
- Corrected the generated CMake preset from nonexistent `/usr/local/Ascend/latest` to the actual CANN 8.0.RC3 aarch64 path.
- CMake now starts, but opbuild does not generate `build_out/autogen/op_proto.cc`, so host compilation cannot proceed. No kernel package or benchmark result exists yet.

## 2026-10-09: Ascend C AddCustom package built

- Corrected `ASCEND_CANN_PACKAGE_PATH` to the CANN root so `toolkit/tools/opbuild/op_build` is found; headers remain available through the root `include` symlink.
- Official AddCustom Ascend C sample compiled for `ascend910b` and produced/verified `custom_opp_ubuntu_aarch64.run`.
- Package deployment failed only because default OPP `vendors` is root-owned. No benchmark claim yet; next step is user-writable custom OPP deployment and a standalone Add correctness run.

## 2026-10-09: Ascend C AddCustom runtime gate

- Deployed the generated package into user-writable `custom_opp` and compiled the official aclnn invocation sample against its headers/library.
- Runtime call reached `aclnnAddCustomGetWorkspaceSize` but returned status `161001`; the Add sample did not pass its correctness gate.
- The status maps to `ACLNN_ERR_PARAM_NULLPTR`; package files and exported API symbols are present, so the remaining issue is runtime/operator registration or the sample API contract, not compilation.
- Diagnostic rebuild showed all `aclCreateTensor` calls return non-null pointers/status 0. The error changes between 361001 and 161001 depending on custom OPP environment variables, confirming registration/environment mismatch rather than null tensor creation.

## 2026-10-10: custom-op runtime corrected

- Root installed the generated package into the standard CANN OPP path and granted `davinci` read/traverse permissions for CANN OPP metadata and custom directories.
- Standalone `aclnnAddCustom` correctness now passes: sampled outputs are `3.0`; the program prints `test pass`.
- The custom Ascend C pipeline is now unblocked for Phase 3 kernel microbenchmarks. The Add sample is a generic correctness test, not yet an LLM hotspot optimization.

## 2026-10-09: Phase 3 NPU critical-path trace

- Analyzed the strict 127-task 16-slot trace with `bench/analyze_steps.py`.
- `engine.step()` accounts for 343.11 s (52.3%), while gaps over 50 ms account for 299.06 s (45.6%). Average step is 245.08 ms, live-row fill 88.3%, padding 11.7%.
- Amdahl decision: prioritize prefill/tool/state-transition gaps and an NPU-aware scheduling simulator. Decode kernel micro-optimizations alone cannot address the 45.6% gap fraction.
- The CUDA-oriented `sim_steps.py` cannot consume this NPU eager run because no CUDA bucket profile exists; this is recorded as a tooling gap rather than using an invalid simulation.

## 2026-10-09: NPU-aware simulator and 24-slot attempt

- Added and corrected `bench/sim_npu_steps.py` to put prefill before decode, model async tool waits without blocking other rows, and include EOS step counts. Mean measured step costs are 180.18/221.07/245.08 ms for buckets 4/8/16.
- Calibration: 8-slot prediction 873.57 s vs 907.54 s measured (-3.74%); 16-slot prediction 644.97 s vs 656.37 s measured (-1.74%). Both are within the PLAN's 5% simulation tolerance.
- The simulator predicts 565.08 s at 24 slots. A real 24-slot attempt failed during GE initialization; logs later showed user processes cannot read `opp/scene.info` (root:root 0640) and cannot traverse root-only `opp/vendors`/custom directories. This runtime permission issue must be fixed before retesting 24 slots.
- Simulator evaluated longest-chain-first admission using measured per-task prompt+generation lengths: predicted 930.18 s at 8 slots, 691.33 s at 16, and 612.92 s at 24, all worse than file-order predictions. The candidate was rejected before implementation by the simulate-first gate; an earlier unverified implementation stub was removed.
- Decision: retain 16 slots as the largest stable measured configuration. The simulator and failure evidence are recorded; 24-slot remains unverified.

## 2026-10-09: final system-level comparison

- Designated the validated 16-slot full workload (`656.37 s`, accuracy `0.9449`) as the Ascend Phase 3 system final. It is the same measured configuration as the Phase 2 endpoint because subsequent system candidates (compaction, 24 slots) provided no safe improvement.
- Added `results/FINAL_COMPARISON_REPORT.md` comparing strict H100 and Ascend curves and separating the system track from the blocked custom-kernel track.
- Tool-worker candidate rejected by Amdahl: strict serial profiling measured only 5.10 s of tool time in a 3466.21 s run (~0.15%), and the 16-slot run reports ~0.1% tool share. Even infinite tool speedup cannot materially reduce makespan; no implementation was made.

## 2026-10-09: Phase 3 longest-chain admission candidate

- Candidate: prioritize admissions by historical per-task prompt+generation length, based on the observed critical-path/gap share.
- The corrected NPU simulator predicts this policy is worse than file order at every modeled slot count (16-slot: 691.33 s vs 644.97 s, +7.2%).
- Decision: reject before implementation/benchmark per the simulate-first gate. The earlier unverified implementation stub was removed; its GE startup failure was from the environment permission issue and is not used as candidate evidence.

## 2026-10-09: Ascend long-context compatibility probe

- Attempted the Phase 2 16-slot engine on `gsm8k_longctx_128`.
- The workload's first prompt exceeds the engine's fixed `max_len=2048` cache and aborts before timing (`prompt exceeds max_len`).
- Decision: do not fabricate a longctx comparison. A future longctx implementation requires paged/larger KV allocation and an explicit HBM budget; this is a Phase 3 systems/kernel candidate.

## 2026-10-09: longctx DynamicCache fallback probe

- Ran two longctx tasks with the serial DynamicCache engine (max 64 tokens, 2 turns) to separate capacity from model/runtime failure.
- Result: 30.62 s, 8,500 prompt tokens, prefill 1,039 token/s, decode 177.84 ms/token. Accuracy is intentionally invalid (0.0) under the short probe cap.
- This confirms the workload itself runs; the blocker is the fixed 2048-token Phase 2 SlotCache. Paged/larger KV/state allocation remains a Phase 3 systems candidate.

## 2026-10-09: existing FLA/causal-conv fast-path attempt

- Installed pure-Python `flash-linear-attention==0.5.2`/`fla-core==0.5.2` from the Huawei mirror.
- `causal-conv1d==1.7.0` has no aarch64 wheel; its source build requires CUDA/nvcc and fails. A direct import probe confirms `causal_conv1d` is unavailable, so the Qwen fast path cannot be constructed.
- Decision: reject this candidate for the current Ascend package/runtime. This is an evidence-based dependency limitation, not a conclusion that Phase 1 has no theoretical room.

## 2026-10-09: Phase 3 prefix/state reuse and forced suffix

- Fixed GDN fallback prefix extension: when `causal_conv1d_fn=None`, HF bypassed the function wrapper; extension now wraps the existing Conv1d module and preserves cached K-1 input history. CPU reference checks: GDN output/state max abs error 0; Conv1d suffix max abs error 1.49e-8.
- NPU `check_batch.py`: extend-vs-full 8/8 identical; shared-prefix 7/8 identical (one first-turn divergence); alone-vs-concurrent 8/8 identical.
- 16-task prefix reuse: 84.87 s vs 87.05 s baseline; accuracy unchanged at 0.9375.
- Strict 127-task prefix reuse + forced suffix: 556.76 s, accuracy 0.9449, vs Phase 2 656.37 s (15.2% better). Prefill 7.08 s; decode 548.18 s (98.5%). Prefix-only full run was 559.16 s, so forced suffix contributes ~0.4% incremental gain.
- Amdahl interpretation: prefix/state reuse is the material system gain; forced suffix mostly moves eager prefill work into batched decode at 16 slots. Next bottleneck is decode/step cost.

## 2026-10-09: correction to interim Phase 3 label

- An earlier report incorrectly called the 656.37 s Phase 2 16-slot result the Phase 3 final. That was not a new Phase 3 optimization.
- The actual Phase 3 system result is the measured 16-slot prefix/state reuse + forced-suffix run at 556.76 s, accuracy 0.9449. Prefix-only measured 559.16 s; forced suffix adds only 0.4% and is below the 5% step threshold. The combined Phase 3 gain over Phase 2 is 15.2%.
- `results/FINAL_COMPARISON_REPORT.md` and `results/ASCEND_FINAL_REPORT.md` now use the measured Phase 3 value and explain the attribution.

## 2026-10-09: Prefix/recurrent continuation reference repair

- HF bypassed the causal-conv wrapper when `causal_conv1d_fn=None`, directly invoking `nn.Conv1d` and dropping cached history. Added an extension-mode Conv1d wrapper and corrected its suffix crop to `[K-1:K-1+n]`.
- CPU reference validation: GDN full vs prefix+initial_state output/state max abs error 0; causal Conv1d suffix max abs error 1.49e-8.
- NPU token-level check is pending because CANN OPP metadata/directory permissions currently prevent ordinary users from initializing GE. Do not claim the optimization until `bench/check_batch.py` passes on NPU.

## 2026-10-10: NPU eager batch initialization cleanup

- The NPU eager path now disables the CUDA-only `torch.compile` setup even when the shared engine default is `compile_decode=True`.
- Initialization warmup is bounded by the configured slot count (`min(16, num_slots)`) instead of probing 16 rows unconditionally. This avoids touching inactive rows and makes 8-slot probes use only their allocated cache.
- Validation: Python bytecode compilation and `git diff --check` pass. Hardware e2e re-profile remains required on the Ascend node.

## 2026-10-10: Phase 3 layer-0 Ascend-C integration

- Integrated the verified GDN/Conv Ascend-C ACLNN bridge into Qwen3.5 linear-attention layer 0 decode only. Prefill and other layers remain on PyTorch/CANN.
- Real layer token check at C=10240: output max abs 0.00390625, state max abs 0; baseline and replacement first greedy tokens both `[271, 16]`.
- 8-task short probe: baseline 48.118 s, layer-0 49.197 s (2.24% slower); both use a 64-token/short probe with accuracy 0.0, so this is a timing/equivalence probe rather than a quality result.
- 16-task layer-0 probe: 111.368 s, but the matching 16-task baseline did not produce metrics after repeated independent attempts; no speedup claim is made.
- Blocking gate: fair 16-task/127-task replay A/B, L3 profile, full accuracy, and long-context checks require a clean baseline run. Existing NPU OOM/runtime termination is recorded; the custom kernel is not promoted as an optimization until the paired baseline exists.

## 2026-10-12: Phase 3 layer-0 paired performance gate

- Clean paired scripts now capture stdout/stderr/exit code and environment snapshots. Both baseline and Ascend-C exit code 0.
- 16-task paired result: baseline 440.812 s, Ascend-C layer-0 462.559 s; both accuracy 0.9375, 31 turns, 2,329 generated tokens.
- Ascend-C regression: 4.93% end-to-end and 4.99% decode. Task 214 standalone answer correctness is 1.0 in both paths.
- Amdahl/performance decision: reject this single-layer kernel integration for the current model path; its regression is at the 5% exit threshold, so no 127-task replay or L3 claim is justified. Retain the verified kernel and bridge as infrastructure for a future fused multi-layer/state boundary.
# 2026-10-10: Phase 3 kernel candidate selected from L3 profile

- AddCustom remains a compiler/deployment/runtime measurement smoke test only;
  it is excluded from LLM speedup claims.
- Selected a general BF16/FP16 GDN/Conv state-update candidate combining the
  profiled TransData, causal Conv state update, gate/state Mul, and recurrent
  reduction work. Evidence and acceptance gates are recorded in
  `results/p3_gdn_conv_candidate/README.md`.
- MatMulV2 remains on the CANN path for this iteration because its vendor GEMM
  implementation is already highly optimized and a custom replacement has a
  high regression risk.

# 2026-10-12: Phase 1 StaticCache strict 127-task validation

- Configuration: `p1_npu_static`, serial runner, `gsm8k_128` with task 423
  excluded, `max_new_tokens=512`, `max_turns=6`, Ascend 910B3.
- Amdahl hypothesis: preallocated KV storage should remove DynamicCache growth
  and reduce cache-management overhead; the representative 16-task probe
  suggested approximately 10.8% makespan improvement.
- Full measured result: makespan 3552.287 s, accuracy 0.9449, 253 turns,
  19,650 generated tokens, 169.689 ms/token decode, 492.859 tok/s prefill.
- Strict DynamicCache baseline is 3466.21 s, so the full-run result is 2.48%
  slower. The 16-task gain does not generalize to the 127-task workload;
  StaticCache is retained as a tested candidate but is not claimed as a
  reliable end-to-end Phase 1 improvement.

# 2026-10-12: p1_npu_reuse_input probe

- Single change: reuse a fixed `(1, 1)` NPU input tensor for decode instead of
  allocating a device tensor on every token.
- Two-task probe, task 423 excluded, max 64 new tokens, serial eager NPU:
  baseline 22.044 s / 161.41 ms per token; candidate 22.732 s / 166.72 ms
  per token. The 128-token trajectory was identical.
- Allocation was not a positive fraction of the decode path. The candidate
  regressed 3.1% and was REJECTED before 16/127-task expansion.

# 2026-10-12: Phase 1 plan execution audit

- Re-ran the documented CLI discovery under `source env_ascend.sh` and
  `conda run -n ascend-e2e`; `run_bench.py`, `profile_npu.py`,
  `check_equiv.py`, and `check_batch.py` all expose the expected interfaces.
- The environment is usable on Ascend 910B3 (`davinci` membership and readable
  CANN `opp/scene.info`); the toolkit emits only an owner-mismatch warning.
- Static validation passed with `python -m compileall -q agentopt bench`.
  `git diff --check` reports only the pre-existing blank line at EOF in this
  log, with no whitespace errors in source files.
- The strict 127-task DynamicCache and StaticCache artifacts remain the
  authoritative Phase 1 comparison: accuracy 0.9449, 253 turns, and 126 tool
  calls match; StaticCache is 3552.287 s versus 3466.208 s (2.48% slower).
- Decision: REJECT StaticCache as a Phase 1 success candidate. The tested
  compile/backend, internal-format, NZ-layout, and async-readback candidates
  are also rejected or blocked in their recorded result directories. No
  candidate meets the plan's stable >=5% single-request and non-regression
  gates, so this software stack has no qualifying Phase 1 end-to-end gain.

