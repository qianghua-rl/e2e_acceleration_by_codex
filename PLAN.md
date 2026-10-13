# PLAN: profile-driven optimization of an LLM agent on Ascend 910B3

The H100/Claude experiment is a reference for measurement discipline and useful ideas, not a requirement to reproduce CUDA implementation details. The Ascend plan selects the next change from profiles of the actual NPU stack and keeps the same agent workload, accuracy guardrail, and Amdahl loop.

## Scenario (fixed for all stages)
- Offline batch math agent: 128 fixed GSM8K test questions (`data/gsm8k_128.jsonl`, seed 0), all available at t=0.
- ReAct-style agent with Qwen native tool-call format, one `python` tool (stdout returned); 27B uses the tool on ~97% of tasks unprompted.
- Model: Qwen3.5-27B (hybrid: 48 Gated-DeltaNet linear-attn layers + 16 full-attn GQA layers, 248k vocab), bf16 ≈ 54 GB.
  (Switched from Qwen3.5-2B: it rarely used tools and fell into greedy repetition loops.)
- Non-thinking mode, greedy decoding, ≤6 turns/task, ≤512 new tokens/turn.
- Hardware: 1×Ascend 910B3 (64 GB HBM), CANN 8.0.RC3, `torch_npu`.

## Two benchmarks
| | Microbench (`bench/microbench.py`, ~5 min) | End-to-end (`bench/run_bench.py`) |
|---|---|---|
| measures | single-request prefill ms @ ctx 512/1024/2048; decode ms/token (cold + warm); % of HBM roofline | makespan of the 128-task agent workload; accuracy; tok/s; latency p50/p90 |
| role | fast inner loop, local speedup `s` | ground truth, overall speedup |

Guardrail: accuracy within 2 points of baseline; output-preserving changes should also keep greedy outputs (near-)identical.

The 8/16-task runs are iteration probes with reduced token limits. Final platform comparisons must use the strict 127-task workload (task 423 excluded, max 512 new tokens, max 6 turns); their makespans must not be compared directly with probe makespans.

## Iteration loop (all phases)
1. profile: L1 section timers, L2 PyTorch/torch-npu profiler, L3 CANN/Ascend profiling where available.
2. Amdahl: for each candidate estimate fraction `p` and local speedup `s` → `1/((1-p)+p/s)`; pick best gain/effort.
3. Implement ONE change → microbench (phase 1) / e2e bench → record in `results/<stage>/` + `OPTIMIZATION_LOG.md` → git tag.

Stage naming: `stage00_baseline` (shared start), `p1_sNN_<name>` (phase 1), `p2_sNN_<name>` (phase 2).

## Phase 1 — single-request inference engine (batch = 1; no custom kernels)
Goal: make one generate() call as fast as possible. Measured mainly by the microbench; e2e run at phase end.
Physical ceiling: decode must stream all weights per token → 54 GB / 3.35 TB/s ≈ **16 ms/token**.
Candidate directions (order decided by profiles, not this list): CANN-provided operators, static/preallocated cache, `torch.compile` with the Ascend `npu` backend, `torch.npu.NPUGraph`/`npugraphs`, launch configuration, and host-sync reduction. No new Triton, CUDA, or Ascend C kernel is allowed in this phase.
Exit: decode ≥ ~75–80% of roofline (~20–21 ms/token), or 3 stages <5% gain, or only kernel work remains.

## Phase 2 — multi-agent system (serving + agent layers; no custom kernels)
Goal: minimize makespan of the 128-task workload. Measured by the e2e bench.
Expected ceiling of phase 1 alone is ~3× (58 → ~20 ms/token); larger wins need the GPU to serve many agents at once.
Candidate directions:
- batching across agents (static → continuous batching), within the ~25 GB left for KV + recurrent state
- cross-turn cache reuse (append-only history; hybrid model keeps KV *and* recurrent state) and shared-prefix reuse
- memory headroom (e.g. FP8 weights) to allow larger batches
- tool execution: persistent worker pool, overlap tool calls with GPU generation
- reference ceiling: vLLM on the same workload
Exit: 3 stages <5% gain, or close to the vLLM reference, or only kernel work remains.

## Phase 3 — hardware-specific general optimization
Custom kernels are allowed only here and must be general (no task IDs, fixed prompts, or workload-specific branches). Precision is fixed to FP8 per-tensor linear layers where used; no MTP or speculative decoding. Candidates include general Ascend C/Triton kernels, FP8 GEMM/epilogue integration, persistent scheduling, and fused attention/state operations. Every kernel requires numerical tests, long-context accuracy, and an e2e re-profile.

## Scope and comparison
H100 results are a reference ceiling, not a directly comparable implementation: they use CUDA Graphs, FP8/cuBLASLt, Hopper-specific kernels and PDL. The Ascend report must show both absolute makespan and the speedup curve within each platform, and identify which H100 ideas were replicated, adapted, or rejected with evidence.
Final deliverable: `results/ASCEND_FINAL_REPORT.md` plus the chronological `OPTIMIZATION_LOG.md`.
