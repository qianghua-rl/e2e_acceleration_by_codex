# Phase 2 NPU eager continuous batching (16-task validation)

- Same 16-task workload as `stage00_npu_baseline16`: task 423 excluded, max 256 new tokens, max 6 turns.
- Four concurrent slots, existing slot cache and continuous asynchronous runner.
- Makespan: **152.12 s**; accuracy **0.9375**; 31 turns and 15 tool calls.
- Decode: 122.57 s (80.6%); 53.38 ms/token.
- Prefill: 29.36 s (19.3%); 509.65 token/s.
- Compared with serial baseline: 473.66 s to 152.12 s (**3.11x makespan speedup**), with identical accuracy; decode latency 191.84 to 53.38 ms/token (3.59x).

The result validates that the gain is not limited to the 4-task probe. CUDA Graphs and custom kernels remain disabled; this is an upper-layer batching/scheduling change using existing torch-npu operators.
