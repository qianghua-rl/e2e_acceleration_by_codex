# Phase 1 StaticCache validation (16 tasks)

- Same workload as `stage00_npu_baseline16`: 16 tasks, task 423 excluded, max 256 new tokens, max 6 turns.
- DynamicCache baseline: 473.66 s, accuracy 0.9375, decode 191.84 ms/token.
- StaticCache: **422.44 s**, accuracy **0.9375**, decode 175.14 ms/token.
- Improvement: **10.8% makespan**, 8.7% decode latency; prefill throughput 524.22 token/s.

StaticCache is retained as the first Phase 1 optimization. The accuracy and answer/turn profile remain equivalent at this validation size.
