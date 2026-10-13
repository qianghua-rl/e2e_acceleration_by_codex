# Phase 1 StaticCache validation

- Single change: use Transformers' existing `StaticCache` for each batch-1 request; no graph capture, precision change, or custom kernel.
- Workload: 8 tasks, task 423 excluded, max 128 new tokens, max 6 turns.
- DynamicCache reference (`stage00_npu_probe8`): 197.86 s, accuracy 0.500, 186.33 ms/token.
- StaticCache: **174.52 s**, accuracy **0.500**, 166.69 ms/token.
- Measured improvement: **11.8% makespan**, 10.5% decode latency; prefill throughput improved from 456.14 to 522.21 token/s.

The generation trajectory is close (996 vs 1,010 generated tokens) and the accuracy guardrail is unchanged. This validates StaticCache as a real Phase 1 optimization candidate; a same-trajectory replay should be added before claiming the exact system-only gain.

## Amdahl estimate

The baseline decode fraction was 94.0%; a 1.1x decode improvement predicts `1/(0.06+0.94/1.1)=1.089x`. The measured 1.134x makespan gain is above this estimate, likely because StaticCache also reduces prefill allocation/copy overhead.
