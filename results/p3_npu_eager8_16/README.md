# Phase 3 NPU system-level scheduling (8 slots)

- Change: increase existing eager batch scheduler from 4 to 8 concurrent slots; no kernel or operator changes.
- Same 16-task workload as the prior validation, max 256 new tokens and 6 turns.
- Makespan: **121.65 s**; accuracy **0.9375**; 31 turns and 15 tool calls.
- Decode: 93.32 s (76.7%); 40.64 ms/token.
- Prefill: 28.17 s (23.2%); 531.12 token/s.
- Compared with 4 slots: 152.12 to 121.65 s (**1.25x**), decode 53.38 to 40.64 ms/token; accuracy unchanged.

The higher slot count improves decode throughput, while prefill becomes the larger fraction. This is the current Phase 3 system-level result under the no-custom-kernel constraint.
