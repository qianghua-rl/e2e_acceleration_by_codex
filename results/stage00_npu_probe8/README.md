# Stage 0 Ascend baseline probe (8 tasks)

- Hardware: one Ascend 910B3 (device 1), bf16.
- Model: local Qwen3.5 checkpoint (`27,356,728,560` parameters).
- Runner: serial, 8 GSM8K tasks, max 128 new tokens, max 6 turns.
- Makespan: **197.86 s**; accuracy **0.500**.
- Decode: 185.95 s (94.0%); prefill: 11.76 s (5.9%); tools: 0.08 s.
- Generated tokens: 1,010; decode 186.33 ms/token; prefill 456.14 token/s.
- Tool calls: 4; redundant prefill tokens: 1,954.

The probe establishes a valid agent workload and shows decode as the dominant cost. The 0.500 accuracy is a probe result with a 128-token cap, not the final baseline.

## Amdahl choice

Using the measured decode fraction `p=0.940`, a hypothetical 1.5x decode improvement has an upper-bound whole-run speedup of `1 / (0.06 + 0.94/1.5) = 1.415x`. Decode scheduling/launch overhead is therefore the first Phase 1 profiling target.
