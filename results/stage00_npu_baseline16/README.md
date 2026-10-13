# Stage 0 Ascend baseline (16-task representative workload)

- Hardware: one Ascend 910B3 (device 1), CANN 8.0.RC3.
- Software: Python 3.12, PyTorch 2.9.1, torch-npu 2.9.1, Transformers 5.5.4.
- Model: local Qwen3.5 checkpoint, bf16, 27,356,728,560 parameters.
- Workload: 16 GSM8K tasks (task 423 excluded), serial runner, max 256 new tokens and 6 turns.
- Makespan: **473.66 s**; accuracy **0.9375**; 31 turns; 15 tool calls.
- Decode: **440.85 s (93.1%)**, 191.84 ms/token.
- Prefill: 32.31 s (6.8%), 463.02 token/s.
- Tools: 0.31 s (0.1%); redundant prefill tokens: 7,995.

## Amdahl selection

The measured decode fraction is `p=0.931`. If an upper-layer change makes decode 1.5x faster, Amdahl predicts `1/(1-0.931+0.931/1.5)=1.399x` whole-run speedup. Phase 1 therefore starts with profiling decode dispatch, synchronization, and cache behavior. No custom kernels are used.

This is the reproducible representative baseline while the full 127-task run is deferred until the per-step profile and optimized runner are stable.
