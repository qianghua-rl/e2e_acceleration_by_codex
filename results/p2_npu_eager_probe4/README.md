# Phase 2 NPU eager continuous batching probe

- Change: reuse the existing slot-cache and continuous runner, while replacing CUDA Graph/Event execution with eager `torch_npu` execution. No custom kernel.
- Configuration: 4 slots, 4 GSM8K tasks, max 128 new tokens, max 6 turns, asynchronous tools.
- Makespan: **38.69 s**; accuracy **0.75**; 515 generated tokens.
- Decode: 31.54 s (81.5%); 62.09 ms/token.
- Prefill: 7.10 s (18.4%); 448.09 token/s.
- Engine steps: 167; live-row steps 514; computed-row steps 668.

## Amdahl estimate and result

The serial 8-task probe measured 185.7 ms/token. Assuming the 4-row batch reduces weight-read cost by 3x, the decode fraction (0.94) predicts roughly `1/(0.06 + 0.94/3) = 2.54x` speedup. The observed 62.1 ms/token is 2.99x faster per token; makespan is workload-dependent because task lengths differ. Accuracy remains a probe value because of the 128-token cap.
