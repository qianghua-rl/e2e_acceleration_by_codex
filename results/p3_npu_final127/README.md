# Final Ascend 910B run (127-task workload)

- Hardware: one Ascend 910B3, 8-slot eager continuous batching.
- Workload: 127 GSM8K tasks (task 423 excluded), max 512 new tokens, max 6 turns.
- No custom kernels; CUDA Graph, Triton, FP8 and CUDA-only paths are disabled.
- Makespan: **907.54 s** (15.13 min); accuracy **0.9370** (119/127 equivalent answer rate); 252 turns and 125 tool calls.
- Decode: 607.04 s (66.9%); 31.03 ms/token.
- Prefill: 299.09 s (33.0%); 418.41 token/s.
- Generated tokens: 19,812; end-to-end 21.83 token/s.
- Finish reasons: 124 answers, 3 no-answer cases.

This is the final measured result for the no-custom-kernel Ascend migration. It is directly comparable with the repository's fixed 127-task workload, but hardware and implementation differ from the original H100/CUDA experiment.
