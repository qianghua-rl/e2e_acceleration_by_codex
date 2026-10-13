# Phase 2 final 16-slot confirmation

- Workload: full 127 tasks, task 423 excluded, max 512 tokens, six turns.
- 8-slot final: 907.54 s, accuracy 0.9370.
- 16-slot eager: **656.37 s**, accuracy **0.9449**.
- Speedup: **1.38x** over 8 slots and **5.29x** over strict serial baseline 3466.21 s.
- Decode: 17.58 ms/token; prefill is now 47.5% of makespan, so the bottleneck has shifted from decode to eager per-turn prefill.

The 16-slot result is the current Phase 2 best under bf16/no-custom-kernel constraints. 32 slots failed with NPU OOM.
