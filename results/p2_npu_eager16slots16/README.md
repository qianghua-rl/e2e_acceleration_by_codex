# Phase 2 16-slot fixed batching

- Single change from the validated 8-slot engine: fixed concurrent slots 8 -> 16.
- Same 16-task workload, task 423 excluded, max 256 tokens, six turns.
- 8 slots: 121.65 s, accuracy 0.9375, decode 40.64 ms/token.
- 16 slots: **87.05 s**, accuracy **0.9375**, decode **25.44 ms/token**.
- Speedup: **1.40x** makespan over 8 slots; 31 turns, 15 tool calls.
- Computed row steps: 4080 for 2376 live row steps (padding rises), but weight-read amortization dominates.

## Amdahl estimate

The 8-slot trace spent 76.7% in decode steps. A conservative 1.3x local decode improvement predicts `1/(0.233+0.767/1.3)=1.19x`; measured makespan gain is 1.40x because 16 slots also reduce the critical-path tail and overlap more requests.
