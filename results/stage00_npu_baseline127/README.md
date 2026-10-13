# Ascend 910B3 strict serial baseline (127 tasks)

- Same adjusted workload as the H100 baseline: `gsm8k_128`, task 423 excluded, 127 tasks.
- Serial HF/Transformers runner, bf16, max 512 new tokens, max 6 turns.
- Makespan: **3466.21 s** (57.77 min).
- Accuracy: **0.9449**; 253 turns; 126 tool calls.
- Decode: 3191.45 s (92.1%), 165.94 ms/token.
- Prefill: 268.63 s (7.7%), 468.26 token/s.

## Direct H100 baseline comparison

The H100 adjusted baseline is 1470.6 s on the same 127-task workload. Ascend is therefore `3466.21 / 1470.6 = 2.36x` slower in makespan. Decode is the dominant difference: Ascend 165.94 ms/token versus the H100 baseline's 71.84 ms/token (2.31x slower).
