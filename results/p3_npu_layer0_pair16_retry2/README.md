# Phase 3 layer-0 Ascend-C paired result

- Same 16-task configuration for both processes: task 423 excluded, max 256 new tokens, max 6 turns, serial runner.
- Baseline: **440.812 s**, accuracy 0.9375, decode 178.49 ms/token.
- Ascend-C layer-0 decode hook: **462.559 s**, accuracy 0.9375, decode 187.41 ms/token.
- Regression: **4.93% slower** end-to-end and 4.99% slower decode.
- Correctness: 31 turns and 2,329 generated tokens match; task 214 separately reaches answer 8 with accuracy 1.0 in both paths.

The kernel integration is correct but does not pass the performance gate. It is rejected for the current model path; no 127-task replay is run because the paired 16-task result already shows a regression near the 5% exit threshold. The standalone Ascend-C pipeline remains available for a future fused multi-stage kernel.
