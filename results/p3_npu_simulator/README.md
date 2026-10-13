# NPU-aware scheduler simulator validation

`bench/sim_npu_steps.py` now models synchronous eager prefill before each turn, decode steps per live row (including the generated EOS step removed from `gen_len`), asynchronous tool completion, slot availability, and measured NPU step times by bucket.

Step costs were aggregated from measured NPU traces for 4, 8, and 16 slots:

| bucket | measured mean step |
|---:|---:|
| 4 | 180.18 ms |
| 8 | 221.07 ms |
| 16 | 245.08 ms |

Using the full 127-task turn sequence from the 16-slot run:

| slots | measured makespan | simulated makespan | error |
|---:|---:|---:|---:|
| 8 | 907.54 s | 873.57 s | -3.74% |
| 16 | 656.37 s | 644.97 s | -1.74% |
| 24 | not measured | 565.08 s | prediction only |

Both measured configurations are within the PLAN's 5% simulator error gate. The 24-slot prediction is worth validating once CANN GE initialization succeeds in the user runtime; 32 slots previously OOMed.

The simulator also compared longest-chain-first ordering using historical prompt+generation lengths. It predicted 930.18/691.33/612.92 s at 8/16/24 slots versus 873.57/644.97/565.08 s for file order, so the reordering candidate was rejected before implementation.
