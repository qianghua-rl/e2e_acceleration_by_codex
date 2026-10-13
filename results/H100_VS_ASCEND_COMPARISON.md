# H100 vs Ascend baseline comparison

## Strict adjusted 127-task baseline

| system | workload | makespan | accuracy | decode |
|---|---|---:|---:|---:|
| H100 HF baseline | 127 tasks, task 423 excluded | 1470.6 s | 0.937 | 71.84 ms/token |
| Ascend 910B3 HF/torch-npu baseline | 127 tasks, task 423 excluded | 3466.21 s | 0.9449 | 165.94 ms/token |

Ascend is **2.36x slower** in makespan and **2.31x slower** per decode token. Accuracy is effectively comparable (Ascend 0.9449 versus H100 0.937); the performance gap is not an accuracy tradeoff.

## Platform-internal Ascend curve

| configuration | workload | makespan |
|---|---|---:|
| serial baseline | 127 tasks, max 512 | 3466.21 s |
| 8-slot eager | 127 tasks, max 512 | 907.54 s |
| 16-slot eager | 127 tasks, max 512 | **656.37 s** |

This is the valid Ascend optimization curve. The earlier 422.44 s StaticCache and 121.65 s 8-slot values are 16-task/256-token iteration probes and are labeled separately.

The Ascend measurement is now a true full serial run, not the earlier 16-task extrapolation. Its dominant cost is decode (92.1% of makespan), matching the H100 observation that model-weight streaming and per-token execution dominate batch-1 inference.
