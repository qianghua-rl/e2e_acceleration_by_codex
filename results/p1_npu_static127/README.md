# Phase 1 StaticCache: strict 127-task result

## Configuration

```text
engine: p1_npu_static (Transformers StaticCache)
runner: serial
workload: gsm8k_128, task 423 excluded
max_new_tokens: 512
max_turns: 6
device: Ascend 910B3 NPU
```

Command:

```bash
python bench/run_bench.py --stage p1_npu_static127 --engine p1_npu_static --runner serial --num-tasks 127 --exclude-tasks 423 --max-new-tokens 512 --max-turns 6 --device npu
```

## Measured result

| Metric | Value |
|---|---:|
| Makespan | 3552.287 s |
| Accuracy | 0.9449 (120/127) |
| Turns | 253 |
| Tool calls | 126 |
| Generated tokens | 19,650 |
| Prompt tokens | 125,662 |
| Decode | 169.689 ms/token |
| Prefill | 492.859 tok/s |

The run completed successfully with exit code 0. Raw metrics are in
`metrics.json`.

## Interpretation

StaticCache preallocates KV storage and avoids DynamicCache growth. This strict
run measured 3552.287 s versus the strict DynamicCache baseline of 3466.21 s,
a 2.48% regression. The earlier 16-task probe showed a gain, so that result
does not generalize to the full workload. Compared with the H100 adjusted
serial baseline (1470.6 s), this result is 2.42x slower; this is a system
comparison, not an isolated hardware claim.
