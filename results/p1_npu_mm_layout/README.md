# Phase 1 MatMul layout experiments

Both runs use StaticCache, bf16, the same 8-task probe (task 423 excluded, max 128 tokens, 6 turns).

| layout setting | makespan | accuracy | gen tokens | decode |
|---|---:|---:|---:|---:|
| StaticCache default/ND | 174.52 s | 0.500 | 996 | 166.69 ms/token |
| `set_mm_bmm_format_nd(False)` | 200.00 s | 0.500 | 996 | 190.94 ms/token |

The NZ layout setting is **14.6% slower** with the same trajectory and is rejected. The explicit ND=True run measured 177.87 s, but source inspection indicates this is the default format, so it was not a meaningful change and is not treated as an optimization result.
