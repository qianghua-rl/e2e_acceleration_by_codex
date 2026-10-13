# Phase 1 L2/L3 Ascend profile: serial baseline

- Profile method: `torch_npu.profiler` with NPU activity plus CANN `msprof` export.
- Workload: short fixed sample (2 profiled tasks after one warmup), bf16 naive engine, max 64 new tokens.
- The profile trace is under `profile_npu/trace/`; `op_summary_*.csv` is the authoritative L3 operator report.

Top aggregated operator time from the Ascend op summary:

| operator family | device time | calls |
|---|---:|---:|
| `MatMulV2` | 6.595 s | 67,095 |
| convolution/layout `TransDataFzToDst` | 1.559 s | 6,480 |
| elementwise `Mul` | 0.998 s | 160,302 |
| `ReduceSum` | 0.348 s | 34,704 |
| Cast/copy/slice families | ~0.9 s | >190,000 |
| `FlashAttentionScore` | 0.091 s | 2,160 |

## Amdahl selection

The profile confirms that FlashAttention is not the first target. The measured hot region is GEMM plus GDN/conv layout and elementwise overhead. A hypothetical 1.2x improvement to the MatMul family alone has an upper-bound local speedup of about 1.08x for this profile (`p` approximately 0.65). Phase 1 candidates therefore prioritize existing fast-path libraries, static/preallocated cache, and launch/synchronization reduction. No precision change or custom kernel is allowed in this phase.
