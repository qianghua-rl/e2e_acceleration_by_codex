# Phase 2 32-slot scan (rejected: OOM)

Increasing fixed eager batching from 16 to 32 slots initializes an 8.6 GiB slot cache, but the first real workload step exhausts NPU memory:

```text
torch.OutOfMemoryError: 60.96 GiB total; 59.66 GiB allocated; 5.30 MiB free
```

No performance or accuracy result is claimed. The scan establishes the current 16-slot memory/concurrency boundary for bf16 eager execution.
