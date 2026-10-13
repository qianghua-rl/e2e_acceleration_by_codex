# Ascend long-context probe (cache capacity limit)

The Phase 2 16-slot eager engine was tested on `gsm8k_longctx_128`. The first prompt exceeds the engine's fixed `max_len=2048` slot cache, so the run stops with `AssertionError: prompt exceeds max_len` before measurement.

No performance or accuracy result is claimed. Supporting the H100 longctx workload requires a larger/paged KV allocation and a separate memory budget study; increasing the existing bf16 cache blindly would invalidate the 16-slot memory result.
