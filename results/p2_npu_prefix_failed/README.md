# Phase 2 prefix reuse attempt (rejected)

The existing prefix/state reuse wrapper was tested on the NPU eager engine. The run failed during the first prefill because the installed Transformers path has no `causal_conv1d_fn` (the optional flash-linear-attention/causal-conv1d fast path is unavailable), so the wrapper receives `None` and cannot extend the recurrent state.

No custom kernel was added. The candidate is rejected for this environment and the validated eager batching result is retained.
