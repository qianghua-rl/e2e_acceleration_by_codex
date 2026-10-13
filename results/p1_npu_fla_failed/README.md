# Phase 1 existing FLA fast-path attempt (rejected)

`flash-linear-attention==0.5.2` and `fla-core==0.5.2` install as pure Python packages, but Qwen3.5's fast path also requires `causal-conv1d`. The available `causal-conv1d==1.7.0` source package attempts a CUDA/nvcc build and fails (`bare_metal_version`/nvcc path); no aarch64 Ascend wheel is available. The model therefore cannot construct the fast path and falls back to the PyTorch Conv1D/GDN implementation.

No performance result is claimed and no custom kernel was written. The candidate is rejected for the current package index/runtime.
