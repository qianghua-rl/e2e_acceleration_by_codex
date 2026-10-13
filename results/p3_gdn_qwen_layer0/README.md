# Qwen3.5 Layer-0 Ascend-C Integration

The verified ACLNN bridge is installed only on linear-attention layer 0's
decode-time causal convolution update. Prefill remains on the PyTorch path.

The bridge converts Qwen BF16 input/state to the validated FP16 ABI, converts
the layer's taps to FP32 `[4,C]`, calls the ACLNN op on torch_npu's current
stream, and copies state back to Qwen's BF16 cache.

Reference checks:

```text
synthetic bridge, C=10240: output max abs 0.015625, state max abs 0
real layer direct token:    output max abs 0.00390625, state max abs 0
```

For prompt `What is 1+1?`, the baseline and layer-0 replacement both produce
the first two greedy tokens:

```text
baseline: [271, 16]
Ascend-C layer 0: [271, 16]
```

The earlier large error was traced to fixed `384/768` state writeback offsets
in the kernel. Those offsets are now derived from dynamic `channels`.

The 8-task, 16-task, 127-task replay, L3 profile, accuracy, and long-context
gates remain pending until a clean paired baseline can run. The 8-task short
probe measured 48.118 s baseline versus 49.197 s with layer 0 (2.24% slower),
with identical first tokens but intentionally invalid accuracy under the short
token cap. The 16-task layer-0 probe measured 111.368 s without a matching
baseline, so it is not a speed claim.
