# Phase 3 Ascend quantized matmul probe

The installed `torch_npu.npu_quant_matmul` API was tested with int8 activations and weights and bf16 output. CANN 8.0.RC3 rejects the operation:

```text
only aclnnQuantMatmulV5 support A8W4, please upgrade CANN
```

The A8W8/per-tensor path needed for a direct H100-style FP8 comparison is unavailable in this runtime. No model weights were changed and no performance claim is made.
