# Phase 3 Ascend weight-only quant probe

The installed runtime rejects float8/A8W8 paths, but the existing `torch_npu.npu_weight_quant_batchmatmul` API works for bf16 activations, int8 weights, bf16 per-output-channel antiquant scales, and bf16 output.

Small shape probe result: `(4,512) @ (512,1024)` returned bf16 `(4,1024)` successfully. This is a capability result only; model-wide quantization has not yet been applied because it requires calibration/weight conversion and a full accuracy validation.
