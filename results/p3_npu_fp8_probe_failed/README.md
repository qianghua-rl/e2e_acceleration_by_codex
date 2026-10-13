# Phase 3 FP8 probe (runtime unsupported)

The first Phase 3 capability probe attempted per-tensor FP8 casting and `torch._scaled_mm` on NPU. The bf16 -> `float8_e4m3fn` cast fails in torch-npu/CANN 8.0.RC3 with `dtype not support ... DT_UNDEFINED`; no scaled matmul was executed.

This is not a performance result. An Ascend-specific quantization API or a general Phase 3 kernel implementation is required before FP8 can be evaluated.
