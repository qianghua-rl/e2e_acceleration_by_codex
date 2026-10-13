# Phase 1 internal-format option (rejected)

Tested `torch.npu.config.allow_internal_format=True` with StaticCache, without changing precision or adding kernels. The run fails in the Gated-DeltaNet fallback path: CANN cannot find a supported `Conv2D` kernel in the `tbe-custom` op store (`Conv2D is not found in this op store`).

The candidate is rejected for the installed CANN 8.0.RC3 operator package and remains disabled by default.
