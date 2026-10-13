# Phase 3 Candidate: GDN/Conv State Update

## Selection evidence

The candidate is selected from the Ascend L3 profile at
`results/p1_npu_profile_baseline2/profile_npu/trace/.../ASCEND_PROFILER_OUTPUT`:

| operator | calls | total device time (us) | role |
|---|---:|---:|---|
| TransData | 25,920 | 1,979,916 | GDN/Conv layout conversion |
| Mul | 167,187 | 1,012,769 | gates and state elementwise updates |
| Conv2D | 6,480 | 165,449 | causal convolution state update |
| ReduceSum | 34,704 | 347,787 | recurrent reductions |
| MatMulV2 | 67,095 | 6,594,913 | model GEMM; keep CANN path for the first kernel |

The earlier AddCustom work is only an Ascend C build/deploy/call measurement
smoke test. It is not counted as an LLM optimization result.

## Kernel contract

The first implementation target is a general BF16/FP16 state-update operator,
without FP8 or workload-specific branches:

The current implementation provides the exact decode-side contract in
`agentopt/v3/kernels.py`:

```text
x:          [T, C] BF16, contiguous rows; C = 2*HK*DK + HV*DV
xs:         [T] FP32 row scales
w:          [C, 4] FP32 causal-conv taps
conv_state: [S, 3, C] BF16, indexed by seq_slot, updated in place
q/k:        [T, HK, DK] BF16
v:          [T, HV, DV] BF16
a/b:        [T, HV] BF16 GEMM outputs
A_log/bias: [HV] FP32 gate parameters
g/beta:     [T, HV] FP32 gate outputs
```

The first Ascend-C kernel should cover the decode path (`T=1` per active
sequence, batched through `seq_slot`) and combine the layout conversion needed
by the GDN/Conv path with the causal window update and gate elementwise work.
The existing GDN matrix recurrence remains a separate CANN/custom path until
this fused boundary is numerically and performance validated. Shapes and
strides are runtime parameters; no prompt/task-specific branches are allowed.

## Acceptance gates

1. CPU/PyTorch reference comparison for random B, T, C, H, D and K values,
   including `T=1` decode and multi-token prefill.
2. BF16 and FP16 numerical checks against the existing fallback, with explicit
   max-absolute and relative-error thresholds.
3. Standalone Ascend C invocation and repeated execution test. Each iteration
   follows the proven ACLNN lifecycle: fresh descriptors, workspace query,
   workspace allocation, launch, synchronization, and cleanup.
4. Microbenchmark reports layout conversion, state update, launch/synchronize,
   and end-to-end times against the current CANN fallback.
5. L3 profile and end-to-end replay must show a win; otherwise reject the
   kernel and retain the CANN implementation.

## Current implementation status

The reference contract is implemented in
`bench/micro_gdn_conv_candidate.py` and passes eight random shape cases after
loading the CANN environment. The checked-in `custom_opp` tree is a generated
and installed OPP package containing only `AddCustom`; it has no Ascend-C host
project or registration sources from which a second operator can be safely
built in place. The next implementation step therefore needs a separate
Ascend-C project with its own op prototype, tiling registration, host API, and
kernel binary before changing the LLM path.

No GEMM replacement is part of this first candidate. No AddCustom result is
included in speedup or final platform comparison.

The standalone ACLNN caller compiles and reaches
`aclnnGdnConvStateUpdateGetWorkspaceSize`. Runtime currently returns `161001`
because the generated `binary_info_config.json` is not discovered by the
active OPP search path. Device correctness remains pending until the generated
OPP package is installed or exposed through the runtime's supported custom-OPP
path.

The OPP search path is now validated. The runtime error was caused by the
standalone caller passing `weight` as FP16 while the generated op contract
requires FP32. After fixing that ABI mismatch, the device run passes:

```text
gdn_conv_state stage0: PASS (y0=6.0)
```

This proves only the stage-0 `x * scale` path and OPP registration. It is not
the final GDN/Conv kernel result.

The state-window stage now passes on device:

```text
gdn_conv_state state-window: PASS (y0=12.0, state=[2.0,3.0,12.0])
```

This validates the three-history causal window read, four-term accumulation,
state shift, and in-place state writeback.

Per-channel FP32 taps are now enabled with weight layout `[4, C]`. The
standalone device check uses taps `[2, 3, 4, 5]`, `state=[1,2,3]`, and
`x*scale=6`, and passes:

```text
gdn_conv_state weighted-4tap: PASS (y0=50.0, state=[2.0,3.0,6.0])
```

The stage-0 device binary now builds successfully. It implements only
`y = x * scale` using `TPipe` local tensors and `AscendC::Mul`; weight/state
arguments remain in the ABI but are intentionally unused until a standalone
correctness caller is added. The generated `.o` and `.json` are under
`scratch/ascendc_gdn_conv_state/build_out/op_kernel/binary/ascend910b/bin/`.

The current device prototype uses weight layout `[4, C]`, BF16 state/input,
FP32 weights, and a single decode row with `C=384`. It is a correctness
prototype, not yet an LLM integration or performance result; the next gate is
multi-row/shape coverage and comparison against the existing CANN fallback.

A four-row decode check passes with independent state slices. The device now
uses the generated `TilingData` (`rows`, `channels`) and the host restores
dynamic `SetBlockDim(rows)`; the four-row caller passes with the same weighted
4-tap check. opc still emits a non-fatal tiling-registration diagnostic, but
the generated binary and runtime mapping are functional.

The repeated 10-iteration standalone measurement resets state and creates a
fresh executor/workspace each iteration:

```text
GetWorkspaceSize = 63.274 ms
kernel launch    = 0.071 ms
stream sync      = 0.048 ms
end-to-end       = 63.392 ms
```

These are prototype measurements for `[4,384]`, not an LLM speedup claim.

## CANN comparison boundary

The available CANN API is generic `aclnnConvolution`, whose input and weight
must have the same dtype. The prototype intentionally keeps BF16 input/state
and FP32 taps, so a direct call is not an equivalent baseline. A valid CANN
comparison must cast taps to FP16/BF16 and express the operation as a depthwise
2D convolution over `[N,C,1,4]` with weights `[C,1,1,4]`, then compare the
last output column and state writeback. Until that reference is implemented,
the prototype timings remain engineering measurements rather than a claimed
speedup over CANN.

The equivalent CANN depthwise baseline now runs through `torch_npu` after
loading the driver libraries. For `[N=4,C=384]`, FP16 taps, 20 warmup calls and
1000 measured calls report:

```text
cann depthwise conv: value=50.0 rows=4 channels=384 avg_ms=0.059
cann conv+state: value=50.0 rows=4 channels=384 avg_ms=0.158
```

This is a kernel-only CANN measurement with prebuilt `[N,C,1,4]` input and no
state writeback. The custom prototype's corresponding launch plus synchronize
cost is `0.039 ms`, while its full fresh-executor lifecycle is `0.702 ms`.
The numbers are therefore not an end-to-end speedup claim; state writeback and
the custom workspace-query lifecycle must be included in any production-level
comparison.

The same caller supports a larger run (`./standalone 100`). It also passes
weighted correctness and reports:

```text
GetWorkspaceSize = 6.937 ms
kernel launch    = 0.013 ms
stream sync      = 0.033 ms
end-to-end       = 6.983 ms
```

The 10-vs-100 difference shows substantial first-call/runtime cache effects;
future comparisons must include warmup and report both cold and steady-state
measurements.

A 1000-iteration run also passes correctness:

```text
GetWorkspaceSize = 0.663 ms
kernel launch    = 0.007 ms
stream sync      = 0.032 ms
end-to-end       = 0.702 ms
```

This confirms the query cost converges after runtime cache warmup; the result
is still an isolated prototype measurement and has not been compared against
the model's CANN Conv fallback.
