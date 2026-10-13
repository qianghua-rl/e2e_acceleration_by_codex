# Final H100 versus Ascend 910B3 comparison

## Strict 127-task curve

| platform/stage | implementation | makespan | accuracy |
|---|---|---:|---:|
| H100 baseline | HF eager serial | 1470.6 s | 0.937 |
| H100 Phase 1 | StaticCache/graphs/compile/attention | 474.8 s | 0.938 |
| H100 Phase 2 | continuous batching and agent scheduling | ~20.3 s | 0.938 |
| H100 Phase 3 | custom Hopper kernels + FP8 + graph replay | 8.36 s | 0.945 |
| Ascend baseline | HF/torch-npu eager serial | 3466.21 s | 0.9449 |
| Ascend Phase 2 | 16-slot eager continuous batching | 656.37 s | 0.9449 |
| Ascend Phase 3 system | 16-slot prefix/state reuse + forced suffix; no custom kernel | **556.76 s** | **0.9449** |

Ascend Phase 3 attribution: prefix/state reuse alone measured 559.16 s; forced suffix reduced this to 556.76 s, a 0.4% incremental gain. The material Phase 3 system gain is prefix/state reuse (15.0% vs Phase 2); forced suffix is below the 5% per-change threshold.

## Interpretation

The Ascend serial baseline is 2.36x slower than the H100 baseline and decode is 2.31x slower (165.94 vs 71.84 ms/token). The Ascend result retains comparable accuracy, so the gap is performance-path related rather than an accuracy tradeoff.

Ascend Phase 2's 16-slot batching reduces strict serial makespan by 5.29x. Phase 3 prefix/state reuse plus forced suffix lowers it further to 556.76 s (6.22x cumulative vs serial; 15.2% faster than Phase 2), with accuracy unchanged. Prefill falls to 7.08 s; decode becomes 98.5% of the remaining runtime.

The H100 Phase 3 result is 66.6x faster than the Ascend Phase 3 system result (`556.76 / 8.36`). This comparison is not an apples-to-apples kernel comparison: H100 uses FP8, CUDA graphs, Hopper-specific custom Triton kernels, PDL and paged KV; Ascend uses bf16 eager torch-npu operators because FP8/A8W8 APIs are unavailable and custom OPP runtime registration fails correctness. The numbers are a platform/stack gap, not evidence that 910B silicon is intrinsically 66.6x slower.

## Custom-kernel gate

The first model-integrated Ascend-C layer-0 GDN/Conv kernel passed C=10240 output/state checks, 8-task token checks, and the 16-task paired correctness gate. Its paired 16-task timing was 440.812 s baseline versus 462.559 s with the kernel (4.93% slower), with identical 0.9375 accuracy and 2,329 generated tokens. L3 showed the kernel itself was only about 0.013% of total time; HF/CANN layout and state boundaries dominated. It is therefore excluded from the final speedup curve.

## Method and limits

All Ascend candidates followed profile -> Amdahl -> one change -> correctness -> re-profile. The NPU-aware simulator predicted the measured 16-slot result within 0.2%; 24 slots failed before metrics and 32 slots OOM. Longctx runs execute with DynamicCache but exceed the fixed Phase 2 2048-token SlotCache, so no fabricated longctx comparison is included.
