# Ascend 910B3 final report

## Objective

Optimize the Qwen3.5 mathematical Python-tool agent end to end on Ascend 910B3. The H100 experiment supplies the workload, metrics, and profile/Amdahl methodology; implementation choices follow the NPU's supported stack.

## Current measured curve

| stage | configuration | workload | makespan | accuracy |
|---|---|---:|---:|---:|
| Stage 0 representative validation | serial HF/torch-npu eager | 16 tasks, max 256 | 473.66 s | 0.9375 |
| Phase 2 representative validation | 4-slot eager continuous batching | 16 tasks, max 256 | 152.12 s | 0.9375 |
| Phase 2 representative validation | 8-slot eager system scheduling | 16 tasks, max 256 | 121.65 s | 0.9375 |
| Stage 0 strict baseline | serial HF/torch-npu eager | 127 tasks, max 512 | **3466.21 s** | 0.9449 |
| Phase 2 strict final | 16-slot eager continuous batching | 127 tasks, max 512 | **656.37 s** | 0.9449 |
| Phase 3 system final | 16-slot prefix/state reuse + forced suffix | 127 tasks, max 512 | **556.76 s** | 0.9449 |

The representative rows are for fast iteration and are not directly comparable with the strict 127-task rows. The strict platform curve is `3466.21 s -> 656.37 s -> 556.76 s` (6.22x cumulative). Phase 3 adds prefix/state reuse and forced suffix scheduling at 16 slots; 24 slots is blocked by OPP permissions and 32 slots OOM.

Phase 3 attribution: prefix/state reuse alone measured 559.16 s; adding forced suffix measured 556.76 s, an incremental gain of only 0.4% (below the 5% per-step threshold). The combined Phase 3 system result is 15.2% faster than Phase 2. The `prefill_s` section counts eager prefill only; forced tokens are accounted in decode time.

## Method status

The measured work includes profile evidence, Amdahl estimates, implementation, correctness checks, and re-profiling. Phase 1 graph/compiler candidates are still being re-evaluated with the correct Ascend backends (`npu`, `npugraphs`, `npugraph_ex`); the earlier `torch_npu` backend failure was a backend-name error and is not a valid technical rejection.

The Phase 3 system track now has a strict measured result of 556.76 s. Prefix/state reuse passed NPU token-level checks; forced suffix preserved strict accuracy. The custom-kernel track has passed the standalone AddCustom correctness gate after root OPP deployment and permissions, but still requires an LLM-hotspot kernel and full integration before claiming additional end-to-end speedup.

The latest complete curve is: strict serial 3466.21 s; StaticCache Phase 1 validation 422.44 s on 16 tasks; Phase 2 16-slot full workload 656.37 s; Phase 3 16-slot prefix reuse + forced suffix 556.76 s. Float8/A8W8 probes are unsupported by the installed CANN runtime.

The first model-integrated Ascend-C layer-0 kernel passed token/state correctness but regressed the paired 16-task run by 4.93% (462.559 s vs 440.812 s). It is excluded from the final speedup curve under the 5% gate.

An Ascend-specific weight-only quantization primitive (`npu_weight_quant_batchmatmul`) has now passed a small bf16/int8 capability probe and is the next Phase 3 implementation candidate; it has not yet been applied to the model.

The first real model-integrated Ascend-C kernel (layer-0 GDN/Conv decode) passes C=10240 token/state equivalence and first-token checks. Its short 8-task probe is 2.24% slower than baseline; paired full-workload A/B is pending because the baseline process currently terminates under the observed NPU runtime/OOM condition. It is excluded from the final speedup curve until that gate is resolved.

The paired gate has since completed: baseline 440.812 s versus Ascend-C layer-0 462.559 s on the same 16-task configuration, both accuracy 0.9375 and identical 2,329 generated tokens. The 4.93% regression rejects the single-layer integration for production; the verified kernel remains an infrastructure starting point for a larger fused boundary.
