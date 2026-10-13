# Phase 1 Ascend backend capability matrix

Tested on a small bf16 Linear-SiLU-Linear module with the installed PyTorch 2.9.1 / torch-npu 2.9.1 / CANN 8.0.RC3 stack:

| backend | result | evidence |
|---|---|---|
| `npu` | rejected | imports `torchair`, but it is not installed/available |
| `npugraphs` | rejected | `ModuleNotFoundError: No module named 'triton'` |
| `npugraph_ex` | rejected | `Failed to find function aclmdlRICaptureBegin` |

This is a capability/environment matrix, not a model performance result. It identifies the next installation/configuration work: obtain the CANN-matched `torchair` and Triton packages, and check whether the installed driver exposes the RICapture API. No custom kernel was written.
