# Phase 1 dependency audit

- `triton==3.6.0` was installed successfully from the Huawei mirror.
- TorchAir was built from `Ascend/torchair` master and installed editable because no public PyPI package exists.
- Official TorchAir compatibility documentation lists CANN 8.3/8.5 combinations, but no CANN 8.0.RC3 + PyTorch 2.9.1 combination.
- Small-module retest: `npu` remains in the TorchAir import/compiler failure path; `npugraphs` reports zero active drivers; `npugraph_ex` cannot resolve `aclmdlRICaptureBegin`.

The existing 8-slot eager batching result remains the valid baseline. A graph implementation requires a matched vendor runtime/driver upgrade or internal TorchAir package; it was not silently substituted with an incompatible version.
