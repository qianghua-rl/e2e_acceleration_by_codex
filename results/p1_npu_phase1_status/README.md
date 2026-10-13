# Phase 1 optimization status

Phase 1 is not at a hardware ceiling, but several upper-layer candidates have been measured:

| candidate | result | decision |
|---|---|---|
| Transformers StaticCache | 473.66 → 422.44 s on 16 tasks; accuracy unchanged | retain, +10.8% |
| chunked host token readback (8 tokens) | 1214.45 s, accuracy 0.0, trajectory diverged | reject |
| NPU internal-format option | fails: CANN 8.0.RC3 op store has no supported Conv2D for the GDN fallback | reject for this environment |
| MatMul ND layout | same as default | not a change |
| MatMul NZ layout | 14.6% slower on same 8-task trajectory | reject |
| `torch.compile` / graph paths | TorchAir/driver compatibility blockers, recorded separately | defer until matched vendor stack |

The L3 profile still shows room in GDN/conv layout transforms and many small operators. Phase 1 has not reached a physical ceiling; the remaining promising no-custom-kernel options are installing an Ascend-compatible GDN fast path or obtaining a CANN/TorchAir/driver-matched graph stack. Until that stack is available, StaticCache is the best validated Phase 1 result.
