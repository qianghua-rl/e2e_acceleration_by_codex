# Phase 3 longest-chain admission attempt (unverified)

The NPU trace showed a large critical-path/gap fraction, so a general offline-batch scheduling candidate was implemented: order admissions by previously measured per-task turn length while keeping the model, slots, and kernels unchanged.

The 16-task validation failed during CANN GE/op-store initialization before metrics were produced. No speed or accuracy claim is made. The scheduling code remains isolated behind the `continuous_longest` runner and requires a clean NPU runtime before retesting.
