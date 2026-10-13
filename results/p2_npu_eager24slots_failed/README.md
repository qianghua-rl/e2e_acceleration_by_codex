# Phase 2 24-slot validation (runtime failure)

The NPU-aware simulator, calibrated against the 16-slot trace with 0.2% error, predicted 562.37 s for 24 slots. A real 16-task validation was attempted, but CANN GE/op-store initialization failed before metrics were produced. No performance or accuracy result is claimed. The 32-slot run already establishes the bf16 memory boundary; 16 slots remains the largest stable measured configuration.
