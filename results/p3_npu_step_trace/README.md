# Phase 3 NPU step-trace analysis

Analysis of the strict 127-task, 16-slot run (`p2_npu_final127_16slots`):

- makespan: 656.37 s;
- inside `engine.step()`: 343.11 s (52.3%);
- long gaps between steps (>50 ms): 299.06 s (45.6%);
- average step: 245.08 ms;
- live row fill: 88.3%; padding: 11.7%.

This changes the next Amdahl target: Phase 3 should reduce prefill/tool/state transition gaps, not focus first on small decode kernels. The existing `bench/sim_steps.py` requires CUDA `profile/batch_step.json` and cannot be applied directly to this NPU eager trace; an NPU-aware simulator must accept measured per-step timings and explicit gap/prefill costs before using it for scheduling decisions.

Tool concurrency was considered and rejected: tool execution is approximately 0.1% of the full 16-slot run (5.10 s in the strict serial baseline), so its theoretical Amdahl gain is negligible.
