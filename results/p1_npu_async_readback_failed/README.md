# Phase 1 asynchronous/chunked token readback attempt (rejected)

The candidate kept StaticCache but processed decode outputs in chunks of 8 tokens before reading them to the host. It was intended to reduce one NPU-to-CPU synchronization per token.

Validation on the same 16-task workload failed: makespan **1214.45 s**, accuracy **0.0**, 61 turns and 6,957 generated tokens versus the StaticCache reference's 31 turns, 2,276 tokens, 422.44 s and 0.9375 accuracy. The chunked feedback/cache-position semantics changed the greedy trajectory and caused repeated/no-answer turns.

The implementation was reverted. StaticCache remains the valid Phase 1 base; no speed claim is made for this candidate.
