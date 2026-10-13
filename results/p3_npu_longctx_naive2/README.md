# Ascend long-context DynamicCache probe

- Two tasks from `gsm8k_longctx_128`, serial DynamicCache engine, max 64 new tokens and 2 turns.
- The workload runs without the 2048-token SlotCache limit: makespan 30.62 s, prompt tokens 8,500, prefill throughput 1,039 token/s, decode 177.84 ms/token.
- Accuracy is 0.0 because the intentionally short 64-token/2-turn probe produced no answers; this is a capacity/performance probe only.

The result confirms that longctx can execute with DynamicCache, while the Phase 2 fixed SlotCache cannot hold these prompts. A production longctx Phase 2/3 implementation needs paged or larger KV/state allocation.
