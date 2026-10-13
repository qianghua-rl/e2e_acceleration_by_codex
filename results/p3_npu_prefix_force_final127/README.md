# Phase 3 system optimization: prefix/state reuse + forced suffix

- Strict workload: 127 tasks, task 423 excluded, max 512 tokens/turn, max 6 turns.
- Base: 16-slot eager continuous batching with prefix/state reuse.
- Change: feed reusable suffixes of up to 128 tokens through decode steps instead of separate eager prefill calls.
- Makespan: **556.76 s**; accuracy **0.9449**; 253 turns, 126 tool calls.
- Phase 2 16-slot baseline: 656.37 s; net Phase 3 system improvement: **15.2%**.
- Prefix reuse without forcing: 559.16 s; force suffix adds only ~0.4% improvement over that.
- Prefill: 7.08 s (1.3%); decode: 548.18 s (98.5%); 27.72 ms/token.
- Reuse: 66,906 turn-history tokens and 46,112 shared-prefix tokens; 11,687 forced tokens.

Accuracy matches the strict serial baseline. Forced suffixes largely move work from eager prefill into batched decode at 16 slots, so the incremental gain is small; most Phase 3 gain comes from prefix/state reuse.
