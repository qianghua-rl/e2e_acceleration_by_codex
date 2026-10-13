# Phase 2 compaction retry (correct but slower)

- Same 16-task workload as the fixed 8-slot reference, max 256 tokens, six turns.
- Fixed 8 rows: 121.65 s, accuracy 0.9375.
- Bucketed compaction (`2,4,6,8`): **126.35 s**, accuracy **0.9375**.
- Computed rows: 3376 -> 2486; row swaps: 25.
- Net result: 3.9% slower because state/row swaps and bucket transitions outweighed padding reduction. Reject for this workload.
