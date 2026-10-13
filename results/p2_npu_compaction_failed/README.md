# Phase 2 live-row compaction attempt (failed)

The Phase 2 trace showed 31.1% row padding at a fixed 8-row batch, so a bucketed live-row compaction candidate was selected by Amdahl analysis. The first implementation failed in the GDN fallback path: the model conv state remained batch 8 while the compact bucket supplied batch 2 (`aclnnCat` shape mismatch). Adding `SlotCache.set_rows(B)` before each step fixed the state view.

The corrected implementation completed with accuracy 0.9375 but took 126.35 s versus 121.65 s for fixed 8 rows. Computed rows fell 3376 -> 2486, but 25 row swaps and bucket transitions outweighed the savings, so the candidate is rejected for this workload.
