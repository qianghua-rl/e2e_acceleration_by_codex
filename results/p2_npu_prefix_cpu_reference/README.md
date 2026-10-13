# Prefix/state continuation CPU reference check

The GDN fallback accepts `initial_state`, and the Conv1d fallback can continue from the cached K-1 input window. A CPU reference test compared full-sequence execution with a prefix followed by a suffix:

- GDN output max abs error: 0
- GDN final-state max abs error: 0
- causal Conv1d suffix max abs error: 1.49e-8

The Conv wrapper crops outputs at `[K-1:K-1+n]`, preserving the new suffix positions. This validates recurrence math independent of the NPU runtime. NPU token-level validation remains pending CANN OPP read permissions, which currently prevent ordinary users from initializing GE.
