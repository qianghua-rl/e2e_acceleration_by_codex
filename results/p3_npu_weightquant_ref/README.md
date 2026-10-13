# Phase 3 weight-only quantization reference

Single-layer test on Ascend 910B3/CANN 8.0.RC3: bf16 activation `(M,512)`, per-output-channel symmetric int8 weight, bf16 output.

At `M=8,N=1024`:

- max absolute error: `0.75`
- mean absolute error: `0.134`
- mean relative error: `6.35%` (max relative is dominated by near-zero reference values)
- quant matmul: `0.062 ms`
- bf16 linear: `0.024 ms`

The API is functional but slower at this small M and has nontrivial numerical error. No model layer was replaced. At M=32 the quant operator measured 0.079 ms versus 0.029 ms for bf16 Linear (2.7x slower), with mean absolute error 0.135. It is rejected as a latency optimization and remains only a future memory/concurrency experiment.
