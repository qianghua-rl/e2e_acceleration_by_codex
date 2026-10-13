# AddCustom Benchmark

The benchmark creates fresh `aclTensor` descriptors and calls
`aclnnAddCustomGetWorkspaceSize` for every iteration. It allocates workspace
for that iteration, launches the operation, synchronizes the stream, and then
frees the workspace and descriptors. The generated custom API header exposes no
executor destroy/reset/reuse function, so an executor cannot be explicitly
released by the caller; its lifetime is owned by the ACLNN operation context.

Environment: Ascend 910B3, CANN 8.0.RC3. Inputs are FP16 `[8, 2048]` tensors
with values 1 and 2; correctness requires all outputs to be 3.

| iterations | operation | GetWorkspaceSize (ms) | launch (ms) | synchronize (ms) | end-to-end (ms) | correctness |
|---:|---|---:|---:|---:|---:|---|
| 10 | AddCustom | 0.003 | 0.007 | 0.060 | 0.069 | pass |
| 10 | CANN aclnnAdd | 7.250 | 0.547 | 0.035 | 7.832 | pass |
| 1000 | AddCustom | 0.002 | 0.005 | 0.039 | 0.046 | pass |
| 1000 | CANN aclnnAdd | 0.083 | 0.009 | 0.022 | 0.114 | pass |

Commands:

```bash
./build/execute_add_op 10
./build/execute_add_op 1000
```

Both runs completed without the previous executor-reuse `507053` error.
