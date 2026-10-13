# Ascend C AddCustom runtime validation

The official AddCustom Ascend C sample compiled and installed successfully into a user-writable custom OPP path. The aclnn invocation binary also compiled and exported `aclnnAddCustomGetWorkspaceSize`/`aclnnAddCustom` symbols.

Runtime correctness initially failed with `161001` while the OPP path was malformed/inaccessible. After root installed the package at the standard OPP path and granted `davinci` read/traverse access, the same sample passes: all outputs are `3.0` and it prints `test pass`.

Diagnostic rebuild confirms all three `aclCreateTensor` calls return non-null pointers and status 0. With only the custom library path set the status is 361001; with explicit `ASCEND_CUSTOM_OPP_PATH`/`ASCEND_OPP_PATH` it is 161001. This points to custom-op registration/runtime environment rather than null input tensors.
