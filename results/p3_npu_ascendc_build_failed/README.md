# Phase 3 Ascend C kernel build probe

The official Ascend operator samples were obtained and the generic AddCustom Ascend C sample was invoked for `Ascend910B3` using CANN 8.0.RC3. Adding `/usr/local/python3.9.2/lib` to `LD_LIBRARY_PATH` allowed msOpGen to generate the CustomOp project. Correcting the generated CMake preset to the CANN root (`/usr/local/Ascend/ascend-toolkit/8.0.RC3`, not its `aarch64-linux` subdirectory) allowed opbuild to generate `op_proto.cc` and the Ascend C kernel package successfully.

The package `custom_opp_ubuntu_aarch64.run` was created and verified. Installation into the root-owned default OPP path failed with a permissions error; no kernel was installed or used in the benchmark yet. A user-writable `ASCEND_CUSTOM_OPP_PATH` or root deployment is still required.
