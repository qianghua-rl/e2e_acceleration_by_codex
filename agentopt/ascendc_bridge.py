"""Optional bridge for the verified Ascend-C GDN/Conv ACLNN op.

The extension is intentionally opt-in. It requires a torch_npu C++ build and
the generated custom OPP package; callers must compare it against the Python
reference before enabling a model layer.
"""
import os
from pathlib import Path

_EXT = None


def load():
    global _EXT
    if _EXT is not None:
        return _EXT
    import ctypes
    import torch_npu
    # torch_npu's Python import loads _C locally; the extension needs the
    # stream bridge symbols when it is dlopened afterwards.
    c_ext = Path(torch_npu.__file__).parent / "_C.cpython-312-aarch64-linux-gnu.so"
    ctypes.CDLL(str(c_ext), mode=ctypes.RTLD_GLOBAL)
    from torch.utils.cpp_extension import load as cpp_load
    root = Path(__file__).resolve().parent.parent / "scratch" / "ascendc_gdn_conv_state"
    src = Path(__file__).resolve().parent / "ascendc_bridge.cpp"
    _EXT = cpp_load(
        name="agentopt_ascendc_gdn_conv_v6",
        sources=[str(src)],
        extra_include_paths=[str(root / "build_out" / "autogen"), str(root / "build_out" / "op_host"),
                             str(Path(__import__('torch_npu').__file__).parent.parent / "torch_npu" / "include"),
                             os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/8.0.RC3") + "/include"],
        extra_ldflags=["-L" + str(root / "build_out" / "op_host"),
                       "-L/usr/local/Ascend/ascend-toolkit/8.0.RC3/aarch64-linux/lib64",
                       "-L" + str(Path(__import__('torch_npu').__file__).parent / "lib"),
                       "-Wl,-rpath," + str(root / "build_out" / "op_host"),
                       "-Wl,-rpath,/usr/local/Ascend/ascend-toolkit/8.0.RC3/aarch64-linux/lib64",
                       "-Wl,-rpath," + str(Path(__import__('torch_npu').__file__).parent / "lib"),
                       "-lcust_opapi", "-lopapi", "-lnnopbase", "-lascendcl"],
        verbose=bool(int(os.environ.get("AGENTOPT_ASCENDC_BUILD_VERBOSE", "0"))),
    )
    return _EXT


def gdn_conv_state(x, scale, weight, state):
    """Call the ACLNN op on NPU tensors; `state` is updated in place."""
    import torch_npu
    stream = int(torch_npu.npu.current_stream().npu_stream)
    return load().gdn_conv_state(x, scale, weight, state, stream)
