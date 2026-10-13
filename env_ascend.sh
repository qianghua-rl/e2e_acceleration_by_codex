#!/usr/bin/env bash
set -euo pipefail

export ASCEND_HOME=/usr/local/Ascend
export CANN_HOME=/usr/local/Ascend/ascend-toolkit/8.0.RC3
export PYTHON_HOME=/usr/local/python3.9.2
export CUSTOM_OPP_ROOT=/remote-home/S45160/work/e2e_acceleration_by_claude/scratch/ascendc_gdn_conv_state/build_out/packages
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-1}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}"

# CANN runtime/compiler variables must be initialized first.
if [[ -f "$CANN_HOME/set_env.sh" ]]; then
    source "$CANN_HOME/set_env.sh"
else
    source "$CANN_HOME/aarch64-linux/script/set_env.sh"
fi

# msOpGen and Python 3.9 need the interpreter's shared library.
export LD_LIBRARY_PATH="$PYTHON_HOME/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH}"
export PATH="$PYTHON_HOME/bin:$CANN_HOME/tools/profiler/bin:$PATH"
export PYTHONPATH="$CANN_HOME/python/site-packages:$CANN_HOME/python/site-packages/auto_tune:$PYTHONPATH"

# The installed package exposes its custom vendor directory at this level.
export ASCEND_CUSTOM_OPP_PATH="$CUSTOM_OPP_ROOT/vendors/customize${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
# ASCEND_OPP_PATH is consumed as one concrete built-in OPP directory by
# torch-npu; custom operators are supplied separately through
# ASCEND_CUSTOM_OPP_PATH.
export ASCEND_OPP_PATH="$CANN_HOME/opp"
export LD_LIBRARY_PATH="$CUSTOM_OPP_ROOT/vendors/customize/op_api/lib:${LD_LIBRARY_PATH:-}"

# Keep the driver libraries visible to both ACLNN and torch_npu subprocesses.
export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/inner:${LD_LIBRARY_PATH}"

if [[ -n "${VIRTUAL_ENV:-}" && -d "$VIRTUAL_ENV/lib/python3.9/site-packages" ]]; then
    export PYTHONPATH="$VIRTUAL_ENV/lib/python3.9/site-packages${PYTHONPATH:+:$PYTHONPATH}"
fi
