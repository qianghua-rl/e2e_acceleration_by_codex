#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT/env_ascend.sh"
export PYTHONPATH="$ROOT"
PYTHON_BIN=${PYTHON_BIN:-/remote-home/S45160/miniforge3/envs/ascend-e2e/bin/python}
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python interpreter not executable: $PYTHON_BIN" >&2
    exit 2
fi
export PYTHONPATH="$ROOT:$CANN_HOME/python/site-packages:$CANN_HOME/python/site-packages/auto_tune"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"

stage=${1:-p3_npu_layer0_pair16}
tasks=${2:-16}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
MAX_TURNS=${MAX_TURNS:-6}
extra=(--runner serial --num-tasks "$tasks" --exclude-tasks 423 --max-new-tokens "$MAX_NEW_TOKENS" --max-turns "$MAX_TURNS")

mkdir -p "$ROOT/results/${stage}_baseline" "$ROOT/results/${stage}_ascendc"
capture_env() {
    local out="$1"
    {
        date -u +%FT%TZ
        printf 'python_bin=%s\n' "$PYTHON_BIN"
        "$PYTHON_BIN" -VV 2>&1 || true
        "$PYTHON_BIN" -c 'import torch; print("torch", torch.__version__); print("torch_npu", getattr(__import__("torch_npu"), "__version__", "unknown"))' 2>&1 || true
        env | grep -E '^(ASCEND|OPP|LD_LIBRARY|PYTHONPATH|PATH)=' | sort
    } > "$out"
}
capture_env "$ROOT/results/${stage}_baseline/env.txt"
id > "$ROOT/results/${stage}_baseline/id.txt"
npu-smi info > "$ROOT/results/${stage}_baseline/npu-smi.txt" 2>&1 || true
run_phase() {
    local phase="$1"; shift
    local dir="$ROOT/results/${phase}"
    set +e
    "$@" > >(tee "$dir/stdout.log") 2> >(tee "$dir/stderr.log" >&2)
    local rc=$?
    set -e
    printf '%s\n' "$rc" > "$dir/exit_code.txt"
    return "$rc"
}

unset AGENTOPT_ASCENDC_COMPARE
unset ASCEND_CUSTOM_OPP_PATH
run_phase "${stage}_baseline" "$PYTHON_BIN" "$ROOT/bench/run_bench.py" --stage "${stage}_baseline" --engine naive "${extra[@]}"

source "$ROOT/env_ascend.sh"
export AGENTOPT_ASCENDC_COMPARE=0
capture_env "$ROOT/results/${stage}_ascendc/env.txt"
id > "$ROOT/results/${stage}_ascendc/id.txt"
npu-smi info > "$ROOT/results/${stage}_ascendc/npu-smi.txt" 2>&1 || true
run_phase "${stage}_ascendc" "$PYTHON_BIN" "$ROOT/bench/run_bench.py" --stage "${stage}_ascendc" --engine p3_npu_ascendc_layer0 "${extra[@]}"
