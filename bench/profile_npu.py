"""L2 Ascend profile for a short fixed agent workload.

Uses torch_npu.profiler rather than CUDA-only torch.profiler activities.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch_npu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentopt.engines import make_engine
from agentopt.runner import RUNNERS
from agentopt.timing import Timer
from bench.run_bench import load_tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--engine", default="naive")
    ap.add_argument("--tasks", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()
    out = ROOT / "results" / args.stage / "profile_npu"
    out.mkdir(parents=True, exist_ok=True)
    eng = make_engine(args.engine, device="npu")
    tasks = load_tasks(args.tasks + 1, exclude={423})
    runner = RUNNERS["serial"]
    runner(eng, tasks[:1], Timer(), args.max_new_tokens, 2)
    timer = Timer()
    t0 = time.perf_counter()
    profiler_error = None
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(out / "trace")),
        experimental_config=torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level1),
    ) as prof:
        results, turns = runner(eng, tasks[1:], timer, args.max_new_tokens, 2)
        torch.npu.synchronize()
        prof.step()
    wall = time.perf_counter() - t0
    # torch_npu's profiler object does not expose torch.cuda.profiler's
    # key_averages(); the trace directory is the authoritative artifact.
    # Keep parser failures in the summary instead of discarding the trace.
    try:
        (out / "top_ops.txt").write_text(prof.key_averages().table(sort_by="self_npu_time_total", row_limit=60))
    except Exception as exc:
        profiler_error = repr(exc)
        (out / "top_ops.txt").write_text("torch_npu profiler has no key_averages; inspect trace/\n")
    summary = {"tasks": len(results), "wall_s": wall, "turns": len(turns),
               "gen_tokens": sum(t["gen_len"] for t in turns), "breakdown": timer.breakdown(wall),
               "engine": args.engine, "max_new_tokens": args.max_new_tokens,
               "profiler_error": profiler_error,
               "trace_dir": str(out / "trace")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if profiler_error:
        print(f"profiler note: {profiler_error}")


if __name__ == "__main__":
    main()
