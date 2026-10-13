"""Event simulator for NPU eager continuous batching.

The simulator replays measured task turn lengths, charges synchronous prefill
before each turn, lets async tools sleep without blocking active decode rows,
and advances all live rows by one measured bucket step per scheduler tick.
"""
import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_run(name):
    d = ROOT / "results" / name
    tasks = {}
    for line in open(d / "turns.jsonl"):
        rec = json.loads(line)
        tasks.setdefault(rec["task_id"], []).append(rec)
    for ts in tasks.values():
        ts.sort(key=lambda r: r["turn"])
    metrics = json.load(open(d / "metrics.json"))
    return tasks, metrics


def measured_steps(names):
    by_bucket = {}
    for name in names:
        p = ROOT / "results" / name / "steps.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(line) for line in open(p)]
        per = {}
        for x in rows:
            per.setdefault(int(x["bucket"]), []).append(1000 * (x["t1"] - x["t0"]))
        for b, vals in per.items():
            by_bucket.setdefault(b, []).extend(vals)
    return {b: statistics.fmean(v) for b, v in by_bucket.items()}


def simulate(tasks, order, slots, step_ms):
    # A task's prefill runs synchronously and pauses decode scheduling. Tools
    # run asynchronously; their completion only makes that task ready to prefill.
    queue = list(order)
    live = {}       # slot -> (task id, turn index, remaining decode steps)
    ready = []      # (task id, turn index), ready for synchronous prefill
    waiting = []    # (wake_time, task id, next turn index)
    now = 0.0
    by_bucket = sorted(step_ms)
    steps, prefills, gaps = 0, 0, 0.0

    while queue or live or ready or waiting:
        # Tools that completed during the previous decode/prefill interval.
        waiting.sort()
        while waiting and waiting[0][0] <= now:
            _, tid, ti = waiting.pop(0)
            ready.append((tid, ti))

        # Admit tasks and perform one synchronous prefill at a time. Other live
        # rows do not advance while the host is inside an eager model prefill.
        while len(live) < slots and (ready or queue):
            if ready:
                tid, ti = ready.pop(0)
            else:
                tid, ti = queue.pop(0), 0
            turn = tasks[tid][ti]
            now += float(turn.get("prefill_s", 0.0))
            prefills += 1
            # Runner removes the stop token from gen_len after generation, but
            # the device step that produced EOS is still present in the trace.
            generated_steps = int(turn["gen_len"]) + int(bool(turn.get("stopped", False)))
            rem = max(0, generated_steps - 1)
            if rem:
                slot = next(s for s in range(slots) if s not in live)
                live[slot] = (tid, ti, rem)
            else:
                _finish_turn(tasks, tid, ti, turn, now, live, ready, waiting)

        if not live:
            if waiting:
                wake = min(x[0] for x in waiting)
                gaps += max(0.0, wake - now)
                now = max(now, wake)
                continue
            if ready or queue:
                continue
            break

        nlive = len(live)
        bucket = next((b for b in by_bucket if b >= nlive), None)
        if bucket is None:
            # Extrapolate step cost only beyond observed buckets, with the last
            # measured marginal cost per row; report predictions as extrapolated.
            if len(by_bucket) < 2:
                raise ValueError("need at least two measured bucket sizes to extrapolate")
            b1, b2 = by_bucket[-2:]
            cost = step_ms[b2] + (nlive - b2) * (step_ms[b2] - step_ms[b1]) / (b2 - b1)
            bucket = nlive
        else:
            cost = step_ms[bucket]
        now += cost / 1000.0
        steps += 1
        for slot, (tid, ti, rem) in list(live.items()):
            rem -= 1
            if rem > 0:
                live[slot] = (tid, ti, rem)
            else:
                del live[slot]
                turn = tasks[tid][ti]
                _finish_turn(tasks, tid, ti, turn, now, live, ready, waiting)

    return {"makespan_s": now, "decode_steps": steps, "prefills": prefills, "tool_idle_s": gaps}


def _finish_turn(tasks, tid, ti, turn, now, live, ready, waiting):
    if ti + 1 >= len(tasks[tid]):
        return
    nxt = ti + 1
    tool_s = float(turn.get("tool_s", 0.0))
    if tool_s > 0:
        waiting.append((now + tool_s, tid, nxt))
    else:
        ready.append((tid, nxt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="measured turn structure")
    ap.add_argument("--step-runs", nargs="+", required=True, help="runs with NPU steps.jsonl")
    ap.add_argument("--slots", nargs="+", type=int, default=[8, 16, 24])
    ap.add_argument("--order", choices=("file", "longest_first"), default="file")
    args = ap.parse_args()
    tasks, metrics = read_run(args.run)
    costs = measured_steps(args.step_runs)
    if not costs:
        raise SystemExit("no measured NPU step traces found")
    order = sorted(tasks)
    if args.order == "longest_first":
        order.sort(key=lambda tid: -sum(t["prompt_len"] + t["gen_len"] for t in tasks[tid]))
    print(f"tasks={len(tasks)} order={args.order} measured={metrics['makespan_s']:.2f}s step_ms_by_bucket="
          f"{ {b: round(v, 2) for b, v in sorted(costs.items())} }")
    for s in args.slots:
        result = simulate(tasks, order, s, costs)
        err = 100 * (result["makespan_s"] - metrics["makespan_s"]) / metrics["makespan_s"] if s == metrics["engine_options"].get("num_slots") else None
        suffix = f" calibration_error={err:+.2f}%" if err is not None else ""
        print(f"slots={s} predicted={result['makespan_s']:.2f}s steps={result['decode_steps']} "
              f"prefills={result['prefills']}{suffix}")


if __name__ == "__main__":
    main()
