"""Measure cold versus repeatable-executor ACLNN calls."""
import argparse
import json
import time
import torch
import torch_npu

from agentopt.ascendc_bridge import gdn_conv_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1)
    ap.add_argument("--channels", type=int, default=10240)
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.npu.set_device(0)
    x = torch.randn(args.rows, args.channels, device="npu", dtype=torch.float16)
    scale = torch.ones_like(x)
    weight = torch.randn(4, args.channels, device="npu", dtype=torch.float32)
    state = torch.zeros(args.rows, 3, args.channels, device="npu", dtype=torch.float16)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    gdn_conv_state(x, scale, weight, state)
    torch.npu.synchronize()
    cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(args.iterations):
        gdn_conv_state(x, scale, weight, state)
    torch.npu.synchronize()
    steady = (time.perf_counter() - t0) / args.iterations
    result = {"rows": args.rows, "channels": args.channels, "iterations": args.iterations,
              "cold_s": cold, "steady_s": steady,
              "cold_ms": cold * 1e3, "steady_ms": steady * 1e3}
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
