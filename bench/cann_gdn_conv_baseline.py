"""Equivalent CANN baseline for the GDN/Conv four-tap prototype."""
import argparse
import time
import torch
import torch.nn.functional as F


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=4)
    p.add_argument("--channels", type=int, default=384)
    p.add_argument("--iters", type=int, default=1000)
    args = p.parse_args()
    torch.npu.set_device(0)
    n, c = args.rows, args.channels
    x = torch.full((n, c), 2, dtype=torch.float16, device="npu")
    scale = torch.full((n, c), 3, dtype=torch.float16, device="npu")
    initial_state = torch.stack([torch.full((n, c), v, dtype=torch.float16, device="npu") for v in (1, 2, 3)], 1)
    state = initial_state.clone()
    taps = torch.tensor([2, 3, 4, 5], dtype=torch.float16, device="npu").view(4, 1).expand(4, c)
    # [N,C,1,4], [C,1,1,4], groups=C; last output is the causal result.
    weight = taps.transpose(0, 1).contiguous().reshape(c, 1, 1, 4)
    inp = torch.stack((state[:, 0], state[:, 1], state[:, 2], x * scale), dim=-1).unsqueeze(2)
    for _ in range(20):
        F.conv2d(inp, weight, groups=c)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        out = F.conv2d(inp, weight, groups=c)
    torch.npu.synchronize()
    ms = (time.perf_counter() - t0) * 1000 / args.iters
    value = float(out.flatten()[0].cpu())
    print(f"cann depthwise conv: value={value:.1f} rows={n} channels={c} avg_ms={ms:.3f}")

    state = torch.stack([torch.full((n, c), v, dtype=torch.float16, device="npu") for v in (1, 2, 3)], 1)
    for _ in range(20):
        xt = x * scale
        inp = torch.stack((state[:, 0], state[:, 1], state[:, 2], xt), dim=-1).unsqueeze(2)
        out = F.conv2d(inp, weight, groups=c)
        state = torch.stack((state[:, 1], state[:, 2], xt), dim=1)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        state = initial_state
        xt = x * scale
        inp = torch.stack((state[:, 0], state[:, 1], state[:, 2], xt), dim=-1).unsqueeze(2)
        out = F.conv2d(inp, weight, groups=c)
        state = torch.stack((state[:, 1], state[:, 2], xt), dim=1)
    torch.npu.synchronize()
    ms = (time.perf_counter() - t0) * 1000 / args.iters
    print(f"cann conv+state: value={float(out.flatten()[0].cpu()):.1f} rows={n} channels={c} avg_ms={ms:.3f}")


if __name__ == "__main__":
    main()
