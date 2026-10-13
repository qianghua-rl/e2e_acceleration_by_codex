"""Reference and shape sweep for the first GDN/Conv Ascend-C candidate.

This intentionally contains no custom kernel. It freezes the numerical contract
that the Ascend-C implementation must match: causal four-tap convolution over
the slot's three-value state, followed by the GDN gate transform.
"""
import argparse
import torch


def conv_state_reference(x, scale, weight, state):
    """Return q/k/v projection and updated state for x [T,C], state [3,C]."""
    t, c = x.shape
    hist = [state[i].to(torch.float32) for i in range(3)]
    out = torch.empty_like(x, dtype=torch.float32)
    for i in range(t):
        xi = (x[i].float() * scale[i]).to(torch.bfloat16).float()
        out[i] = sum(hist[j] * weight[:, j].float() for j in range(3)) + xi * weight[:, 3].float()
        hist = [hist[1], hist[2], xi]
    return out, torch.stack(hist)


def gate_reference(a, b, scale, a_log, dt_bias):
    """The gate/state elementwise portion fused after the layout conversion."""
    af = a.float() * scale[:, None].float()
    bf = b.float() * scale[:, None].float()
    g = -torch.exp(a_log.float())[None, :] * torch.nn.functional.softplus(af + dt_bias.float())
    beta = torch.sigmoid(bf)
    return g, beta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--cases", type=int, default=8)
    args = p.parse_args()
    dev = torch.device(args.device)
    torch.manual_seed(0)
    for i in range(args.cases):
        t, c, hv = (1, 384, 48) if i == 0 else (2 ** (i % 7), 384 * (1 + i % 3), 48)
        x = torch.randn(t, c, device=dev, dtype=torch.bfloat16)
        scale = torch.randn(t, device=dev, dtype=torch.float32)
        w = torch.randn(c, 4, device=dev, dtype=torch.float32)
        state = torch.randn(3, c, device=dev, dtype=torch.bfloat16)
        y, new_state = conv_state_reference(x, scale, w, state)
        assert y.shape == (t, c) and new_state.shape == (3, c)
        a = torch.randn(t, hv, device=dev, dtype=torch.bfloat16)
        b = torch.randn_like(a)
        g, beta = gate_reference(a, b, scale, torch.randn(hv, device=dev), torch.randn(hv, device=dev))
        assert g.shape == beta.shape == (t, hv)
    print(f"reference contract passed: {args.cases} cases on {dev}")


if __name__ == "__main__":
    main()
