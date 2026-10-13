"""Device-independent reference for the fused 4-tap causal state update."""

import torch


def gdn_conv_state_reference(x, scale, weight, state):
    """Compute one independent causal update per row.

    Args: x [T,C], scale [T] or [T,1], weight [4,C], state [T,3,C].
    Returns: output [T,C] and the updated state tensor.
    """
    if x.ndim != 2 or state.ndim != 3 or state.shape[:2] != (x.shape[0], 3):
        raise ValueError("expected x [T,C] and state [T,3,C]")
    if weight.shape != (4, x.shape[1]):
        raise ValueError("weight must have shape [4,C]")
    if scale.numel() == x.shape[0]:
        s = scale.reshape(x.shape[0], 1)
    elif scale.shape == x.shape:
        s = scale
    else:
        raise ValueError("scale must have shape [T], [T,1], or [T,C]")
    s = s.to(dtype=x.dtype)
    # Match the kernel's BF16/FP16 input arithmetic before applying FP32 taps.
    xt = (x * s).to(dtype=x.dtype)
    old = state.clone()
    y = (old[:, 0] * weight[0].to(x.dtype) +
         old[:, 1] * weight[1].to(x.dtype) +
         old[:, 2] * weight[2].to(x.dtype) +
         xt * weight[3].to(x.dtype)).to(x.dtype)
    new_state = torch.stack((old[:, 1], old[:, 2], xt), dim=1)
    return y, new_state


def repeated_reference(xs, scales, weight, state):
    """Run successive decode steps, preserving state between calls."""
    ys = []
    current = state
    for x, scale in zip(xs, scales):
        y, current = gdn_conv_state_reference(x, scale, weight, current)
        ys.append(y)
    return torch.stack(ys), current
