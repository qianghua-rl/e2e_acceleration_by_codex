"""PyTorch reference and token-level check for the Ascend-C Qwen Conv bridge."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch_npu

from agentopt.ascendc_bridge import gdn_conv_state
from agentopt.gdn_conv_reference import gdn_conv_state_reference


def main():
    torch.npu.set_device(0)
    b, c = 4, 10240
    x = torch.randn(b, c, device="npu", dtype=torch.float16)
    scale = torch.ones_like(x)
    taps = torch.stack([torch.full((c,), v, device="npu", dtype=torch.float32) for v in (2, 3, 4, 5)])
    state = torch.stack([torch.randn(b, c, device="npu", dtype=torch.float16) for _ in range(3)], 1)
    ref_state = state.float().clone()
    ref = (ref_state[:, 0] * 2 + ref_state[:, 1] * 3 + ref_state[:, 2] * 4 + x.float() * 5).to(torch.float16)
    got_state = state.clone()
    got = gdn_conv_state(x, scale, taps, got_state)
    torch_npu.npu.synchronize()
    err = (got.float() - ref.float()).abs().max().item()
    state_ref = torch.stack((ref_state[:, 1], ref_state[:, 2], x.float()), 1).to(torch.float16)
    state_err = (got_state.float() - state_ref.float()).abs().max().item()
    print(f"bridge token check: output_max_abs={err:.6f} state_max_abs={state_err:.6f}")
    # A second launch verifies state persistence and row isolation, rather than
    # only checking the initial zero/previous-window calculation.
    x2 = torch.randn_like(x)
    got2 = gdn_conv_state(x2, scale, taps, got_state)
    torch_npu.npu.synchronize()
    ref2, state_ref2 = gdn_conv_state_reference(x2.cpu(), scale.cpu(), taps.cpu(), state_ref.cpu().to(torch.float16))
    err2 = (got2.cpu().float() - ref2.float()).abs().max().item()
    state_err2 = (got_state.cpu().float() - state_ref2.float()).abs().max().item()
    print(f"bridge repeated check: output_max_abs={err2:.6f} state_max_abs={state_err2:.6f}")
    # Suspend/resume: checkpoint the state after one token and resume from the
    # checkpoint in a fresh tensor; it must match an uninterrupted second step.
    checkpoint = state_ref.to(device="npu")
    resumed = checkpoint.clone()
    direct = state.clone()
    gdn_conv_state(x, scale, taps, direct)
    direct_out = gdn_conv_state(x2, scale, taps, direct)
    resumed_out = gdn_conv_state(x2, scale, taps, resumed)
    torch_npu.npu.synchronize()
    resume_err = (direct_out.float() - resumed_out.float()).abs().max().item()
    resume_state_err = (direct.float() - resumed.float()).abs().max().item()
    print(f"bridge suspend/resume: output_max_abs={resume_err:.6f} state_max_abs={resume_state_err:.6f}")
    raise SystemExit(0 if max(err, state_err, err2, state_err2, resume_err, resume_state_err) <= 0.05 else 1)


if __name__ == "__main__":
    main()
