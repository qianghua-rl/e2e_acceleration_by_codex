import pytest
import torch

from agentopt.gdn_conv_reference import gdn_conv_state_reference, repeated_reference


@pytest.mark.parametrize("rows,channels", [(1, 7), (4, 16), (8, 33), (16, 64)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_reference_shapes_and_state_isolation(rows, channels, dtype):
    torch.manual_seed(rows + channels)
    x = torch.randn(rows, channels, dtype=dtype)
    scale = torch.randn(rows, dtype=dtype)
    weight = torch.tensor([[0.25], [-0.5], [1.25], [2.0]], dtype=torch.float32).expand(4, channels).contiguous()
    state = torch.randn(rows, 3, channels, dtype=dtype)
    y, new_state = gdn_conv_state_reference(x, scale, weight, state)
    assert y.shape == x.shape
    assert new_state.shape == state.shape
    assert torch.equal(new_state[:, 0], state[:, 1])
    assert torch.equal(new_state[:, 1], state[:, 2])
    assert torch.equal(new_state[:, 2], (x * scale[:, None]).to(dtype))


def test_reference_repeated_decode_matches_manual_window():
    dtype = torch.float16
    weight = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=torch.float32)
    xs = torch.tensor([[[1., 2.]], [[3., 4.]], [[5., 6.]]], dtype=dtype)
    scales = torch.ones(3, 1, dtype=dtype)
    initial = torch.zeros(1, 3, 2, dtype=dtype)
    ys, final = repeated_reference(xs, scales, weight, initial)
    assert torch.allclose(ys[0], torch.tensor([[0.7, 1.6]], dtype=dtype), atol=1e-3)
    assert torch.allclose(ys[1], torch.tensor([[2.6, 4.4]], dtype=dtype), atol=1e-3)
    assert torch.allclose(final[0, 2], xs[-1, 0], atol=1e-3)
