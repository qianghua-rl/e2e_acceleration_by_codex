"""Explicit control over HF Qwen3.5 linear-attention fast paths.

HF silently switches Gated-DeltaNet layers to `flash-linear-attention` / `causal-conv1d` kernels when those
packages are importable. Engines choose explicitly so results do not depend on what happens to be installed.
Must be called before the model is constructed (layers bind the functions in __init__).
"""
import os
import transformers.models.qwen3_5.modeling_qwen3_5 as qm

_ASCENDC_CONST_CACHE = {}


def set_linear_attn_fast_path(enabled: bool):
    if enabled:
        from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
        from fla.modules import FusedRMSNormGated
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    else:
        causal_conv1d_fn = causal_conv1d_update = FusedRMSNormGated = None
        chunk_gated_delta_rule = fused_recurrent_gated_delta_rule = None
    qm.causal_conv1d_fn, qm.causal_conv1d_update = causal_conv1d_fn, causal_conv1d_update
    qm.chunk_gated_delta_rule = chunk_gated_delta_rule
    qm.fused_recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule
    qm.FusedRMSNormGated = FusedRMSNormGated
    qm.is_fast_path_available = enabled


def install_ascendc_conv_layer(model, layer_index: int):
    """Replace one Qwen linear-attention decode Conv update with the ACLNN bridge."""
    from ..ascendc_bridge import gdn_conv_state
    import torch

    text_model = getattr(getattr(model, "model", model), "language_model", getattr(model, "language_model", model))
    layer = text_model.layers[layer_index]
    if layer.layer_type != "linear_attention":
        raise ValueError(f"layer {layer_index} is {layer.layer_type}, expected linear_attention")
    la = layer.linear_attn

    def ascendc_update(hidden_states, conv_state, weight, bias=None, activation=None):
        if hidden_states.shape[-1] != 1:
            raise RuntimeError("Ascend-C decode bridge only supports seq_len=1")
        # The validated Ascend-C ABI is FP16 input/state + FP32 taps. Qwen
        # checkpoints commonly run BF16, so convert only this bridge boundary.
        compare = os.environ.get("AGENTOPT_ASCENDC_COMPARE", "0") == "1"
        if compare:
            print("ascendc shapes", tuple(hidden_states.shape), tuple(conv_state.shape), tuple(weight.shape),
                  "strides", hidden_states.stride(), conv_state.stride(), weight.stride(), flush=True)
        ref_state = conv_state.clone() if compare else None
        x = hidden_states[:, :, 0].contiguous().to(torch.float16)
        scale_key = (x.device, tuple(x.shape), x.dtype)
        scale = _ASCENDC_CONST_CACHE.get(("scale", scale_key))
        if scale is None:
            scale = torch.ones_like(x)
            _ASCENDC_CONST_CACHE[("scale", scale_key)] = scale
        state = conv_state.transpose(1, 2).contiguous().to(torch.float16)
        taps_key = ("taps", weight.data_ptr(), tuple(weight.shape), weight.dtype, weight.device)
        taps = _ASCENDC_CONST_CACHE.get(taps_key)
        if taps is None:
            taps = weight.to(torch.float32).contiguous().transpose(0, 1).contiguous()
            _ASCENDC_CONST_CACHE[taps_key] = taps
        out = gdn_conv_state(x, scale, taps, state)
        conv_state.copy_(state.transpose(1, 2).to(conv_state.dtype))
        result = torch.nn.functional.silu(out).unsqueeze(-1).to(hidden_states.dtype)
        if compare:
            hs = torch.cat((ref_state, hidden_states), dim=-1).to(weight.dtype)
            ref_state.copy_(hs[:, :, -ref_state.shape[-1]:])
            ref = torch.nn.functional.silu(torch.nn.functional.conv1d(
                hs, weight.unsqueeze(1), bias, groups=hs.shape[1])[:, :, -1:]).to(hidden_states.dtype)
            print("ascendc layer diff", float((result.float() - ref.float()).abs().max()),
                  float((conv_state.float() - ref_state.float()).abs().max()), flush=True)
        return result

    la.causal_conv1d_update = ascendc_update
    return layer_index
