"""Stage 00 engine: plain HF transformers, one request at a time.

Semantically equivalent to `model.generate(do_sample=False)`: prefill computes only the last logit,
then a Python loop decodes one token per forward with an HF DynamicCache and a host sync per token
(argmax -> int) for the stop check. Every turn re-prefills the whole conversation from scratch.
"""
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration, StaticCache

from .. import MODEL_PATH
from .hf_patches import install_ascendc_conv_layer, set_linear_attn_fast_path


class NaiveEngine:
    def __init__(self, model_path: str = MODEL_PATH, device: str = "npu", dtype=torch.bfloat16,
                 linear_attn_fast_path: bool = False, cudnn_sdpa: bool = True,
                 static_cache: bool = False, allow_internal_format: bool = False,
                 mm_bmm_format_nd: bool | None = None, ascendc_conv_layer: int | None = None,
                 reuse_decode_input: bool = False):
        self.device = device
        self.options = {"linear_attn_fast_path": linear_attn_fast_path, "cudnn_sdpa": cudnn_sdpa,
                        "static_cache": static_cache, "allow_internal_format": allow_internal_format,
                        "mm_bmm_format_nd": mm_bmm_format_nd, "reuse_decode_input": reuse_decode_input}
        if allow_internal_format:
            if not hasattr(torch, "npu"):
                raise RuntimeError("allow_internal_format requires torch_npu")
            torch.npu.config.allow_internal_format = True
        if mm_bmm_format_nd is not None:
            if not hasattr(torch, "npu"):
                raise RuntimeError("mm_bmm_format_nd requires torch_npu")
            torch.npu.set_mm_bmm_format_nd(mm_bmm_format_nd)
        set_linear_attn_fast_path(linear_attn_fast_path)
        # cuDNN SDPA builds an execution plan per new (q_len, kv_len) shape: ~2.4 ms host time per call,
        # and during decode kv_len grows every step, so nearly every step is a "new" shape.
        # Disabling it lets SDPA dispatch to the flash / mem-efficient kernels, which need no plan.
        if device.startswith("cuda"):
            torch.backends.cuda.enable_cudnn_sdp(cudnn_sdpa)
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, dtype=dtype, low_cpu_mem_usage=True
        ).to(device).eval()
        if ascendc_conv_layer is not None:
            if not device.startswith("npu"):
                raise ValueError("ascendc_conv_layer requires an NPU device")
            install_ascendc_conv_layer(self.model, ascendc_conv_layer)
            self.options["ascendc_conv_layer"] = ascendc_conv_layer
        self.static_cache = static_cache
        self.reuse_decode_input = reuse_decode_input
        self.decode_input = torch.empty((1, 1), dtype=torch.long, device=device) if reuse_decode_input else None
        self.cache = StaticCache(config=self.model.config, max_cache_len=4096) if static_cache else None
        self.stop_ids = {self.tok.convert_tokens_to_ids("<|im_end|>"), self.tok.convert_tokens_to_ids("<|endoftext|>")}

    @torch.inference_mode()
    def generate(self, input_ids: list[int], max_new_tokens: int, timer, ignore_eos: bool = False,
                 forced_tokens: list[int] | None = None):
        """Returns (generated token ids without the stop token, stats dict)."""
        stop_ids = set() if ignore_eos else self.stop_ids
        ids = torch.tensor([input_ids], device=self.device)
        if self.static_cache:
            self.cache.reset()
        with timer("prefill") as t_pre:
            out = self.model(input_ids=ids, use_cache=True, past_key_values=self.cache,
                             logits_to_keep=1)
            past = out.past_key_values
            next_id = int(out.logits[0, -1].argmax())
        forced = list(forced_tokens or [])
        if forced and not ignore_eos:
            forced.append(next(iter(stop_ids)))
        if forced:
            next_id = forced[0]
        gen = [next_id]
        with timer("decode") as t_dec:
            while next_id not in stop_ids and len(gen) < max_new_tokens:
                if self.reuse_decode_input:
                    self.decode_input[0, 0] = next_id
                    decode_ids = self.decode_input
                else:
                    decode_ids = torch.tensor([[next_id]], device=self.device)
                out = self.model(input_ids=decode_ids,
                                 past_key_values=past, use_cache=True)
                past = out.past_key_values
                next_id = int(out.logits[0, -1].argmax())
                if forced and len(gen) < len(forced):
                    next_id = forced[len(gen)]
                gen.append(next_id)
        stopped = gen[-1] in stop_ids
        if stopped:
            gen = gen[:-1]
        return gen, {"prefill_s": t_pre.elapsed, "decode_s": t_dec.elapsed, "stopped": stopped}
