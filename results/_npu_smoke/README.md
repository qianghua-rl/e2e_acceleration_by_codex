# NPU smoke baseline

This is a connectivity and execution smoke test, not a quality baseline.

- Hardware: one Ascend 910B3 (device 1), CANN 8.0.RC3, `torch==2.9.1`, `torch_npu==2.9.1`.
- Model: local `model/qwen3.5-27b`, bf16, `Qwen3_5ForConditionalGeneration`.
- Workload: first 2 GSM8K tasks, serial runner, max 32 new tokens and 2 turns.
- Result: makespan 13.45 s; prefill 1.93 s (14.3%); decode 11.52 s (85.6%); 185.7 ms/decode token.
- Accuracy is 0.0 because the deliberately short 32-token limit prevented tool calls and answers (`no_answer` for both tasks).

The result proves model loading, NPU execution, tokenizer, runner, and result serialization. It must not be used for accuracy or full-workload comparison.
