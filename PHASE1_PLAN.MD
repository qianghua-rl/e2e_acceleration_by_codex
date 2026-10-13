# Phase 1：Ascend 910B batch=1 推理优化执行手册

本文档自包含；执行者不需要聊天上下文。目标是降低数学 agent 的单请求 batch=1 端到端延迟。

## 1. 严格边界

允许安装已有推理库，选择已有 backend，调整 `torch.compile`、graph、warmup、stream、同步、生成循环和请求级对象复用方式。禁止写或修改 kernel、Ascend C/TBE 算子、算子融合/替换、layout/precision 改写、MTP/speculative decoding 和 workload 特化分支。模型权重、greedy、工具协议、prompt、`max_new_tokens` 和 `max_turns` 固定。

StaticCache 已完成完整测试但比 DynamicCache baseline 慢，属于 rejected 候选，不能算 Phase 1 成果。

## 2. 固定 workload

```text
repo=/remote-home/S45160/work/e2e_acceleration_by_claude
model=$repo/model/qwen3.5-27b
device=npu (Ascend 910B3)
data=gsm8k_128; tasks=127; exclude=423
batch=1; runner=serial; greedy
max_new_tokens=512; max_turns=6
```

单请求是筛选目标；16-task 是回归筛选；127-task 是最终确认。每次都记录 accuracy、完整 token 轨迹、turns、tool calls、generated/prompt tokens。

## 3. 环境准备

```bash
cd /remote-home/S45160/work/e2e_acceleration_by_claude
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/inner:/usr/local/Ascend/ascend-toolkit/8.0.RC3/aarch64-linux/lib64:/usr/local/Ascend/ascend-toolkit/8.0.RC3/compiler/lib64:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1
id; npu-smi info
conda run -n ascend-e2e python -c 'import torch,transformers,torch_npu; print(torch.__version__, transformers.__version__)'
```

必须能读取 CANN `opp/scene.info`，用户必须在 `davinci` 组；否则先修复权限，不复制或混用其他 CANN OPP。所有 Python 命令均用 `conda run -n ascend-e2e`。

## 4. 文件和记录规则

每个候选使用唯一编号 `p1_s01`、`p1_s02`；结果写入 `results/p1_<candidate>_<scope>/`，其中 scope 为 `single`、`16`、`127`。目录必须保存 `metrics.json`、`README.md`、命令、环境快照和 stdout/stderr。每次改动、profile、Amdahl 估算、正确性、实测和 KEEP/REJECT 决策都追加到 `OPTIMIZATION_LOG.md`。不得覆盖已有结果。

## 5. Baseline

先查看脚本实际参数：

```bash
conda run -n ascend-e2e python bench/run_bench.py --help
conda run -n ascend-e2e python bench/profile_npu.py --help
```

使用仓库中标记为 DynamicCache/eager 的 baseline preset，执行：

```bash
conda run -n ascend-e2e python bench/run_bench.py \
 --stage p1_baseline127_recheck --engine naive --runner serial \
 --num-tasks 127 --exclude-tasks 423 --max-new-tokens 512 \
 --max-turns 6 --device npu
```

若 `naive` 不是 DynamicCache preset，以 `--help` 和 `agentopt/engines/__init__.py` 为准并把实际命令写入 README。重复两次；makespan 差异超过 1% 时先调查噪声。baseline 必须报告总时间、load、prefill、decode、Python 调度、同步/回读、工具时间、decode ms/token、prefill tok/s、accuracy、turns 和 tokens。

## 6. Profile 与 Amdahl

按 L1→L2→L3：L1 分段计时；L2 `torch.profiler` 查看已有调用（compile、同步、`item/cpu`、生成循环）；L3 仅在前两层指向 launch/runtime 问题时使用系统 profile。不得因为 profile 而修改算子。

对每个候选记录热点比例 `p`、局部预估加速 `s`，计算：

```text
expected_speedup = 1 / ((1-p) + p/s)
expected_time = baseline_time / expected_speedup
```

若 `p < 0.05` 且没有可信的 `s`，停止该方向。按收益/工作量选择下一候选。

## 7. 候选顺序

每次只开一个候选，不叠加开关。

### p1_s01：NPU launch/runtime 清理

关闭当前代码中仅适用于 CUDA 的 compile、graph、allocator、cudnn 路径和重复设备探测；只改设备分支/launch 参数。确认没有 fallback 或重复编译。

### p1_s02：已有 `torch.compile` backend

分别测试 eager、仓库支持的 backend、仅 decode compile、仅 model-forward compile。冷启动、warmup、steady-state 必须分开计时；编译成本不能隐藏。

### p1_s03：已有生成 API

减少外层 Python forward 循环，使用已有 batch=1 连续 decode/generate API。保持 stopping、工具边界和 greedy token 不变，不得 speculative decoding。

### p1_s04：请求级对象复用

复用模型、tokenizer、工具执行器、固定 device tensor 和非请求状态，避免重复初始化；每个请求的 KV/cache/state 必须隔离。分别报告 cold 和 warm 请求。

### p1_s05：host 回读/同步

只有 profile 证明它显著占时才执行：合并已有 stopping 判断的回读，减少不必要 `.item()`、`.cpu()`、同步和日志。不得改变 EOS 判断时机；若占比低于 5%，按 Amdahl 淘汰。

### p1_s06：已有推理库

仅测试已安装且明确支持当前模型、工具调用和 910B 的 vLLM-Ascend、MindIE 或其他 backend。固定权重、precision、greedy、tokens 和协议；不满足完整 token 轨迹一致时不得比较性能。

attention/layout/cache 算子改写、StaticCache、融合和任何 kernel 不属于 Phase 1。

## 8. 每个候选的强制流程

```text
1. 保存 baseline profile、git diff 和环境
2. 写 p、s、Amdahl 预期和风险
3. 只实现一个 launch/runtime/backend 改动
4. python -m compileall -q agentopt bench && git diff --check
5. correctness：单题完整 greedy token、答案、turn、state 隔离
6. 单请求：warmup 后至少 3 次，分开 cold/steady-state
7. 16-task：无 accuracy/turn/token 异常且有净收益才继续
8. 重新 profile，解释时间变化
9. 127-task：仅对通过 16-task 的候选执行
10. 更新 README 和 OPTIMIZATION_LOG.md，KEEP 或 REJECT
```

检查脚本参数先看 `--help`：

```bash
conda run -n ascend-e2e python bench/check_equiv.py --help
conda run -n ascend-e2e python bench/check_batch.py --help
```

## 9. 成功标准

只有同时满足以下条件才算 Phase 1 成功：batch=1 steady-state 单请求总延迟稳定下降至少 5%；16-task 不慢超过 1%；127-task 净下降且重复差异不超过 1%；accuracy、完整 token 轨迹、turns、tool calls 和 tokens 无异常；无 kernel、算子替换/融合、layout/precision 改写和 speculative decoding；profile 能解释收益。

任何失败、回归、噪声过大或无法解释的候选均标记 `REJECTED` 并保留证据。若所有候选失败，报告“当前软件栈 Phase 1 未获得稳定端到端收益”，不要把筛选结果包装成优化。

## 10. README 必填字段

候选编号/日期/提交；完整命令和固定配置；边界检查；profile 热点及证据；Amdahl 的 `p/s` 和预期；单一改动；token/答案/turn/state correctness；single/16/127 指标；重复性；KEEP/REJECT 及原因。
