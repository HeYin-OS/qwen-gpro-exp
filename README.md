# qwen-grpo

基于 **GRPO**（Group Relative Policy Optimization，组相对策略优化）对 **Qwen2.5-1.5B-Instruct** 做数学推理后训练，训练栈为 TRL + PEFT（LoRA）。奖励完全由规则验证，无需 judge 模型：答案正确性（1.0）+ 格式合规（0.2），与 GSM8K 标准答案比对。

## 工作原理

```
GSM8K 题目 ──► GRPOTrainer (TRL)
               ├── 每个 prompt 采样 num_generations=8 个候选（组采样）
               ├── 由规则奖励计算组内相对优势
               ├── PPO 式裁剪 (ε=0.2) + KL 正则 (β=0.04)
               └── LoRA adapter (r=32，仅注意力投影；底座权重冻结)
```

- **训练**：`src/qwen_grpo/train.py` — 通过 `configs/*.yaml` 配置（未知键会被拒绝）
- **评测**：`src/qwen_grpo/evaluate.py` — base 与微调模型在 500 道 held-out 题上对比（准确率 / 格式合规率 / 生成长度）
- **奖励**：`src/qwen_grpo/rewards.py` + `answer_utils.py` — 用 `Fraction` 做精确数值等价比较（"0.5" ≡ "1/2" ≡ "0.50"）

## 结果

训练 rollout 正确率（correctness 奖励均值）：**早期 ~5.5% → 后期 ~29%，峰值 39%**，共 125 步、1,000 道 GSM8K 题目，单张 RTX 4070 Laptop（8 GB），耗时约 4 小时。

| 指标 (200 道 held-out，贪心解码) | Base | GRPO |
|---|---|---|
| 答案准确率（可验证格式下） | 0.0% | **38.5%** |
| 格式遵循率 | 0.5% | **55.5%** |
| 平均生成长度（字符） | 931 | 860 |

注：Base 准确率为 0 是因为它几乎不按 `#### <答案>` 格式输出（0.5%），答案无法解析即计 0 —— 该口径衡量的是"在可验证答案格式约束下"的准确率，而非模型的原始数学能力。

## 环境配置

要求 Python ≥ 3.10、[uv](https://docs.astral.sh/uv/)、NVIDIA GPU（支持 bf16）、约 7 GB 空闲显存。

```bash
uv sync          # 安装 CUDA 版 torch (cu128 源) 及全部依赖到 .venv
```

如果 huggingface.co 无法直连，使用镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com   # Windows: set HF_ENDPOINT=https://hf-mirror.com
```

## 使用方法

```bash
# 训练（超参数见 configs/gsm8k_4070.yaml）
uv run qwen-grpo-train --config configs/gsm8k_4070.yaml

# 也可以用命令行覆盖个别参数
uv run qwen-grpo-train --config configs/gsm8k_4070.yaml --train-size 500 --lr 2e-5

# 在 held-out 测试集上评测 base vs 微调模型
uv run qwen-grpo-eval --base Qwen/Qwen2.5-1.5B-Instruct --adapter outputs/grpo \
    --out outputs/eval_results.json

# 运行测试
uv run pytest
```

## 项目结构

```
configs/            gsm8k_4070.yaml (单张 8GB 显卡配置), default.yaml
src/qwen_grpo/
  prompts.py        指令模板，"#### <answer>" 答案格式
  answer_utils.py   答案解析 + 精确数值等价比较
  rewards.py        正确性 / 格式奖励函数
  data.py           GSM8K 加载，确定性 train/test 划分（无泄漏）
  train.py          GRPO 训练入口
  evaluate.py       base vs 微调模型评测
tests/              23 个单元测试
```
