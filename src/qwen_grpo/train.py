"""GRPO post-training of Qwen2.5 with PEFT/LoRA on a single GPU.

Pipeline (see project README-equivalent entry points):
  1. Build the reward-verifiable math dataset (data.py).
  2. Load Qwen2.5 and attach a LoRA adapter (PEFT).
  3. Train with TRL's GRPOTrainer:
       - num_generations candidates sampled per prompt (online rollout)
       - group-relative advantage computed from the rewards
       - PPO-style clipping on the old-policy probability ratio
       - KL regularization against a reference policy
  4. Save adapter checkpoints (safe serialization, PEFT format).

Run: qwen-grpo-train --model Qwen/Qwen2.5-1.5B-Instruct --output-dir outputs/grpo
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, fields
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from .data import build_datasets
from .rewards import correctness_reward, format_reward

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_OUTPUT_DIR = "outputs/grpo"


@dataclass
class TrainConfig:
    """All hyperparameters for one GRPO run, with sensible single-GPU defaults."""

    model_name: str = DEFAULT_MODEL
    output_dir: str = DEFAULT_OUTPUT_DIR

    # Dataset
    train_size: int = 7200
    test_size: int = 500
    seed: int = 42

    # Online rollout (group sampling)
    num_generations: int = 8      # candidates per prompt (group size)
    max_completion_length: int = 512

    # Generation sampling for rollouts
    temperature: float = 1.0
    top_p: float = 1.0

    # GRPO optimization
    learning_rate: float = 1e-5
    per_device_train_batch_size: int = 8   # completions per device batch
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 1
    beta: float = 0.04             # KL regularization coefficient
    epsilon: float = 0.2           # PPO-style clip epsilon
    loss_type: str = "dapo"        # TRL loss variant with importance-ratio clipping

    # PEFT / LoRA
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Memory: BF16 + gradient checkpointing
    bf16: bool = True
    gradient_checkpointing: bool = True

    # Logging / checkpointing
    logging_steps: int = 5
    save_steps: int = 100
    save_total_limit: int = 3
    log_completions: bool = True


def build_grpo_config(cfg: TrainConfig) -> GRPOConfig:
    """Translate TrainConfig into TRL's GRPOConfig."""
    use_bf16 = cfg.bf16 and torch.cuda.is_bf16_supported()

    return GRPOConfig(
        output_dir=cfg.output_dir,
        seed=cfg.seed,
        # Rollout
        num_generations=cfg.num_generations,
        max_completion_length=cfg.max_completion_length,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        # Optimization
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        beta=cfg.beta,
        epsilon=cfg.epsilon,
        loss_type=cfg.loss_type,
        # Precision & memory
        bf16=use_bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        # Logging & checkpoints
        logging_steps=logging_steps_arg(cfg.logging_steps),
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        log_completions=cfg.log_completions,
        report_to="none",
    )


def logging_steps_arg(v: int) -> int:
    """Keep logging_steps within the valid range expected by TrainingArguments."""
    return max(1, v)


def load_config(path: str | Path) -> TrainConfig:
    """Load a TrainConfig from a YAML file (see configs/default.yaml).

    Unknown keys in the file are rejected so a typo silently changing a
    hyperparameter can't slip through.
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    valid = {f.name for f in fields(TrainConfig)}
    unknown = set(data) - valid
    if unknown:
        raise ValueError(
            f"Unknown config keys in {path}: {sorted(unknown)}; valid keys: {sorted(valid)}"
        )
    return TrainConfig(**data)


def build_lora_config(cfg: TrainConfig) -> LoraConfig:
    """LoRA over attention projections; base model weights stay frozen."""
    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg.lora_target_modules,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="GRPO post-training for Qwen2.5")
    parser.add_argument(
        "--config",
        default=None,
        help="YAML config file (see configs/default.yaml); CLI flags override it",
    )
    parser.add_argument(
        "--model", default=None, help="Base model id (Hugging Face hub)"
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--num-generations", type=int, default=None)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args(argv)

    # Config file first (if given); explicit CLI flags then override it.
    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = TrainConfig()
    if args.model is not None:
        cfg.model_name = args.model
    if args.output_dir != DEFAULT_OUTPUT_DIR:
        cfg.output_dir = args.output_dir
    if args.train_size is not None:
        cfg.train_size = args.train_size
    if args.test_size is not None:
        cfg.test_size = args.test_size
    if args.num_generations is not None:
        cfg.num_generations = args.num_generations
    if args.epochs is not None:
        cfg.num_train_epochs = args.epochs
    if args.lr is not None:
        cfg.learning_rate = args.lr

    # 1. Data
    train_ds, _ = build_datasets(
        train_size=cfg.train_size, test_size=cfg.test_size, seed=cfg.seed
    )

    # 2. Base model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
    )

    # 3. GRPO trainer with LoRA (base weights frozen) and rule-based rewards
    trainer = GRPOTrainer(
        model=model,
        args=build_grpo_config(cfg),
        train_dataset=train_ds,
        processing_class=tokenizer,
        reward_funcs=[correctness_reward, format_reward],
        peft_config=build_lora_config(cfg),
    )
    trainer.train()

    # 4. Save final adapter
    trainer.save_model(cfg.output_dir)
    print(f"Saved GRPO adapter to {cfg.output_dir}")


if __name__ == "__main__":
    main()
