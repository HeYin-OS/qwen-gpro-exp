"""Post-training evaluation: base model vs GRPO adapter.

Measures on a held-out test set:
  - answer accuracy   (parsed final answer matches ground truth)
  - format compliance (response ends with "#### <answer>")
  - generation length (mean completion tokens)

Run: qwen-grpo-eval --base Qwen/Qwen2.5-1.5B-Instruct --adapter outputs/grpo
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .answer_utils import answers_match, is_format_correct, parse_final_answer
from .data import TEST_SIZE, load_gsm8k_samples, split_train_test
from .prompts import build_chat_messages

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_ADAPTER_DIR = "outputs/grpo"


@dataclass
class EvalMetrics:
    """Aggregates reported for one model on the test set."""

    n: int
    accuracy: float
    format_compliance: float
    mean_completion_length: float


@torch.no_grad()
def generate_answers(
    model,
    tokenizer,
    problems: list[str],
    batch_size: int = 8,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> list[str]:
    """Greedy (or low-temperature) generation for a list of problems."""
    model.eval()
    completions: list[str] = []
    device = next(model.parameters()).device
    for i in range(0, len(problems), batch_size):
        batch = problems[i : i + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                build_chat_messages(p), tokenize=False, add_generation_prompt=True
            )
            for p in batch
        ]
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, padding_side="left"
        ).to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        # Strip the prompt tokens; keep only the completion.
        new_tokens = outputs[:, inputs["input_ids"].shape[1]:]
        completions.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    return completions


def compute_metrics(completions: list[str], answers: list[str]) -> EvalMetrics:
    n = len(answers)
    correct = sum(
        answers_match(parse_final_answer(c), a) for c, a in zip(completions, answers)
    )
    formatted = sum(is_format_correct(c) for c in completions)
    lengths = [len(c) for c in completions]
    return EvalMetrics(
        n=n,
        accuracy=correct / n if n else 0.0,
        format_compliance=formatted / n if n else 0.0,
        mean_completion_length=sum(lengths) / len(lengths) if lengths else 0.0,
    )


def evaluate_model(model, tokenizer, test_samples, batch_size: int) -> EvalMetrics:
    problems = [s.problem for s in test_samples]
    answers = [s.answer for s in test_samples]
    completions = generate_answers(model, tokenizer, problems, batch_size=batch_size)
    return compute_metrics(completions, answers)


def load_adapter(base_model_id: str, adapter_dir: str):
    """Load the base model and attach the trained LoRA adapter."""
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    return model


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate base vs GRPO model")
    parser.add_argument("--base", default=DEFAULT_MODEL, help="Base model id")
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER_DIR, help="LoRA adapter dir")
    parser.add_argument("--test-size", type=int, default=TEST_SIZE)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out", default="outputs/eval_results.json")
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Held-out test set (same split logic as training, so no leakage)
    samples = load_gsm8k_samples("train")
    _, test_samples = split_train_test(samples, test_size=args.test_size)

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    results = {}

    # Base model
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16
    ).to(device)
    results["base"] = asdict(evaluate_model(base_model, tokenizer, test_samples, args.batch_size))
    del base_model
    if device == "cuda":
        torch.cuda.empty_cache()

    # GRPO-tuned model (base + adapter)
    tuned_model = load_adapter(args.base, args.adapter).to(device)
    results["grpo"] = asdict(evaluate_model(tuned_model, tokenizer, test_samples, args.batch_size))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
