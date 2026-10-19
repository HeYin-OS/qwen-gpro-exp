"""Dataset construction for GRPO post-training.

Each training example carries:
  - prompt       : the rendered instruction (see prompts.py)
  - ground_truth : the canonical answer string, e.g. "42"
  - rule         : optional verification rule name (e.g. "math_verify")

The reward functions (rewards.py) consume the ground truth and compare it to
whatever the policy generated inside the GRPO rollout.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from datasets import Dataset, load_dataset

from .prompts import build_prompt

# Split sizes for the default GSM8K-style dataset.
TRAIN_SIZE = 7200
TEST_SIZE = 500
SEED = 42


@dataclass
class Sample:
    """A single problem with its verifiable answer."""

    problem: str
    answer: str
    rule: str = "math_verify"


def _extract_gsm8k_answer(answer_with_rationale: str) -> str:
    """Pull the numeric answer out of a GSM8K '<<...>>' annotation.

    GSM8K answers look like: "... reasoning ... #### 18".
    """
    if "####" in answer_with_rationale:
        return answer_with_rationale.split("####", 1)[1].strip()
    return answer_with_rationale.strip()


def load_gsm8k_samples(split: str = "train") -> list[Sample]:
    """Load GSM8K as Sample objects with prompt-normalized answers."""
    raw = load_dataset("openai/gsm8k", "main", split=split)
    return [
        Sample(problem=row["question"], answer=_extract_gsm8k_answer(row["answer"]))
        for row in raw
    ]


def split_train_test(
    samples: list[Sample],
    test_size: int = TEST_SIZE,
    seed: int = SEED,
) -> tuple[list[Sample], list[Sample]]:
    """Shuffle and split samples into train/test lists (deterministic)."""
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    if len(shuffled) <= test_size:
        raise ValueError(
            f"Need more than {test_size} samples for a test split, got {len(shuffled)}"
        )
    return shuffled[test_size:], shuffled[:test_size]


def samples_to_dataset(samples: list[Sample]) -> Dataset:
    """Convert Sample objects to a Hugging Face Dataset with prompt columns."""
    return Dataset.from_list(
        [
            {"prompt": build_prompt(s.problem), "ground_truth": s.answer, "rule": s.rule}
            for s in samples
        ]
    )


def build_datasets(
    train_size: int = TRAIN_SIZE,
    test_size: int = TEST_SIZE,
    seed: int = SEED,
) -> tuple[Dataset, Dataset]:
    """Build the train and evaluation datasets used by GRPO training.

    Returns (train_ds, test_ds) where train_ds has a `prompt` column plus
    reward metadata, and test_ds keeps raw problem/answer fields for
    accuracy evaluation (see evaluate.py).
    """
    samples = load_gsm8k_samples("train")
    train_samples, test_samples = split_train_test(samples, test_size=test_size, seed=seed)
    train_samples = train_samples[:train_size]

    train_ds = samples_to_dataset(train_samples)

    test_ds = Dataset.from_list(
        [
            {"problem": s.problem, "answer": s.answer}
            for s in test_samples
        ]
    )
    return train_ds, test_ds
