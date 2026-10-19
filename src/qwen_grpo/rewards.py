"""Reward functions for GRPO.

Two verifiable, rule-based rewards (no judge model):
  - correctness: the parsed final answer matches the ground truth
  - format     : the response ends with the required "#### <answer>" line

TRL's GRPOTrainer calls these with (prompts, completions, **dataset_cols);
the extra `ground_truth` column from data.py is forwarded as a keyword.
"""

from __future__ import annotations

from .answer_utils import answers_match, is_format_correct, parse_final_answer

CORRECTNESS_REWARD = 1.0
FORMAT_REWARD = 0.2


def correctness_reward(
    completions: list[str],
    ground_truth: list[str],
    **kwargs,
) -> list[float]:
    """1.0 when the final answer matches the ground truth, else 0.0."""
    rewards = []
    for completion, gt in zip(completions, ground_truth):
        predicted = parse_final_answer(completion)
        rewards.append(CORRECTNESS_REWARD if answers_match(predicted, gt) else 0.0)
    return rewards


def format_reward(completions: list[str], **kwargs) -> list[float]:
    """A small reward for following the required answer format."""
    return [FORMAT_REWARD if is_format_correct(c) else 0.0 for c in completions]
