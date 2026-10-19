"""Answer parsing and verification used by the reward functions.

The policy is asked to end every response with "#### <answer>" (see
prompts.py). Parsing extracts that final line; verification then compares
the extracted answer against the ground truth with some numeric
normalization so "42", "42.0", " 42 " and "42$" all match.
"""

from __future__ import annotations

import re
from fractions import Fraction

from .prompts import ANSWER_MARKER

# Matches the last "#### ..." line of a completion.
_ANSWER_LINE_RE = re.compile(
    rf"^\s*{re.escape(ANSWER_MARKER)}\s*(.+?)\s*$", re.MULTILINE
)


def parse_final_answer(completion: str) -> str | None:
    """Extract the answer from the last '#### <answer>' line, if present."""
    matches = _ANSWER_LINE_RE.findall(completion)
    if not matches:
        return None
    return matches[-1].strip()


def _to_number(text: str) -> Fraction | None:
    """Parse a string into an exact Fraction for numeric comparison."""
    cleaned = (
        text.replace(",", "")
        .replace("$", "")
        .replace("%", "")
        .strip()
        .rstrip(".")
    )
    if not cleaned:
        return None
    try:
        return Fraction(cleaned)
    except (ValueError, ZeroDivisionError):
        return None


def answers_match(predicted: str | None, ground_truth: str) -> bool:
    """Check whether a parsed answer is equivalent to the ground truth.

    Numeric comparison is exact (Fractions), so "1/2" == "0.5" == "0.50".
    Falls back to case-insensitive string equality for non-numeric answers.
    """
    if predicted is None:
        return False
    pred_num = _to_number(predicted)
    gt_num = _to_number(ground_truth)
    if pred_num is not None and gt_num is not None:
        return pred_num == gt_num
    return predicted.strip().lower() == ground_truth.strip().lower()


def is_format_correct(completion: str) -> bool:
    """True if the completion contains a parseable '#### <answer>' line."""
    return parse_final_answer(completion) is not None
