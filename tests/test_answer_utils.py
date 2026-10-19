import pytest

from qwen_grpo.answer_utils import (
    answers_match,
    is_format_correct,
    parse_final_answer,
)
from qwen_grpo.prompts import ANSWER_MARKER, build_prompt
from qwen_grpo.rewards import correctness_reward, format_reward


class TestParseFinalAnswer:
    def test_parses_marker_line(self):
        completion = "Let me think.\n2 + 2 = 4.\n#### 4"
        assert parse_final_answer(completion) == "4"

    def test_uses_last_marker_line(self):
        completion = "First draft.\n#### 3\nOn reflection.\n#### 5"
        assert parse_final_answer(completion) == "5"

    def test_missing_marker_returns_none(self):
        assert parse_final_answer("The answer is 4.") is None

    def test_empty_answer_returns_none(self):
        assert parse_final_answer("reasoning\n####") is None


class TestAnswersMatch:
    @pytest.mark.parametrize(
        "pred,gt",
        [
            ("4", "4"),
            (" 4 ", "4"),
            ("4.0", "4"),
            ("0.5", "1/2"),
            ("1,000", "1000"),
            ("$42", "42"),
        ],
    )
    def test_numeric_equivalences(self, pred, gt):
        assert answers_match(pred, gt) is True

    def test_none_never_matches(self):
        assert answers_match(None, "4") is False

    def test_string_fallback_case_insensitive(self):
        assert answers_match("Yes", "yes") is True

    def test_wrong_number(self):
        assert answers_match("5", "4") is False


class TestFormatReward:
    def test_formatted_gets_reward(self):
        assert format_reward(["thinking...\n#### 42"]) == [0.2]

    def test_unformatted_gets_zero(self):
        assert format_reward(["no marker here"]) == [0.0]


class TestCorrectnessReward:
    def test_correct_answer(self):
        rewards = correctness_reward(
            ["#### 18"], ground_truth=["18"]
        )
        assert rewards == [1.0]

    def test_wrong_answer(self):
        rewards = correctness_reward(
            ["#### 19"], ground_truth=["18"]
        )
        assert rewards == [0.0]

    def test_batch_mixed(self):
        rewards = correctness_reward(
            ["#### 18", "no marker", "#### 20.0"],
            ground_truth=["18", "18", "20"],
        )
        assert rewards == [1.0, 0.0, 1.0]


class TestPromptTemplate:
    def test_template_contains_marker(self):
        prompt = build_prompt("What is 2+2?")
        assert ANSWER_MARKER in prompt
        assert "What is 2+2?" in prompt
