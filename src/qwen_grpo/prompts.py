"""Instruction template and answer format shared by training and evaluation.

All prompts ask the model to reason freely, then give the final answer in a
verifiable format so the reward functions can parse it without a judge model.
"""

SYSTEM_PROMPT = "You are a helpful assistant that solves math problems step by step."

# The model must end its response with a line of the form:
#   #### <answer>
# e.g. "#### 42" or "#### 3/4". Anything before the marker is treated as
# reasoning and is not parsed.
ANSWER_MARKER = "####"

INSTRUCTION_TEMPLATE = (
    "Answer the math problem below.\n"
    "Think step by step, then give your final answer on the last line in the exact format:\n"
    f"{ANSWER_MARKER} <final_answer>\n\n"
    "Problem:\n"
    "{problem}"
)


def build_prompt(problem: str) -> str:
    """Render the instruction template for a single problem."""
    return INSTRUCTION_TEMPLATE.format(problem=problem)


def build_chat_messages(problem: str) -> list[dict]:
    """Chat-formatted messages for tokenizer.apply_chat_template."""
    return [{"role": "user", "content": build_prompt(problem)}]
