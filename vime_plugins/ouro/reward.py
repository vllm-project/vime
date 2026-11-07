"""Exact boxed-answer reward for short integer arithmetic validation."""

import re


def boxed_answer(response: str, label: str) -> int:
    answers = re.findall(r"\\boxed\{([^{}]+)\}", response.split("</think>")[-1])
    return int(bool(answers) and answers[-1].strip() == str(label).strip())
