from __future__ import annotations

import re


_INTEGER = r"[+-]?\d(?:[\d,\s]*\d)?|[+-]?\d"
_ANSWER_PATTERNS = (
    re.compile(rf"<answer>\s*({_INTEGER})\s*</answer>", re.IGNORECASE),
    re.compile(rf"\\boxed\s*\{{\s*({_INTEGER})\s*\}}", re.IGNORECASE),
    re.compile(
        rf"(?:final\s+answer|answer|정답|답)\s*(?:is|=|:|：)?\s*({_INTEGER})",
        re.IGNORECASE,
    ),
)
_ANY_INTEGER = re.compile(r"(?<![\w.])[+-]?\d[\d,]*(?![\w.])")


def normalize_integer(value: str) -> int | None:
    """Parse a plain integer while rejecting decimals and other expressions."""
    compact = re.sub(r"[\s,]", "", value.strip())
    if not re.fullmatch(r"[+-]?\d+", compact):
        return None
    try:
        return int(compact)
    except ValueError:
        return None


def extract_answer(text: str) -> int | None:
    """Extract the last, most explicit integer answer from generated text."""
    for pattern in _ANSWER_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            answer = normalize_integer(matches[-1])
            if answer is not None:
                return answer

    matches = _ANY_INTEGER.findall(text)
    return normalize_integer(matches[-1]) if matches else None


def ensure_answer_tag(solution: str, answer: int) -> str:
    """Return a completion ending in one canonical answer tag."""
    solution = re.sub(
        r"\s*<answer>\s*[+-]?[\d,\s]+\s*</answer>\s*$",
        "",
        solution.strip(),
        flags=re.IGNORECASE,
    )
    return f"{solution}\n<answer>{answer}</answer>" if solution else f"<answer>{answer}</answer>"
