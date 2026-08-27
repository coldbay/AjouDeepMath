from __future__ import annotations


SYSTEM_PROMPT = """You are a careful mathematical problem solver.
Solve the problem step by step and check arithmetic and constraints before answering.
The official answer is always an integer.
End with exactly <answer>INTEGER</answer>. Do not write anything after the closing tag."""


def messages_for(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
