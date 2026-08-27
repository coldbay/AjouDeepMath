from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .parsing import extract_answer
from .prompting import messages_for


@dataclass
class GenerationResult:
    answer: int | None
    candidates: list[int]
    texts: list[str]


def generate_solutions(
    model,
    tokenizer,
    question: str,
    num_samples: int = 1,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> GenerationResult:
    return generate_batch_solutions(
        model,
        tokenizer,
        [question],
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )[0]


def generate_batch_solutions(
    model,
    tokenizer,
    questions: list[str],
    num_samples: int = 1,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> list[GenerationResult]:
    import torch

    if not questions:
        return []
    prompts = [
        tokenizer.apply_chat_template(
            messages_for(question), tokenize=False, add_generation_prompt=True
        )
        for question in questions
    ]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    sampled = num_samples > 1
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": sampled,
        "num_return_sequences": num_samples,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if sampled:
        generation_kwargs.update(temperature=temperature, top_p=top_p)
    with torch.inference_mode():
        outputs = model.generate(**inputs, **generation_kwargs)
    prompt_length = inputs["input_ids"].shape[1]
    texts = tokenizer.batch_decode(outputs[:, prompt_length:], skip_special_tokens=True)
    results = []
    for question_index in range(len(questions)):
        start = question_index * num_samples
        group_texts = texts[start : start + num_samples]
        candidates = [
            answer for text in group_texts if (answer := extract_answer(text)) is not None
        ]
        if not candidates:
            results.append(GenerationResult(answer=None, candidates=[], texts=group_texts))
            continue
        counts = Counter(candidates)
        first_position = {answer: candidates.index(answer) for answer in counts}
        winner = max(counts, key=lambda answer: (counts[answer], -first_position[answer]))
        results.append(
            GenerationResult(answer=winner, candidates=candidates, texts=group_texts)
        )
    return results
