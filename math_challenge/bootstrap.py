from __future__ import annotations

import json
from pathlib import Path

from .data import append_jsonl, is_validation, read_jsonl
from .generation import generate_batch_solutions


def run_bootstrap(args, problems, model, tokenizer) -> None:
    output = Path(args.output)
    progress = output.with_suffix(output.suffix + ".progress")
    completed: set[str] = set()
    if output.exists() and not args.overwrite:
        completed = {str(record["id"]) for record in read_jsonl(output)}
    if progress.exists() and not args.overwrite:
        completed.update(str(record["id"]) for record in read_jsonl(progress))
    if completed:
        print(f"기존 처리 완료 {len(completed)}개 문항을 건너뜁니다: {output}")
    if args.overwrite:
        output.unlink(missing_ok=True)
        progress.unlink(missing_ok=True)

    eligible = [
        problem
        for problem in problems
        if problem.id not in completed
        and (args.include_validation or not is_validation(problem.id, args.validation_fraction))
    ]
    if args.max_rows:
        eligible = eligible[: args.max_rows]

    accepted = 0
    processed = 0
    for batch_start in range(0, len(eligible), args.generation_batch_size):
        batch = eligible[batch_start : batch_start + args.generation_batch_size]
        results = generate_batch_solutions(
            model,
            tokenizer,
            [problem.question for problem in batch],
            num_samples=args.samples_per_question,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        for problem, result in zip(batch, results):
            records = []
            seen: set[str] = set()
            for generated_text in result.texts:
                from .parsing import ensure_answer_tag, extract_answer

                if extract_answer(generated_text) == problem.answer:
                    solution = ensure_answer_tag(generated_text, int(problem.answer))
                    if solution not in seen:
                        seen.add(solution)
                        records.append(
                            {
                                "id": problem.id,
                                "question": problem.question,
                                "answer": problem.answer,
                                "solution": solution,
                            }
                        )
            append_jsonl(output, records)
            append_jsonl(progress, [{"id": problem.id}])
            accepted += len(records)
            processed += 1
            print(
                f"[{processed}/{len(eligible)}] {problem.id}: "
                f"{len(records)}/{len(result.texts)} accepted (total={accepted})",
                flush=True,
            )

    accepted_total = sum(1 for _ in read_jsonl(output)) if output.exists() else 0
    metadata = {
        "source_model": str(args.model_path),
        "input": str(args.train),
        "samples_per_question": args.samples_per_question,
        "accepted_solutions": accepted_total,
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
    }
    output.with_suffix(".meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
