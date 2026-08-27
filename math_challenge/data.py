from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True)
class MathProblem:
    id: str
    question: str
    answer: int | None = None


def _find_column(fieldnames: list[str], wanted: str) -> str | None:
    return next((name for name in fieldnames if name.strip().lower() == wanted), None)


def read_problems(path: str | Path, require_answers: bool = False) -> list[MathProblem]:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        id_column = _find_column(fields, "id")
        question_column = _find_column(fields, "question")
        answer_column = _find_column(fields, "answer")
        if not id_column or not question_column:
            raise ValueError(f"{path}: id/question columns are required; found {fields}")
        if require_answers and not answer_column:
            raise ValueError(f"{path}: answer column is required")

        problems: list[MathProblem] = []
        for line_number, row in enumerate(reader, start=2):
            raw_answer = (row.get(answer_column, "") if answer_column else "").strip()
            answer = None
            if raw_answer:
                try:
                    answer = int(raw_answer.replace(",", ""))
                except ValueError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: answer is not an integer: {raw_answer!r}"
                    ) from exc
            if require_answers and answer is None:
                raise ValueError(f"{path}:{line_number}: answer is empty")
            problems.append(
                MathProblem(
                    id=(row[id_column] or "").strip(),
                    question=(row[question_column] or "").strip(),
                    answer=answer,
                )
            )
    if not problems:
        raise ValueError(f"{path}: no rows found")
    return problems


def read_filtered_ids(path: str | Path | None) -> set[str]:
    if not path:
        return set()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        return set()
    start = 1 if rows[0] and rows[0][0].strip().lower() == "id" else 0
    return {row[0].strip() for row in rows[start:] if row and row[0].strip()}


def filter_problems(problems: Iterable[MathProblem], excluded_ids: set[str]) -> list[MathProblem]:
    return [problem for problem in problems if problem.id not in excluded_ids]


def is_validation(problem_id: str, fraction: float = 0.1) -> bool:
    if not 0.0 <= fraction < 1.0:
        raise ValueError("validation fraction must be in [0, 1)")
    bucket = int.from_bytes(hashlib.sha256(problem_id.encode("utf-8")).digest()[:8], "big")
    return bucket / 2**64 < fraction


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc


def write_submission(path: str | Path, predictions: Iterable[tuple[str, int]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["id", "answer"])
        for problem_id, answer in predictions:
            writer.writerow([problem_id, int(answer)])
