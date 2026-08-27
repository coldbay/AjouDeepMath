from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .data import (
    filter_problems,
    is_validation,
    read_filtered_ids,
    read_problems,
    write_submission,
)


def _data_arguments(parser, answers_required=True):
    parser.add_argument("--train" if answers_required else "--test", required=True, type=Path)
    if answers_required:
        parser.add_argument("--filter-ids", type=Path, default=None)


def _model_arguments(parser, adapter=True):
    parser.add_argument("--model-path", type=Path, default=None)
    if adapter:
        parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument(
        "--quantization",
        choices=("bf16", "4bit"),
        default="bf16",
        help="A5000 기본값은 bf16 LoRA이며 4bit는 QLoRA입니다",
    )


def _generation_arguments(parser, default_samples=5):
    parser.add_argument("--num-samples", type=int, default=default_samples)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--generation-batch-size", type=int, default=4)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen2.5-3B LLM Math Challenge pipeline")
    parser.add_argument("--seed", type=int, default=42)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect", help="데이터와 필터 적용 결과 확인")
    _data_arguments(inspect)

    bootstrap = subparsers.add_parser("bootstrap", help="정답으로 검증된 자기 풀이 생성")
    _data_arguments(bootstrap)
    _model_arguments(bootstrap, adapter=False)
    bootstrap.add_argument("--output", type=Path, default=Path("artifacts/pseudo_solutions.jsonl"))
    bootstrap.add_argument("--samples-per-question", type=int, default=4)
    bootstrap.add_argument("--max-new-tokens", type=int, default=512)
    bootstrap.add_argument("--temperature", type=float, default=0.8)
    bootstrap.add_argument("--top-p", type=float, default=0.95)
    bootstrap.add_argument("--generation-batch-size", type=int, default=4)
    bootstrap.add_argument("--max-rows", type=int, default=0)
    bootstrap.add_argument("--validation-fraction", type=float, default=0.1)
    bootstrap.add_argument("--include-validation", action="store_true")
    bootstrap.add_argument("--overwrite", action="store_true")

    train = subparsers.add_parser("train", help="BF16 LoRA / 4-bit QLoRA SFT")
    _data_arguments(train)
    _model_arguments(train, adapter=False)
    train.add_argument("--pseudo-data", type=Path, default=None)
    train.add_argument("--max-pseudo-per-problem", type=int, default=2)
    train.add_argument("--output-dir", type=Path, default=Path("artifacts/qwen-math-lora"))
    train.add_argument("--validation-fraction", type=float, default=0.1)
    train.add_argument("--train-all", action="store_true")
    train.add_argument("--max-length", type=int, default=2048)
    train.add_argument("--epochs", type=float, default=2.0)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--gradient-accumulation-steps", type=int, default=8)
    train.add_argument("--lora-r", type=int, default=32)
    train.add_argument("--lora-alpha", type=int, default=64)
    train.add_argument("--lora-dropout", type=float, default=0.05)
    train.add_argument("--resume-from-checkpoint", type=Path, default=None)

    evaluate = subparsers.add_parser("evaluate", help="고정 검증 세트 Exact Match 평가")
    _data_arguments(evaluate)
    _model_arguments(evaluate)
    _generation_arguments(evaluate)
    evaluate.add_argument("--validation-fraction", type=float, default=0.1)
    evaluate.add_argument("--max-rows", type=int, default=0)
    evaluate.add_argument("--output", type=Path, default=Path("artifacts/validation_predictions.jsonl"))

    predict = subparsers.add_parser("predict", help="submission.csv 생성")
    _data_arguments(predict, answers_required=False)
    _model_arguments(predict)
    _generation_arguments(predict)
    predict.add_argument("--output", type=Path, default=Path("submission.csv"))
    return parser


def _load_train(args):
    problems = read_problems(args.train, require_answers=True)
    excluded = read_filtered_ids(args.filter_ids)
    filtered = filter_problems(problems, excluded)
    print(f"학습 데이터: {len(problems)}개, 오류 제외: {len(problems) - len(filtered)}개")
    return filtered


def _load_runtime(args, for_training=False):
    import torch

    from .modeling import load_model, load_tokenizer, resolve_model_path

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model_path = resolve_model_path(args.model_path)
    args.model_path = model_path
    print(f"베이스 모델: {model_path}")
    tokenizer = load_tokenizer(model_path)
    model = load_model(
        model_path,
        adapter_path=getattr(args, "adapter_path", None),
        quantization=args.quantization,
        for_training=for_training,
    )
    return model, tokenizer


def _validate_generation_args(args):
    if args.num_samples < 1:
        raise ValueError("--num-samples must be >= 1")
    if args.generation_batch_size < 1:
        raise ValueError("--generation-batch-size must be >= 1")
    if args.num_samples > 1 and args.temperature <= 0:
        raise ValueError("sampling requires --temperature > 0")


def _validate_training_args(args):
    if args.max_length < 256:
        raise ValueError("--max-length must be >= 256")
    if args.batch_size < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError("batch sizes must be >= 1")
    if args.max_pseudo_per_problem < 1:
        raise ValueError("--max-pseudo-per-problem must be >= 1")
    if args.epochs <= 0 or args.learning_rate <= 0:
        raise ValueError("epochs and learning rate must be > 0")


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    random.seed(args.seed)

    if args.command == "inspect":
        problems = _load_train(args)
        validation = sum(is_validation(p.id) for p in problems)
        lengths = sorted(len(p.question) for p in problems)
        print(
            json.dumps(
                {
                    "usable_rows": len(problems),
                    "validation_rows": validation,
                    "question_chars": {
                        "min": lengths[0],
                        "median": lengths[len(lengths) // 2],
                        "max": lengths[-1],
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "bootstrap":
        from .bootstrap import run_bootstrap

        if args.samples_per_question < 1 or args.generation_batch_size < 1:
            raise ValueError("sample count and generation batch size must be >= 1")
        problems = _load_train(args)
        model, tokenizer = _load_runtime(args)
        run_bootstrap(args, problems, model, tokenizer)
        return

    if args.command == "train":
        from .training import run_training

        _validate_training_args(args)
        problems = _load_train(args)
        model, tokenizer = _load_runtime(args, for_training=True)
        run_training(args, problems, model, tokenizer)
        return

    if args.command in {"evaluate", "predict"}:
        from .generation import generate_batch_solutions

        _validate_generation_args(args)
        model, tokenizer = _load_runtime(args)
        if args.command == "evaluate":
            problems = [
                problem
                for problem in _load_train(args)
                if is_validation(problem.id, args.validation_fraction)
            ]
            if args.max_rows:
                problems = problems[: args.max_rows]
        else:
            problems = read_problems(args.test, require_answers=False)

        predictions = []
        details = []
        correct = 0
        processed = 0
        for batch_start in range(0, len(problems), args.generation_batch_size):
            batch = problems[batch_start : batch_start + args.generation_batch_size]
            results = generate_batch_solutions(
                model,
                tokenizer,
                [problem.question for problem in batch],
                num_samples=args.num_samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            for problem, result in zip(batch, results):
                processed += 1
                answer = result.answer if result.answer is not None else 0
                predictions.append((problem.id, answer))
                if args.command == "evaluate":
                    correct += int(answer == problem.answer)
                    details.append(
                        {
                            "id": problem.id,
                            "gold": problem.answer,
                            "prediction": answer,
                            "candidates": result.candidates,
                            "correct": answer == problem.answer,
                        }
                    )
                    print(
                        f"[{processed}/{len(problems)}] accuracy={correct / processed:.4f} "
                        f"id={problem.id} pred={answer} gold={problem.answer}",
                        flush=True,
                    )
                else:
                    print(
                        f"[{processed}/{len(problems)}] {problem.id} -> {answer}",
                        flush=True,
                    )

        if args.command == "evaluate":
            if not problems:
                raise ValueError("검증 세트가 비어 있습니다. validation fraction/max rows를 확인하세요.")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8") as handle:
                for record in details:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"Exact Match: {correct}/{len(problems)} = {correct / len(problems):.6f}")
        else:
            write_submission(args.output, predictions)
            print(f"제출 파일 저장: {args.output.resolve()}")


if __name__ == "__main__":
    main()
