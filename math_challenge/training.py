from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .data import is_validation, read_jsonl
from .parsing import ensure_answer_tag, extract_answer
from .prompting import messages_for


class CompletionDataset:
    def __init__(self, examples, tokenizer, max_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        import torch

        question, completion = self.examples[index]
        prompt = self.tokenizer.apply_chat_template(
            messages_for(question), tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        completion_ids = self.tokenizer.encode(
            completion + self.tokenizer.eos_token, add_special_tokens=False
        )

        if len(prompt_ids) + len(completion_ids) > self.max_length:
            # Preserve the chat prefix/question tail and, most importantly, the final answer.
            completion_budget = min(len(completion_ids), max(128, self.max_length // 2))
            completion_ids = completion_ids[-completion_budget:]
            prompt_budget = self.max_length - completion_budget
            prefix_budget = min(128, prompt_budget)
            tail_budget = prompt_budget - prefix_budget
            prompt_prefix = prompt_ids[:prefix_budget]
            prompt_tail = prompt_ids[-tail_budget:] if tail_budget else []
            prompt_ids = prompt_prefix
            if tail_budget:
                prompt_ids += prompt_tail
        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids.copy()
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class CompletionCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        import torch

        max_length = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            batch["input_ids"].append(
                torch.cat(
                    [feature["input_ids"], torch.full((padding,), self.pad_token_id)]
                )
            )
            batch["attention_mask"].append(
                torch.cat([feature["attention_mask"], torch.zeros(padding, dtype=torch.long)])
            )
            batch["labels"].append(
                torch.cat([feature["labels"], torch.full((padding,), -100, dtype=torch.long)])
            )
        return {key: torch.stack(value) for key, value in batch.items()}


def build_examples(
    problems,
    pseudo_path,
    validation_fraction: float,
    train_all: bool,
    max_pseudo_per_problem: int,
):
    pseudo = defaultdict(list)
    if pseudo_path:
        for record in read_jsonl(pseudo_path):
            answer = int(record["answer"])
            solution = str(record["solution"])
            if extract_answer(solution) == answer:
                pseudo[str(record["id"])].append(ensure_answer_tag(solution, answer))

    examples = []
    covered = 0
    for problem in problems:
        if not train_all and is_validation(problem.id, validation_fraction):
            continue
        solutions = pseudo.get(problem.id, [])[:max_pseudo_per_problem]
        if solutions:
            covered += 1
            examples.extend((problem.question, solution) for solution in solutions)
        else:
            examples.append((problem.question, f"<answer>{problem.answer}</answer>"))
    return examples, covered


def run_training(args, problems, model, tokenizer) -> None:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Trainer, TrainingArguments

    examples, covered = build_examples(
        problems,
        args.pseudo_data,
        args.validation_fraction,
        args.train_all,
        args.max_pseudo_per_problem,
    )
    print(f"학습 예제 {len(examples)}개 (검증된 풀이 보유 문항 {covered}개)")
    dataset = CompletionDataset(examples, tokenizer, args.max_length)

    if args.quantization == "4bit":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    model.print_trainable_parameters()

    output = Path(args.output_dir)
    training_args = TrainingArguments(
        output_dir=str(output),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit" if args.quantization == "4bit" else "adamw_torch_fused",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=CompletionCollator(tokenizer.pad_token_id),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    metadata = {
        "base_model": str(args.model_path),
        "train_data": str(args.train),
        "filtered_ids": str(args.filter_ids) if args.filter_ids else None,
        "pseudo_data": str(args.pseudo_data) if args.pseudo_data else None,
        "example_count": len(examples),
        "validation_fraction": args.validation_fraction,
        "train_all": args.train_all,
        "hyperparameters": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "quantization": args.quantization,
            "max_pseudo_per_problem": args.max_pseudo_per_problem,
        },
    }
    (output / "challenge_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
