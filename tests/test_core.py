import csv
import tempfile
import unittest
from pathlib import Path

from math_challenge.data import is_validation, read_filtered_ids, read_problems, write_submission
from math_challenge.generation import generate_batch_solutions
from math_challenge.parsing import ensure_answer_tag, extract_answer
from math_challenge.training import CompletionDataset


class ParsingTests(unittest.TestCase):
    def test_explicit_answer_has_priority(self):
        self.assertEqual(extract_answer("2 + 3 = 5. <answer>-1,234</answer>"), -1234)

    def test_boxed_and_fallback(self):
        self.assertEqual(extract_answer(r"Thus \\boxed{42}"), 42)
        self.assertEqual(extract_answer("reason 10 then 27"), 27)

    def test_decimal_is_not_taken_as_whole_answer(self):
        self.assertIsNone(extract_answer("The approximation is 3.14"))

    def test_canonical_tag(self):
        self.assertEqual(ensure_answer_tag("work\n<answer>3</answer>", 3), "work\n<answer>3</answer>")


class DataTests(unittest.TestCase):
    def test_csv_round_trip_and_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.csv"
            train.write_text('id,question,answer\na,"1, plus 1",2\n', encoding="utf-8")
            problems = read_problems(train, require_answers=True)
            self.assertEqual((problems[0].id, problems[0].answer), ("a", 2))

            filtered = root / "filtered.csv"
            filtered.write_text("id\na\n", encoding="utf-8")
            self.assertEqual(read_filtered_ids(filtered), {"a"})

            output = root / "submission.csv"
            write_submission(output, [("a", 2)])
            with output.open(encoding="utf-8-sig", newline="") as handle:
                self.assertEqual(list(csv.reader(handle)), [["id", "answer"], ["a", "2"]])

    def test_split_is_stable(self):
        self.assertEqual(is_validation("train-123"), is_validation("train-123"))


class GenerationTests(unittest.TestCase):
    def test_batched_outputs_are_grouped_per_question(self):
        import torch

        decoded = [
            "<answer>1</answer>",
            "<answer>1</answer>",
            "<answer>2</answer>",
            "<answer>7</answer>",
            "<answer>8</answer>",
            "<answer>8</answer>",
        ]

        class Batch(dict):
            def to(self, _device):
                return self

        class Tokenizer:
            pad_token_id = 0
            eos_token_id = 0

            def apply_chat_template(self, messages, **_kwargs):
                return messages[-1]["content"]

            def __call__(self, prompts, **_kwargs):
                return Batch(
                    input_ids=torch.ones((len(prompts), 2), dtype=torch.long),
                    attention_mask=torch.ones((len(prompts), 2), dtype=torch.long),
                )

            def batch_decode(self, _tokens, **_kwargs):
                return decoded

        class Model:
            device = torch.device("cpu")

            def generate(self, input_ids, num_return_sequences, **_kwargs):
                rows = input_ids.shape[0] * num_return_sequences
                return torch.ones((rows, input_ids.shape[1] + 1), dtype=torch.long)

        results = generate_batch_solutions(
            Model(), Tokenizer(), ["q1", "q2"], num_samples=3
        )
        self.assertEqual(results[0].answer, 1)
        self.assertEqual(results[1].answer, 8)


class TrainingTests(unittest.TestCase):
    def test_truncation_preserves_masked_prompt_and_completion(self):
        class Tokenizer:
            eos_token = "E"

            def apply_chat_template(self, _messages, **_kwargs):
                return "P" * 300

            def encode(self, text, **_kwargs):
                return [1 if char == "P" else 2 for char in text]

        dataset = CompletionDataset([("question", "R" * 200)], Tokenizer(), max_length=256)
        example = dataset[0]
        self.assertEqual(len(example["input_ids"]), 256)
        self.assertTrue(all(label == -100 for label in example["labels"][:128]))
        self.assertTrue(all(label != -100 for label in example["labels"][128:]))


if __name__ == "__main__":
    unittest.main()
