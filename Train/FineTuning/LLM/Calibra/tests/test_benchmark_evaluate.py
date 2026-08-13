import json

import pytest

from scripts.evaluate_benchmarks import evaluate_questions, format_prompt, load_questions


class TinyTokenizer:
    def __call__(self, text, add_special_tokens=True, return_tensors=None):
        # A deterministic character tokenizer is enough to exercise scoring.
        values = [ord(char) % 31 + 1 for char in text]
        if add_special_tokens:
            values = [2] + values
        return {"input_ids": values}


class TinyModel:
    class Output:
        def __init__(self, logits):
            self.logits = logits

    def __init__(self):
        import torch
        self._parameter = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self._parameter])

    def __call__(self, input_ids, attention_mask=None):
        import torch
        # Make token 1 the most likely next token at every position.
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 64)
        logits[..., 1] = 1.0
        return self.Output(logits)


def test_load_questions_and_prompt(tmp_path):
    split = tmp_path / "val"
    split.mkdir()
    (split / "computer.csv").write_text(
        "question,A,B,C,D,answer\n何为CPU,中央处理器,内存,硬盘,显卡,A\n", encoding="utf-8"
    )
    questions = load_questions(tmp_path, "val")
    assert questions[0]["subject"] == "computer"
    assert questions[0]["answer"] == "A"
    assert "A. 中央处理器" in format_prompt(questions[0], "ceval")


def test_evaluate_questions_returns_subject_and_overall_metrics():
    question = {"subject": "demo", "question": "x", "A": "a", "B": "b", "C": "c", "D": "d", "answer": "A"}
    result = evaluate_questions(TinyModel(), TinyTokenizer(), [question], "cmmlu", 64, "cpu")
    assert result["total"] == 1
    assert result["correct"] in {0, 1}
    assert result["subjects"]["demo"]["total"] == 1
    assert result["predictions"][0]["prediction"] in {"A", "B", "C", "D"}


def test_load_questions_rejects_missing_options(tmp_path):
    (tmp_path / "bad.json").write_text(json.dumps({"question": "q", "A": "a"}), encoding="utf-8")
    with pytest.raises(ValueError, match="question,A,B,C,D"):
        load_questions(tmp_path)
