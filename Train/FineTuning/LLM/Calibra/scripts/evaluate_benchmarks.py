"""Evaluate causal language models on C-Eval and CMMLU.

Both benchmarks are multiple-choice exams.  This script scores the four answer
letters by continuation log-likelihood, which avoids brittle free-form answer
parsing and works for Chinese tokenizers as well as English ones.

Examples::

    python scripts/evaluate_benchmarks.py --config configs/sft_config.yaml \
        --checkpoint output/sft/final --benchmark ceval --data-dir data/ceval
    python scripts/evaluate_benchmarks.py --config configs/sft_config.yaml \
        --benchmark both --data-dir data/benchmarks --split val

The expected files follow the official layouts (``<split>/*.csv``), with
columns ``question,A,B,C,D,answer``.  JSON/JSONL records with the same keys are
also accepted.  Test files without an ``answer`` column are loaded but are not
scored; use ``--predictions`` to save model predictions for those files.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

# Make ``python scripts/evaluate_benchmarks.py`` work without installing first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CHOICES = ("A", "B", "C", "D")


def _read_file(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            return [dict(row) for row in csv.DictReader(stream)]
    if suffix == ".jsonl":
        records = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} must contain an object")
                records.append(value)
        return records
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(value, dict) and isinstance(value.get("data"), list):
            value = value["data"]
        if isinstance(value, dict):
            value = [value]
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise ValueError(f"{path} must contain an object, list, or data list")
        return value
    return []


def discover_files(data_dir: str | Path, split: str = "val") -> list[Path]:
    """Find benchmark files in an official split directory or a flat export."""
    root = Path(data_dir)
    if root.is_file():
        return [root]
    split_dir = root / split
    if split_dir.is_dir():
        files = [p for p in split_dir.iterdir() if p.suffix.lower() in {".csv", ".json", ".jsonl"}]
    else:
        files = [
            p for p in root.rglob("*")
            if p.is_file()
            and p.suffix.lower() in {".csv", ".json", ".jsonl"}
            and (p.parent.name.lower() == split.lower() or split.lower() in p.stem.lower())
        ]
        if not files:
            files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in {".csv", ".json", ".jsonl"}]
    return sorted(set(files[0:2]))
    # return sorted(set(files))


def load_questions(data_dir: str | Path, split: str = "val", subjects: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """Load and normalize questions from C-Eval/CMMLU files."""
    wanted = {item.strip().lower() for item in subjects or () if item.strip()}
    questions: list[dict[str, Any]] = []
    for path in discover_files(data_dir, split):
        file_subject = path.stem
        for index, raw in enumerate(_read_file(path), 1):
            row = {str(key).strip().lower(): value for key, value in raw.items()}
            options = {choice: str(row.get(choice.lower(), "")).strip() for choice in CHOICES}
            if not row.get("question") or any(not options[choice] for choice in CHOICES):
                raise ValueError(f"{path}:{index} must contain question,A,B,C,D columns")
            subject = str(row.get("subject") or file_subject).strip()
            if wanted and subject.lower() not in wanted and file_subject.lower() not in wanted:
                continue
            answer = row.get("answer")
            answer = str(answer).strip().upper() if answer is not None and str(answer).strip() else None
            if answer in {"0", "1", "2", "3"}:
                answer = CHOICES[int(answer)]
            if answer not in CHOICES:
                answer = None
            questions.append({"subject": subject, "question": str(row["question"]).strip(), **options, "answer": answer, "source": str(path)})
    if not questions:
        raise ValueError(f"No benchmark questions found under {Path(data_dir).resolve()} (split={split!r})")
    return questions


def format_prompt(question: Mapping[str, Any], benchmark: str) -> str:
    """Format a question using the prompt convention used by both benchmarks."""
    subject = question.get("subject", "")
    intro = "下面是中国关于" + str(subject) + "的单项选择题，请选出其中的正确答案。"
    options = "\n".join(f"{choice}. {question[choice]}" for choice in CHOICES)
    return f"{intro}\n题目：{question['question']}\n{options}\n答案："


def _continuation_score(model: Any, tokenizer: Any, prompt: str, continuation: str, max_length: int, device: Any) -> float:
    """Return sum log P(continuation | prompt), truncating only old context."""
    import torch

    prompt_ids = tokenizer(prompt, add_special_tokens=True, return_tensors=None)["input_ids"]
    option_ids = tokenizer(continuation, add_special_tokens=False, return_tensors=None)["input_ids"]
    if prompt_ids and isinstance(prompt_ids[0], list):
        prompt_ids = prompt_ids[0]
    if option_ids and isinstance(option_ids[0], list):
        option_ids = option_ids[0]
    if not option_ids:
        raise ValueError("Tokenizer produced no tokens for answer choice")
    prompt_ids = list(prompt_ids)
    option_ids = list(option_ids)
    prompt_ids = prompt_ids[-max(1, max_length - len(option_ids)):]
    input_ids = torch.tensor([prompt_ids + option_ids], dtype=torch.long, device=device)
    attention = torch.ones_like(input_ids)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention).logits
    log_probs = torch.log_softmax(logits[0, :-1], dim=-1)
    start = len(prompt_ids) - 1
    targets = torch.tensor(option_ids, dtype=torch.long, device=device)
    return float(log_probs[start:start + len(option_ids)].gather(1, targets[:, None]).sum().item())


def predict_choice(model: Any, tokenizer: Any, question: Mapping[str, Any], benchmark: str, max_length: int, device: Any) -> tuple[str, dict[str, float]]:
    prompt = format_prompt(question, benchmark)
    scores = {choice: _continuation_score(model, tokenizer, prompt, choice, max_length, device) for choice in CHOICES}
    return max(scores, key=scores.get), scores


def evaluate_questions(model: Any, tokenizer: Any, questions: Iterable[Mapping[str, Any]], benchmark: str, max_length: int, device: Any) -> dict[str, Any]:
    by_subject: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0, "predicted": 0})
    predictions = []
    for question in questions:
        predicted, scores = predict_choice(model, tokenizer, question, benchmark, max_length, device)
        answer = question.get("answer")
        stats = by_subject[str(question.get("subject", "unknown"))]
        stats["predicted"] += 1
        if answer in CHOICES:
            stats["total"] += 1
            stats["correct"] += int(predicted == answer)
        predictions.append({**dict(question), "prediction": predicted, "scores": scores})
    total = sum(item["total"] for item in by_subject.values())
    correct = sum(item["correct"] for item in by_subject.values())
    subjects_result = {
        subject: {**stats, "accuracy": stats["correct"] / stats["total"] if stats["total"] else None}
        for subject, stats in sorted(by_subject.items())
    }
    return {"benchmark": benchmark, "subjects": subjects_result, "correct": correct, "total": total, "accuracy": correct / total if total else None, "predictions": predictions}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a Calibra checkpoint on C-Eval or CMMLU")
    parser.add_argument("--config", required=True, help="Calibra YAML/JSON config")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path; overrides model.path")
    parser.add_argument("--benchmark", choices=("ceval", "cmmlu", "both"), default="ceval")
    parser.add_argument("--data-dir", required=True, help="Benchmark root, split directory, or one data file")
    parser.add_argument("--split", default="val", help="Dataset split (usually val; test has no labels)")
    parser.add_argument("--subjects", nargs="*", default=None, help="Optional subject names to evaluate")
    parser.add_argument("--max-length", type=int, default=None, help="Maximum model input length")
    parser.add_argument("--output", default=None, help="Write JSON results to this path")
    parser.add_argument("--predictions", default=None, help="Write per-question predictions to JSONL")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from calibra.config import configure_runtime, load_config, set_seed
    config = load_config(args.config)
    if args.checkpoint:
        config.model.path = args.checkpoint
    configure_runtime(config)
    set_seed(config.runtime.seed)
    from calibra.models import load_model
    import torch

    model, tokenizer = load_model(config, trainable=False)
    device = next(model.parameters()).device
    max_length = args.max_length or config.data.max_length
    if max_length < 2:
        raise ValueError("--max-length must be at least 2")
    benchmarks = ("ceval", "cmmlu") if args.benchmark == "both" else (args.benchmark,)
    all_results = {}
    for benchmark in benchmarks:
        root = Path(args.data_dir)
        candidate = root / benchmark
        benchmark_root = candidate if len(benchmarks) > 1 and candidate.exists() else root
        questions = load_questions(benchmark_root, args.split, args.subjects)
        evaluated = evaluate_questions(model, tokenizer, questions, benchmark, max_length, device)
        result = {key: value for key, value in evaluated.items() if key != "predictions"}
        all_results[benchmark] = result
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.predictions:
            prediction_path = Path(args.predictions)
            if len(benchmarks) > 1:
                prediction_path = prediction_path.with_name(f"{prediction_path.stem}.{benchmark}{prediction_path.suffix or '.jsonl'}")
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            with prediction_path.open("w", encoding="utf-8") as stream:
                for question in evaluated["predictions"]:
                    stream.write(json.dumps(question, ensure_ascii=False) + "\n")
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(all_results if len(all_results) > 1 else next(iter(all_results.values())), ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
