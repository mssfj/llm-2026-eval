#!/usr/bin/env python
"""
Build a few-shot candidate pool from EleutherAI/hendrycks_math while excluding
problems that appear in HuggingFaceH4/MATH-500.

Default behavior:
- Source pool: MATH train split only
- Exclusion set: MATH-500 test split
- Matching key: normalized problem text

Example:
  UV_CACHE_DIR=/tmp/.uv uv run python scripts/extract_math_fewshot_candidates.py \
    --output-path outputs/math_train_minus_math500.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Set

from datasets import load_dataset


MATH_DATASET = "EleutherAI/hendrycks_math"
MATH500_DATASET = "HuggingFaceH4/MATH-500"
MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def normalize_problem_text(text: str) -> str:
    text = text.strip()
    text = text.replace("\u3000", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\\!", "")
    text = text.replace("\\,", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract MATH few-shot candidates excluding MATH-500 overlaps."
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/math_train_minus_math500.jsonl"),
        help="JSONL path for filtered candidate examples.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Optional summary JSON path. Defaults to <output-path>.summary.json.",
    )
    parser.add_argument(
        "--source-split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Which hendrycks_math split to use as the candidate pool.",
    )
    parser.add_argument(
        "--include-math500-fields",
        action="store_true",
        help="Include the matching MATH-500 normalized problem keys in the summary.",
    )
    return parser.parse_args()


def load_math500_problem_keys() -> Set[str]:
    ds = load_dataset(MATH500_DATASET, split="test")
    keys: Set[str] = set()
    for ex in ds:
        problem = ex.get("problem", "")
        if not isinstance(problem, str) or not problem.strip():
            continue
        keys.add(normalize_problem_text(problem))
    return keys


def iter_math_examples(split: str) -> Iterable[Dict]:
    for subject in MATH_SUBJECTS:
        ds = load_dataset(MATH_DATASET, subject, split=split)
        for idx, ex in enumerate(ds):
            yield {
                "subject": subject,
                "split": split,
                "index_in_subject_split": idx,
                "problem": ex.get("problem", ""),
                "solution": ex.get("solution", ""),
                "level": ex.get("level", ""),
                "type": ex.get("type", ""),
            }


def main() -> None:
    args = parse_args()
    output_path = args.output_path
    summary_path = args.summary_path or Path(f"{output_path}.summary.json")

    math500_problem_keys = load_math500_problem_keys()

    kept_rows: List[Dict] = []
    total_rows = 0
    dropped_rows = 0
    kept_by_subject: Counter = Counter()
    dropped_by_subject: Counter = Counter()

    for row in iter_math_examples(args.source_split):
        total_rows += 1
        normalized_problem = normalize_problem_text(row["problem"])
        row["normalized_problem"] = normalized_problem

        if normalized_problem in math500_problem_keys:
            dropped_rows += 1
            dropped_by_subject[row["subject"]] += 1
            continue

        kept_by_subject[row["subject"]] += 1
        kept_rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in kept_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "math_dataset": MATH_DATASET,
        "math500_dataset": MATH500_DATASET,
        "source_split": args.source_split,
        "num_math500_problem_keys": len(math500_problem_keys),
        "num_source_rows": total_rows,
        "num_kept_rows": len(kept_rows),
        "num_dropped_rows": dropped_rows,
        "kept_by_subject": dict(kept_by_subject),
        "dropped_by_subject": dict(dropped_by_subject),
        "output_path": str(output_path),
    }
    if args.include_math500_fields:
        summary["math500_problem_keys"] = sorted(math500_problem_keys)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
