#!/usr/bin/env python
"""
Sample balanced few-shot examples from the filtered MATH candidate pool.

Input:
- JSONL produced by scripts/extract_math_fewshot_candidates.py

Default behavior:
- Sample evenly across subjects
- Prefer shorter problems/solutions to reduce prompt bloat
- Output JSON with selected examples and a prompt-ready text block

Example:
  UV_CACHE_DIR=/tmp/.uv uv run python scripts/sample_math_fewshot_examples.py \
    --input-path outputs/math_train_minus_math500.jsonl \
    --num-examples 6 \
    --output-path outputs/math_fewshot_examples.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample balanced math few-shot examples.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("outputs/math_train_minus_math500.jsonl"),
        help="Filtered candidate JSONL path.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/math_fewshot_examples.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=6,
        help="Number of few-shot examples to sample.",
    )
    parser.add_argument(
        "--max-problem-chars",
        type=int,
        default=700,
        help="Drop examples with longer problem text.",
    )
    parser.add_argument(
        "--max-solution-chars",
        type=int,
        default=1200,
        help="Drop examples with longer solution text.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducible sampling.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_whitespace(text: str) -> str:
    text = text.strip()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_final_answer_text(solution: str) -> str:
    solution = normalize_whitespace(solution)
    lines = [ln.strip() for ln in solution.splitlines() if ln.strip()]
    if not lines:
        return ""

    last_line = lines[-1]
    patterns = [
        r"\\boxed\{(.+)\}",
        r"The answer is[: ]+(.+)",
        r"Answer[: ]+(.+)",
        r"Thus,?[: ]+(.+)",
        r"So,?[: ]+(.+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, last_line, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip().rstrip(".")

    boxed_matches = re.findall(r"\\boxed\{([^{}]+)\}", solution)
    if boxed_matches:
        return boxed_matches[-1].strip()

    return last_line.rstrip(".")


def build_short_solution(solution: str, max_lines: int = 4) -> str:
    solution = normalize_whitespace(solution)
    lines = [ln.strip() for ln in solution.splitlines() if ln.strip()]
    if not lines:
        return ""
    trimmed = lines[:max_lines]
    return "\n".join(trimmed)


def score_row(row: Dict) -> tuple:
    problem_len = len(row.get("problem", ""))
    solution_len = len(row.get("solution", ""))
    level = row.get("level", "")
    level_rank = int(level.split()[-1]) if isinstance(level, str) and level.split() and level.split()[-1].isdigit() else 99
    return (problem_len + solution_len, level_rank, problem_len, solution_len)


def sample_balanced(rows: List[Dict], num_examples: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    by_subject: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        by_subject[row["subject"]].append(row)

    for subject_rows in by_subject.values():
        subject_rows.sort(key=score_row)

    subjects = sorted(by_subject)
    if not subjects:
        return []

    selected: List[Dict] = []
    base_quota = num_examples // len(subjects)
    remainder = num_examples % len(subjects)

    for i, subject in enumerate(subjects):
        quota = base_quota + (1 if i < remainder else 0)
        candidates = by_subject[subject][: max(quota * 4, quota)]
        if quota > 0 and candidates:
            if len(candidates) <= quota:
                selected.extend(candidates)
            else:
                selected.extend(rng.sample(candidates, quota))

    if len(selected) < num_examples:
        remaining = [row for row in rows if row not in selected]
        remaining.sort(key=score_row)
        pool = remaining[: max((num_examples - len(selected)) * 5, num_examples - len(selected))]
        if pool:
            need = min(num_examples - len(selected), len(pool))
            selected.extend(rng.sample(pool, need))

    selected.sort(key=lambda row: (row["subject"], score_row(row)))
    return selected[:num_examples]


def build_prompt_block(rows: List[Dict]) -> str:
    blocks: List[str] = []
    for i, row in enumerate(rows, start=1):
        short_solution = build_short_solution(row.get("solution", ""))
        final_answer = extract_final_answer_text(row.get("solution", ""))
        blocks.append(
            "\n".join(
                [
                    f"Example {i}",
                    f"Problem: {row.get('problem', '').strip()}",
                    "Solution:",
                    short_solution,
                    f"Final Answer: {final_answer}",
                ]
            ).strip()
        )
    return "\n\n".join(blocks)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_path)

    filtered = [
        row
        for row in rows
        if len(row.get("problem", "")) <= args.max_problem_chars
        and len(row.get("solution", "")) <= args.max_solution_chars
    ]

    selected = sample_balanced(filtered, args.num_examples, args.seed)
    for row in selected:
        row["fewshot_solution"] = build_short_solution(row.get("solution", ""))
        row["fewshot_final_answer"] = extract_final_answer_text(row.get("solution", ""))

    payload = {
        "input_path": str(args.input_path),
        "num_input_rows": len(rows),
        "num_filtered_rows": len(filtered),
        "num_selected_rows": len(selected),
        "selection_constraints": {
            "num_examples": args.num_examples,
            "max_problem_chars": args.max_problem_chars,
            "max_solution_chars": args.max_solution_chars,
            "seed": args.seed,
        },
        "examples": selected,
        "prompt_block": build_prompt_block(selected),
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "num_input_rows": len(rows),
        "num_filtered_rows": len(filtered),
        "num_selected_rows": len(selected),
        "output_path": str(args.output_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
