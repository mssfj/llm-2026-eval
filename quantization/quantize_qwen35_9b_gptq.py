#!/usr/bin/env python3
"""
Quantize Qwen/Qwen3.5-9B to GPTQ and save it locally.

Example:
    uv run python quantization/quantize_qwen35_9b_gptq.py \
        --output-dir /workspace/lowbit-math-reasoning/experiments/models/Qwen3.5-9B-GPTQ-INT4 \
        --calibration-preset math_qa_cot \
        --max-calibration-samples 128 \
        --max-seq-len 16384 \
        --bits 4

Required packages:
    uv add gptqmodel optimum transformers datasets accelerate sentencepiece
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional


DEFAULT_MODEL_NAME = "Qwen/Qwen3.5-9B"
DEFAULT_OUTPUT_DIR = "/workspace/lowbit-math-reasoning/experiments/models/Qwen3.5-9B-GPTQ-INT4"
DEFAULT_DATASET_NAME = "zwhe99/DeepMath-103K"
DEFAULT_DATASET_CONFIG = ""


def build_model_card(args: argparse.Namespace) -> str:
    dataset_ref = args.dataset_name
    if args.dataset_config:
        dataset_ref = f"{dataset_ref}/{args.dataset_config}"

    model_id = Path(args.output_dir).name
    optional_flags = ""
    if args.desc_act:
        optional_flags += " \\\n  --desc-act"
    if args.trust_remote_code:
        optional_flags += " \\\n  --trust-remote-code"
    if args.use_fast_tokenizer:
        optional_flags += " \\\n  --use-fast-tokenizer"

    return f"""---
language:
- en
license: other
base_model: {args.model_name}
tags:
- qwen
- gptq
- quantized
- math
- causal-lm
library_name: transformers
pipeline_tag: text-generation
datasets:
- {dataset_ref}
---

# {model_id}

This model is a GPTQ-quantized version of `{args.model_name}`.

## Quantization

- Method: GPTQ
- Bits: {args.bits}
- Group size: {args.group_size}
- desc_act: {args.desc_act}
- damp_percent: {args.damp_percent}
- Calibration preset: {args.calibration_preset}
- Calibration dataset: `{dataset_ref}` split `{args.dataset_split}`
- Max calibration samples: {args.max_calibration_samples}
- Max sequence length: {args.max_seq_len}

## Recommended Baseline

For `Qwen/Qwen3.5-9B` with the `math_qa_cot` calibration preset, the current recommended baseline on a 48GB-class GPU is:

- `--max-calibration-samples 128`
- `--max-seq-len 16384`
- `--bits 4`

## Intended Use

This checkpoint was created to measure whether quantization degrades math reasoning quality, especially chain-of-thought stability.

## Reproduction

```bash
uv run python quantize_qwen35_9b_gptq.py \\
  --model-name {args.model_name} \\
  --output-dir {args.output_dir} \\
  --dataset-name {args.dataset_name} \\
  --dataset-config {args.dataset_config or "''"} \\
  --dataset-split {args.dataset_split} \\
  --calibration-preset {args.calibration_preset} \\
  --question-column {args.question_column} \\
  --answer-column {args.answer_column} \\
  --text-column {args.text_column} \\
  --max-calibration-samples {args.max_calibration_samples} \\
  --max-seq-len {args.max_seq_len} \\
  --bits {args.bits} \\
  --group-size {args.group_size} \\
  --damp-percent {args.damp_percent}{optional_flags}
```

## Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "{model_id}"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    trust_remote_code=True,
)
```

## Notes

- This repository contains quantized weights only.
- Evaluation should be performed on math benchmarks such as GSM8K or MATH-500 to check answer accuracy and CoT-format failures.
- Long math CoT calibration samples are often truncated heavily below `--max-seq-len 8192`.
- Stop vLLM or other GPU-heavy processes before quantization to avoid OOM during GPTQ.
"""


def write_model_card(output_dir: Path, args: argparse.Namespace) -> None:
    readme_path = output_dir / "README.md"
    readme_path.write_text(build_model_card(args), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize Qwen/Qwen3.5-9B with GPTQ."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Base model on Hugging Face Hub or local path.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the GPTQ model will be saved.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=DEFAULT_DATASET_NAME,
        help="Calibration dataset name for datasets.load_dataset().",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=DEFAULT_DATASET_CONFIG,
        help="Calibration dataset config for datasets.load_dataset(). Use '' if not needed.",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="Dataset split used for calibration.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="r1_solution_1",
        help="Column containing plain text for calibration.",
    )
    parser.add_argument(
        "--question-column",
        type=str,
        default="question",
        help="Question column used by math calibration presets.",
    )
    parser.add_argument(
        "--answer-column",
        type=str,
        default="r1_solution_1",
        help="Answer / rationale column used by math calibration presets.",
    )
    parser.add_argument(
        "--calibration-preset",
        type=str,
        default="math_qa_cot",
        choices=["plain_text", "gsm8k_cot", "math_qa_cot"],
        help=(
            "How to build calibration texts. Use math CoT presets when checking "
            "whether quantization breaks chain-of-thought."
        ),
    )
    parser.add_argument(
        "--max-calibration-samples",
        type=int,
        default=128,
        help="Maximum number of calibration samples. 128 is a good default on 96GB-class GPUs.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=8192,
        help=(
            "Maximum sequence length per calibration sample. "
            "8192 is a practical default for the current math CoT calibration set on 96GB-class GPUs."
        ),
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=8,
        choices=[2, 3, 4, 8],
        help="GPTQ bit width. For CoT breakage checks, sweep 8/4/3/2.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="GPTQ group size. Use -1 for per-column quantization.",
    )
    parser.add_argument(
        "--desc-act",
        action="store_true",
        help="Enable act-order (desc_act=True). Improves quality but can slow quantization.",
    )
    parser.add_argument(
        "--damp-percent",
        type=float,
        default=0.1,
        help="GPTQ dampening percent.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to transformers and the GPTQ backend.",
    )
    parser.add_argument(
        "--use-fast-tokenizer",
        action="store_true",
        help="Use fast tokenizer if available.",
    )
    return parser.parse_args()


def ensure_dependencies() -> None:
    try:
        import datasets  # noqa: F401
        import gptqmodel  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install with:\n"
            "  uv add gptqmodel optimum transformers datasets accelerate sentencepiece"
        ) from exc


def patch_optimum_gptq_pack_model() -> None:
    from optimum.gptq.quantizer import GPTQQuantizer

    original_pack_model = GPTQQuantizer.pack_model
    if getattr(original_pack_model, "_qwen35_hf_device_map_patch", False):
        return

    def pack_model_with_missing_device_map(self, model, quantizers):
        if not hasattr(model, "hf_device_map"):
            model.hf_device_map = None
        return original_pack_model(self, model, quantizers)

    pack_model_with_missing_device_map._qwen35_hf_device_map_patch = True
    GPTQQuantizer.pack_model = pack_model_with_missing_device_map


def cleanup_device_map_for_save(model) -> None:
    # Some GPTQ paths on Qwen3.5 models do not populate hf_device_map, while
    # save_pretrained() assumes that if the attribute exists it is a dict.
    if getattr(model, "hf_device_map", None) is None and hasattr(model, "hf_device_map"):
        delattr(model, "hf_device_map")


def format_math_cot_sample(question: str, answer: str) -> str:
    return (
        "Solve the following math problem step by step.\n"
        "The last line of your response should be in the format: \\boxed{ANSWER}\n"
        f"Problem: {question.strip()}\n\n"
        f"Solution:\n{answer.strip()}"
    )


def load_calibration_texts(
    dataset_name: str,
    dataset_config: Optional[str],
    dataset_split: str,
    text_column: str,
    question_column: str,
    answer_column: str,
    calibration_preset: str,
    max_samples: int,
) -> List[str]:
    from datasets import load_dataset

    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)
    else:
        dataset = load_dataset(dataset_name, split=dataset_split)
    texts: List[str] = []

    for row in dataset:
        if calibration_preset == "plain_text":
            value = row.get(text_column)
            if isinstance(value, str) and value.strip():
                texts.append(value.strip())
        else:
            question = row.get(question_column)
            answer = row.get(answer_column)
            if isinstance(question, str) and isinstance(answer, str):
                question = question.strip()
                answer = answer.strip()
                if question and answer:
                    texts.append(format_math_cot_sample(question, answer))
        if len(texts) >= max_samples:
            break

    if not texts:
        if calibration_preset == "plain_text":
            raise ValueError(
                f"No calibration text found in column '{text_column}' "
                f"from {dataset_name}/{dataset_config}:{dataset_split}."
            )
        raise ValueError(
            "No math CoT calibration samples found. Check "
            f"question_column='{question_column}' and answer_column='{answer_column}' "
            f"for {dataset_name}/{dataset_config}:{dataset_split}."
        )

    return texts


def build_quantization_examples(
    texts: Iterable[str],
    tokenizer,
    max_seq_len: int,
) -> List[dict]:
    examples: List[dict] = []

    for text in texts:
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        if input_ids.numel() == 0:
            continue

        examples.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        )

    if not examples:
        raise ValueError("Tokenizer produced no valid calibration examples.")

    return examples


def main() -> None:
    ensure_dependencies()
    patch_optimum_gptq_pack_model()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=args.use_fast_tokenizer,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(
        "Loading calibration dataset: "
        f"{args.dataset_name}/{args.dataset_config}:{args.dataset_split} "
        f"(preset={args.calibration_preset})"
    )
    calibration_texts = load_calibration_texts(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config or None,
        dataset_split=args.dataset_split,
        text_column=args.text_column,
        question_column=args.question_column,
        answer_column=args.answer_column,
        calibration_preset=args.calibration_preset,
        max_samples=args.max_calibration_samples,
    )

    print(f"Preparing {len(calibration_texts)} calibration samples")
    quantize_examples = calibration_texts

    quantize_config = GPTQConfig(
        bits=args.bits,
        dataset=quantize_examples,
        tokenizer=tokenizer,
        group_size=args.group_size,
        damp_percent=args.damp_percent,
        desc_act=args.desc_act,
        model_seqlen=args.max_seq_len,
    )

    print("Loading base model for GPTQ quantization")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantize_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=args.trust_remote_code,
    )

    print(f"Saving quantized model to: {output_dir}")
    cleanup_device_map_for_save(model)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    write_model_card(output_dir, args)

    print("Done")


if __name__ == "__main__":
    main()
