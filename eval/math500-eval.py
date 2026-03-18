#!/usr/bin/env python/
# eval.py
"""
Base Model + vLLM で HuggingFaceH4/MATH-500 を評価するスクリプト。
"""

import argparse
import json
import os
from collections import Counter
from typing import List, Dict, Any, Optional

from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from mymath_verify_math500 import verify_math_answer, MathVerifyConfig, MathVerifyResult

from transformers import AutoTokenizer

WANDB_PROJECT = "qwen2.5-7b-instruct-math500"
WANDB_ENTITY = "mssfj-1"
WANDB_RUNNAME = "qwen2.5-7b-instruct"
DATASET_NAME = "HuggingFaceH4/MATH-500"

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
VLLM_TENSOR_PARALLEL_SIZE = 1
VLLM_MAX_MODEL_LEN = 8192
VLLM_GPU_MEMORY_UTILIZATION = 0.9
VLLM_BATCH_SIZE = 2
VLLM_ENFORCE_EAGER = False
VLLM_QUANTIZATION = "none"
VLLM_LOAD_FORMAT = "none"
VLLM_MAX_TOKENS = 4096
MAX_SAMPLES = 50

PROJECT_HOME_PATH = "/workspace/lowbit-math-reasoning"
SPRIT_MODEL_NAME = MODEL_NAME.rsplit("/", 1)[-1]
#LORA_PATH = "/workspace/model/qwen3_sft_lora_openmathinst2-1000/"
LORA_PATH = ""
OUTPUT_PATH = f"{PROJECT_HOME_PATH}/eval/outputs/math500_{SPRIT_MODEL_NAME}.jsonl"


def extract_math500_gold_answer(ex: Dict[str, Any]) -> str:
    for key in ("answer", "final_answer", "expected_answer", "target"):
        value = ex.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""

def build_prompt(question: str, tokenizer, final_answer_only: bool = False) -> str:
    if final_answer_only:
        user_content = (
            "Solve the following math problem.\n"
            "Return only your final answer within \\boxed{}.\n"
            "Do not include any other text.\n\n"
            f"Problem:\n{question}"
        )
    else:
        user_content = (
            "Solve the following math problem step by step.\n"
            "The last line of your response should be in the format: \\boxed{ANSWER}\n"
            f"Problem: {question}"
        )

    messages = [
        {"role": "system", "content": "You are a careful mathematical problem solver."},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def evaluate_with_vllm(
    model_name: str,
    lora_path: Optional[str] = None,
    max_samples: Optional[int] = None,
    batch_size: int = VLLM_BATCH_SIZE,
    max_tokens: int = VLLM_MAX_TOKENS,
    enforce_eager: bool = VLLM_ENFORCE_EAGER,
    quantization: str = VLLM_QUANTIZATION,
    load_format: str = VLLM_LOAD_FORMAT,
    output_path: Optional[str] = None,
    wandb_run=None,
    wandb_log_artifacts: bool = False,
) -> Dict[str, Any]:
    print(f"Loading Tokenizer from: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    ds = load_dataset(DATASET_NAME, split="test")
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    print(f"Loading Base Model: {model_name}")
    print(f"Quantization: {quantization}")
    print(f"Load format: {load_format}")

    if lora_path:
        print(f"Enabling LoRA with adapter: {lora_path}")

    use_lora = bool(lora_path)

    llm_kwargs = {
        "model": model_name,
        "trust_remote_code": True,
        "tensor_parallel_size": VLLM_TENSOR_PARALLEL_SIZE,
        "max_model_len": VLLM_MAX_MODEL_LEN,
        "enforce_eager": enforce_eager,
        "gpu_memory_utilization": VLLM_GPU_MEMORY_UTILIZATION,
        "max_num_seqs": batch_size,
        "enable_lora": use_lora,
        "max_lora_rank": 32 if use_lora else 16,
    }
    if quantization != "none":
        llm_kwargs["quantization"] = quantization
    if load_format != "none":
        llm_kwargs["load_format"] = load_format

    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=1.0,
        max_tokens=max_tokens,
        stop=None,
    )

    gold_answers: List[str] = []
    raw_questions: List[str] = []
    prompts: List[str] = []

    for ex in ds:
        q = ex.get("problem") or ex.get("question") or ""
        gold = extract_math500_gold_answer(ex)
        raw_questions.append(q)
        gold_answers.append(gold)
        prompts.append(build_prompt(q, tokenizer))

    print("Running vLLM generation...")
    
    lora_request = None
    if use_lora:
        lora_request = LoRARequest("adapter", 1, lora_path)

    outputs: List[Any] = llm.generate(prompts, sampling_params, lora_request=lora_request)

    config = MathVerifyConfig(use_exact=True, use_numeric=True, use_sympy=True, require_final_answer=True)
    detailed_results: List[Dict[str, Any]] = []
    retry_indices: List[int] = []

    for i, (out, q, gold) in enumerate(zip(outputs, raw_questions, gold_answers)):
        pred_text = out.outputs[0].text if out.outputs else ""
        res: MathVerifyResult = verify_math_answer(pred_text, gold, config=config)
        detailed_results.append({
            "index": i, "question": q, "gold_answer": gold, "model_output": pred_text,
            "extracted_pred_answer": res.pred_answer, "is_correct": res.is_correct, "reason": res.reason
        })
        if res.reason == "missing_final_answer":
            retry_indices.append(i)

    if retry_indices:
        print(f"Retrying {len(retry_indices)} samples with strict final-answer-only prompt...")
        retry_prompts = [build_prompt(raw_questions[i], tokenizer, final_answer_only=True) for i in retry_indices]
        retry_outputs = llm.generate(retry_prompts, sampling_params, lora_request=lora_request)

        for row_index, retry_out in zip(retry_indices, retry_outputs):
            retry_text = retry_out.outputs[0].text if retry_out.outputs else ""
            retry_res: MathVerifyResult = verify_math_answer(retry_text, gold_answers[row_index], config=config)
            if retry_res.reason != "missing_final_answer":
                detailed_results[row_index]["model_output"] = retry_text
                detailed_results[row_index]["extracted_pred_answer"] = retry_res.pred_answer
                detailed_results[row_index]["is_correct"] = retry_res.is_correct
                detailed_results[row_index]["reason"] = retry_res.reason

    num_correct = 0
    reason_counter: Counter = Counter()
    num_total = len(detailed_results)
    for i, row in enumerate(detailed_results):
        if row["is_correct"]:
            num_correct += 1
        reason_counter[row["reason"]] += 1
        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{num_total} samples")

    em = num_correct / max(num_total, 1)
    print(f"\n==== Evaluation Result ====")
    print(f"Base Model: {model_name}")
    print(f"Quantization: {quantization}")
    print(f"LoRA Path: {lora_path}")
    print(f"EM: {em:.4f}")

    result_summary = {
        "model_name": model_name, "lora_path": lora_path, "num_samples": num_total,
        "num_correct": num_correct, "em": em, "reason_counts": dict(reason_counter),
    }

    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for row in detailed_results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(output_path + ".summary.json", "w", encoding="utf-8") as f:
            json.dump(result_summary, f, ensure_ascii=False, indent=2)

    if wandb_run is not None:
        log_payload = {
            "eval/em": em,
            "eval/num_correct": num_correct,
            "eval/num_total": num_total,
        }
        for reason_key, reason_count in reason_counter.items():
            log_payload[f"eval/reason/{reason_key}"] = reason_count
        wandb_run.log(log_payload)

        if wandb_log_artifacts and output_path and os.path.exists(output_path):
            import wandb

            artifact = wandb.Artifact("math500_eval_outputs", type="evaluation")
            artifact.add_file(output_path)
            summary_path = output_path + ".summary.json"
            if os.path.exists(summary_path):
                artifact.add_file(summary_path)
            wandb_run.log_artifact(artifact)

    return result_summary

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name",type=str,default=f"{MODEL_NAME}",help="Hugging Face base model name.")
    p.add_argument("--lora-path",type=str,default=LORA_PATH,help="Path to the LoRA adapter.")
    p.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    p.add_argument("--batch-size", type=int,default=VLLM_BATCH_SIZE,help="vLLM batch size (passed to max_num_seqs).")
    p.add_argument("--max-tokens", type=int, default=VLLM_MAX_TOKENS, help="Maximum number of generated tokens per sample.")
    p.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=VLLM_ENFORCE_EAGER,
        help="Force eager execution in vLLM. Disable for better throughput when stable.",
    )
    p.add_argument(
        "--quantization",
        type=str,
        default=VLLM_QUANTIZATION,
        choices=["bitsandbytes", "none"],
        help="vLLM quantization mode. Use 'none' to disable quantization.",
    )
    p.add_argument(
        "--load-format",
        type=str,
        default=VLLM_LOAD_FORMAT,
        choices=["bitsandbytes", "none"],
        help="vLLM load format. Set to 'none' when running without quantization.",
    )
    p.add_argument("--output-path", type=str, default=f"{OUTPUT_PATH}")
    p.add_argument("--wandb-project", type=str, default=f"{WANDB_PROJECT}", help="W&B project name.")
    p.add_argument("--wandb-entity", type=str, default=f"{WANDB_ENTITY}", help="W&B entity/user.")
    p.add_argument("--wandb-run-name", type=str, default=f"{WANDB_RUNNAME}", help="Optional W&B run name.")
    p.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="Set to online/offline to enable W&B logging. Default is online.",
    )
    p.add_argument(
        "--wandb-log-artifacts",
        action="store_true",
        help="Log evaluation outputs as W&B artifacts (requires --wandb-project).",
    )
    return p.parse_args()

def init_wandb(args: argparse.Namespace):
    if args.wandb_mode == "disabled" or not args.wandb_project:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb is not installed but W&B logging was requested.") from exc

    init_kwargs = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "name": args.wandb_run_name,
        "mode": args.wandb_mode,
        "config": {
            "model_name": args.model_name,
            "lora_path": args.lora_path,
            "max_samples": args.max_samples,
            "batch_size": args.batch_size,
            "max_tokens": args.max_tokens,
            "enforce_eager": args.enforce_eager,
            "quantization": args.quantization,
            "load_format": args.load_format,
            "output_path": args.output_path,
        },
    }
    # Remove None values to keep init clean
    init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}
    return wandb.init(**init_kwargs)

def main():
    args = parse_args()
    wandb_run = init_wandb(args)
    try:
        evaluate_with_vllm(
            model_name = args.model_name,
            lora_path = args.lora_path,
            max_samples = args.max_samples if args.max_samples > 0 else None,
            batch_size = args.batch_size,
            max_tokens = args.max_tokens,
            enforce_eager = args.enforce_eager,
            quantization = args.quantization,
            load_format = args.load_format,
            output_path = args.output_path,
            wandb_run = wandb_run,
            wandb_log_artifacts = args.wandb_log_artifacts,
        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()

if __name__ == "__main__":
    main()
