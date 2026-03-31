#!/usr/bin/env python
# eval.py
"""
openai/gsm8k を評価するスクリプト。
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from typing import List, Dict, Any, Optional

from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from mymath_verify import verify_math_answer, MathVerifyConfig, MathVerifyResult

from transformers import AutoTokenizer, PretrainedConfig

WANDB_PROJECT = "qwen3.5-9b-gsm8k-100"
WANDB_ENTITY = "mssfj-1"
WANDB_RUNNAME = "qwen3.5-9b"

MODEL_NAME = "Qwen/Qwen3.5-9B"

#LORA_PATH = "/workspace/model/qwen3_sft_lora_openmathinst2-1000/"
LORA_PATH = ""
BATCH_SIZE = 8
MAX_TOKENS = 2048
OUTPUT_PATH = "/workspace/lowbit-math-reasoning/eval/outputs/gsm8k_eval_qwen3.5-9b.jsonl"


def extract_gsm8k_gold_answer(answer_text: str) -> str:
    lines = [ln.strip() for ln in answer_text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if "####" in ln:
            after = ln.split("####", 1)[1].strip()
            return after
    return lines[-1] if lines else ""

def build_prompt(question: str, tokenizer) -> str:
    user_content = (
        "Solve the following math problem step by step.\n"
        "The last line of your response should be in the format: \\boxed{ANSWER}\n"
        f"Problem: {question}"
    )

    messages = [
        {"role": "system", "content": "You are a careful mathematical problem solver."},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False, add_generation_prompt=True)


def maybe_build_vllm_compat_config(model_name: str) -> Optional[tempfile.TemporaryDirectory]:
    config_dict, _ = PretrainedConfig.get_config_dict(model_name, trust_remote_code=True)
    if config_dict.get("model_type") != "qwen3_5_text":
        return None

    text_config = {
        key: value
        for key, value in config_dict.items()
        if key not in {"architectures", "model_type", "quantization_config", "transformers_version"}
    }
    text_config["model_type"] = "qwen3_5_text"

    compat_config = {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForCausalLM"],
        "text_config": text_config,
        "quantization_config": config_dict.get("quantization_config"),
        "bos_token_id": config_dict.get("bos_token_id"),
        "eos_token_id": config_dict.get("eos_token_id"),
        "tie_word_embeddings": config_dict.get("tie_word_embeddings", False),
    }

    temp_dir = tempfile.TemporaryDirectory(prefix="vllm-hf-config-")
    with open(os.path.join(temp_dir.name, "config.json"), "w", encoding="utf-8") as f:
        json.dump(compat_config, f, ensure_ascii=False, indent=2)

    return temp_dir


def should_force_language_model_only(config_dict: Dict[str, Any]) -> bool:
    return (
        config_dict.get("model_type") in {"qwen3_5", "qwen3_5_text"}
        and config_dict.get("architectures") == ["Qwen3_5ForCausalLM"]
    )


def get_nvidia_smi_gpu_stats() -> Dict[str, Optional[float]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {
            "nvidia_smi_memory_used_mb": None,
            "nvidia_smi_memory_total_mb": None,
            "nvidia_smi_gpu_utilization_percent": None,
        }

    first_line = next((line.strip() for line in result.stdout.splitlines() if line.strip()), "")
    if not first_line:
        return {
            "nvidia_smi_memory_used_mb": None,
            "nvidia_smi_memory_total_mb": None,
            "nvidia_smi_gpu_utilization_percent": None,
        }

    try:
        memory_used_mb_str, memory_total_mb_str, utilization_percent_str = [part.strip() for part in first_line.split(",", 2)]
        return {
            "nvidia_smi_memory_used_mb": float(memory_used_mb_str),
            "nvidia_smi_memory_total_mb": float(memory_total_mb_str),
            "nvidia_smi_gpu_utilization_percent": float(utilization_percent_str),
        }
    except ValueError:
        return {
            "nvidia_smi_memory_used_mb": None,
            "nvidia_smi_memory_total_mb": None,
            "nvidia_smi_gpu_utilization_percent": None,
        }


def _forward_captured_stream(read_fd: int, target_fd: int, chunks: List[str]) -> None:
    while True:
        try:
            data = os.read(read_fd, 4096)
        except OSError:
            break
        if not data:
            break
        chunks.append(data.decode("utf-8", errors="replace"))
        try:
            os.write(target_fd, data)
        except OSError:
            pass


def capture_vllm_init_metrics(build_llm) -> tuple[LLM, Dict[str, Optional[float]]]:
    captured_chunks: List[str] = []
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)
    stdout_read_fd, stdout_write_fd = os.pipe()
    stderr_read_fd, stderr_write_fd = os.pipe()

    stdout_thread = threading.Thread(
        target=_forward_captured_stream,
        args=(stdout_read_fd, saved_stdout_fd, captured_chunks),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_forward_captured_stream,
        args=(stderr_read_fd, saved_stderr_fd, captured_chunks),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    try:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(stdout_write_fd, stdout_fd)
        os.dup2(stderr_write_fd, stderr_fd)
        llm = build_llm()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout_fd, stdout_fd)
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(stdout_write_fd)
        os.close(stderr_write_fd)
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
        os.close(stdout_read_fd)
        os.close(stderr_read_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)

    captured_output = ''.join(captured_chunks)
    metrics: Dict[str, Optional[float]] = {
        'model_loading_vram_gib': None,
        'model_loading_time_seconds': None,
        'available_kv_cache_memory_gib': None,
        'gpu_kv_cache_size_tokens': None,
    }

    match = re.search(r'Model loading took\s+([0-9.]+)\s+GiB memory and\s+([0-9.]+)\s+seconds', captured_output)
    if match:
        metrics['model_loading_vram_gib'] = float(match.group(1))
        metrics['model_loading_time_seconds'] = float(match.group(2))

    match = re.search(r'Available KV cache memory:\s*([0-9.]+)\s+GiB', captured_output)
    if match:
        metrics['available_kv_cache_memory_gib'] = float(match.group(1))

    match = re.search(r'GPU KV cache size:\s*([0-9,]+)\s+tokens', captured_output)
    if match:
        metrics['gpu_kv_cache_size_tokens'] = int(match.group(1).replace(',', ''))

    return llm, metrics


def evaluate_gsm8k_with_vllm(
    model_name: str,
    lora_path: Optional[str] = None,
    max_samples: Optional[int] = None,
    batch_size: int = 8,
    max_tokens: int = 512,
    output_path: Optional[str] = None,
    wandb_run=None,
    wandb_log_artifacts: bool = False,
) -> Dict[str, Any]:
    
    print(f"Loading Tokenizer from: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # データ読み込み
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    print(f"Loading base model: {model_name}")

    if lora_path:
        print(f"Enabling LoRA with adapter: {lora_path}")

    config_dict, _ = PretrainedConfig.get_config_dict(model_name, trust_remote_code=True)
    compat_config_dir = maybe_build_vllm_compat_config(model_name)
    if compat_config_dir is not None:
        print(f"Using vLLM compatibility config: {compat_config_dir.name}")

    llm_kwargs = {
        "model": model_name,
        "trust_remote_code": True,
        "tensor_parallel_size": 1,
        "max_model_len": 4096,
        "enforce_eager": True,
        "gpu_memory_utilization": 0.8,
        "enable_lora": bool(lora_path),
        "max_lora_rank": 32 if lora_path else 16,
        "max_num_seqs": batch_size,
    }
    if compat_config_dir is not None:
        llm_kwargs["hf_config_path"] = compat_config_dir.name
    if should_force_language_model_only(config_dict):
        llm_kwargs["language_model_only"] = True
    llm, vllm_init_metrics = capture_vllm_init_metrics(lambda: LLM(**llm_kwargs))

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        stop=["\n\nQ:", "\n\nProblem:", "<|im_end|>"], # 無駄な生成を防ぐためのストップトークン
    )
    
    gold_answers: List[str] = []
    raw_questions: List[str] = []
    prompts: List[str] = []

    for ex in ds:
        q = ex["question"]
        gold_full = ex["answer"]
        gold = extract_gsm8k_gold_answer(gold_full)
        raw_questions.append(q)
        gold_answers.append(gold)
        prompts.append(build_prompt(q, tokenizer)) 

    print("Running vLLM generation...")
    
    lora_request = None
    if lora_path:
        lora_request = LoRARequest("adapter", 1, lora_path)

    config = MathVerifyConfig(use_exact=True, use_numeric=True, use_sympy=True, require_final_answer=True)
    detailed_results: List[Dict[str, Any]] = []
    generation_tokens_per_second: List[float] = []
    total_generation_tokens = 0
    total_generation_time_seconds = 0.0

    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    # vLLMに全プロンプトを一度に渡してContinuous Batchingを活用する
    generation_start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    generation_elapsed_time_seconds = time.perf_counter() - generation_start_time

    for i, (out, q, gold) in enumerate(zip(outputs, raw_questions, gold_answers)):
        pred_text = out.outputs[0].text if out.outputs else ""
        generated_token_ids = list(out.outputs[0].token_ids) if out.outputs else []
        total_generation_tokens += len(generated_token_ids)

        if out.metrics is not None:
            num_generation_tokens = out.metrics.num_generation_tokens or len(generated_token_ids)
            generation_duration = out.metrics.last_token_ts - out.metrics.first_token_ts
            if num_generation_tokens > 0 and generation_duration > 0:
                generation_tokens_per_second.append(num_generation_tokens / generation_duration)
                total_generation_time_seconds += generation_duration

        res: MathVerifyResult = verify_math_answer(pred_text, gold, config=config)

        row = {
            "index": i,
            "question": q,
            "gold_answer": gold,
            "model_output": pred_text,
            "extracted_pred_answer": res.pred_answer,
            "is_correct": res.is_correct,
            "reason": res.reason,
        }
        detailed_results.append(row)

    num_correct = 0
    reason_counter: Counter = Counter()
    num_total = len(detailed_results)
    for i, row in enumerate(detailed_results):
        if row["is_correct"]:
            num_correct += 1
        reason_counter[row["reason"]] += 1

    nvidia_smi_gpu_stats = get_nvidia_smi_gpu_stats()

    em = num_correct / max(num_total, 1)
    avg_generation_tokens_per_second = (
        sum(generation_tokens_per_second) / len(generation_tokens_per_second)
        if generation_tokens_per_second
        else None
    )
    if avg_generation_tokens_per_second is None and generation_elapsed_time_seconds > 0:
        avg_generation_tokens_per_second = total_generation_tokens / generation_elapsed_time_seconds

    overall_generation_tokens_per_second = (
        total_generation_tokens / total_generation_time_seconds
        if total_generation_time_seconds > 0
        else None
    )
    if overall_generation_tokens_per_second is None and generation_elapsed_time_seconds > 0:
        overall_generation_tokens_per_second = total_generation_tokens / generation_elapsed_time_seconds
    print(f"\n==== Evaluation Result ====")
    print(f"Base Model: {model_name}")
    print(f"LoRA Path: {lora_path}")
    print(f"EM: {em:.4f}")
    if avg_generation_tokens_per_second is not None:
        print(f"Avg generation tokens/sec: {avg_generation_tokens_per_second:.4f}")
    if overall_generation_tokens_per_second is not None:
        print(f"Overall generation tokens/sec: {overall_generation_tokens_per_second:.4f}")
    if nvidia_smi_gpu_stats["nvidia_smi_memory_used_mb"] is not None:
        print(
            "nvidia-smi memory used / total (MiB): "
            f"{nvidia_smi_gpu_stats['nvidia_smi_memory_used_mb']:.0f} / "
            f"{nvidia_smi_gpu_stats['nvidia_smi_memory_total_mb']:.0f}"
        )
    if nvidia_smi_gpu_stats["nvidia_smi_gpu_utilization_percent"] is not None:
        print(
            "nvidia-smi GPU utilization (%): "
            f"{nvidia_smi_gpu_stats['nvidia_smi_gpu_utilization_percent']:.1f}"
        )
    if vllm_init_metrics["model_loading_vram_gib"] is not None:
        print(
            "vLLM model loading VRAM (GiB): "
            f"{vllm_init_metrics['model_loading_vram_gib']:.2f}"
        )
    if vllm_init_metrics["available_kv_cache_memory_gib"] is not None:
        print(
            "vLLM available KV cache memory (GiB): "
            f"{vllm_init_metrics['available_kv_cache_memory_gib']:.2f}"
        )
    if vllm_init_metrics["gpu_kv_cache_size_tokens"] is not None:
        print(
            "vLLM GPU KV cache size (tokens): "
            f"{vllm_init_metrics['gpu_kv_cache_size_tokens']}"
        )

    wandb_eval_records = [
        {
            "question": row["question"],
            "gold_answer": row["gold_answer"],
            "model_output": row["model_output"],
            "extracted_pred_answer": row["extracted_pred_answer"],
        }
        for row in detailed_results
    ]

    result_summary = {
        "model_name": model_name, "lora_path": lora_path, "num_samples": num_total,
        "num_correct": num_correct, "em": em, "reason_counts": dict(reason_counter),
        "avg_generation_tokens_per_second": avg_generation_tokens_per_second,
        "overall_generation_tokens_per_second": overall_generation_tokens_per_second,
        "generation_elapsed_time_seconds": generation_elapsed_time_seconds,
        "total_generation_tokens": total_generation_tokens,
        **nvidia_smi_gpu_stats,
        **vllm_init_metrics,
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            for row in detailed_results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(output_path + ".summary.json", "w", encoding="utf-8") as f:
            json.dump(result_summary, f, ensure_ascii=False, indent=2)

    if wandb_run is not None:
        import wandb

        metrics_row = {
            "model_name": model_name,
            "lora_path": lora_path,
            "num_samples": num_total,
            "num_correct": num_correct,
            "em": em,
            "avg_generation_tokens_per_second": avg_generation_tokens_per_second,
            "overall_generation_tokens_per_second": overall_generation_tokens_per_second,
            "generation_elapsed_time_seconds": generation_elapsed_time_seconds,
            "total_generation_tokens": total_generation_tokens,
            **nvidia_smi_gpu_stats,
            **vllm_init_metrics,
        }
        for reason_key, reason_count in reason_counter.items():
            metrics_row[f"reason_{reason_key}"] = reason_count

        metrics_columns = list(metrics_row.keys())
        metrics_table = wandb.Table(columns=metrics_columns)
        metrics_table.add_data(*[metrics_row[column] for column in metrics_columns])
        wandb_run.log({"eval/metrics_table": metrics_table})

        samples_table = wandb.Table(
            columns=["question", "gold_answer", "model_output", "extracted_pred_answer"]
        )
        for row in wandb_eval_records:
            samples_table.add_data(
                row["question"],
                row["gold_answer"],
                row["model_output"],
                row["extracted_pred_answer"],
            )
        wandb_run.log({"eval/samples_table": samples_table})

        if wandb_log_artifacts and output_path and os.path.exists(output_path):
            artifact = wandb.Artifact("gsm8k_eval_outputs", type="evaluation")
            artifact.add_file(output_path)
            summary_path = output_path + ".summary.json"
            if os.path.exists(summary_path):
                artifact.add_file(summary_path)
            wandb_run.log_artifact(artifact)

    return result_summary

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name",type=str,default=f"{MODEL_NAME}",help="Hugging Face 4-bit base model name.")
    p.add_argument("--lora-path",type=str,default=LORA_PATH,help="Path to the LoRA adapter.")
    p.add_argument("--max-samples", type=int, default=100)
    p.add_argument("--batch-size", type=int,default=BATCH_SIZE,help="vLLM batch size (passed to max_num_seqs).")
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help="Maximum number of tokens to generate per sample.")
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
        evaluate_gsm8k_with_vllm(
            model_name = args.model_name,
            lora_path = args.lora_path,
            max_samples = args.max_samples if args.max_samples > 0 else None,
            batch_size = args.batch_size,
            max_tokens = args.max_tokens,
            output_path = args.output_path,
            wandb_run = wandb_run,
            wandb_log_artifacts = args.wandb_log_artifacts,
        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()

if __name__ == "__main__":
    main()
