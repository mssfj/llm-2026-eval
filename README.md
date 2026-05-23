# Low-bit Math Reasoning

This repository provides scripts for low-bit quantization of math reasoning models and evaluation before and after quantization.

- `eval/`: vLLM-based evaluation scripts for GSM8K and MATH-500
- `quantization/`: GPTQ quantization scripts for Qwen3.5-9B
- `vastai-setup_uv.sh`: helper setup script for GPU environments such as Vast.ai

## Requirements

- Python `>=3.10,<3.12`
- CUDA-capable GPU environment
- Network access to download models and datasets from Hugging Face Hub
- `uv` for Python dependency management

The evaluation and quantization workflows use different dependency stacks. For that reason, `eval/` and `quantization/` each have their own `pyproject.toml`.

## Setup

### Manual Setup

```bash
# Evaluation environment
cd eval
uv sync --dev

# Quantization environment
cd ../quantization
uv sync --dev
```

### Vast.ai Setup

To install system packages, `uv`, Node.js, Codex, and both Python environments on a GPU instance, run the helper script with the repository root path.

```bash
bash vastai-setup_uv.sh /workspace/lowbit-math-reasoning
```

## GSM8K Evaluation

`eval/gsm8k-eval.py` evaluates the `openai/gsm8k` test split with vLLM. The default model is `Qwen/Qwen3.5-9B`.

```bash
cd eval
uv run python gsm8k-eval.py \
  --model-name Qwen/Qwen3.5-9B \
  --max-samples 100 \
  --batch-size 8 \
  --max-tokens 2048 \
  --output-path outputs/gsm8k_eval_qwen3.5-9b.jsonl \
  --wandb-mode disabled
```

Main options:

- `--model-name`: Hugging Face model ID or local model path
- `--lora-path`: LoRA adapter path. If empty, LoRA is disabled
- `--max-samples`: number of samples to evaluate. Use `0` or a negative value to evaluate the full split
- `--batch-size`: batch size passed to vLLM as `max_num_seqs`
- `--max-tokens`: maximum generated tokens per sample
- `--output-path`: JSONL path for detailed evaluation results
- `--wandb-mode`: `online`, `offline`, or `disabled`

The script writes both detailed JSONL results and a summary JSON file.

```text
outputs/gsm8k_eval_qwen3.5-9b.jsonl
outputs/gsm8k_eval_qwen3.5-9b.jsonl.summary.json
```

The JSONL file contains one record per sample, including the question, gold answer, model output, extracted prediction, correctness, and verification reason. The summary JSON contains EM accuracy, number of correct answers, generation speed, and vLLM initialization metrics such as VRAM usage.

## MATH-500 Evaluation

`eval/math500-eval.py` evaluates the `HuggingFaceH4/MATH-500` test split with vLLM. The default model is `mssfj/Qwen3.5-9B-GPTQ-INT8`.

```bash
cd eval
uv run python math500-eval.py \
  --model-name mssfj/Qwen3.5-9B-GPTQ-INT8 \
  --max-samples 50 \
  --batch-size 2 \
  --max-tokens 4096 \
  --output-path outputs/math500_Qwen3.5-9B-GPTQ-INT8.jsonl \
  --wandb-mode disabled
```

To load a quantized model with vLLM's bitsandbytes mode, use:

```bash
cd eval
uv run python math500-eval.py \
  --model-name /path/to/model \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --max-samples 50 \
  --wandb-mode disabled
```

Main options:

- `--model-name`: Hugging Face model ID or local model path
- `--lora-path`: LoRA adapter path
- `--max-samples`: number of samples to evaluate. Use `0` or a negative value to evaluate the full split
- `--batch-size`: batch size passed to vLLM as `max_num_seqs`
- `--max-tokens`: maximum generated tokens per sample
- `--enforce-eager` / `--no-enforce-eager`: vLLM eager execution setting
- `--quantization`: `none` or `bitsandbytes`
- `--load-format`: `none` or `bitsandbytes`
- `--output-path`: JSONL path for detailed evaluation results
- `--wandb-mode`: `online`, `offline`, or `disabled`

For MATH-500, samples with missing final answers are automatically retried with a final-answer-only prompt.

## GPTQ Quantization

`quantization/quantize_qwen35_9b_gptq.py` quantizes `Qwen/Qwen3.5-9B` with GPTQ and saves the result to a local directory. By default, it uses `zwhe99/DeepMath-103K` as the calibration dataset.

```bash
cd quantization
uv run python quantize_qwen35_9b_gptq.py \
  --model-name Qwen/Qwen3.5-9B \
  --output-dir /workspace/lowbit-math-reasoning/experiments/models/Qwen3.5-9B-GPTQ-INT4 \
  --calibration-preset math_qa_cot \
  --max-calibration-samples 128 \
  --max-seq-len 16384 \
  --bits 4 \
  --trust-remote-code
```

Main options:

- `--model-name`: source model ID or local model path
- `--output-dir`: output directory for the quantized model
- `--dataset-name`: calibration dataset name
- `--dataset-config`: dataset config. Use an empty string when not needed
- `--dataset-split`: dataset split used for calibration
- `--calibration-preset`: `plain_text`, `gsm8k_cot`, or `math_qa_cot`
- `--text-column`: text column used by the `plain_text` preset
- `--question-column`: question column used by math CoT presets
- `--answer-column`: answer or rationale column used by math CoT presets
- `--max-calibration-samples`: number of calibration samples
- `--max-seq-len`: maximum calibration sample length
- `--bits`: GPTQ bit width. Supported values are `2`, `3`, `4`, and `8`
- `--group-size`: GPTQ group size. Use `-1` for per-column quantization
- `--desc-act`: enable activation-order quantization
- `--damp-percent`: GPTQ dampening percent
- `--trust-remote-code`: pass `trust_remote_code=True` to Transformers and the GPTQ backend

The output directory contains the quantized weights, tokenizer files, normalized `config.json`, and a generated model card `README.md` with the reproduction command.

## Reading Evaluation Results

The evaluation scripts print metrics to stdout and also write them to the summary JSON file.

- `em`: exact match accuracy
- `num_samples`: number of evaluated samples
- `num_correct`: number of correct answers
- `reason_counts`: counts grouped by verification reason
- `avg_generation_tokens_per_second`: average per-sample generation speed
- `overall_generation_tokens_per_second`: overall generation speed
- `generation_elapsed_time_seconds`: wall-clock generation time
- `total_generation_tokens`: total number of generated tokens
- `model_loading_vram_gib`: VRAM used by vLLM while loading the model
- `available_kv_cache_memory_gib`: available KV cache memory reported by vLLM

## W&B Logging

By default, the evaluation scripts use `--wandb-mode online`. Disable W&B explicitly when you only want local output files.

```bash
--wandb-mode disabled
```

Add `--wandb-log-artifacts` to log the JSONL output and summary JSON as W&B artifacts.

## Helper Script

`eval/chat_cli.py` is a small interactive CLI for testing a LoRA adapter on math problems. The base model and adapter path are defined as constants in the file. Update `BASE_MODEL_NAME` and `ADAPTER_PATH` before using it in your environment.

## Notes

- Keep evaluation and quantization in separate environments. Evaluation uses vLLM, while quantization uses GPTQModel, Optimum, and Transformers.
- Quantization can consume a large amount of VRAM. Stop vLLM servers or other GPU-heavy processes before running GPTQ.
- For gated or private Hugging Face models, authenticate first with a tool such as `huggingface-cli login`.
- Evaluation outputs are written under `eval/outputs/`. Check disk usage when running full evaluations.
