# lowbit-math-reasoning

This repository explores how low-bit representations affect mathematical reasoning in large language models, with a primary focus on BitNet b1.58 and post-training quantization methods such as GPTQ.

## Objective

To systematically evaluate how aggressive low-bit constraints impact multi-step mathematical reasoning, using benchmarks such as GSM8K.

## Scope

- Post-training quantization (GPTQ, AWQ, etc.)
- Native low-bit architectures (BitNet b1.58)
- Mathematical reasoning benchmarks (GSM8K, MATH)
- Failure mode analysis (reasoning collapse, arithmetic errors, hallucination)

## Current Status

- Qwen3.5-9B GPTQ (8-bit / 4-bit) prepared
- GSM8K evaluation in progress

## Research Questions

- At what bit-width does mathematical reasoning collapse?
- How do failure patterns change under quantization?
- Can low-bit architectures (e.g., BitNet) preserve reasoning better than quantized models?

## Repository Structure

```text
lowbit-math-reasoning/
├── eval/ # evaluation scripts (GSM8K, etc.)
├── quantization/ # GPTQ / AWQ configs and scripts
├── prompts/ # reasoning prompts
├── experiments/ # experiment logs (CRITICAL)
├── results/ # tables / plots
└── README.md
```
