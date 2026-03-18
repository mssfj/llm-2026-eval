#!/usr/bin/env bash
set -euxo pipefail

# ==== 設定 ====
if [ "$#" -lt 1 ]; then
  echo "プロジェクトのルートディレクトリの指定が必要です。例) bash vastai-setup_uv.sh /workspace/lowbit-math-reasoning" >&2
  exit 1
fi
PROJECT_ROOT="$(realpath -m "$1")"
EVAL_ROOT="${PROJECT_ROOT}"
QUANTIZATION_ROOT="${PROJECT_ROOT}/quantization"

# ==== 0. 基本パッケージ ====
sudo apt-get update
sudo apt-get install -y \
  git wget curl build-essential \
  python3-dev python3-pip \
  pkg-config nodejs npm unzip

# ==== codexのインストール ====
npm i -g @openai/codex

curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install --update
aws configure

mkdir -p ~/.codex
aws s3 cp s3://llm-train-dev/codex/auth.json ~/.codex/auth.json

npm cache clean -f
npm install -g n
n lts

export PATH="/usr/local/bin:$HOME/.local/bin:$PATH"
hash -r

npm install -g @openai/codex@latest
npm install -g @google/gemini-cli
hash -r

# ==== 1. uv インストール ====
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"
hash -r

# ==== 2. プロジェクトディレクトリ ====
mkdir -p \
  "${PROJECT_ROOT}" \
  "${PROJECT_ROOT}/eval" \
  "${QUANTIZATION_ROOT}" \
  "${PROJECT_ROOT}/prompts" \
  "${PROJECT_ROOT}/experiments" \
  "${PROJECT_ROOT}/results"
cd "${PROJECT_ROOT}"

# ==== 3. eval / quantization の uv プロジェクト ====
if [ ! -f "${EVAL_ROOT}/pyproject.toml" ]; then
  echo "${EVAL_ROOT}/pyproject.toml がありません。ルート pyproject.toml を eval 用 uv プロジェクトとして配置してください。" >&2
  exit 1
fi

cat > "${QUANTIZATION_ROOT}/pyproject.toml" << PYPROJECT_QUANTIZATION
[project]
name = "llm-quantization"
version = "0.1.0"
description = "Separate quantization environment for GPTQ/AWQ and model conversion"
requires-python = ">=3.10,<3.12"
dependencies = [
    "transformers @ git+https://github.com/huggingface/transformers.git",
    "accelerate",
    "datasets",
    "sentencepiece",
    "gptqmodel>=5.7.0",
    "optimum>=2.1.0",
    "bitsandbytes",
]

[dependency-groups]
dev = [
    "ipykernel",
]
PYPROJECT_QUANTIZATION

# ==== 4. lock と sync ====
cd "${EVAL_ROOT}"
uv lock
uv sync --group sft --group grpo --group dev --group eval

cd "${QUANTIZATION_ROOT}"
uv lock
uv sync
cd "${PROJECT_ROOT}"

# ==== 5. PyTorch (CUDA 12.1 wheel) ====
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu121 \
    "torch==2.4.0" \
    "torchvision==0.19.0" \
    "torchaudio==2.4.0"

uv pip install --python "${QUANTIZATION_ROOT}/.venv/bin/python" --index-url https://download.pytorch.org/whl/cu121 \
    "torch==2.4.0" \
    "torchvision==0.19.0" \
    "torchaudio==2.4.0"

# ==== 6. 動作確認 ====
cd "${EVAL_ROOT}"
uv run --group eval python - << PYCODE
import torch
import vllm
import transformers
print("eval torch version:", torch.__version__)
print("eval cuda available:", torch.cuda.is_available())
print("eval cuda version:", torch.version.cuda)
print("eval vllm version:", vllm.__version__)
print("eval transformers version:", transformers.__version__)
PYCODE

(
  cd "${QUANTIZATION_ROOT}"
  uv run python - << PYCODE
import torch
import transformers
print("quantization torch version:", torch.__version__)
print("quantization cuda available:", torch.cuda.is_available())
print("quantization cuda version:", torch.version.cuda)
print("quantization transformers version:", transformers.__version__)
PYCODE
)

echo "=== setup done. ==="
echo "eval環境:"
echo "  cd ${EVAL_ROOT}"
echo "  uv run --group eval python eval/gsm8k-eval.py --model-name ./experiments/models/Qwen3.5-9B-GPTQ-INT4"
echo
echo "quantization環境:"
echo "  cd ${QUANTIZATION_ROOT}"
echo "  uv run python quantize_qwen35_9b_gptq.py \\\n    --output-dir ${PROJECT_ROOT}/experiments/models/Qwen3.5-9B-GPTQ-INT4 \\\n    --calibration-preset math_qa_cot \\\n    --max-calibration-samples 32 \\\n    --max-seq-len 512 \\\n    --bits 4"
echo "  ※ vLLM など GPU を占有するプロセスは事前に停止すること。"

# ==== 7. git 初期化 ====
git config --global user.email "mss.fujimoto@gmail.com"
git config --global user.name "Masashi Fujimoto"

# ==== 8. クリーニング ====
rm -rf ./aws
rm -f ./awscliv2.zip
