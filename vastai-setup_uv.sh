#!/usr/bin/env bash
set -euxo pipefail

# ==== 設定 ====
if [ "$#" -lt 1 ]; then
  echo "プロジェクトのルートディレクトリの指定が必要です。例) bash vastai-setup_uv.sh /workspace/lowbit-math-reasoning" >&2
  exit 1
fi
PROJECT_ROOT="$(realpath -m "$1")"
EVAL_ROOT="${PROJECT_ROOT}/eval"
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

# ==== gemini-cliのインストール ====
npm install -g @google/gemini-cli
hash -r

# ==== 1. uv インストール ====
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"
hash -r

# ==== 2. プロジェクトディレクトリ ====
mkdir -p \
  "${PROJECT_ROOT}" \
  "${EVAL_ROOT}" \
  "${QUANTIZATION_ROOT}"
cd "${PROJECT_ROOT}"

# ==== 3. eval / quantization の uv プロジェクト ====
cat > "${EVAL_ROOT}/pyproject.toml" << PYPROJECT_EVAL
[project]
name = "llm-eval"
version = "0.1.0"
description = "Separate evaluation environment for vLLM-based inference and verification"
requires-python = ">=3.10,<3.12"
dependencies = [
    "vllm",
    "transformers",
    "datasets",
    "sympy",
    "peft",
    "sentencepiece",
]

[dependency-groups]
dev = [
    "ipykernel",
]
PYPROJECT_EVAL

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
uv sync --dev

cd "${QUANTIZATION_ROOT}"
uv lock
uv sync --dev
cd "${PROJECT_ROOT}"

# ==== 5. 動作確認 ====
# NOTE:
# vllm / gptqmodel が互換性のある torch をそれぞれ解決するため、
# ここで別バージョンの torch を上書きインストールしない。
# 後から torch/cuDNN/NCCL を混在させると import error の原因になる。
cd "${EVAL_ROOT}"
uv run python - << PYCODE
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

# ==== 6. git 初期化 ====
git config --global user.email "mss.fujimoto@gmail.com"
git config --global user.name "Masashi Fujimoto"

# ==== 7. クリーニング ====
rm -rf ./aws
rm -f ./awscliv2.zip

echo "=== setup done. ==="
