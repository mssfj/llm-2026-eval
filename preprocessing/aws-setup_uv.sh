#!/usr/bin/env bash
set -euxo pipefail

# ==== 設定 ====
if [ "$#" -lt 1 ]; then
  echo "プロジェクトのルートディレクトリの指定が必要です。例) bash vastai-setup_uv.sh /workspace/lowbit-math-reasoning" >&2
  exit 1
fi
PROJECT_ROOT="$(realpath -m "$1")"
PREPROCESSING_ROOT="${PROJECT_ROOT}/preprocessing"

# ==== 0. 基本パッケージ ====
sudo apt-get update
sudo apt-get install -y tmux nodejs npm

if [ -z "${TMUX:-}" ]; then
  tmux
fi

# ==== codexのインストール ====
npm i -g @openai/codex

npm cache clean -f
npm install -g n
n lts

export PATH="/usr/local/bin:$HOME/.local/bin:$PATH"
hash -r

npm install -g @openai/codex@latest

# ==== 1. uv インストール ====
curl -LsSf https://astral.sh/uv/install.sh | sh
UV_BIN_DIR="${HOME}/.local/bin"
export PATH="${UV_BIN_DIR}:/usr/local/bin:${PATH}"
hash -r

# uv は通常 ~/.local/bin に入る。Vast.ai の新しいシェルや非ログインシェルでも
# `uv run ...` がそのまま動くよう、PATH 永続化と /usr/local/bin へのリンクを行う。
for shell_rc in "${HOME}/.bashrc" "${HOME}/.profile"; do
  touch "${shell_rc}"
  if ! grep -Fq "${UV_BIN_DIR}" "${shell_rc}"; then
    printf "\n# uv installed by vastai-setup_uv.sh\nexport PATH=\"%s:/usr/local/bin:\${PATH}\"\n" "${UV_BIN_DIR}" >> "${shell_rc}"
  fi
done

if [ -x "${UV_BIN_DIR}/uv" ]; then
  sudo ln -sf "${UV_BIN_DIR}/uv" /usr/local/bin/uv
fi
if [ -x "${UV_BIN_DIR}/uvx" ]; then
  sudo ln -sf "${UV_BIN_DIR}/uvx" /usr/local/bin/uvx
fi
hash -r
uv --version

# ==== 2. プロジェクトディレクトリ ====
mkdir -p \
  "${PROJECT_ROOT}" \
  "${PREPROCESSING_ROOT}"
cd "${PROJECT_ROOT}"

# ==== 3.  preprocessing の uv プロジェクト ====
cat > "${PREPROCESSING_ROOT}/pyproject.toml" << PYPROJECT_PREPROCESSING
[project]
name = "llm-preprocessing"
version = "0.1.0"
description = "Separate preprocessing environment for vLLM-based inference and verification"
requires-python = ">=3.10,<3.12"
dependencies = [
    "datasets",
    "sympy",
    "sentencepiece",
]

[dependency-groups]
dev = [
    "ipykernel",
]
PYPROJECT_PREPROCESSING

# ==== 4. lock と sync ====
cd "${PREPROCESSIN_ROOT}"
uv lock
uv sync --dev

# ==== 5. git 初期化 ====
git config --global user.email "mss.fujimoto@gmail.com"
git config --global user.name "Masashi Fujimoto"
git config --global credential.helper

echo "=== setup done. ==="
