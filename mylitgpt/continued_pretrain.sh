set -euo pipefail

export HF_SOURCE_DATASET_REPO_ID="${HF_SOURCE_DATASET_REPO_ID:-HuggingFaceTB/finemath}"
export HF_SOURCE_DATASET_CONFIG="${HF_SOURCE_DATASET_CONFIG:-finemath-4plus}"
export HF_PRETOKENIZED_DATASET_REPO_ID="${HF_PRETOKENIZED_DATASET_REPO_ID:-mssfj/finemath-4plus-qwen25}"
export HF_BASE_LIT_MODEL_REPO_ID="${HF_BASE_LIT_MODEL_REPO_ID:-mssfj/qwen25-0.5b-fineweb-edu-10bt-litgpt}"
export HF_MODEL_REPO_ID="${HF_MODEL_REPO_ID:-mssfj/qwen25-0.5b-finemath-4plus}"

export BASE_LIT_CHECKPOINT_DIR="${BASE_LIT_CHECKPOINT_DIR:-out/pretrain/qwen25-0.5b-fineweb-edu-10bt-lit}"
export TOKENIZER_DIR="${TOKENIZER_DIR:-checkpoints/Qwen/Qwen2.5-0.5B}"
export RAW_DATA_DIR="${RAW_DATA_DIR:-data/finemath-4plus/raw}"
export LITDATA_DIR="${LITDATA_DIR:-data/finemath-4plus/qwen25/train}"
export OUT_DIR="${OUT_DIR:-out/pretrain/qwen25-0.5b-finemath-4plus}"
export LIT_OUT_DIR="${LIT_OUT_DIR:-out/pretrain/qwen25-0.5b-finemath-4plus-lit}"
export HF_WEIGHTS_DIR="${HF_WEIGHTS_DIR:-out/pretrain/qwen25-0.5b-finemath-4plus-hf-weights}"
export HF_DIR="${HF_DIR:-out/pretrain/qwen25-0.5b-finemath-4plus-hf}"

# huggingface-cli login
hf auth login
wandb login

cd /root/lowbit-math-reasoning/mylitgpt

# 環境構築
python -m venv .venv
source .venv/bin/activate

pip install -e ".[extra]"
pip install -U 'wandb>=0.12.10'

# トークナイザだけダウンロード
litgpt download Qwen/Qwen2.5-0.5B --tokenizer_only true

# FineWeb-Edu で事前学習済みの LitGPT checkpoint を取得
mkdir -pv "$BASE_LIT_CHECKPOINT_DIR"
HF_HUB_ENABLE_HF_TRANSFER=1 hf download \
  "$HF_BASE_LIT_MODEL_REPO_ID" \
  --repo-type model \
  --local-dir "$BASE_LIT_CHECKPOINT_DIR"

python -c 'import sys; from pathlib import Path; p = Path(sys.argv[1]) / "lit_model.pth"; assert p.is_file(), f"lit_model.pth not found: {p}"; print(f"Validated base LitGPT checkpoint: {p}")' "$BASE_LIT_CHECKPOINT_DIR"

# FineMath-4plus を Hugging Face Datasets の subset/config として読み込み、parquet として保存
mkdir -pv "$RAW_DATA_DIR"
python - <<'PY_LOAD_FINEMATH'
import os
from pathlib import Path

from datasets import load_dataset

repo_id = os.environ["HF_SOURCE_DATASET_REPO_ID"]
config_name = os.environ["HF_SOURCE_DATASET_CONFIG"]
raw_data_dir = Path(os.environ["RAW_DATA_DIR"])
raw_data_dir.mkdir(parents=True, exist_ok=True)
output_file = raw_data_dir / "train.parquet"

if output_file.exists():
    print(f"Using existing FineMath parquet: {output_file}")
else:
    dataset = load_dataset(repo_id, config_name, split="train", num_proc=8)
    dataset.to_parquet(str(output_file))
    print(f"Saved FineMath parquet: {output_file}")
PY_LOAD_FINEMATH

# litgptで読み込めるよう前処理
# - litdata.optimize + TokensLoader 形式（LitGPT pretrain の StreamingDataset と互換）
# - *.parquet を再帰的に読み、text カラムを Qwen tokenizer で tokenize
# - LitData chunks を data/finemath-4plus/qwen25/train に出力
mkdir -pv "$LITDATA_DIR"
python litgpt/data/prepare_fineweb_edu.py \
  --input_dir "$RAW_DATA_DIR" \
  --output_dir "$LITDATA_DIR" \
  --tokenizer_path "$TOKENIZER_DIR"

# TokensLoader形式とチャンク次元を検証（旧DataChunkRecipe/PyTreeLoader形式の混入を防止）
python -c 'import json, sys; from pathlib import Path; p = Path(sys.argv[1]) / "index.json"; index = json.loads(p.read_text()); assert index["config"]["item_loader"] == "TokensLoader", index["config"]; assert all(isinstance(chunk["dim"], int) and chunk["dim"] > 0 for chunk in index["chunks"]), "invalid chunk dim"; print("Validated TokensLoader dataset:", len(index["chunks"]), "chunks")' "$LITDATA_DIR"

# 前処理済みデータをHugging Face Hubへアップロード
# 事前に書き込み権限のあるトークンでログインし、アップロード先を指定する
if [ -z "${HF_PRETOKENIZED_DATASET_REPO_ID:-}" ]; then
  echo "HF_PRETOKENIZED_DATASET_REPO_IDを設定してください（例: username/finemath-4plus-qwen25）" >&2
  exit 1
fi

HF_HUB_ENABLE_HF_TRANSFER=1 hf upload-large-folder \
  "$HF_PRETOKENIZED_DATASET_REPO_ID" \
  "$LITDATA_DIR" \
  --repo-type dataset

# 学習開始
# 学習パラメータは.yamlに記載
litgpt pretrain \
  --config config_hub/pretrain/qwen25-0.5b-finemath-4plus.yaml \
  --initial_checkpoint_dir "$BASE_LIT_CHECKPOINT_DIR" \
  --out_dir "$OUT_DIR"

# 前回の変換生成物を削除して再実行可能にする
rm -rf \
  "$LIT_OUT_DIR" \
  "$HF_WEIGHTS_DIR" \
  "$HF_DIR"

# 学習後のcheckpointからoptimizer stateを除去
litgpt convert_pretrained_checkpoint \
  "$OUT_DIR/final" \
  "$LIT_OUT_DIR"

# LitGPTの重み名をHugging Face Transformersの重み名へ変換
litgpt convert_from_litgpt \
  "$LIT_OUT_DIR" \
  "$HF_WEIGHTS_DIR"

# from_pretrained()で直接ロードできる標準Hugging Face形式に保存
python - <<'PY_CONVERT_TO_HF'
import os
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

lit_dir = Path(os.environ["LIT_OUT_DIR"])
weights_dir = Path(os.environ["HF_WEIGHTS_DIR"])
hf_dir = Path(os.environ["HF_DIR"])

config = AutoConfig.from_pretrained(lit_dir, local_files_only=True)
state_dict = torch.load(weights_dir / "model.pth", map_location="cpu", mmap=True, weights_only=True)

with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config)
missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
if missing or unexpected:
    raise RuntimeError(f"State dict mismatch: missing={missing}, unexpected={unexpected}")

hf_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained(hf_dir, safe_serialization=True, max_shard_size="5GB")
AutoTokenizer.from_pretrained(lit_dir, local_files_only=True).save_pretrained(hf_dir)

# アップロード前に、実際にfrom_pretrained()でロードできることを検証
loaded = AutoModelForCausalLM.from_pretrained(
    hf_dir,
    local_files_only=True,
    low_cpu_mem_usage=True,
)
del loaded
print(f"Validated Hugging Face checkpoint: {hf_dir}")
PY_CONVERT_TO_HF

# 学習済みモデルをHugging Face Hubへアップロード
# 事前に書き込み権限のあるトークンでログインし、アップロード先を指定する
if [ -z "${HF_MODEL_REPO_ID:-}" ]; then
  echo "HF_MODEL_REPO_IDを設定してください（例: username/qwen25-0.5b-finemath-4plus）" >&2
  exit 1
fi

HF_HUB_ENABLE_HF_TRANSFER=1 hf upload-large-folder \
  "$HF_MODEL_REPO_ID" \
  "$HF_DIR" \
  --repo-type model
