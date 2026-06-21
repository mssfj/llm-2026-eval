set -euo pipefail

# export HF_DATASET_REPO_ID="mssfj/qwen25-0.5b-fineweb-edu-10bt"  # 別リポジトリを使う場合は上書き
# export HF_MODEL_REPO_ID="username/qwen25-0.5b-fineweb-edu-10bt"

# huggingface-cli login
hf auth login
wandb login

cd /root/lowbit-math-reasoning/mylitgpt

# 環境構築
python -m venv .venv
source .venv/bin/activate 

pip install -e ".[all]"
pip install datasets huggingface_hub pyarrow litdata 

# huggingface_hubをlitgptのdowndload.pyにあわせダウングレード
python -m pip install -U --force-reinstall "huggingface_hub[hf-transfer]>=0.30,<1.0" 

# トークナイザだけダウンロード
litgpt download Qwen/Qwen2.5-0.5B --tokenizer_only true

## コーパスをダウンロード
# HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download HuggingFaceFW/fineweb-edu --repo-type dataset --include "sample/10BT/*.parquet" --local-dir data/fineweb-edu/sample-10BT
#
## litgptで読み込めるよう前処理（事前にローカルへの保存が必要）
## - litdata.optimize + TokensLoader 形式（LitGPT pretrain の StreamingDataset と互換）
## - *.parquet を再帰的に読み、text カラムを Qwen tokenizer で tokenize
## - LitData chunks を data/fineweb-edu-10bt/qwen25/train に出力
# mkdir -pv data/fineweb-edu-10bt/qwen25/train
# python litgpt/data/prepare_fineweb_edu.py --input_dir data/fineweb-edu/sample-10BT --output_dir data/fineweb-edu-10bt/qwen25/train --tokenizer_path checkpoints/Qwen/Qwen2.5-0.5B
## TokensLoader形式とチャンク次元を検証（旧DataChunkRecipe/PyTreeLoader形式の混入を防止）
# python -c 'import json; from pathlib import Path; p = Path("data/fineweb-edu-10bt/qwen25/train/index.json"); index = json.loads(p.read_text()); assert index["config"]["item_loader"] == "TokensLoader", index["config"]; assert all(isinstance(chunk["dim"], int) and chunk["dim"] > 0 for chunk in index["chunks"]), "invalid chunk dim"; print("Validated TokensLoader dataset:", len(index["chunks"]), "chunks")'
#
## 前処理済みデータをHugging Face Hubへアップロード
## 事前に書き込み権限のあるトークンでログインし、アップロード先を指定する
## huggingface-cli login
## export HF_DATASET_REPO_ID=<Hugging Faceユーザー名>/fineweb-edu-10bt-qwen25
# if [ -z "${HF_DATASET_REPO_ID:-}" ]; then
#     echo "HF_DATASET_REPO_IDを設定してください（例: username/fineweb-edu-10bt-qwen25）" >&2
#     exit 1
# fi
# HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli upload-large-folder "$HF_DATASET_REPO_ID" data/fineweb-edu-10bt/qwen25/train --repo-type dataset
# Hugging Face Hubから前処理済みTokensLoaderデータセットを取得
HF_DATASET_REPO_ID="${HF_DATASET_REPO_ID:-mssfj/qwen25-0.5b-fineweb-edu-10bt}"
HF_DATASET_DIR="data/fineweb-edu-10bt/qwen25/train"
mkdir -pv "$HF_DATASET_DIR"
HF_HUB_ENABLE_HF_TRANSFER=1 hf download "$HF_DATASET_REPO_ID" --repo-type dataset --local-dir "$HF_DATASET_DIR"

# TokensLoader形式とチャンク次元を検証
python -c 'import json, sys; from pathlib import Path; p = Path(sys.argv[1]) / "index.json"; index = json.loads(p.read_text()); assert index["config"]["item_loader"] == "TokensLoader", index["config"]; assert all(isinstance(chunk["dim"], int) and chunk["dim"] > 0 for chunk in index["chunks"]), "invalid chunk dim"; print("Validated TokensLoader dataset:", len(index["chunks"]), "chunks")' "$HF_DATASET_DIR"

# 学習開始
# 学習パラメータは.yamlに記載
litgpt pretrain --config config_hub/pretrain/qwen25-0.5b-fineweb-edu-10bt.yaml

# 学習後のHuggingface形式への変換
litgpt convert_pretrained_checkpoint out/pretrain/qwen25-0.5b-fineweb-edu-10bt/final out/pretrain/qwen25-0.5b-fineweb-edu-10bt-hf

# 学習済みモデルをHugging Face Hubへアップロード
# 事前に書き込み権限のあるトークンでログインし、アップロード先を指定する
# huggingface-cli login
# export HF_MODEL_REPO_ID=<Hugging Faceユーザー名>/qwen25-0.5b-fineweb-edu-10bt
if [ -z "${HF_MODEL_REPO_ID:-}" ]; then
    echo "HF_MODEL_REPO_IDを設定してください（例: username/qwen25-0.5b-fineweb-edu-10bt）" >&2
    exit 1
fi
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli upload-large-folder "$HF_MODEL_REPO_ID" out/pretrain/qwen25-0.5b-fineweb-edu-10bt-hf --repo-type model
