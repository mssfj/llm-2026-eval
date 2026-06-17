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

# コーパスをダウンロード
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download HuggingFaceFW/fineweb-edu --repo-type dataset --include "sample/10BT/*.parquet" --local-dir data/fineweb-edu/sample-10BT

# litgptで読み込めるよう前処理（事前にローカルへの保存が必要）
# - prepare_starcoder.py と同じ DataChunkRecipe 形式
# - *.parquet を再帰的に読み、text カラムを Qwen tokenizer で tokenize
# - LitData chunks を data/fineweb-edu-10bt/qwen25/train に出力
mkdir -pv data/fineweb-edu-10bt/qwen25/train
python litgpt/data/prepare_fineweb_edu.py --input_dir data/fineweb-edu/sample-10BT --output_dir data/fineweb-edu-10bt/qwen25/train --tokenizer_path checkpoints/Qwen/Qwen2.5-0.5B

# 学習開始
# 学習パラメータは.yamlに記載
litgpt pretrain --config config_hub/pretrain/qwen25-0.5b-fineweb-edu-10bt.yaml

# 学習後のHuggingface形式への変換
litgpt convert_pretrained_checkpoint out/pretrain/qwen25-0.5b-fineweb-edu-10bt/final out/pretrain/qwen25-0.5b-fineweb-edu-10bt-hf
