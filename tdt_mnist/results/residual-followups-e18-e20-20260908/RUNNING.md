# E18〜E20 実行・再現手順

事前登録はE18_E20_PREREGISTRATION.md（commit 16485f4）、学習実装と初期確認は52e7d8f。
E17のモデル・TDT更新則・量子化・データ前処理は変更していない。

リポジトリルートから、PyTorch/torchvision/numpyを持つPythonで実行する。

```bash
eval/.venv/bin/python -m unittest discover -s tdt_mnist -p test_residual_followups.py -v
eval/.venv/bin/python tdt_mnist/run_residual_followups.py preflight
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 eval/.venv/bin/python tdt_mnist/run_residual_followups.py run --workers 12
eval/.venv/bin/python tdt_mnist/analyze_residual_followups.py
eval/.venv/bin/python tdt_mnist/plot_residual_followups.py
```

結果は`results/residual-followups-e18-e20-20260908/`。preflight/run/解析は`--root PATH`で別の空ディレクトリを指定できる。学習ハイパーパラメータを変更するCLIは設けていない。

E18dは幅54・98,712重みと100,000±1%の指定が矛盾する。例外をユーザーが確認するまではE18dの3 runを開始せず、他の21 runを進める。確認は結果先のauthorizations.jsonに確認内容と時刻を保存する。起動プロセスはこの記録を読み取ってE18dを追加する。応答がないことを承認と解釈しない。

- `preflight.json`, `preflight_initial_validation.csv`, `preflight_runtime.csv`: 全条件の初期validation、重み内訳、3区間測定の時間。testや重み更新は行わない。
- `manifest.json`, `sources/`: 学習開始時の実行ソースコピー・SHA-256・Git commit・MNIST rawハッシュ・E17成果物ハッシュ。
- `runtime_workers.json`: CPU予算と各workerのPID・開始終了時刻・同時実行数。E17との実時間比は公平な速度ベンチマークではない。
- `per_seed/E18*-seed*`, `per_seed/E19a-seed*`: 全12,000区間のmetrics.csv、12,000×64のabs_y.npy、全区間×全行列のlayer_metrics.csv、25時点のsignal/activation/rms_ratios、初期/最終の行列単独64候補対probes、config、checkpoint/model、summary、hash。
- `per_seed/E20*-seed*/attempt0`: 学習初回の全記録。E20cが事前登録の発散条件を満たす場合のみattempt1を追加。失敗attemptは上書きしない。
- BPの`training.csv`: epoch末validation・選択最良epoch・LR・失敗フラグ。`update_validation.csv`: 500更新ごとの診断専用validation（選択不可）。
- BPの`gradient_metrics.csv`: 全epoch×18行列の訓練バッチ平均/最大勾配ノルムとゼロ勾配バッチ数。救済でもクリップ前を記録。
- BPの`signal.csv`等は初期・毎epoch、`selected_signal.csv`等はvalidationで選んだモデル。step列はBPではepoch番号、TDTでは区間番号であり混同しない。
- E20cの`model.pt`は潜在FP32、`quantized_model.pt`はそこから復元したINT8コード・α・実効重みを別々に保存。weight_metrics.csvに全epochのコード数・α・RMS。
- BPにはTDTの発火・候補対の概念がないため、これらは適用外。勾配とW3診断で置き換える。
- `audit_cache/`: 完了済みBPの入力ハッシュ付き独立監査。testの数値を集計せずvalidation・LR選択・保存W3のみを確認する。
- 全24 runが確定した後、`aggregate/`の精度・対応差・判定、`signal/activation/firing/gradient/`の一覧、図を生成する。testを監査で再評価しない。

RMSのoutputは各線形行列の出力。W1のReLU後はbranch_activation、W2出力はbranch_output、加算前/後はstream_before/after。比は検証集合全体の枝RMS/加算前ストリームRMS。
E20c−E17aはW3スケール・初期化・学習予算・選択方法も異なる比較差であり、学習則だけの純粋な因果効果ではない。
