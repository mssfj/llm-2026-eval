# E17実行と保存形式

事前登録: E17_PREREGISTRATION.md、commit `1ad7b12`。実装・preflight: commit `46a25f8`。
既存train.py、activation_quantization.py、depth_diagnostics.pyは無変更。

リポジトリルートから実行する。

```bash
eval/.venv/bin/python -m unittest discover -s tdt_mnist -p test_residual_stream.py -v
eval/.venv/bin/python tdt_mnist/run_residual_e17.py preflight
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 eval/.venv/bin/python tdt_mnist/run_residual_e17.py run
eval/.venv/bin/python tdt_mnist/audit_residual_e17.py
eval/.venv/bin/python tdt_mnist/plot_residual_e17.py
```

既存結果は上書きしない。再現時はpreflight/runの両方で同じ`--root PATH`を指定し、監査にもPATHを位置引数で渡す。`--data PATH`でMNISTキャッシュを指定できる。条件・学習設定を変更するCLIは設けていない。各runのCPU threadsは1。

結果先: `results/residual-stream-a8-e17-20260908/`。

- `preflight.json`, `per_seed/initial_validation.csv`: test未評価の事前確認。
- `per_seed/E17*-seed*/config.json`: 実行条件、実際のshapes、固定スケール、行列名、ソースハッシュ。モデル構造はarchitecture=residual_streamとwidth/blocks/shapesで定義し、旧CLIのhidden_size/hidden_sizesはこのモデルに使わない。
- `metrics.csv`: 区間1〜12,000の全更新診断、S、mean abs(y)、厳密ゼロ率。初期validationは上記CSVおよびsummaryに保存。
- `abs_y.npy`: 12,000×64のFP32候補損失差絶対値。
- `layer_metrics.csv`: 全区間×18行列の選択座標数、選択区間フラグ、発火数。
- `signal.csv`: 初期・500区間ごと。stream_before/afterはブロック加算前後、branch_outputはW2出力、branch_activationはReLU/identity後。outputは各行列の線形出力であり、W1のReLU後はbranch_activationを参照。layer17のoutputはlogits。
- `rms_ratios.csv`: 検証集合全体の枝RMS/加算前ストリームRMS。サンプルごとの比の平均とは異なる。
- `activation.csv`: 18行列入力の量子化相対二乗誤差・有効例平均コサイン・未定義例数・コード分布。FP32条件も同じ観測点を記録。
- `probes.csv`: 初期・最終×18行列×64候補対、各16辺摂動。訓練データ・専用乱数を用いる。
- `checkpoint.pt`: 直近500区間の重み・更新/バッチ乱数・S。自動resumeは実装していない。
- `model.pt`, `summary.json`: 最終重みと最終testを含む結果。test評価は各run完了時の1回のみ。全run完了前の判断に読まない。
- `aggregate`, `signal`, `activation`, `firing`: 9run完了後のCSV集計。README.mdとLAYER_TABLES.mdに比較・層別一覧。
- `manifest.json`, `sources/`: 学習開始時のソースコピーとSHA-256、Git commit、MNIST rawハッシュ。
- `audit.json`: 全区間のS更新・発火・候補値・乱数対応を照合。初期・最終validation/診断/probesを再現し、testは再評価しない。
- `artifacts_sha256.json`: 完成成果物のSHA-256。監査スクリプトもsourcesに保存する。

RMSNormは学習パラメータを持たないが非線形である。E17a−E17bはこの構造でReLUを追加する効果として解釈する。E17c−E17aは学習経路の変化を含むA8の効果であり、同一重みを量子化した推論誤差だけを意味しない。
