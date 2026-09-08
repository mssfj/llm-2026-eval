# E17 結果

各条件12,000区間×3seed。平均±標本標準偏差。testは最終モデルのみ。

| 条件 | test精度 (%) |
| --- | ---: |
| E17a | 90.6367 ± 0.4300 |
| E17b | 89.2533 ± 0.4140 |
| E17c | 90.7000 ± 0.4744 |
| E16 A32（既存） | 87.31 ± 0.471 |
| E16 A8（既存） | 87.03 ± 0.104 |
| E14 backprop（参考・異なる学習則と重み数） | 93.89 |

事前登録主判定: 合格（非線形性が機能した）。

ReLU_effect: 1.3833 ± 0.6447ポイント（対応seed差）。
A8_cost: 0.0633 ± 0.5064ポイント（対応seed差）。

E17a−E17bはReLU追加の効果。E17bにもRMSNorm・動的量子化の非線形性がある。
最終枝/ストリームRMS比>0.5: 39件。logits RMS>10: 0件。詳細はverification.json。
層別一覧: signal/metrics.csv、signal/rms_ratios.csv、signal/isolated_candidates.csv、firing/matrices.csv。
量子化診断: activation/metrics.csv。全候補・区間・行列の生記録はper_seed配下。

独立監査: audit.json。全区間ログ・S更新・発火集計・ソース/データハッシュを照合し、初期/最終validationと全層単独プローブを完全再現。監査でtestは評価していない。

層別平均の一覧は[LAYER_TABLES.md](LAYER_TABLES.md)。各CSVの*_aggregate.csvに標本標準偏差を併記。

E17aの固定対照87.31%への平均差: +3.3267ポイント。各seed差: +3.80 / +2.96 / +3.22ポイント。

E16の丸め前平均はA32 87.313333…%、A8 87.03%。主判定は事前登録の87.31%を固定使用。既存seed別の対応差はaggregate/historical_paired_differences.csv。E14は95,274個の連続重み、93.89±0.551%の参考値。

三値重み100,016個、18行列、FP32ストリーム幅76、8ブロック。学習forwardは1run 1,536,000回、9run合計13,824,000回。初期・最終の層単独診断は1run 4,608 forwardで別計上。

図: [精度比較](figures/accuracy.png)、[層別診断](figures/diagnostics.png)。同名SVGも保存。帯は3seedの標本標準偏差。
