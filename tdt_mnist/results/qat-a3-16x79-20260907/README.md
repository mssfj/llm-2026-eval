# 16層・幅79：A3 STE付きQAT

ReLUあり、入力90、隠れ幅79×15層＋10クラス出力、バイアスなし、95,274個のFP32重み。
通常FP32対照と同じseedの乱数初期重みから学習し、初期重みハッシュの完全一致を確認。FP32学習済みモデルからの微調整ではない。
入力・全隠れ層を線形層直前でA3化する。閾値τ=.5 mean(abs(x))、復元値β=mean(abs(x)|selected)。各例・各層で動的計算。ReLU後のコードは0,+1。
順伝播は既存量子化器と完全一致するカスタムautograd関数。逆伝播だけdQ/dx=1とする恒等STEで、閾値・復元スケール経由の微分は行わない。ReLUと線形層の逆伝播は通常どおり。
重み・スケール・積和・logitsはFP32。重みを三値化したTDTとは異なる。
データ・前処理・初期化・Adam・学習率スケジュール・最大100epoch・早期終了規則はFP32対照と共通。validation損失最小のモデルを選ぶ。QATでは選択時もA3。testはモデル選択に使わない。

| 方法 | validation精度 % | test精度 % |
| --- | ---: | ---: |
| FP32 | 94.400 ± 1.044 | 93.890 ± 0.551 |
| PTQ_A3 | 10.267 ± 0.493 | 10.287 ± 0.591 |
| QAT_A3 | 10.100 ± 2.771 | 10.587 ± 1.363 |

平均±seed間標本標準偏差。PTQ_A3は既存FP32学習後に入力・全隠れ層をA3化した結果。QAT_A3も同じ量子化位置・閾値・復元規則。
paired_effects.csvに各seedのQAT−PTQ、QAT−FP32の差。signal_metrics.csvに全16層のRMS、activation_metrics.csvに量子化誤差・コード分布・コサイン。各seedのgradient_metrics.csvにepochごとの層別平均勾配ノルム。
fp32_ablationはQAT重みの量子化を外した診断であり、通常FP32学習対照とは異なる。
保存モデルのvalidation再現、推論中の重み不変、既存量子化器との順伝播一致、全予測からの精度再集計を検証。

## 層別診断で確認した失敗の形

seed 0・1の選択モデルでは、第14隠れ層のRMSが約2.06×10^7・2.94×10^6まで増大し、第15隠れ層とlogitsはtest全例でゼロ。単純な減衰だけでなく、振幅増大とReLUの全ゼロ化が観測された。
全3seedで最終学習epochの全層平均勾配ノルムがゼロ、train損失は約log(10)。seed 2の報告精度は崩壊前のepoch 1モデルをvalidation損失で選んだ結果である。選択モデルのRMSと最終学習epochの勾配は異なる時点なので区別する。
恒等STE・この初期化／Adam設定では学習が安定しなかったという結果であり、QAT一般で改善できないという結論ではない。追加の設定探索は実施していない。

collapse_diagnostics.csv、qat_per_seed_rms.pngに各seedの層別診断を保存。effective_config.jsonは設定説明を補正したもの。元のconfig.jsonにFP32対照から継承したquantization説明の誤記が残るため、metadata_corrections.jsonに明記した。元の学習記録・モデルは変更していない。
