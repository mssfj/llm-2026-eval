# CUDA Graph追加最適化：16残差ブロック

RTX5090、幅76、34行列、192,432三値重み、A8/ReLU。CPU threads1・affinity15。各条件3 seed×100更新区間、保存済みE18a重みから開始。各変更は新たに測定した元gpu_graphから独立。CPU整理は提案1〜3をまとめた条件。組合せやtest評価は行っていない。

| 条件 | ms/区間 平均±標本SD | 基準Graph比 | CPU schedule ms | GPU処理 ms | 予約MiB |
|---|---:|---:|---:|---:|---:|
| gpu_graph | 17.795 ± 0.028 | 1.000倍 | 3.603 | 2.139 | 224.0 |
| cpu_compact | 11.082 ± 0.084 | 1.606倍 | 3.458 | 2.139 | 224.0 |
| persistent_candidates | 17.862 ± 0.134 | 0.996倍 | 3.602 | 2.053 | 224.0 |
| transfer_buffers | 17.687 ± 0.079 | 1.006倍 | 3.604 | 2.148 | 224.0 |
| fused_graph | 16.727 ± 0.081 | 1.064倍 | 3.607 | 1.012 | 328.0 |

区間時間はCPUの候補生成・判定、転送、GPU計算、受理通知、CPUモデル更新を含む。初期化・コンパイル・Graph捕捉・3区間ウォームアップ・ログのディスク書込みは除外し別記録。GPU処理のCUDAイベント区間は転送後からloss計算まで。transfer_buffersだけ受理座標更新をGraph内部へ含めており、GPU内訳の境界差を全区間時間と混同しない。

## 数値一致性

固定共通状態での検査：初期/学習済み×3 seed×3連続更新、9216候補損失比較。構造・乱数の一致：True。完全一致を期待したCPU整理・候補再利用・転送整理の合格：True。融合版の相対損失<1e-5かつ投票/カウンタ/発火一致：False。

| seed | 条件 | 最初の発火分岐 | 損失ビット不一致数 | 最終重み一致 | 最終S一致 |
|---|---|---:|---:|---|---|
| 0 | gpu_graph | None | 0 | True | True |
| 0 | cpu_compact | None | 0 | True | True |
| 0 | persistent_candidates | None | 0 | True | True |
| 0 | transfer_buffers | None | 0 | True | True |
| 0 | fused_graph | 2 | 12799 | False | False |
| 1 | gpu_graph | None | 0 | True | True |
| 1 | cpu_compact | None | 0 | True | True |
| 1 | persistent_candidates | None | 0 | True | True |
| 1 | transfer_buffers | None | 0 | True | True |
| 1 | fused_graph | 4 | 12798 | False | False |
| 2 | gpu_graph | None | 0 | True | True |
| 2 | cpu_compact | None | 0 | True | True |
| 2 | persistent_candidates | None | 0 | True | True |
| 2 | transfer_buffers | None | 0 | True | True |
| 2 | fused_graph | 14 | 12800 | False | False |

固定状態検査は基準Graphの重みに揃えて比較する。100区間測定は各条件が自分の損失で更新するため、分岐後の損失差にはモデル差も含まれる。診断だけの丸め差はvalidation/diagnostic_differences.jsonに別記する。融合による数値差をCPU整理の完全一致結果と混同しない。

## 実装と限界

cpu_compactは元candidate_pairを16座標配列に適用し、元accumulateに保存した一様乱数を供給、元select_actionsを使う。カウンタ診断は全体座標順に揃え、未選択ゼロの分母を復元する。persistent_candidatesはGPU候補の前回座標を現在baseへ戻し、新候補を直接代入する。transfer_buffersはpinned固定バッファ、受理座標の直接通知、イベント再利用、1回のmetadata H2Dとloss D2H。fused_graphはInductorをfullgraphで使用し、内部自動CUDA Graphを無効にして手動捕捉する。

GPU予約量はPyTorchプールであり、CUDAコンテキストを含むプロセス全VRAMではない。Graph割当カウンタのみから実際の生存テンソルピークを断定しない。過去の測定との時間差は環境・負荷が異なり得るため、倍率には今回の基準Graphを使う。3 seedの短い動作測定で、最終test精度や普遍的な速度保証は主張しない。すべての失敗・遅い条件も報告する。


## 独立監査

全15測定・1500区間について、保存した128候補損失を元のCPU epochへ渡し、発火・S・両乱数状態・最終重みを再構成して一致を確認。ソース/モデル/データ20ハッシュを照合。診断集計だけの差は0件。test再評価なし。


## 固定状態検査の数値と測定の補足

| 条件 | 最大相対損失誤差 | 損失ビット不一致 | 投票不一致 | 全時点カウンタ不一致 | 発火不一致ケース |
|---|---:|---:|---:|---:|---:|
| cpu_compact | 0 | 0 | 0 | 0 | 0 |
| persistent_candidates | 0 | 0 | 0 | 0 | 0 |
| transfer_buffers | 0 | 0 | 0 | 0 | 0 |
| fused_graph | 0.00252989959 | 2304 | 270 | 5782 | 1 |

各条件18ケース・2,304候補。完全一致を期待した3条件は固定検査と100区間×3 seedの全候補で一致。融合版は最大相対誤差0.00252990で1e-5基準に未達、100区間中の最初の発火分岐はseed順2/4/14区間。融合版の精度低下を示した実験ではなく、既存Graphとの数値一致を満たさない結果である。

最も有効だったCPU整理は平均37.73%の時間削減（基準17.795 ms→11.082 ms）。候補再利用ではGPU区間だけは2.139→2.053 msだが全体は高速化しなかった。この実装には同期対象を確認する追加のCPU差分走査もあり、候補行列コピーだけの理想的下限を測った値ではない。転送整理の約0.6%差は小さく、3 seedの測定揺らぎと併せて読む。

setup_secondsは各測定workerの準備費用（融合版平均1.862秒）で、事前検証後のコンパイルキャッシュを利用した状態。事前検証での最初のコールドコンパイル時間は独立に記録していないため、コールド起動の総費用は未測定とする。TF32有効化のライブラリ警告は採用せず、全条件FP32 IEEE/TF32無効を維持した。

CPU整理は提案1〜3の合算効果であり、それぞれの寄与を個別分離していない。今回の結果から12,000区間の最終精度や8ブロックE17aの追加速度を推定した実測値として報告しない。
