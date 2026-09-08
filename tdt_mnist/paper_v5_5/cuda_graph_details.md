# CUDA Graph版の処理方法：実行時ソースに基づく技術解説

## G1. 最適化した部分と維持した部分

本節の一次資料はgpu_evaluation_engines.pyとrun_gpu_e17a.pyの実行時ソースである。説明対象は今回実行したgpu_graphであり、提案のみの方式とは区別する。GPU短期測定の16残差ブロック（34行列）と全長E17a再現の8残差ブロック（18行列）は、同じ評価器を構造情報で切り替えて用いた。幅はともに76。

CPU上のINT8三値重みを学習状態の正本とする。GPUには学習データ、三値コード、固定スケール、復元済みFP32重みを常駐させる。GPUが返す128個の候補損失を、既存train.epochと同じ候補順にCPUへ渡す。投票・C8・leak・タイブレーク・発火・S更新は元のCPUコードを使う。逆伝播、STE、Adam、潜在FP32学習重みは導入しない。

CPU fastの低ランク出力補正とは別の方式である。GPU版は候補ごとの完全な重み行列を構成し、全層のforwardをBMMで評価する。プレフィックス省略、低ランク出力補正、CPUでの候補損失再評価を使わない。高速化は候補並列化とGPU演算の投入負荷低減によるもので、論理的な候補数を減らしてはいない。

## G2. データ配置とテンソル形状

候補数P=128、候補対数K=64、評価例数B=128、幅d=76。Nは三値重み総数。行列は実装のF.linearに合わせ[out_features, in_features]で保持する。

| 所在 | 名前・形状 | 型・用途 |
|---|---|---|
| CPU | model.weights [N] | INT8。受理された三値状態の正本 |
| CPU | g / bg、S、カウンタ | 候補・確率丸め用乱数、バッチ用乱数、証拠の集約 |
| GPU | x [10000,90] / y [10000] | FP32学習入力とINT64ラベル。初期化時に転送 |
| GPU | codes [N] / scales [N] / base [N] | INT8コード、FP32固定層スケールを座標に展開、FP32復元重み |
| GPU固定入力 | indices [16] | INT64。区間で選択された全体座標 |
| GPU固定入力 | batches [64,128] | INT64。各候補対のミニバッチ添字 |
| GPU固定入力 | candidate_codes [128,16] | INT8。候補ごとの選択座標の三値コード |
| GPU一時値 | weights [128,N] | FP32。128候補分の完全な有効重み |
| GPU一時値 | x [128,128,90] / h [128,128,76] | 候補×例×特徴。残差ストリームはFP32 |
| GPU固定出力 | graph_losses [128] | FP32。候補ごとの128例平均交差エントロピー |

N=100,016のE17aでは、候補重みテンソルだけで128×N×4=51,208,192 bytes（48.84 MiB）。16ブロックでは98,525,184 bytes（93.96 MiB）。入力バッチは5.625 MiB、幅76のストリーム1本は4.75 MiB。この計算値は単一テンソルの容量であり、実測VRAMピークではない。量子化中間値、複数の活性、ライブラリ作業領域、Graph専用プールも必要となる。

三値重みをFP32へ復元するのは、固定スケールを掛けた有効重みで既存と同じFP32線形演算を行うためである。GPU版もINT8行列積や三値専用カーネルではない。FP32のbaseはコードから再生成可能な計算用キャッシュで、勾配更新を蓄積する潜在重みではない。

## G3. CPUで候補スケジュールを先読みする

1区間の冒頭で、候補用gとバッチ用bgの状態を複製したローカルGeneratorを作る。正本Generatorはこの段階で進めない。gの初期seedはseed+1、bgはseed+100000。初期重みは元のseed 0/1/2で生成する。

複製gのrandperm(N)から16座標を選ぶ。64候補対それぞれで、複製bgから128例の添字を生成し、元のcandidate_pair由来のコードでplus/minus候補を作る。GPU転送用には選択した16座標の三値コードだけを保存する。候補順はplus_0, minus_0, plus_1, minus_1, ...で固定する。

各候補対の後でtorch.rand([16], generator=g)を消費し、元のaccumulateの確率的丸めで使われる乱数消費も先読み時に再現する。これを省くと、次の候補生成の乱数がずれる。先読みは乱数消費数が固定された今回の学習コードに対応しており、将来の判定コード変更に自動追従する保証はない。

その後、正本のg/bgを使って元のtrain.epochを実行する。候補生成と乱数消費は従来どおり行い、loss関数だけが先に計算したGPU損失を順に返す。候補を先読みする計算はCPUで重複するが、乱数・判定経路を維持するための構成である。trace検査では渡された候補コードとミニバッチを先読み結果に照合し、最後に128損失すべてが消費されたことを確認する。

## G4. GPU上で128候補のforwardを構成する

prepared()はbase.expand(128,-1).clone()で128候補分の有効重みを作り、weights[:, indices]をcandidate_codes.float()×scales[indices]で上書きする。FP32差分の加算ではなく、三値コードと固定スケールの積を直接代入する。選択された16座標が複数行列にまたがっても、この全体座標方式で反映される。

データはx[batches]で[64,128,90]に集め、repeat_interleave(2,dim=0)で[128,128,90]とする。同じ候補対のplus/minusは同じ128例を使い、異なる候補対のバッチは元の系列どおり異なる。ラベルも同じ対応で複製する。

各層では候補重みの該当区間を[128,out,in]にviewし、torch.bmm(Q(x), W.transpose(1,2))を実行する。すべての候補が同じ重みを共有する単一の大きなmatmulではなく、候補ごとに異なる行列を使うBMMである。候補軸Pを保つことで128通りの全forwardを並列評価する。

入力射影：h = BMM(Q_A8(x), W_in^T)。各ブロック：u = RMSNorm(h)、a = BMM(Q_A8(u), W1^T)、b = BMM(Q_A8(ReLU(a)), W2^T)、h = h + b。出力：logits = BMM(Q_A8(RMSNorm(h)), W_out^T)。正規化と量子化は最後の特徴次元だけで集約し、候補間・例間を混ぜない。

RMSNormはx / sqrt(mean(x^2, dim=-1)+1e-8)。A8は各候補・各例・各点でabsmax/127を計算し、round(x/scale)を[-127,127]へクリップしてINT8化後FP32に復元する。丸めはties-to-even。全ゼロ行ではscale=1、正の極小scaleには元コードのFP32 tiny下限を適用する。ストリーム自身は量子化しない。

logits [128,128,10]を[16384,10]にreshapeし、reduction='none'の交差エントロピーを計算する。[128,128]へ戻し例軸だけをmean(1)することで、順序を保持した128候補損失を得る。候補間で平均しない。

## G5. CUDA Graphの捕捉、固定バッファ、再生

GPUEvaluatorの初期化時に、データ・重み・固定入力バッファをGPUへ確保する。gpu_graphモードでは別CUDA streamでparallel()を3回ウォームアップし、stream間の待機とcuda.synchronize()で準備を完了する。

続いてtorch.cuda.CUDAGraph()を作り、with torch.cuda.graph(self.graph): の中で一度parallel()を呼ぶ。この呼出しはprepared()による候補行列・バッチ構成、A8、RMSNorm、ReLU、残差加算、BMM、交差エントロピーまでを捕捉し、graph_lossesという出力テンソルを保持する。

各区間ではindices.copy_、batches.copy_、candidate_codes.copy_で同じ入力バッファの内容を更新してからgraph.replay()を呼ぶ。Graphが参照するバッファを別のテンソルへ差し替えない。候補やバッチが変わっても、形状と参照先を維持して内容を更新するため、再捕捉なしで再生できる。候補重みのcloneをコード上から除去するのではなく、捕捉時に用意したGraph用メモリを再利用する。

| Graph内で再生する処理 | Graph外で毎区間行う処理 |
|---|---|
| GPU上の候補重み展開・16座標代入 | CPUの乱数先読み・候補生成 |
| GPU常駐データから候補別バッチ構成 | 受理済み重みの同期、固定入力へのcopy_ |
| 全層のA8・RMSNorm・BMM・ReLU・残差加算 | CUDAイベント計測と損失のCPU転送 |
| 全候補のFP32交差エントロピー・例平均 | 元のCPU epochによる投票・C8・発火・S更新 |
| graph_lossesへの出力 | validation、checkpoint、ログ、最終test |

Graphは学習ループ全体、CPU乱数、分岐する判定ロジックを捕捉していない。固定P=128、K=64、B=128、16座標に合わせた実装であり、これらの変更は本実験の対象外。CUDA GraphはGPU演算を一つの数学演算へ融合する仕組みとして使ったわけではなく、同じGPUワークフローの繰り返し投入をまとめている。

## G6. 重み同期、転送、CPU判定への復帰

sync_weights()はCPU上に持つ前回同期時の重みコピーと正本を比較し、変わった座標だけをGPUへ送る。codesとbaseの該当座標を更新し、CPU側の同期用コピーも更新する。今回の発火上限は1なので、通常は0または1座標の変更となる。GPUのbase更新もcode.float()×scaleの直接代入であり、丸め誤差を蓄積する差分加算ではない。

区間開始の同期は、前の区間末にCPUで受理された更新を次のGPU評価へ反映する。最終区間の更新後にGPUコピーを改めて同期する必要はなく、保存モデルと最終評価にはCPUの正本を使う。GPU上の乱数生成や独自の発火判断はない。

固定入力の内容量はindices 128 bytes、batches 65,536 bytes、candidate_codes 2,048 bytes、計67,712 bytes/区間（約66.1 KiB）に、受理更新の座標・コードが加わる。これはテンソルの論理ペイロードで、転送呼出しの内部費用を含む実測帯域ではない。学習入力全体や候補行列全体を区間ごとにCPUから送る方式ではない。

graph.replay()後、graph_losses.cpu()で128個のFP32損失（512 bytes）をまとめてCPUへ戻す。このCPU転送で結果を待ち、有限値を確認する。戻った損失をplus/minus順に既存loss呼出しへ渡すことで、元のCPUコードが差、確率丸め、カウンタ、発火、Sを計算する。GPUイベントの開始は重み同期・入力更新の後、終了はGPU評価後に記録するため、イベント時間は区間全体の時間とは異なる。

この方式でもCPUのバッチ「添字生成」とTDT判定は必要である。CPUで128候補のforwardを逐次計算する必要はなくなるが、CPU処理自体がなくなるわけではない。また、今回のvalidation・testは比較条件を揃えるためCPU推論を使用する。

## G7. 再現性、速度、数値差の読み方

GPU設定はFP32 IEEE、TF32無効、torch.use_deterministic_algorithms(True)、CUBLAS_WORKSPACE_CONFIG=:4096:8。実行記録はPyTorch 2.13.0+cu130、PyTorch CUDA 13.0、driver 580.105.08、RTX 5090。nvidia-smiのCUDA Version 13.2表示と、PyTorchビルドのCUDA 13.0を区別する。

決定的なGPU実行を設定してもCPUとのFP32演算順序までは一致しない。行列積・損失の微差でA8の丸め境界をまたぐと量子化後の差が大きくなり、損失差・S・確率投票・発火へ伝わりうる。A32対照では最大相対損失誤差が約3e-7以下だったのに対し、GPU並列/GraphのA8検査では0.00279344となった。この診断は丸めによる増幅と整合するが、全差異の単一原因を証明したものではない。

GPU並列版とGraph版は、初期/学習済み×3 seedの入力更新検査6ケースで128損失がビット一致し、100区間後の重み・S・乱数状態・発火も一致した。これに対しCPUとは発火が分岐する。CPU互換性、GPU eagerとGraphの同一性、最終testの許容幅は三つの別の検証事項である。

16ブロックのGPU処理区間は逐次596.209 ms、並列5.188 ms、Graph2.142 ms。CPU処理・転送等を含む全区間はそれぞれ613.028 / 21.019 / 17.952 ms。GraphによるGPU処理区間の短縮は約2.42倍だが、並列版から区間全体への上積みは約1.17倍である。CPU復元キャッシュ全区間271.573 msに対しては15.127倍。異なる時間範囲の倍率を混同しない。

Graphは128個の候補forwardに相当する全層の計算を残す。8ブロックでは18回の候補BMM、16ブロックでは34回の候補BMMがワークフローに含まれるが、各BMMは128候補を処理するため、「forward回数が128から1へ減った」と解釈しない。FLOPs削減と、並列実行・投入負荷削減による実時間短縮を区別する。

## G8. 実装箇所と検証の対応

| 実装箇所 | 処理 | 検証記録 |
|---|---|---|
| gpu_evaluation_engines.py / schedule | RNG先読み・候補順・バッチ対応 | 短期validationのindices/rng一致、全長audit |
| GPUEvaluator / prepared, parallel | 候補別完全行列、A8、BMM、候補別損失 | CPU/GPU相対誤差、A32対照、全候補loss CSV |
| GPUEvaluator / __init__, inputs, evaluate | 捕捉・固定バッファ更新・replay | graph_input_refresh.csv、GPU eagerとのビット一致 |
| GPUEvaluator / sync_weights | 受理座標の同期・復元キャッシュ更新 | 更新後Graph入力検査、学習後重み監査 |
| gpu_evaluation_engines.py / epoch | 元のtrain.epochへ128損失を供給 | 候補・バッチtrace、消費数128、発火記録 |
| run_gpu_e17a.py / worker | 12,000区間・CPU評価・記録 | seed別checkpoint、分岐記録、test各1回 |
| audit_gpu_e17a.py | 独立監査 | 全区間loss差/S/RNG、重み再構成、15区間のGPU再評価 |

上表のファイルと関数は、PDF添付資料の実行時sourcesに収録する。GPU短期測定と全長再現で保存したgpu_evaluation_engines.pyが同じ内容であることをPDF生成時にも確認する。報告書作成にあたって新たな学習やtest再評価は行わない。
