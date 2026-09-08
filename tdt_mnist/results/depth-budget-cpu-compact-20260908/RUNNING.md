# 深さ・更新予算実験：開始時の案内

8ブロックは6,000 / 12,000 / 24,000 / 48,000区間、32ブロックはこれに96,000区間を追加。各seed 0/1/2、6学習軌跡・27固定checkpoint評価。CUDA Graph CPU整理版、d76/A8/ReLU、TDT設定固定。

最新状態はstatus.json、個別の最新区間・validation・発火数はper_seed/blocks{8|32}-seed{0|1|2}/progress.json。全学習が完了してall_training_frozen.jsonへ27モデルのハッシュを固定するまでtestは評価しない。完了後は独立監査、27点のtest、集計と図の作成が自動実行される。最終レポートはREADME.md、aggregate/results.csv、aggregate/paired_comparisons.csv、figures/depth_budget.png。

事前登録a36cb75、実装49edde5。開始前検証では両深さ・全seedの初期validationが既存記録と一致し、初期/学習済み状態の局所3区間でCPU整理と元Graphが一致した。preflight.jsonに保存。短期100区間の換算では6軌跡の学習エンジン合計は約89分。validation・層単独プローブ・保存・監査・最終評価は別途時間を要する。

この文書は開始時の記録であり、実行中かどうかはstatus.jsonを確認すること。failure.jsonがあれば異常停止の記録。区間末500ごとcheckpoint.ptを更新し、指定budgetではbudgets/<区間>/model.ptを固定保存する。未完了時の事前確保NPYはprogressの保存済み区間までのみ使用する。
