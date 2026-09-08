# TDT-v5.5要約版：LLM引き継ぎ資料

配布用成果物はdoc/TDT-v5.5要約版_引き継ぎ資料.pdfと同名Markdown。12ページの独立要約で、旧版全文を再添付していない。v5理論と成立条件、共通設定、E1〜E20、CPU/GPUエンジン、CUDA Graph処理、終了状態・解釈上の制約・出典の案内を含む。

PDFにはMarkdown原文、handoff_sources_sha256.json、key_results.csvを埋め込んだ。既存モデルの学習・test再評価は実施せず、12条件の最終精度を保存済みseed記録から再集計した。生成時の照合結果はpdf_validation.json。

再生成（リポジトリルート）：

```sh
uv run --offline --with pymupdf --with reportlab python tdt_mnist/paper_v5_5_summary/build_pdf.py
```

組版関数のみpaper_v5_5/build_pdf.pyから継承。要約原稿はdoc側のMarkdownを編集する。出典ハッシュは相対パスを用いるが、他LLMへ生データまで引き継ぐには別途results配下のファイルを渡す必要がある。

全12ページのプレビューを目視確認済み。日本語・数式・表がページ内に収まること、主要数値・未完了区間数・関数名の抽出を確認した。本文の±は標本標準偏差であり信頼区間ではない。
