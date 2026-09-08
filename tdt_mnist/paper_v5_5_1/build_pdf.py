"""v5.5.1: audited follow-up benchmark supplement, complete v5.5 preserved."""
from pathlib import Path
import csv,json,hashlib,html,re,statistics,zipfile
import pymupdf as fitz
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,PageBreak,Table,TableStyle,Spacer
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;DOC=ROOT/'doc'
R=ROOT/'tdt_mnist/results/gpu-graph-optimizations-16blocks-20260908'
OLD=DOC/'TDT-v5.5_離散状態遷移学習理論.pdf';DEST=DOC/'TDT-v5.5.1_離散状態遷移学習理論.pdf';ARCHIVE=DOC/'TDT-v5.5.1_追加実験データ.zip'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return list(csv.DictReader(p.open()))
def j(p):return json.loads(p.read_text())
manifest=j(R/'artifacts_sha256.json')
for name,h in manifest.items():assert sha(R/name)==h,name
assert j(R/'status.json')['complete'] and j(R/'status.json')['audited'] and j(R/'audit.json')['passed']
raw=read(R/'per_seed.csv');agg=read(R/'aggregate.csv');val=read(R/'validation/aggregate.csv');traj=read(R/'trajectory_comparison.csv')
assert len(raw)==15
for a in agg:
 rs=[r for r in raw if r['engine']==a['engine']];assert len(rs)==3
 values=[float(r['seconds_per_interval']) for r in rs]
 assert abs(statistics.mean(values)-float(a['seconds_per_interval_mean']))<1e-12
 assert abs(statistics.stdev(values)-float(a['seconds_per_interval_sample_sd']))<1e-12
 for r in rs:
  intervals=read(R/'benchmarks'/f"seed{r['seed']}-{r['engine']}"/'intervals.csv');assert len(intervals)==100
  assert abs(statistics.mean(float(x['seconds']) for x in intervals)-float(r['seconds_per_interval']))<1e-12
baseline=float(agg[0]['seconds_per_interval_mean'])
for a in agg:assert abs(baseline/float(a['seconds_per_interval_mean'])-float(a['speedup_vs_original_graph']))<1e-12
helper=(ROOT/'tdt_mnist/paper_v5_5/build_pdf.py').read_text();helper=helper[helper.index('pdfmetrics.registerFont'):helper.index('def label(')];exec(helper)
def footer(c,d):
 c.setFont('CJK',8);c.setFillColor(blue);c.drawString(48,815,'TDT-v5.5.1 | CUDA Graph追加最適化 | 2026-09-08');c.drawRightString(547,25,str(d.page))
labels={'gpu_graph':'基準Graph','cpu_compact':'CPU整理','persistent_candidates':'候補重み再利用','transfer_buffers':'転送整理','fused_graph':'演算融合'}
p('TDT-v5.5.1\n離散状態遷移学習理論','title')
p('CUDA Graph版の追加高速化\nCPU処理整理・候補重み再利用・転送・演算融合','sub')
p('実験追補版 · 2026年9月8日')
table(['今回の結論','確定結果'],[['CPU整理が最も有効','1区間17.795→11.082 ms、1.606倍。時間を37.73%削減。候補損失・発火・最終重み・S・乱数が検査範囲で一致。'],['候補再利用・転送整理','速度はそれぞれ0.996倍・1.006倍。候補再利用は全体の高速化に至らず、転送整理の効果は小さい。'],['演算融合','1.064倍だが最大相対損失誤差0.00252990、投票・発火が分岐。一致基準未達。'],['測定範囲','16残差ブロック、5条件×3 seed×100区間。全15測定・1,500区間を監査。test評価・全長学習なし。']])
p('本追加部はv5.5後に実施したエンジン最適化の追試を収録する。後半にv5.5全186ページを原版のまま継承し、CUDA Graphの基本処理とE17a全長再現、v5〜v5.4の理論・科学実験を引き続き参照できる。旧版の記述は当時の検証範囲である。')
p('値の±はseed0/1/2の平均±標本標準偏差。速度倍率は今回新たに計測した基準Graphに対する比。精度のパーセントポイントや旧CPU naive比とは異なる。')
section('1. 事前登録・共通条件・比較範囲')
table(['項目','条件'],[['構造','E18a相当：16残差ブロック、幅76、34行列、192,432三値重み、A8＋ReLU、pre-norm、FP32残差ストリーム。16線形層ではない。'],['開始状態','保存済みE18aのseed0/1/2学習済み重みから開始。各条件3区間ウォームアップ後に、重み・S・乱数と評価器を開始状態へ戻して100更新区間を測定。'],['データと学習則','MNIST、9×10平均プーリング・元の正規化・data_seed0・batch128。閾値8、K64、16座標、最大1発火、C8、leak1、全リセット、S初期0.02/EMA0.1/下限1e-5を継承。'],['精度・機器','RTX5090、FP32 IEEE、TF32無効、決定的アルゴリズム、CUBLAS_WORKSPACE_CONFIG=:4096:8。CPU threads1・affinity15。'],['実行順','5条件を逐次実行し、seedごとに条件順を循環。各条件は基準Graphから独立。CPU整理内の3変更は一つの条件として測定。'],['検証','初期/保存済み学習済み×3 seed×3連続更新の共通状態比較。候補順・バッチ・乱数、128損失、全64時点の票/カウンタ、発火・Sを照合。']])
p('事前登録コミット5e27fdc、固定実装e1b1792、確定結果b83dd91。既存gpu_evaluation_engines.pyとtrain.pyは変更せず、新規gpu_graph_optimizations.pyに独立経路を追加した。')
p('CPU整理・候補再利用・転送整理は基準Graphとのビット一致を期待する条件。融合版は相対損失誤差<1e-5かつ票・カウンタ・発火一致を数値判定基準とし、未達でも時間と分岐を報告する。計算精度の変更や不一致を隠す条件探索はしない。')
section('2. CPU整理版：同じ判断に必要な処理へ絞る')
p('従来Graph版はGPU評価前の先読みで候補とバッチを作り、さらに元のtrain.epochで候補・バッチを再生成していた。GPU損失を返すだけのloss呼出しに対してもCPU入力の取得と全重み候補のコピーが残っていた。')
table(['変更','具体的処理'],[['候補を一度だけ生成','正本CPU Generatorで元と同じrandperm・randint・randの呼出し順を維持。辺、向きphi、確率丸め用一様乱数、バッチ、候補コードを保存し、後段で使う。'],['候補コピー・バッチ取得削減','元candidate_pairを選択済み16座標の配列へ適用。CPUでN重み全体の候補を128通り作らず、判定段階のx[batch]も不要にする。'],['局所カウンタ','証拠と測定回数を[16,2]に縮小。元accumulateに保存した一様乱数を供給し、元select_actionsで同じ候補順・タイブレーク・最大1発火を維持。'],['統計・更新','カウンタ統計は全体座標順へ並べ、全体平均の未選択ゼロ分母を復元。受理された16座標内の結果をCPU正本へ反映し、上側中央値によるS更新は変更しない。']])
p('GPU評価器・BMM・A8・RMSNorm・FP32損失は基準Graphと同じである。今回のCPU整理は提案1〜3の合算効果で、候補再生成、コピー、カウンタ確保それぞれの寄与を単独に分離していない。')
p('CPU scheduleの測定値は3.603→3.458 msと小幅な変化だが、区間全体は17.795→11.082 msとなった。schedule欄に含まれない後段の再生成・CPU判定準備も削減されるためである。GPU処理区間はどちらも約2.139 msで変わらない。')
section('3. GPU側の独立した3変更')
table(['条件','実装と制約'],[['候補重み再利用','128×NのFP32候補行列を常駐。前回の摂動座標を現在baseへ戻してから新しい16座標を直接代入。受理base更新も候補行列へ反映する。浮動小数点の差分加算で累積しない。元CPU判定経路を維持。'],['転送整理','pinned固定バッファへメタデータをまとめ、H2Dを1回にする。受理座標を呼出し側から通知して評価器の全重み比較を省き、CUDAイベントとD2H出力を再利用。完了イベントを待ってからhostバッファを再使用する。'],['演算融合','torch.compileのInductorをfullgraphで使い、内部の自動CUDA Graphを無効にして手動捕捉する。GPUの候補並列forwardの融合を試す。FP32・TF32無効は維持するが、演算順序の変化によるA8丸め差は許容基準で判定する。']])
p('候補再利用はGPU処理区間を2.139→2.053 msへ短縮したが、区間全体は17.862 msとなり高速化しなかった。本実装には同期対象を調べる追加のCPU差分走査があり、候補行列コピーだけの理想的な下限を測定したものではない。')
p('転送整理は17.687 ms、約0.6%の短縮にとどまる。3 seedの揺らぎと併せて解釈し、確実な大幅改善とはしない。融合版はGPU処理区間1.012 ms、全体16.727 msで、GPU部分が約半分になっても全体の改善は約6%だった。')
p('5条件すべてで候補対64、各候補128例、全行列の論理forwardを維持する。GPU処理回数を減らす学習則変更、test利用、旧CPU実験の再開、組合せ最適化の探索は行っていない。')
section('4. 性能・メモリ・計測境界')
table(['条件','ms/区間\n平均±標本SD','基準比','GPU ms','予約 MiB'],[[labels[a['engine']],f"{float(a['seconds_per_interval_mean'])*1000:.3f} ± {float(a['seconds_per_interval_sample_sd'])*1000:.3f}",f"{float(a['speedup_vs_original_graph']):.3f}",f"{float(a['gpu_workflow_ms_mean']):.3f}",f"{float(a['gpu_reserved_mib_max']):.1f}"] for a in agg],[119,140,65,85,90])
p('区間時間はCPU候補生成・判定、転送、GPU評価、受理通知、CPUモデル更新を含み、100区間の計測値を平均したもの。初期化、コンパイル、Graph捕捉、ウォームアップ、結果ログのディスク書込みは含めない。記録・診断を減らしただけの高速化ではない。')
p('CUDAイベントは入力転送後から損失計算終了まで。転送整理版だけ受理座標更新をGraph内部に含めるため、GPU欄の境界は完全には同じでない。性能の主比較には区間全体を用いる。')
table(['条件','worker準備 秒','最大割当 MiB','予約 MiB'],[[labels[a['engine']],f"{float(a['setup_seconds_mean']):.3f}",f"{float(a['gpu_allocated_mib_max']):.1f}",f"{float(a['gpu_reserved_mib_max']):.1f}"] for a in agg],[145,118,118,118])
p('準備は事前検証後のコンパイルキャッシュを利用した測定workerの値。融合版は平均1.862秒だが、最初のコールドコンパイル時間を独立計測していないため、コールド起動の総費用は未測定。')
p('メモリはPyTorchの割当・予約カウンタ。Graph専用プールの再生時に割当カウンタだけでは生存テンソル全体のピークを表さない。予約量にはプールを含むがCUDAコンテキスト等は含まない。融合版の予約328 MiBは基準224 MiBから104 MiB増加した。')
section('5. 数値一致性と発火系列')
table(['条件','最大相対損失誤差','損失ビット不一致','票不一致','カウンタ不一致','発火不一致'],[[labels[a['engine']],f"{float(a['max_relative_loss_error']):.8g}",a['loss_bit_mismatches'],a['vote_mismatches'],a['counter_mismatches'],a['action_mismatch_cases']] for a in val],[105,112,82,63,75,62])
p('各条件18ケース×128候補＝2,304損失、全4変更で9,216比較。カウンタは64測定時点すべて、発火は共通状態のケース数。CPU整理・候補再利用・転送整理は固定検査と各100区間×3 seedで候補損失、発火、最終重み・S・乱数が一致した。任意入力での普遍的な同一性の証明ではない。')
p('融合版は2,304候補すべてで損失ビット差があり、うち2,212候補で相対誤差1e-5以上。票270、全時点カウンタ5,782、発火1ケースに差が出た。最大誤差0.00252990で一致基準は未達。FP32融合による演算順序差とA8丸めの関係は既存知見と整合するが、原因を個々のカーネルへ完全分離した検証ではない。')
table(['融合版seed','最初の損失ビット差','最初の発火分岐','発火相違区間数','最終重み/S','乱数'],[[r['seed'],r['first_loss_bit_difference'],r['first_action_divergence'],r['action_difference_intervals'],'不一致','一致'] for r in traj if r['engine']=='fused_graph'],[70,110,100,99,70,50])
p('固定検査は常に基準Graphの重みに揃えて比較する。一方、100区間の測定では各条件が自分の損失で更新するため、分岐後の損失差にはモデル差も含まれる。融合版の不一致は「test精度が低下した」という結果ではない。testは未評価である。')
section('6. 全15測定の個別結果')
table(['条件','seed','ms/区間','GPU ms','CPU schedule ms','準備 秒'],[[labels[r['engine']],r['seed'],f"{float(r['seconds_per_interval'])*1000:.3f}",f"{float(r['gpu_workflow_milliseconds_mean']):.3f}",f"{float(r['schedule_seconds_mean'])*1000:.3f}",f"{float(r['setup_seconds']):.3f}"] for r in raw],[121,36,88,78,98,78])
p('同一seed内の条件は同じ保存重み・S初期値・乱数状態から開始。条件順の循環は順序バイアスを減らすためであり、3 seedの短期測定から普遍的な速度差を保証するものではない。')
p('今回の比較基準は新たに測定した元CUDA Graph版である。v5.5のCPU復元キャッシュ比15.127倍や、E17a全長での過去CPU比11.13倍とは分母・深さ・測定条件が異なるため、掛け合わせた倍率を実測として報告しない。')
section('7. 監査・成果物・v5.5から更新する結論')
p('独立監査は全15測定・1,500区間の保存済み128候補損失を元のCPU epochへ渡し、発火、S、両乱数状態、最終重みを再構成して一致を確認した。候補損失192,000個の記録と絶対差、選択座標を照合し、診断集計だけの差も0件。ソース・モデル・データ20ハッシュを検証した。この監査は融合版の損失が基準と同じことを意味せず、記録した損失から各条件の更新を再現できることを示す。')
p(f'PDF生成時には成果物{len(manifest)}ファイルのSHA-256を再照合し、全15測定の100区間ログから区間平均を、3 seedから平均・標本SD・速度比を再計算した。学習やtestの再評価は実施していない。')
p('v5.5の「CPU側の残存費用が大きい」という課題に対し、CPU処理整理で結果を保った1.606倍の追加高速化を確認した。候補再利用と転送整理は今回の独立実装では小さい効果にとどまった。融合版はGPU部分を短縮したが数値一致基準を満たさなかった。')
p('この追加版は16残差ブロック・保存済み重みから100区間のベンチマークである。8ブロックE17aの12,000区間再現は旧Graph版の結果であり、今回のCPU整理版について最終test精度や全長速度を追加検証したものではない。CPU整理内の個別寄与、最適化同士の組合せ、他構造・他GPUも未検証。')
p('追加実験の全記録・実行時ソースをTDT-v5.5.1_追加実験データ.zipとして外部保存し、PDFにも埋め込む。今回は短期実験のため、候補損失NPY・最終状態PT・逐区間CSVも収録する。全ファイルのsource_hashes.jsonを付す。v5.5の全186ページと旧添付資料も保持する。')
p(str(R.relative_to(ROOT)),'small')
p('実装：gpu_graph_optimizations.py。実行：benchmark_gpu_graph_optimizations.py。独立監査：audit_gpu_graph_optimizations.py。事前登録：GPU_GRAPH_OPTIMIZATION_PROTOCOL.md。各ファイルの実行時版は上記results/sourcesが正本。','small')
main=OUT/'v5.5.1-update.pdf'
Doc(str(main),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=48,bottomMargin=43).build(story,onFirstPage=footer,onLaterPages=footer)
(OUT/'TDT-v5.5.1_追加本文.txt').write_text('\n\n'.join(logs)+'\n')
files=[p for p in R.rglob('*') if p.is_file()]+[Path(__file__).resolve(),OUT/'TDT-v5.5.1_追加本文.txt',ROOT/'tdt_mnist/paper_v5_5/build_pdf.py']
hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))};(OUT/'source_hashes.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2)+'\n')
with zipfile.ZipFile(ARCHIVE,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(set(files)):z.write(p,str(p.relative_to(ROOT)))
 z.write(OUT/'source_hashes.json','source_hashes.json')
pdf=fitz.open(main);n=len(pdf);old=fitz.open(OLD);toc=pdf.get_toc();pdf.insert_pdf(old)
toc.append([1,'付録：TDT-v5.5全文（旧版原記録）',n+1]);toc.extend([l+1,t,p+n] for l,t,p in old.get_toc());pdf.set_toc(toc)
pdf.embfile_add('v5.5.1-evidence.zip',ARCHIVE.read_bytes(),filename=ARCHIVE.name)
for i in range(old.embfile_count()):pdf.embfile_add(f'inherited-v5.5-{i+1}',old.embfile_get(i),filename=old.embfile_info(i).get('filename',f'old-{i+1}.zip'))
pdf.set_metadata({'title':'TDT-v5.5.1 離散状態遷移学習理論 — CUDA Graph追加最適化','author':'TDT experimental study','subject':'CPU compact 1.606x; independent GPU ablations; 15 audited short benchmarks; complete v5.5 preserved','creator':'ReportLab / PyMuPDF'})
pdf.save(DEST,garbage=4,deflate=True);pdf.close();pdf=fitz.open(DEST)
for i in range(len(old)):
 assert pdf[n+i].get_text()==old[i].get_text()
 assert pdf[n+i].get_pixmap(matrix=fitz.Matrix(.4,.4)).samples==old[i].get_pixmap(matrix=fitz.Matrix(.4,.4)).samples
outside=[]
for i in range(n):
 for b in pdf[i].get_text('dict')['blocks']:
  for l in b.get('lines',[]):
   for s in l['spans']:
    if not pdf[i].rect.contains(fitz.Rect(s['bbox'])):outside.append([i+1,s['text']])
 pdf[i].get_pixmap(matrix=fitz.Matrix(.7,.7)).save(OUT/f'preview-{i+1:02}.png')
assert not outside,outside
assert pdf.embfile_get('v5.5.1-evidence.zip')==ARCHIVE.read_bytes()
for i in range(old.embfile_count()):assert pdf.embfile_get(f'inherited-v5.5-{i+1}')==old.embfile_get(i)
with zipfile.ZipFile(ARCHIVE) as z:
 assert z.testzip() is None
 for name,h in hashes.items():assert hashlib.sha256(z.read(name)).hexdigest()==h
text='\n'.join(pdf[i].get_text() for i in range(n))
for key in ['1.606','11.082','0.00252990','192,000','未評価']:assert key in text,key
validation=dict(passed=True,update_pages=n,preserved_v5_5_pages=len(old),total_pages=len(pdf),verified_artifact_hashes=len(manifest),recomputed_seed_measurements=len(raw),old_text_and_pixels_preserved=True,attachments_verified=True,no_text_outside_page=True,archive_files=len(hashes)+1,new_training=False,test_evaluated=False,pdf_sha256=sha(DEST),original_sha256=sha(OLD),archive_sha256=sha(ARCHIVE))
(OUT/'pdf_validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n');print(json.dumps(validation,ensure_ascii=False,indent=2))
