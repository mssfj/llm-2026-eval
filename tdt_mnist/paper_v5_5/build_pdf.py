"""Publish v5.5 from frozen records; no training or test evaluation."""
from pathlib import Path
import csv, json, hashlib, html, re, statistics, zipfile
import pymupdf as fitz
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak, Table, TableStyle, Spacer
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon, Circle
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent; DOC=ROOT/'doc'
DEST=DOC/'TDT-v5.5_離散状態遷移学習理論.pdf'; OLD=DOC/'TDT-v5.4_離散状態遷移学習理論.pdf'
ARCHIVE=DOC/'TDT-v5.5_追加実験データ.zip'
base=ROOT/'tdt_mnist/results'
C=base/'fast-engine-e17a-20260908'; A=base/'allocation-ablations-16blocks-20260908'; G=base/'gpu-evaluation-16blocks-20260908'; R=base/'gpu-e17a-reproduction-20260908'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
def j(p): return json.loads(p.read_text())
def rows(p): return list(csv.DictReader(p.open()))
verified={}
for root in [A,G,R,C]:
 manifest=j(root/('stop_manifest_sha256.json' if root==C else 'artifacts_sha256.json'))
 if root==C: manifest=manifest['files']
 for name,digest in manifest.items(): assert sha(root/name)==digest,(root,name)
 verified[root.name]=len(manifest)
assert j(R/'audit.json')['passed']
assert j(C/'status.json')['stopped'] and not j(C/'status.json')['active_seeds']
assert sha(G/'sources/gpu_evaluation_engines.py')==sha(R/'sources/gpu_evaluation_engines.py')==sha(ROOT/'tdt_mnist/gpu_evaluation_engines.py')
summaries=[j(R/f'per_seed/seed{s}/summary.json') for s in range(3)]
vals=[100*s['test']['accuracy'] for s in summaries];report=j(R/'report.json')
assert abs(statistics.mean(vals)-report['gpu_test_mean_percent'])<1e-10
assert abs(statistics.stdev(vals)-report['gpu_test_sample_std_percent'])<1e-10
assert all(s['steps']==12000 and s['test_evaluations']==1 for s in summaries)
for d in [A,G]:
 for r in rows(d/'aggregate.csv'):
  if 'speedup_vs_cpu_cache' in r:
   ref=float(rows(d/'aggregate.csv')[0]['seconds_per_interval_mean'])
   assert abs(ref/float(r['seconds_per_interval_mean'])-float(r['speedup_vs_cpu_cache']))<1e-10
pdfmetrics.registerFont(TTFont('CJK',str(ROOT/'tdt_mnist/paper_v5/CJK-font.ttf')))
pdfmetrics.registerFont(TTFont('Latin','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
blue=colors.HexColor('#173c61'); teal=colors.HexColor('#167e80'); muted=colors.HexColor('#526476')
styles={k:ParagraphStyle(k,fontName='CJK',fontSize=s,leading=l,wordWrap='CJK',spaceAfter=a,textColor=blue if k in ['title','h'] else colors.black) for k,s,l,a in [('title',24,35,20),('h',16,23,15),('sub',11,17,9),('body',9,15,9),('small',7,11,6),('cell',7,10.7,0)]}
story=[]; logs=[]
def clean(t):
 t=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'\1（\2）',str(t));return t.replace('**','').replace('`','')
def esc(t):
 t=html.escape(clean(t)).replace('\n','<br/>')
 for c in '±θτβΔΣ√−→×α≤≥δ':t=t.replace(c,f'<font name="Latin">{c}</font>')
 return t
def p(t,style='body'):
 logs.append(clean(t));q=Paragraph(esc(t),styles[style]);story.append(q);return q
def section(t):
 if story:story.append(PageBreak())
 q=p(t,'h');q.bookmark='s'+str(len(logs))
def table(headers,data,widths=None):
 logs.extend(' | '.join(map(str,r)) for r in [headers,*data])
 if widths is None:
  k=len(headers);widths=([112,387] if k==2 else [116,165,218] if k==3 else [125]+[(499-125)/(k-1)]*(k-1))
 t=Table([[Paragraph(esc(v),styles['cell']) for v in r] for r in [headers,*data]],colWidths=widths,repeatRows=1,hAlign='LEFT')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6eef5')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f7fa')]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]));story.extend([t,Spacer(1,10)])
def md(path):
 lines=path.read_text().splitlines();i=0
 while i<len(lines):
  t=lines[i].strip()
  if t.startswith('## '): section(t[3:])
  elif t.startswith('### '):p(t[4:],'sub')
  elif t.startswith('|'):
   values=[]
   while i<len(lines) and lines[i].strip().startswith('|'):
    cells=[c.strip() for c in lines[i].strip().strip('|').split('|')]
    if not all(set(c)<=set('-: ') for c in cells):values.append(cells)
    i+=1
   table(values[0],values[1:]);continue
  elif t and not t.startswith('#'):p(t)
  i+=1
class Doc(SimpleDocTemplate):
 def afterFlowable(self,f):
  if hasattr(f,'bookmark'):
   self.canv.bookmarkPage(f.bookmark);self.canv.addOutlineEntry(f.getPlainText(),f.bookmark,0)
def footer(c,d):
 c.setFont('CJK',8);c.setFillColor(blue);c.drawString(48,815,'TDT-v5.5 | 候補評価の高速化・CUDA Graph | 2026-09-08');c.drawRightString(547,25,str(d.page))
def label(d,x,y,t,size=9,color=blue):d.add(String(x,y,t,fontName='Latin',fontSize=size,fillColor=color))
def arrow(d,x1,y1,x2,y2):
 d.add(Line(x1,y1,x2,y2,strokeColor=muted,strokeWidth=1))
 if abs(y2-y1)>abs(x2-x1):
  sign=1 if y2>y1 else -1;d.add(Polygon([x2,y2,x2-3,y2-sign*6,x2+3,y2-sign*6],fillColor=muted,strokeColor=None))
 else:
  sign=1 if x2>x1 else -1;d.add(Polygon([x2,y2,x2-sign*6,y2-3,x2-sign*6,y2+3],fillColor=muted,strokeColor=None))
def box(d,x,y,w,h,ls,fill):
 d.add(Rect(x,y,w,h,rx=5,ry=5,fillColor=fill,strokeColor=muted,strokeWidth=.6))
 for i,t in enumerate(ls):label(d,x+10,y+h-17-i*14,t,8)
p('TDT-v5.5\n離散状態遷移学習理論','title')
p('候補評価エンジンの高速化とE17a再現検証\nCUDA Graph版の処理方法・数値一致性・性能・監査','sub')
p('実験終了版 · 2026年9月8日')
table(['結果の要点','今回確認できたこと'],[['CPUの完全一致最適化','復元キャッシュ1.107倍、候補コピー削減1.013倍。検査範囲でFP32ビット一致。'],['GPUの速度','16残差ブロック・100区間×3 seedでCUDA Graph版15.127倍（CPU復元キャッシュ比）。'],['E17aの全長再現','8残差ブロック・12,000区間×3 seed：90.587 ± 0.398%。既存90.637 ± 0.430%との差−0.050 pt、精度許容基準に合格。'],['一致性と終了状態','GPUはCPUと発火が分岐。元のCPU fastは遅くなり、seed 1・2を途中終了。完全一致と精度維持を区別する。']])
p('v5.4以降の実験を追加し、後半にv5.4全文とその添付資料を継承する。追加部は確定結果・途中終了記録・実行時ソースに基づく。PDFのしおりから各節および旧版へ移動できる。表の±は3 seedの標本標準偏差。ptは精度のパーセントポイント。')
section('読者案内・共通条件')
table(['読む目的','参照箇所'],[['全体結果','終了報告の結論、1〜4節、精度と速度の図'],['CUDA Graphの実装詳細','処理図、G1〜G8：データ配置、乱数、BMM、捕捉・再生、同期、判定'],['検証と追跡','監査・添付資料一覧、埋め込みZIP内のCSV・設定・実行時ソース'],['v5.4以前の科学実験','付録のTDT-v5.4全文：E17〜E20と過去版']])
table(['固定した項目','内容'],[['データ','MNIST、9×10平均プーリングで90次元。(x/255−0.1307)/0.3081。train 10,000 / val 1,000 / test 10,000、data_seed=0。'],['TDT','閾値8、選択16辺、K=64候補対、batch128、最大1発火/区間、C8、leak1、区間末リセット。S初期0.02、EMA0.1、下限1e-5。全長再現は12,000区間。'],['重み・構造','INT8三値、初期ゼロ率1/3、gain1、固定スケール1/sqrt(fan-in×2/3)、バイアスなし。pre-norm残差、RMSNorm eps1e-8、ゲインなし。'],['精度・乱数','演算・復元・ストリーム・logits・損失FP32、A8 absmax/127。seed0/1/2、batch_seed=seed+100000、CPU threads1。GPUで逆伝播・STEは使わない。']])
p('短期ベンチマークは保存済みE18a重みから各100区間の動作測定であり、新規の収束実験として数えない。全長GPU再現はE17a初期状態から3 run。元のCPU fastは1 run完了・2 run途中終了。旧E17/E18/E20対照は再学習していない。')
md(ROOT/'tdt_mnist/ENGINE_OPTIMIZATION_SUMMARY.md')
section('図1. 16残差ブロックの区間時間')
d=Drawing(499,300)
items=rows(G/'aggregate.csv'); names=['CPU cache','GPU sequential','GPU batched','GPU + Graph']
for i,(r,name) in enumerate(zip(items,names)):
 y=235-i*59; v=float(r['seconds_per_interval_mean'])*1000
 label(d,0,y+13,name,9);w=v/650*325
 d.add(Rect(145,y, w,24,fillColor=teal if i==3 else blue,strokeColor=None));label(d,145+w+5,y+8,f'{v:.2f}',8)
label(d,145,12,'Milliseconds per interval (lower is faster)',9);story.append(d)
p('3 seed平均。CPU候補生成・判定、転送、GPU候補評価、受理更新を含む。Graph捕捉・ウォームアップは別。CPU復元キャッシュ比15.127倍。GPUに逐次移すだけでは遅くなった。元のCPU fastベンチマークとは異なる比較基準。')
section('図2. E17a全長再現の最終精度')
d=Drawing(499,275);x0=110;x1=465
for tick in [90,90.337,90.637,90.937,91.2]:
 x=x0+(tick-90)/1.2*(x1-x0);d.add(Line(x,40,x,235,strokeColor=colors.HexColor('#dddddd')));label(d,x-12,24,f'{tick:.3f}',7)
lo=x0+.337/1.2*(x1-x0);hi=x0+.937/1.2*(x1-x0)
d.add(Rect(lo,45,hi-lo,180,fillColor=colors.HexColor('#e5f4ec'),strokeColor=None))
for i,(name,mean,sd,raw) in enumerate([('CPU E17a',report['cpu_test_mean_percent'],report['cpu_test_sample_std_percent'],[91.11,90.27,90.53]),('GPU Graph',statistics.mean(vals),statistics.stdev(vals),vals)]):
 y=180-i*85;label(d,0,y,name,9)
 xx=lambda a:x0+(a-90)/1.2*(x1-x0)
 d.add(Line(xx(mean-sd),y,xx(mean+sd),y,strokeColor=blue,strokeWidth=2));d.add(Circle(xx(mean),y,4,fillColor=blue,strokeColor=None))
 for v in raw:d.add(Circle(xx(v),y-15,2.5,fillColor=teal,strokeColor=None))
 label(d,115,y+20,f'{mean:.3f} +/- {sd:.3f} %',9)
story.append(d)
p('大点と横線は平均±標本SD、小点はseed値。背景帯はGPU平均の事前登録許容範囲90.337〜90.937%。個別seedに課す基準ではない。平均差−0.050 ptで合格だが、統計的同等性やCPUとの発火系列完全一致を示すものではない。')
section('図3. CUDA Graph版の1区間の処理')
d=Drawing(499,505);light=colors.HexColor('#edf2f7');green=colors.HexColor('#e5f4ec')
label(d,15,485,'CPU: authoritative TDT state',10);label(d,273,485,'GPU: candidate evaluation',10)
box(d,5,390,223,65,['1. Clone RNG states; schedule','16 coordinates / 64 pair batches','128 x 16 candidate INT8 codes'],light)
box(d,268,390,225,65,['2. Sync accepted weight change','copy_ into fixed input buffers','base / indices / batches / codes'],light);arrow(d,228,422,268,422)
box(d,268,250,225,112,['3. graph.replay()','Build candidate weights [128,N]','Gather paired input batches','A8 / RMSNorm / ReLU / BMM','Residual FP32 stream / CE'],green);arrow(d,380,390,380,362)
box(d,268,170,225,52,['4. graph_losses [128]','One .cpu() call: 512 bytes'],light);arrow(d,380,250,380,222)
box(d,5,154,223,80,['5. Original train.epoch','Consume losses in candidate order','Votes / C8 / threshold / tie-break','At most one accepted change'],light);arrow(d,268,195,228,195)
box(d,5,62,223,62,['6. Update CPU INT8 weights + S','Log / checkpoint / CPU validation','Next interval: sync accepted change'],light);arrow(d,116,154,116,124)
arrow(d,245,90,245,422);arrow(d,228,90,245,90);arrow(d,245,422,268,422)
label(d,12,12,'Captured: GPU green box only. RNG and decisions remain on CPU.',8)
story.append(d)
p('Graph捕捉は開始時に実施。区間ごとには固定バッファへ内容をコピーし、同じGPU処理を再生する。図のCPU損失消費では正本Generatorが元のコードどおり進む。先読みGeneratorとは分離されている。')
md(OUT/'cuda_graph_details.md')
section('計測範囲と全長runの時間内訳')
table(['seed','Graph準備 秒','engine 秒','validation 秒','naive分岐確認 秒','学習ループ 秒'],[[i,f"{s['gpu_setup_seconds']:.3f}",f"{s['engine_seconds']:.3f}",f"{s['validation_seconds']:.3f}",f"{s['naive_reference_seconds']:.3f}",f"{s['elapsed_seconds']:.3f}"] for i,s in enumerate(summaries)],[35,84,85,85,110,100])
p('engine時間はgpu_epoch呼出し全体。学習ループ時間は初期準備後から12,000区間終了までで、validation・途中のnaive比較・ログ・checkpointを含む。初期/最終の層単独プローブ、Graph準備、最終testはこの学習ループ時間の外。上表は相互排他的な全実行時間の分解ではないため、行の値を単純加算しない。')
p('同一マシン・CPU threads1の短期比較でも、先行CPU実験との共有資源がある。GPU系列のCPU affinityは15。既存CPU E17aの過去時間との比11.13倍は参考値であり、同一負荷下の専有速度保証ではない。各rootのruntime_workers.jsonとoverlap記録を添付する。')
section('監査・添付資料と版の継承')
table(['実験記録','PDF生成時に再照合した成果物ハッシュ数'],[[name,n] for name,n in verified.items()],[350,149])
p('GPU全長再現の既存独立監査は77ハッシュ、3 seed全36,000区間の損失差・S・乱数状態・発火からの最終重み再構成などを確認した。ここでのPDF生成時ハッシュ照合数は、各成果物マニフェストの登録ファイル数であり、独立監査の77件と対象・数え方が異なる。PDF生成では学習やtestの再評価をせず、保存済み3 seedから平均と標本SDを再計算した。')
p('添付のTDT-v5.5_追加実験データ.zipは本文原稿、生成スクリプト、集計CSV、設定、監査、事前登録、停止記録、実行時sourcesを含む。各ファイルのSHA-256はsource_hashes.json。大容量の逐区間CSV、候補損失NPY、モデルPTは実験結果ディレクトリに保存済みで、ZIPには含めない。これらの追跡用マニフェストは添付する。')
for root in [C,A,G,R]:p(str(root.relative_to(ROOT)),'small')
p('v5.4の165ページを後半へ継承し、本文・図の画素と埋め込み資料の一致を検査する。旧版の日付・結論・ページ番号は当時の記録として読む。旧版でFP32演算を用いていたことと、今回GPUで候補を並列評価したことを区別する。新規の科学条件や追加test探索は行っていない。')
main=OUT/'v5.5-update.pdf'
Doc(str(main),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=48,bottomMargin=43,title='TDT-v5.5 追加実験・CUDA Graph処理詳細',author='TDT experimental study').build(story,onFirstPage=footer,onLaterPages=footer)
(OUT/'TDT-v5.5_追加本文.txt').write_text('\n\n'.join(logs)+'\n')
files=[Path(__file__).resolve(),OUT/'cuda_graph_details.md',OUT/'TDT-v5.5_追加本文.txt',ROOT/'tdt_mnist/ENGINE_OPTIMIZATION_SUMMARY.md']
large={'metrics.csv','layer_metrics.csv','firing.csv','intervals.csv','losses.csv','mismatches.csv'}
for root in [C,A,G,R]:
 for f in root.rglob('*'):
  if not f.is_file() or f.suffix not in ['.json','.csv','.md','.py','.patch']:continue
  if f.name in large or f.stat().st_size>2_000_000:continue
  files.append(f)
for name in ['FAST_ENGINE_PREREGISTRATION.md','GPU_ENGINE_BENCHMARK_PROTOCOL.md','GPU_E17A_REPRODUCTION_PREREGISTRATION.md']:
 files.append(ROOT/'tdt_mnist'/name)
hashes={str(f.relative_to(ROOT)):sha(f) for f in sorted(set(files))}
(OUT/'source_hashes.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2)+'\n')
with zipfile.ZipFile(ARCHIVE,'w',zipfile.ZIP_DEFLATED) as z:
 for f in sorted(set(files)):z.write(f,str(f.relative_to(ROOT)))
 z.write(OUT/'source_hashes.json','source_hashes.json')
pdf=fitz.open(main);n=len(pdf);old=fitz.open(OLD);toc=pdf.get_toc()
pdf.insert_pdf(old);toc.append([1,'付録：TDT-v5.4全文（旧版の原記録）',n+1]);toc.extend([level+1,title,page+n] for level,title,page in old.get_toc());pdf.set_toc(toc)
pdf.embfile_add('v5.5-evidence.zip',ARCHIVE.read_bytes(),filename=ARCHIVE.name)
for i in range(old.embfile_count()):
 info=old.embfile_info(i);pdf.embfile_add(f'inherited-v5.4-attachment-{i+1}',old.embfile_get(i),filename=info.get('filename',f'old-{i+1}.zip'))
pdf.set_metadata({'title':'TDT-v5.5 離散状態遷移学習理論 — 候補評価高速化・CUDA Graph版処理詳細','author':'TDT experimental study','subject':'CPU/GPU optimization and E17a accuracy reproduction; stopped CPU study; complete v5.4 preserved','creator':'ReportLab / PyMuPDF'})
pdf.save(DEST,garbage=4,deflate=True);pdf.close();result=fitz.open(DEST)
for i in range(len(old)):
 assert result[n+i].get_text()==old[i].get_text(),i
 assert result[n+i].get_pixmap(matrix=fitz.Matrix(.4,.4)).samples==old[i].get_pixmap(matrix=fitz.Matrix(.4,.4)).samples,i
outside=[]
for i in range(n):
 for block in result[i].get_text('dict')['blocks']:
  for line in block.get('lines',[]):
   for span in line['spans']:
    if not result[i].rect.contains(fitz.Rect(span['bbox'])):outside.append([i+1,span['text']])
 result[i].get_pixmap(matrix=fitz.Matrix(.7,.7)).save(OUT/f'preview-{i+1:02}.png')
assert not outside,outside
assert result.embfile_get('v5.5-evidence.zip')==ARCHIVE.read_bytes()
for i in range(old.embfile_count()):assert result.embfile_get(f'inherited-v5.4-attachment-{i+1}')==old.embfile_get(i)
text='\n'.join(result[i].get_text() for i in range(n))
for key in ['90.587','15.127','G8.','graph.replay()','5,282','5,465']:assert key in text,key
validation=dict(passed=True,update_pages=n,preserved_v5_4_pages=len(old),total_pages=len(result),verified_artifact_hashes=verified,new_training_or_test_evaluation=False,gpu_source_snapshots_identical=True,old_text_and_pixels_preserved=True,no_text_outside_page=True,attachments_verified=True,archive_files=len(hashes)+1,pdf_sha256=sha(DEST),original_sha256=sha(OLD),archive_sha256=sha(ARCHIVE))
(OUT/'pdf_validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n');print(json.dumps(validation,ensure_ascii=False,indent=2))
