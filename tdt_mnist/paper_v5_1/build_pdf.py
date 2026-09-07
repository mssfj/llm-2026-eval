"""Build an audited v5.1 update followed by the complete, unchanged v5 PDF.
PYTHONPATH=/tmp/tdt-pdf-reader python tdt_mnist/paper_v5_1/build_pdf.py
Requires reportlab and pymupdf; source data are the completed experiment CSVs.
"""
from pathlib import Path
import csv, json, hashlib, statistics as st, html, zipfile
import pymupdf as fitz
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak, Table, TableStyle, Image, Spacer
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
DOC=ROOT/'doc'; ORIGINAL=DOC/'TDT-v5_離散状態遷移学習理論.pdf'
DEST=DOC/'TDT-v5.1_離散状態遷移学習理論.pdf'
A=ROOT/'tdt_mnist/results/a3-ablation-100k-20260907'; D=ROOT/'tdt_mnist/results/depth-100k-20260907'
def read(p):
 with p.open() as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
ap=read(A/'per_seed.csv'); dp=read(D/'per_seed.csv'); aa=read(A/'aggregate.csv'); da=read(D/'aggregate.csv')
assert len(ap)==12 and len(dp)==36
for path, rows, aggs, keys in [(A,ap,aa,['hidden_activation','a3_method']),(D,dp,da,['depth','threshold'])]:
 assert json.loads((path/'verification.json').read_text())['passed']
 for a in aggs:
  rs=[r for r in rows if all(r[k]==a[k] for k in keys)];assert len(rs)==3
  for k in ['val_accuracy','test_accuracy','val_loss','total_fires','zero_difference_fraction']:
   vs=[float(r[k]) for r in rs]
   assert abs(st.mean(vs)-float(a[k+'_mean']))<1e-10
   assert abs(st.stdev(vs)-float(a[k+'_std']))<1e-10
 for r in rows:assert int(r['train_forward_calls'])==1536000
pdfmetrics.registerFont(TTFont('CJK',str(ROOT/'tdt_mnist/paper_v5/CJK-font.ttf')))
pdfmetrics.registerFont(TTFont('Latin','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
blue=colors.HexColor('#173c61')
styles={k:ParagraphStyle(k,fontName='CJK',fontSize=size,leading=leading,wordWrap='CJK',spaceAfter=space,textColor=blue if k in ['title','h'] else colors.black) for k,size,leading,space in [('title',24,36,20),('h',16,24,14),('body',9,15,8),('small',7,10,5),('cell',7,10,0)]}
story=[]; logs=[]
def p(t,style='body'):
 logs.append(t); e=html.escape(str(t)).replace('\n','<br/>')
 for c in set(t):
  if c in '±θτβΔΣ√−→×':e=e.replace(c,f'<font name="Latin">{c}</font>')
 q=Paragraph(e,styles[style]);story.append(q);return q
def section(t):
 if story:story.append(PageBreak())
 q=p(t,'h');q.bookmark='s'+str(len(logs))
def table(headers,rows,widths=None):
 logs.extend(' | '.join(map(str,r)) for r in [headers,*rows])
 data=[[Paragraph(html.escape(str(v)),styles['cell']) for v in r] for r in [headers,*rows]]
 t=Table(data,colWidths=widths or [499/len(headers)]*len(headers),repeatRows=1,hAlign='LEFT')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6eef5')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,blue),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f7fa')]),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]));story.append(t);story.append(Spacer(1,9))
def ms(r,k,m=1,n=3):return f"{float(r[k+'_mean'])*m:.{n}f} ± {float(r[k+'_std'])*m:.{n}f}"
def label(r):return ('ReLU' if r['hidden_activation']=='relu' else 'ReLUなし')+' / '+('absmax' if r['a3_method']=='absmax' else '閾値分離')
def fig(path,caption):
 im=fitz.Pixmap(str(path));w=499;h=w*im.height/im.width
 story.append(Image(str(path),width=w,height=h));p(caption,'small')
class Doc(SimpleDocTemplate):
 def afterFlowable(self,f):
  if hasattr(f,'bookmark'):
   self.canv.bookmarkPage(f.bookmark);self.canv.addOutlineEntry(f.getPlainText(),f.bookmark,0)
def footer(c,d):
 c.setFont('CJK',8);c.setFillColor(blue);c.drawString(48,815,'TDT-v5.1 | 追加実験・改訂考察 | 2026-09-07');c.drawRightString(547,25,str(d.page))
p('TDT-v5.1\n離散状態遷移学習理論','title')
p('A3量子化の設計と、深さ方向の信号伝搬・層別発火率')
p('実験追加版 · 2026年9月7日')
p('本版はv5に、E10：A3の2×2比較（4条件・12run）と、E11：層数×カウンタ閾値（12条件・36run）を追加する。追加部に実験条件、定量結果、層別診断、更新した考察をまとめ、後半にv5全52ページを変更せず収録する。v5中の結論・日付・ページ番号は当時の記録として読む。')
p('追加48runはすべて100,000重み、各12,000学習区間、seed 0・1・2で実施した。v5の9実験群・82条件行・244run記録と合わせ、11実験群・98条件行・292run記録となる。v5の重複記録を含むため、292個の独立標本を意味しない。')
table(['追加実験','主な結果'],[['E10：A3','ReLUなし＋閾値分離でtest 83.120 ± 0.214%。現行A3比 +6.240ポイント。'],['E11：深さ','4層・閾値1でtest 87.870 ± 0.052%。16層では全層が発火しても精度が大幅低下。']],[110,389])
p('値は3 seedの平均 ± 標本標準偏差。評価結果は最終12,000区間時点であり、testを用いたモデル選択は行わない。完全なCSV、実験ソース、監査結果、図は付属の実験データZIPに収録する。')
section('1. 共通条件と比較範囲')
table(['項目','条件'],[['データ','MNIST。入力9×10＝90。train 10,000 / validation 1,000 / test 10,000、data_seed=0。前処理はv5実装を継承。'],['重み・演算','バイアスなし、三値重みをINT8保持。復元スケール、行列積、出力logits、損失はFP32。潜在浮動小数点重み、逆伝播、STEなし。'],['TDT','ブロック16、候補対K=64、12,000区間、batch 128、1区間の最大発火数1。各run 768,000候補対、学習forward 1,536,000回。'],['カウンタ','C8、leak=1。初期S=0.02、EMA係数0.1、下限1e-5。閾値はE10では8、E11では1 / 4 / 8 / 16。'],['初期化・評価','gain=1、層スケール=1/sqrt(fan-in × (1−1/3))。seed 0 / 1 / 2、batch_seed=seed+100000。初期・500区間ごと・最終に検証。CPU、threads=1。'],['基準','v5の100k・12,000区間・ブロック16の高精度条件を継承。E10はA3と活性化を比較。E11はv5同様ReLUあり・A32。']],[90,409])
p('候補対数・学習区間数を揃えた比較であり、深さ・幅・量子化による実時間や演算量まで同一という意味ではない。診断値は学習へフィードバックしない。A3では入力と隠れ層の双方を量子化する。')
section('2. E10：A3の2×2実験')
p('構造は90→1000→10（100,000重み）。隠れ層のReLUあり／なしと、現行absmax／閾値・復元スケール分離を交差させた。ReLUなしは恒等関数であり、入力接続数による新たな正規化は加えていない。')
p('absmax：各サンプル・各層でs=max|x|、q=round(x/s)、復元値=sq。丸めは実装の偶数丸めに従う。全ゼロの場合s=1とする。')
p('閾値分離：τ=0.5 mean(|x|)。|x|>τならq=sign(x)、それ以外はq=0。βは選択された|x|の平均、復元値=βqとする。選択値がなければβ=1。係数0.5はこの比較で固定し、精度に基づく探索は行っていない。各候補forwardで決定的に計算する。')
table(['条件','val精度 %','test精度 %','val損失','損失差ゼロ %'],[[label(r),ms(r,'val_accuracy',100),ms(r,'test_accuracy',100),ms(r,'val_loss'),ms(r,'zero_difference_fraction',100,6)] for r in aa],[121,94,94,90,100])
p('ReLUなし＋閾値分離は、現行ReLU＋absmaxに対しtestを6.240ポイント、validationを5.333ポイント改善した。一方、ReLUありで閾値分離だけを導入すると精度は改善するがvalidation交差エントロピーは悪化する。精度と損失は別指標であり、一律の改善とは結論できない。')
p('v5の同規模A32基準（test 88.55 ± 0.22%）は過去の測定値で、今回の再実行ではない。最良A3でも約5.43ポイントの差が残る。')
section('3. E10：層別量子化誤差・コード分布')
p('最終validation全体で診断した3 seed平均。層0は第1線形層への入力、層1は出力線形層への隠れ入力を表す。MSE=mean((x−復元値)^2)、相対二乗誤差=Σ(x−復元値)^2 / Σx^2。コード分布は復元前の三値コードの比率。')
ac=[r for r in read(A/'activation_aggregate.csv') if r['stage']=='final']
table(['条件 / 層','MSE','相対二乗誤差','−1 %','0 %','+1 %'],[[label(r)+' / '+r['layer'],f"{float(r['mse_mean']):.5f}",f"{float(r['relative_squared_error_mean']):.5f}",*[f"{100*float(r['code_'+c+'_fraction_mean']):.3f}" for c in ['-1','0','1']]] for r in ac],[174,65,80,60,60,60])
p('ReLUありでは隠れ入力が非負なのでコード−1は現れない。ReLUを除くだけではabsmaxの隠れコードは約88.57%が0に集中する。閾値分離との組合せで約35.37% / 30.28% / 34.35%となり、3値を使う分布になった。入力側も同時に変わるため、精度改善を隠れ層だけの効果とは分離できない。')
p('モデル間では量子化前の活性分布自体が異なる。MSEの低下やコードの均等化は有用な診断だが、それだけで精度改善の因果説明や最適なコード分布の証明にはならない。初期・最終の全診断と標準偏差は付属CSVに収録。')
section('4. E10：候補間損失差が0になる割合')
p('損失差ゼロ率は、学習中の全候補対においてFP32で評価した2候補の交差エントロピーの差が厳密に0となった回数を、候補対数768,000で割った値である。表示桁への丸め後に0になった割合ではなく、ゼロ投票率や更新しなかった区間率とも異なる。')
p('三値量子化で候補間の変化が消える場合や、候補の出力差が最終的なFP32損失に現れない場合に発生しうる。同じ損失から、内部活性やlogitsまで同一だったとは判断できない。差が0の候補対は、その比較から更新方向を区別する情報を与えない。')
p('現行ReLU＋absmaxでは約0.525391%、ReLUなし＋閾値分離では約0.002127%。ただし両者とも候補対の大多数では差が非ゼロである。ゼロ率の低下だけでは、精度差の大きさやカウンタへの有効な方向情報を説明しきれない。微小な非ゼロ差の安定性も別の論点として残る。')
fig(A/'comparison.png','図E10-1　validation推移、損失差ゼロ率、最終層別量子化誤差、隠れコード分布。図のidentityはReLUなし、mean_thresholdは閾値分離。')
section('5. E11：層数×カウンタ閾値')
p('深さは出力線形層を含む線形層数で、4 / 8 / 16層は隠れ層3 / 7 / 15層に対応する。全隠れ層はReLU、活性値はA32、出力はlogits。残差接続・バイアス・新規の正規化はない。カウンタ閾値1 / 4 / 8 / 16を各3 seedで比較した。')
from importlib.util import spec_from_file_location,module_from_spec
spec=spec_from_file_location('dd',ROOT/'tdt_mnist/depth_diagnostics.py');mod=module_from_spec(spec);spec.loader.exec_module(mod)
for depth in [4,8,16]:
 widths=[90,*mod.DEPTH_WIDTHS[depth],10]
 assert sum(a*b for a,b in zip(widths,widths[1:]))==100000
 p(str(depth)+'層：'+' → '.join(map(str,widths)))
p('隠れ幅は、重み総数を厳密に100,000としつつ概ね均一になるよう選定した。精度を用いた幅探索は行っていない。深くするほど幅が狭くなるため、本実験は固定パラメータ予算下の深さ比較であり、幅固定での純粋な深さ効果を推定する実験ではない。')
table(['層','閾値','val精度 %','test精度 %','val損失','発火回数'],[[r['depth'],r['threshold'],ms(r,'val_accuracy',100),ms(r,'test_accuracy',100),ms(r,'val_loss'),ms(r,'total_fires',1,1)] for r in da],[30,35,112,112,106,104])
section('6. E11：学習結果の解釈')
p('4層ではvalidation平均が閾値4で86.800%、test平均は閾値1で87.870%となった。8層では閾値4・8のvalidation平均がほぼ同水準であり、0.033ポイント差から最適閾値を断定しない。16層では閾値1・4でもtest平均は約48.49%、seed間の変動が大きい。')
p('閾値16では全深さで更新回数と精度が低下し、とくに16層は平均2,598更新、test 14.117 ± 4.084%となった。固定12,000区間では、強いカウンタ蓄積条件が深いモデルの学習進行を制約していることと整合する。閾値の機序を単独で証明する結果ではない。')
fig(D/'learning_curves.png','図E11-1　深さごとのvalidation学習曲線。3 seedの変動と最終精度を併せて解釈する。')
section('7. E11：層別発火率の定義')
p('ここでの発火は、辺カウンタによって重みが実際に更新された事象を指す。ReLUの非ゼロ活性率とは区別する。最大発火数が1なので、各層の更新回数と発火区間数は一致する。')
table(['指標','分母と意味'],[['全区間発火率','当該層で発火した区間数 / 12,000。全学習予算のうち各層の更新が占める比率。'],['選択時発火率','当該層で発火した区間数 / ブロックに当該層が含まれた区間数。層の選択機会を考慮した条件付き比率。'],['選択辺あたり更新','更新回数 / 選択された辺の累積数。同一区間の複数辺も数える。'],['重みあたり更新','更新回数 / 層の重み数。繰り返し更新を含むため確率ではない。']],[110,389])
p('層ごとに重み数が異なり、選択される機会も異なる。全区間発火率のみで層の更新のしやすさを比較せず、選択時発火率を併記する。同じ深さ・seedでは閾値間の初期状態と選択機会が一致することを監査した。全336個の層×runで更新回数は正であった。')
lf=read(D/'layer_firing_aggregate.csv')
for depth in [4,8,16]:
 section(f'8.{depth}. E11：{depth}層の発火率')
 p('各セルは「全区間発火率 / 選択時発火率」（%、3 seed平均）。層番号は入力側から1始まり。最終層は出力層。全標準偏差・回数・分母は付属CSVに保存。')
 rows=[]
 for layer in range(depth):
  cells=[str(layer+1)]
  for threshold in [1,4,8,16]:
   r=next(r for r in lf if int(r['depth'])==depth and int(r['threshold'])==threshold and int(r['layer'])==layer)
   cells.append(f"{100*float(r['fire_interval_rate_mean']):.3f} / {100*float(r['fire_given_selected_interval_rate_mean']):.3f}")
  rows.append(cells)
 table(['層','閾値1','閾値4','閾値8','閾値16'],rows,[35,116,116,116,116])
section('9. E11：深さ方向の信号伝搬')
p('初期・500区間ごと・最終のvalidationで、各層の入力、活性化前、出力についてRMS、標準偏差、ゼロ率、負値率、最大絶対値、非有限値、全検証例でゼロの特徴数を記録した。診断の集計はFP64、学習計算はFP32。既存評価forwardを観測し、追加の学習forwardは使わない。')
sig=read(D/'signal_aggregate.csv')
rows=[]
for depth in [4,8,16]:
 vals=[]
 for step in [0,12000]:
  r=next(r for r in sig if int(r['depth'])==depth and int(r['threshold'])==8 and int(r['layer'])==depth-1 and r['stage']=='output' and int(r['step'])==step)
  vals.append(ms(r,'rms',1,6))
 rows.append([depth,*vals])
table(['層数（閾値8）','初期の出力RMS','最終の出力RMS'],rows)
p('深いほど初期の出力振幅は小さく、16層で顕著に減衰した。最終時点では出力RMSが増加し、非ゼロの信号と更新が全層で存在する。しかし16層の分類精度は低い。振幅が伝わること、辺が更新されること、識別に有用な情報が学習されることは別に評価すべきである。')
p('全検証例でゼロの特徴は、この検証集合上の観測である。あらゆる入力に対して恒久的に死んだニューロンであることは意味しない。出力層にはReLUがないため、隠れ層のゼロ率とは役割が異なる。')
fig(D/'signal_propagation.png','図E11-2　各層出力のRMS（上、対数軸）とゼロ率（下）。初期と最終、各閾値を比較。大判のPNG・SVGもデータZIPに収録。')
section('10. v5.1の結論と限界')
p('A3の情報損失は、単に精度のビット数だけでなく、ReLUによる符号制約と量子化の閾値・復元スケールに依存する。今回の固定条件ではReLUなし＋閾値分離が最も高精度で、相対量子化誤差と損失差ゼロ率も小さかった。ただしA32との差は残る。')
p('深さ4・8・16のすべてで、全層の辺更新と非ゼロ出力を確認した。一方、固定100k重み・固定学習予算・従来スケーリングの構成では、16層の高精度学習は達成できていない。v5の浅いネットワークで得た成功を、そのまま深いネットワークに一般化できないことが明確になった。')
p('各条件は3 seedに限られ、標準偏差は信頼区間や有意差検定ではない。深さと幅、量子化前の分布と量子化方式が同時に変化する点に注意が必要である。閾値16の劣化が学習区間延長で回復するか、残差接続・初期化スケール・層別閾値で深層学習を改善できるかは未検証。')
p('本追加実験は、TDTの計算量優位性、深層での収束保証、第一通過時間の理論式の普遍性、あるいは内部表現の有効次元を証明するものではない。v5の理論的仮説と実装上の実証範囲を引き続き区別する。')
section('付録A. E10：全12runの最終測定値')
table(['条件','seed','val %','test %','val損失','発火数','損失差0 %'],[[label(r),r['seed'],f"{100*float(r['val_accuracy']):.2f}",f"{100*float(r['test_accuracy']):.2f}",f"{float(r['val_loss']):.6f}",r['total_fires'],f"{100*float(r['zero_difference_fraction']):.6f}"] for r in ap],[143,30,55,55,75,60,81])
for depth in [4,8,16]:
 section(f'付録B.{depth}. E11：{depth}層の全12run')
 table(['閾値','seed','val %','test %','val損失','発火数','損失差0 %'],[[r['threshold'],r['seed'],f"{100*float(r['val_accuracy']):.2f}",f"{100*float(r['test_accuracy']):.2f}",f"{float(r['val_loss']):.6f}",r['total_fires'],f"{100*float(r['zero_difference_fraction']):.6f}"] for r in dp if int(r['depth'])==depth])
section('付録C. 再現性・データ所在・v5の収録')
p('E10：tdt_mnist/results/a3-ablation-100k-20260907/\nE11：tdt_mnist/results/depth-100k-20260907/')
p('per_seed.csvを一次集計として平均・標本標準偏差を再計算し、aggregate.csvとの一致をPDF生成時に確認した。両実験のverification.jsonはpassedであり、学習forward予算、保存モデル、データ・ソース整合性等の既存監査を参照した。')
p('E10のactivation_diagnostics.csv / activation_codes.csv / activation_aggregate.csvに層別誤差とコード分布、E11のlayer_firing.csv（336行）と同aggregateに層別更新、signal_metrics.csv（25,200行）と同aggregateに信号伝搬診断を収録する。各runの逐区間の層ログはrunsディレクトリに保存済みで、集計値をZIPへ収録した。チェックポイントと全逐区間ログはZIPに含めない。')
p('付属ZIPは追加実験のCSV・JSON・README・保存時ソース・図と本生成スクリプトを含む。manifest.jsonのソースは各実験時点のものを使い、後から変更されたコードと混同しない。v5の実験データZIPもPDFに別添付する。')
p('次ページからはv5原版全52ページ（v4原文10ページを含む）を収録する。既存の本文・表・図・ページ番号は変更していない。PDFのしおりで追加実験と原版の各章へ移動できる。原版の本文に記載された実験数・最良条件はv5時点の範囲であり、本追加部の結果と併せて読む。')
p('v5原版 SHA-256:\n'+sha(ORIGINAL),'small')
main=OUT/'v5.1-update.pdf'
Doc(str(main),pagesize=A4,leftMargin=48,rightMargin=48,topMargin=48,bottomMargin=45).build(story,onFirstPage=footer,onLaterPages=footer)
(OUT/'TDT-v5.1_追加本文.txt').write_text('\n\n'.join(logs))
# Archive the evidence accompanying the new sections, excluding raw checkpoints.
files=[]
for directory in [A,D]:
 files.extend(f for f in directory.rglob('*') if f.is_file() and f.suffix in ['.csv','.json','.md','.py','.png','.svg'])
files.extend([Path(__file__).resolve(),OUT/'TDT-v5.1_追加本文.txt'])
hashes={str(f.relative_to(ROOT)):sha(f) for f in sorted(files)}
(OUT/'source_hashes.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2))
archive=DOC/'TDT-v5.1_追加実験データ.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for f in sorted(files):z.write(f,str(f.relative_to(ROOT)))
 z.write(OUT/'source_hashes.json','source_hashes.json')
pdf=fitz.open(main); n=len(pdf);toc=pdf.get_toc();old=fitz.open(ORIGINAL)
pdf.insert_pdf(old)
toc.append([1,'付録D：TDT-v5原版（全52ページ）',n+1])
toc.extend([level+1,title,page+n] for level,title,page in old.get_toc())
pdf.set_toc(toc)
pdf.embfile_add("TDT-v5.1-additional-data.zip",archive.read_bytes(),filename=archive.name,desc='E10 / E11 data, figures, source snapshots and build source')
oldzip=DOC/'TDT-v5_実験データ.zip'
if oldzip.exists():pdf.embfile_add("TDT-v5-original-data.zip",oldzip.read_bytes(),filename=oldzip.name,desc='Original v5 experiment archive')
pdf.set_metadata({'title':'TDT-v5.1 離散状態遷移学習理論 — A3量子化と深さ方向の実験','author':'TDT experimental study','subject':'E10: 12 A3 runs; E11: 36 depth runs; original v5 preserved','keywords':'TDT,MNIST,A3,ReLU,depth,counter,layer firing','creator':'ReportLab / PyMuPDF'})
pdf.save(DEST,garbage=4,deflate=True);pdf.close()
check=fitz.open(DEST)
for i in range(len(old)):
 assert check[n+i].get_text()==old[i].get_text(),i
 assert check[n+i].get_pixmap(matrix=fitz.Matrix(.5,.5)).samples==old[i].get_pixmap(matrix=fitz.Matrix(.5,.5)).samples,i
outside=[]
for i in range(n):
 for block in check[i].get_text('dict')['blocks']:
  for line in block.get('lines',[]):
   for span in line['spans']:
    box=fitz.Rect(span['bbox'])
    if not check[i].rect.contains(box):outside.append((i+1,span['text']))
assert not outside,outside
assert check.embfile_get("TDT-v5.1-additional-data.zip")==archive.read_bytes()
assert all(sha(ROOT/k)==v for k,v in hashes.items())
audit={'passed':True,'new_conditions':16,'new_runs':48,'update_pages':n,'original_v5_pages':len(old),'total_pages':len(check),'all_displayed_per_seed_aggregates_verified':True,'original_v5_text_and_pixels_preserved':True,'no_text_outside_page':True,'embedded_archive_matches_external':True,'original_v5_sha256':sha(ORIGINAL),'pdf_sha256':sha(DEST),'archive_sha256':sha(archive)}
(OUT/'pdf_validation.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
for page in [0,2,3,5,6,10,11,12,n-1]:
 if page<n:check[page].get_pixmap(matrix=fitz.Matrix(1,1)).save(OUT/f'preview-{page+1:02d}.png')
print(json.dumps(audit,ensure_ascii=False,indent=2));print(DEST)
