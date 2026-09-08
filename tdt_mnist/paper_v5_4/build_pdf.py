"""Build v5.4 from audited E17-E20 records; preserve the complete v5.3 PDF."""
from pathlib import Path
import csv,json,hashlib,statistics as st,html,zipfile
import pymupdf as fitz
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,PageBreak,Table,TableStyle,Image,Spacer
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;DOC=ROOT/'doc'
ORIGINAL=DOC/'TDT-v5.3_離散状態遷移学習理論.pdf';DEST=DOC/'TDT-v5.4_離散状態遷移学習理論.pdf'
A=ROOT/'tdt_mnist/results/residual-stream-a8-e17-20260908';B=ROOT/'tdt_mnist/results/residual-followups-e18-e20-20260908'
def read(p):return list(csv.DictReader(p.open()))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
verified=0
for root in [A,B]:
 assert json.loads((root/'status.json').read_text())['complete']
 for name,digest in json.loads((root/'artifacts_sha256.json').read_text()).items():
  assert sha(root/name)==digest,(root,name);verified+=1
ap=read(A/'per_seed/results.csv');bp=read(B/'per_seed/results.csv')
ag=read(A/'aggregate/results.csv')+read(B/'aggregate/results.csv')
assert len(ap)==9 and len(bp)==24
for r in ag:
 vals=[float(x['test_accuracy_percent']) for x in ap+bp if x['condition']==r['condition']]
 assert len(vals)==3
 assert abs(st.mean(vals)-float(r['test_mean_percent']))<1e-10
 assert abs(st.stdev(vals)-float(r['test_sample_std_percent']))<1e-10
helper=(ROOT/'tdt_mnist/paper_v5_1/build_pdf.py').read_text()
helper=helper[helper.index('pdfmetrics.registerFont'):helper.index("p('TDT-v5.1\\n")]
helper=helper.replace('TDT-v5.1 | 追加実験・改訂考察 | 2026-09-07','TDT-v5.4 | 残差ストリーム型TDT | 2026-09-08')
exec(helper)
# Use DejaVu Sans for mathematical symbols in table cells as well.
def celltext(s):
 s=html.escape(str(s))
 for c in '±α':s=s.replace(c,f'<font name="Latin">{c}</font>')
 return s
# The original helper's CJK font covers Japanese; ASCII formulas avoid missing glyphs.
def table(headers,rows,widths=None):
 logs.extend(' | '.join(map(str,r)) for r in [headers,*rows])
 t=Table([[Paragraph(celltext(v),styles['cell']) for v in r] for r in [headers,*rows]],colWidths=widths or [499/len(headers)]*len(headers),repeatRows=1,hAlign='LEFT')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6eef5')),('VALIGN',(0,0),(-1,-1),'TOP'),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f7fa')]),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]));story.append(t);story.append(Spacer(1,8))
def figure(path,caption):
 im=fitz.Pixmap(str(path));w=min(499,560*im.width/im.height)
 story.append(Image(str(path),width=w,height=w*im.height/im.width));p(caption,'small')
def md_tables(path,prefix):
 lines=path.read_text().splitlines();i=0
 while i<len(lines):
  s=lines[i]
  if s.startswith('## '):section(prefix+' / '+s[3:])
  elif s.startswith('|'):
   rows=[]
   while i<len(lines) and lines[i].startswith('|'):
    cells=[x.strip() for x in lines[i].strip('|').split('|')]
    if not all(set(c)<=set('-: ') for c in cells):rows.append(cells)
    i+=1
   table(rows[0],rows[1:]);continue
  elif s and not s.startswith('#'):p(s)
  i+=1
p('TDT-v5.4\n離散状態遷移学習理論','title')
p('残差ストリームによる非線形性・深さ・A4耐性\nおよび同一構造のbackprop／STE対照')
p('実験追加版 · 2026年9月8日')
p('v5.3にE17〜E20の11条件・33 runを追加する。E17は9 run、E18は12 run、E19は3 run、E20は9 run。E18以降のE17a対照は保存結果の再利用であり、重複して新規学習に数えない。全条件seed 0/1/2。')
table(['実験','主要な結果'],[['E17：残差ストリーム','A8＋ReLUで90.637%。固定線形対照87.31%を3.327ポイント上回り、事前登録主判定に合格。'],['E18：深さ','16/24/32ブロックで89.357/88.583/87.553%。非劣化基準に未達。'],['E19：全量子化点A4','75.960%。A4コスト14.677ポイントで実用域判定に未達。'],['E20：backprop対照','W3＋A8＋STEで95.987%。E17aとの差は5.350ポイント。ただし学習則のみの因果差ではない。']],[125,374])
p('本追加部の平均±標準偏差は3 seedの平均±標本標準偏差。精度差の単位はパーセントポイント（pt）。後半にv5.3全文をそのまま収録する。旧版の記述は当時の検証範囲を表す。')
section('1. 共通設定と残差ストリーム')
p('入力射影 h0 = W_in Q(x)。各ブロックは h = h + W2 Q(ReLU(W1 Q(RMSNorm(h))))。出力は logits = W_out Q(RMSNorm_final(h))。行列の表記は作用素として記し、実装ではバッチ行ベクトルを用いる。残差ストリームhはFP32で保持し、一切量子化しない。')
p('RMSNorm(h) = h / sqrt(mean(h^2) + 1e-8)。例ごとに計算し、学習ゲインなし。枝の中間値・ReLU出力、復元スケール、行列積、logits、交差エントロピーはFP32。E17bだけReLUを恒等関数に置き換える。')
table(['項目','固定設定'],[['データ','MNIST、9×10平均プーリングで90次元。(x/255 − 0.1307)/0.3081。train10,000 / val1,000 / test10,000、data_seed=0。'],['TDT','発火しきい値8、16辺、64候補対、12,000区間、batch128、最大1発火/区間、C8、leak1、区間末証拠リセット。S初期0.02、EMA0.1、下限1e-5。'],['三値重み','INT8 {-1,0,+1}、初期ゼロ率1/3、gain1、固定スケール1/sqrt(fan-in × 2/3)。バイアス・潜在FP32重み・逆伝播・STEなし（E20は別仕様）。'],['評価・乱数','TDTは初期＋500区間ごとval、最終12,000区間のみtest。seed0/1/2、batch_seed=seed+100000、CPU threads=1。'],['量子化','A8：各例・各点absmax/127、[-127,127]の255値。A4：absmax/7、[-7,7]の15値。丸め後FP32復元。A3の閾値分離方式とは異なる。']],[82,417])
p('1区間の学習forwardは64候補対×2＝128回。1 runでは1,536,000回。今回のTDT新規24 runでは36,864,000回。validation、初期/最終の層単独プローブは別計上。既存TDTの候補・発火機構を継承した。')
section('2. 実験行列と重み数の確認')
rows=[]
for c,b,d,q,act in [('E17a',8,76,'A8','ReLU'),('E17b',8,76,'A8','恒等'),('E17c',8,76,'FP32','ReLU'),('E18a',16,76,'A8','ReLU'),('E18b',24,76,'A8','ReLU'),('E18c',32,76,'A8','ReLU'),('E18d',16,54,'A8','ReLU'),('E19a',8,76,'A4','ReLU')]:
 rows.append([c,b,d,90*d,2*b*d*d,10*d,100*d+2*b*d*d,q+'/'+act])
table(['条件','blocks','幅','入力','枝計','出力','総重み数','枝活性'],rows,[44,37,28,54,67,45,68,156])
p('E17は18行列・100,016重みで100k±1%以内。E18dは4,860＋93,312＋540＝98,712個（−1.288%）のため予算条件を外れる。幅55では102,300個（＋2.3%）となり、一定整数幅で±1%を満たせない。ユーザーの明示承認により幅54を維持した。承認記録コミットc9a635aを開始前に保存した。')
p('ゼロ枝での恒等写像、E17と汎用化した実装のforward・区間動作一致、128 forward、全18行列への勾配、量子化重みと潜在重みの分離、任意上流勾配の恒等STEを単体テストで確認した。')
section('3. 全条件の最終test精度')
table(['条件','test平均 ± 標本SD (%)','実時間平均（分）'],[[r['condition'],f"{float(r['test_mean_percent']):.4f} ± {float(r['test_sample_std_percent']):.4f}",f"{float(r['runtime_mean_seconds'])/60:.2f}" if r.get('runtime_mean_seconds') else 'E17a: 36.38（他は原記録）'] for r in ag],[65,230,204])
p('既存対照：E16 A32 87.31%、A8 87.03%、A4 67.17%。E14 FP32 backpropは93.89±0.551%、95,274重みの異なる直列構造。対照は再学習していない。E16 A32の丸め前平均87.313333…%に対し、E17主判定には事前登録値87.31%を固定使用した。')
section('4. E17：非線形性の獲得に関する判定')
p('事前登録：E17aのtest平均が87.31%を3.0pt以上上回り、全3 seedが改善方向で一致すること。結果は平均差＋3.3267pt、seed別差＋3.80 / ＋2.96 / ＋3.22ptで合格。全seedが3pt以上であることは要件ではない。')
p('E17a−E17b＝＋1.3833±0.6447pt。ReLU追加の効果として報告する。E17bにもRMSNormと動的量子化の非線形性が残るため、全非線形性を除いた線形モデルではない。E17c−E17a＝＋0.0633±0.5064pt。今回の残差構造ではA8とFP32の差は小さいが、3 seedから同等性を証明するものではない。')
figure(A/'figures/accuracy.png','E17：validation推移と最終test。帯・誤差棒は3 seedの標本標準偏差。')
section('5. E18：深さ反転テスト')
p('事前登録の非劣化境界は90.637−1.0＝89.637%。E18a/b/cのすべてがこの境界以上なら「深さ問題は反転した」と判定する設計だった。結果はそれぞれ−1.2803 / −2.0537 / −3.0837ptで全条件が境界を下回り、主判定は未達。固定12,000区間では深さ問題が残存した。')
p('E18a−E18d＝＋0.2700±0.3604pt。幅76と54の差を含む比較であり、パラメータ数のみの純粋な因果効果とはしない。深いモデルも同じ最大12,000回の発火予算で学ぶため、重み数増加に対して更新予算が増えていない点が解釈を制約する。')
figure(B/'figures/e18_depth.png','E18：幅固定の深さ比較と約100k対照。E18dは厳密な±1%条件からの承認済み例外。')
section('6. スケール診断と計算時間')
w=read(B/'signal/scale_warnings.csv')
table(['条件','枝/stream >0.5','割合 %','logits RMS平均','logits >10件数'],[[r['condition'],r['ratio_exceed_count']+'/'+r['ratio_total'],f"{100*float(r['ratio_exceed_fraction']):.2f}",f"{float(r['logits_rms_mean']):.4f}",r['logits_exceed_count']] for r in w],[60,120,80,139,100])
p('最終時点の3 seed×ブロックを分母とする。E18cの超過は9/96（9.375%）で、E18bの8/72より件数は1増えるが割合は下がる。E17aの11/24より件数・割合とも低い。logits RMS>10は全条件で0件。したがって深さ劣化を、この事前登録した振幅暴走指標の増加だけでは説明できない。RMS診断は因果の証明ではない。')
p('E17全9 runでは最終枝/stream比>0.5が39件、logits>10は0件。比の超過は警告として記録し、単独で失敗扱いにしない。枝/stream比は各ブロックの加算前streamと枝出力を別々に測ったRMSの比である。')
table(['条件','E17a実績×重み比（分）','実測平均（分）'],[['E18a','69.99','86.98'],['E18b','103.61','111.56'],['E18c','137.23','134.59'],['E18d','35.91','53.89']],[75,230,194])
p('予想はE17a実績36.38分から重み数（行列積MAC数）比で換算。量子化点・正規化・候補生成費用はこの換算に含まれない。最大12 workerの並行実行と監査の資源共有をruntime_workers.jsonに記録した。実測時間は公平な速度ベンチマークではない。')
section('7. E19：全量子化点のA4化')
p('d=76・8ブロック・ReLUを固定し、入力射影・枝・出力射影の全量子化点をA4にした。枝だけでなく入力と出力の量子化も変更している。')
p('test平均75.9600±1.4781%。固定参照90.637%との差は14.6770pt（保存済みE17aの丸め前平均を用いた対応差は14.6767±1.8126pt）。直列E16のA8−A4＝19.86ptより縮小したが、3.0pt以内の条件を満たさない。全seedはE16-A4の67.17%を上回るが、主判定は未達。')
figure(B/'figures/e19_a4.png','E19：残差構造と直列構造でのA4コスト。構造間で比較条件が異なることに注意。')
section('8. E20：backpropとSTEの実装・学習設定')
p('全条件d=76、8ブロック、18行列、100,016重み。E20aはFP32重み・量子化なし、E20bはFP32重み・学習時からA8＋活性化恒等STE、E20cは潜在FP32重みからW3へ量子化し、重み・活性化とも恒等STE。')
p('E20cは層ごとのalpha=mean(abs(w))、q=clip(round(w/alpha),−1,+1)、有効重みalpha*q。0.5alpha境界の丸めは実装の偶数丸めに従う。alpha=0では全ゼロ。更新は潜在FP32重みだけに蓄積し、量子化重みのin-place更新は行わない。量子化forwardと任意上流勾配の恒等backwardを別々にテストした。')
p('He初期化：隠れstd=sqrt(2/fan-in)、出力std=sqrt(1/fan-in)。同seedの3条件で潜在/FP32初期重みを一致させた。Adam lr0.001、batch128、最大100 epoch。val損失プラトーでLR半減（patience5、下限1e-5）。30 epoch以降20 epoch改善なしで終了し、epoch末val損失最小モデルを選択した。500更新ごとの追加valは診断のみでモデル選択に使わない。')
p('各epochで全18行列の勾配ノルム・出力RMSを記録。非有限値、または事前登録した振幅増大（RMS>1e4）＋ReLU枝全ゼロ＋全18行列の全epoch勾配ゼロを検知したら中断。E20cだけlr0.0003＋全体勾配ノルムclip1.0で同初期値から1回再試行する救済を事前登録した。今回は全9 runが初回成功し、救済は0回。')
table(['条件','seed','終了epoch','選択epoch','選択attempt'],[[r['condition'],r['seed'],r['epochs'],r['best_epoch'],r['selected_attempt']] for r in bp if r['condition'].startswith('E20')],[90,70,110,110,119])
section('9. E20：ギャップ分解')
effects=read(B/'aggregate/paired_effects.csv')
labels={'E18a_minus_E18d':'E18a − E18d','E17a_minus_E19a_A4_cost':'E17a − E19a（A4コスト）','E20a_minus_E20b_A8_cost':'E20a − E20b（A8コスト）','E20b_minus_E20c_W3_cost':'E20b − E20c（W3コスト）','E20c_minus_E17a_TDT_comparison':'E20c − E17a（TDTとの比較）','E20a_minus_E14_reference':'E20a − E14（参考差）'}
table(['比較','平均差 ± 標本SD (pt)'],[[labels[r['comparison']],f"{float(r['mean_pp']):+.4f} ± {float(r['sample_std_pp']):.4f}"] for r in effects],[305,194])
p('A8コスト＋0.4567、W3コスト−0.4733、TDTとの差＋5.3500を足すとE20a−E17a＝＋5.3333ptになる。W3コストの負値は今回W3条件の精度が高かったことを表し、W3化一般の優位性を意味しない。')
p('最重要のE20c−E17aには学習則だけでなく、動的alpha対固定スケール、He対三値初期化、更新予算、val最良対最終モデルという差が含まれる。同一構造でも「TDT学習則の純粋なギャップ」という因果的主張はできない。E20に精度の合否閾値は事前登録されていない。')
figure(B/'figures/e20_decomposition.png','E20：test精度と対応seed差。3 seedによる記述的比較。')
section('10. 診断定義・監査と再現性')
p('量子化相対二乗誤差はsum((x−Q(x))^2)/sum(x^2)。コサインは両ノルムが非ゼロの例で平均し、未定義例を別記。候補差|y|=|L(T+)−L(T−)|は同一128例・FP32損失の差で、S正規化前。ゼロ率はFP32で厳密な差ゼロの割合。層単独プローブは各行列64候補対・16辺、初期と最終で実施した。')
p('全区間発火率＝行列の発火区間数/12,000。選択時発火率＝発火数/その行列が選択された区間数。学習中の行列条件付き候補差は複数行列への重複分類を含み、層単独プローブや加法的寄与とは異なる。')
p('全33 runで記録と監査を保存。TDTの全区間ログ・S更新・発火・初期/最終validation・層単独プローブ、BPの勾配・選択・量子化状態を照合した。監査でtestを再評価せず、testは最終報告にのみ使用した。本PDF作成でも既存集計を使用し、新規学習・モデル選択・test再評価は実施しない。')
p('事前登録コミット：E17 1ad7b12、E18〜E20 16485f4。E18d例外承認 c9a635a。実行時ソース・データ・成果物SHA-256、CPU資源共有、全attemptを保存。PDF生成時に成果物ハッシュと33 seed行から11条件の平均・標本SDを再検証した。')
for root in [A,B]:p(str(root.relative_to(ROOT)),'small')
p('PDF添付のv5.4-evidence.zipには集計CSV、層別一覧、初期validation、設定・監査・事前登録・実行時ソースを収録する。大容量の逐区間ログ、全候補NPY、モデル、全epoch勾配の生データは上記結果ディレクトリを参照する。添付データの全ファイルにSHA-256を付ける。')
section('11. 全33 runの個別testとvalidation')
table(['条件','seed','val %','test %'],[[r['condition'],r['seed'],f"{float(r.get('val_accuracy_percent',r.get('validation_accuracy_percent'))):.2f}",f"{float(r['test_accuracy_percent']):.2f}"] for r in ap+bp],[100,70,160,169])
section('12. 学習中の候補損失差')
table(['条件','seed','mean |y|','差ゼロ率 %'],[[r['condition'],r['seed'],f"{float(r['mean_abs_y']):.7f}",f"{100*float(r['zero_difference_fraction']):.6f}"] for r in ap+bp if r.get('mean_abs_y')],[75,45,189,190])
p('値が非ゼロであることや大きいことだけでは、有効な更新方向が得られるとは判断しない。')
section('13. 更新した結論と限界')
p('残差FP32ストリームとpre-normを備えた8ブロックTDTは、事前登録した線形天井突破の条件を満たした。一方、固定学習予算では深さを16〜32ブロックへ増やすと低下し、全量子化点A4も大きなコストを残した。残差化ですべての深さ・精度問題が解消したとはいえない。')
p('同一構造のbackprop＋STEは今回安定して約96%に達し、E15型の失敗は再現しなかった。TDTとの差は今後の検討対象だが、単一要因へ帰属するにはスケール・初期化・予算・モデル選択まで整合した比較が必要である。追加条件探索は本実験には含めない。')
p('MNIST、各条件3 seed、指定予算に限られる結果であり、統計的有意差、収束保証、整数専用ハードウェア効率は主張しない。FP32ストリーム・復元・行列積を使うため、低ビット保持と全演算の整数化を区別する。')
section('付表A. E17の全ブロック・全行列診断')
md_tables(A/'LAYER_TABLES.md','E17')
section('付表B. E18・E19の全ブロック・全行列診断')
md_tables(B/'LAYER_TABLES.md','E18/E19')
section('付録への案内：v5.3原版')
p('次ページからv5.3全文を、本文・図・ページ番号を変更せず収録する。しおりから版ごとの追加部へ移動できる。旧版の埋め込み添付データも継承する。')
p('v5.3原版 SHA-256: '+sha(ORIGINAL),'small')
main=OUT/'v5.4-update.pdf'
Doc(str(main),pagesize=A4,leftMargin=48,rightMargin=48,topMargin=48,bottomMargin=45).build(story,onFirstPage=footer,onLaterPages=footer)
(OUT/'TDT-v5.4_追加本文.txt').write_text('\n\n'.join(logs))
files=[]
for root in [A,B]:
 for f in root.rglob('*'):
  if not f.is_file():continue
  rel=f.relative_to(root)
  if (len(rel.parts)==1 and f.suffix in ['.md','.json','.patch']) or rel.parts[0] in ['aggregate','sources'] or (rel.parts[0] in ['signal','activation','firing'] and 'aggregate' in f.name) or rel.as_posix()=='per_seed/results.csv':files.append(f)
files.extend([Path(__file__).resolve(),OUT/'TDT-v5.4_追加本文.txt',ROOT/'tdt_mnist/paper_v5_1/build_pdf.py'])
hashes={str(f.relative_to(ROOT)):sha(f) for f in sorted(set(files))}
(OUT/'source_hashes.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2))
archive=OUT/'v5.4-evidence.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for f in sorted(set(files)):z.write(f,str(f.relative_to(ROOT)))
 z.write(OUT/'source_hashes.json','source_hashes.json')
pdf=fitz.open(main);n=len(pdf);old=fitz.open(ORIGINAL);toc=pdf.get_toc()
pdf.insert_pdf(old);toc.append([1,'付録：TDT-v5.3原版',n+1]);toc.extend([lev+1,title,page+n] for lev,title,page in old.get_toc());pdf.set_toc(toc)
pdf.embfile_add('v5.4-evidence.zip',archive.read_bytes(),filename=archive.name)
for i in range(old.embfile_count()):
 info=old.embfile_info(i);pdf.embfile_add(f'previous-version-data-{i+1}.zip',old.embfile_get(i),filename=info.get('filename',f'previous-{i+1}.zip'))
pdf.set_metadata({'title':'TDT-v5.4 離散状態遷移学習理論 — 残差ストリーム・深さ・A4・backprop対照','author':'TDT experimental study','subject':'Audited E17-E20, 33 new runs; complete v5.3 preserved','creator':'ReportLab / PyMuPDF'})
pdf.save(DEST,garbage=4,deflate=True);pdf.close();result=fitz.open(DEST)
for i in range(len(old)):
 assert result[n+i].get_text()==old[i].get_text()
 assert result[n+i].get_pixmap(matrix=fitz.Matrix(.4,.4)).samples==old[i].get_pixmap(matrix=fitz.Matrix(.4,.4)).samples
outside=[]
for i in range(n):
 for block in result[i].get_text('dict')['blocks']:
  for line in block.get('lines',[]):
   for span in line['spans']:
    if not result[i].rect.contains(fitz.Rect(span['bbox'])):outside.append([i+1,span['text']])
assert not outside,outside
assert result.embfile_get('v5.4-evidence.zip')==archive.read_bytes()
for i in range(old.embfile_count()):assert result.embfile_get(f'previous-version-data-{i+1}.zip')==old.embfile_get(i)
for i in range(n):result[i].get_pixmap(matrix=fitz.Matrix(.6,.6)).save(OUT/f'preview-{i+1:02}.png')
validation=dict(passed=True,new_training_runs=33,aggregate_conditions=11,verified_artifact_hashes=verified,update_pages=n,preserved_v5_3_pages=len(old),total_pages=len(result),old_text_and_pixels_preserved=True,no_text_outside_page=True,all_attachments_verified=True,pdf_sha256=sha(DEST),original_sha256=sha(ORIGINAL),archive_sha256=sha(archive))
(OUT/'pdf_validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2));print(json.dumps(validation,ensure_ascii=False,indent=2))
