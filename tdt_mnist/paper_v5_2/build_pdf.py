"""Build v5.2 from audited E12 data, preserving the complete v5.1 PDF.
Run: PYTHONPATH=/tmp/tdt-pdf-reader python tdt_mnist/paper_v5_2/build_pdf.py
"""
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
DATA=ROOT/'tdt_mnist/results/depth-activation-100k-20260907'
ORIGINAL=DOC/'TDT-v5.1_離散状態遷移学習理論.pdf'
DEST=DOC/'TDT-v5.2_離散状態遷移学習理論.pdf'
def read(p):
 with p.open() as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
C=['relu-a3-threshold','identity-a32','identity-a3-threshold']
LABELS=['ReLU＋A3閾値分離','ReLUなし＋A32','ReLUなし＋A3閾値分離']
SHORT=['R＋A3','なし＋A32','なし＋A3']
per=read(DATA/'per_seed.csv');agg=read(DATA/'aggregate.csv')
rms=read(DATA/'rms_h_aggregate.csv');isolated=read(DATA/'layer_isolated_abs_y_aggregate.csv')
yagg=read(DATA/'abs_y_aggregate.csv');mixed=read(DATA/'abs_y_by_perturbed_layer_aggregate.csv')
assert json.loads((DATA/'status.json').read_text())['complete']
audit=json.loads((DATA/'verification.json').read_text());assert audit['passed'] and audit['runs']==108
for name,digest in audit['analysis_sources'].items():assert sha(DATA/'sources'/name)==digest
assert len(per)==108 and len(agg)==36
verified=0
def check_aggregate(raw,summary,keys,fields):
 global verified
 groups={}
 for r in raw:groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
 for a in summary:
  members=groups[tuple(a[k] for k in keys)];assert len(members)==3
  for field in fields:
   vals=[float(r[field]) for r in members]
   assert abs(st.mean(vals)-float(a[field+'_mean']))<1e-10
   assert abs(st.stdev(vals)-float(a[field+'_std']))<1e-10
   verified+=1
check_aggregate(per,agg,['condition','depth','threshold'],['val_accuracy','test_accuracy','val_loss','total_fires','zero_difference_fraction'])
check_aggregate(read(DATA/'rms_h_per_seed.csv'),rms,['condition','depth','threshold','step','layer','quantity'],['rms','zero_fraction'])
check_aggregate(read(DATA/'layer_isolated_abs_y.csv'),isolated,['condition','depth','threshold','stage','layer'],['mean','rms','median','p90','p99','zero_fraction'])
check_aggregate(read(DATA/'abs_y_per_seed.csv'),yagg,['condition','depth','threshold'],['mean','rms','median','p90','p99','max','zero_fraction'])
check_aggregate(read(DATA/'abs_y_by_perturbed_layer.csv'),mixed,['condition','depth','threshold','layer'],['mean','rms','median','p90','p99','zero_fraction'])
for r in per:assert int(r['train_forward_calls'])==1536000
# Use the established v5.1 typesetting helpers without running its build.
helper=(ROOT/'tdt_mnist/paper_v5_1/build_pdf.py').read_text()
helper=helper[helper.index("pdfmetrics.registerFont"):helper.index("p('TDT-v5.1\\n")]
helper=helper.replace('TDT-v5.1 | 追加実験・改訂考察','TDT-v5.2 | 活性化方式・深さ・摂動対象層別診断')
exec(helper)
def get(rows,c,d,t=8,**filters):
 return next(r for r in rows if r['condition']==c and int(r['depth'])==d and int(r['threshold'])==t and all(str(r[k])==str(v) for k,v in filters.items()))
def e(x):return f'{float(x):.5g}'
p('TDT-v5.2\n離散状態遷移学習理論','title')
p('活性化方式と深さの比較\n層別RMS・摂動対象層別の損失差')
p('実験追加版 · 2026年9月7日')
p('本版はv5.1にE12：活性化方式×深さ×カウンタ閾値の実験を追加する。ReLU＋A3閾値分離、ReLUなし＋A32、ReLUなし＋A3閾値分離を、4・8・16層、閾値1・4・8・16、3 seedで比較した。全36条件・108 runと診断を完了し、監査を通過した。')
p('追加部に方法・結果・層別診断・考察をまとめ、後半にはv5.1全70ページを原版のまま収録する。旧版の結論・実験数は各版時点の記録として読む。累計は12実験群・134条件行・400 run記録で、旧版の重複記録を含むため400個の独立標本を意味しない。')
p('主要結果：閾値8でReLUなし＋A32は16層でもtest 87.31%を維持した。ReLU＋A3は8・16層で約11%、ReLUなし＋A3も16層では13.91%となった。ReLUを除くだけでは、この条件下の深いA3の精度劣化を解消できなかった。')
table(['層数',*SHORT],[[d,*[ms(get(agg,c,d),'test_accuracy',100,2)+'%' for c in C]] for d in [4,8,16]],[40,153,153,153])
p('表：閾値8の最終test精度。3 seed平均 ± 標本標準偏差。A3はすべて閾値・復元スケール分離型。略号RはReLUあり、「なし」はReLUなし。')
section('1. 実験設計と共通条件')
table(['項目','条件'],[['比較','3方式 × 線形層数4 / 8 / 16 × カウンタ閾値1 / 4 / 8 / 16 × seed 0 / 1 / 2。'],['構造','100,000重みを厳密に固定。深さは出力層を含む。バイアス・残差接続・追加正規化なし。'],['データ','MNIST、入力9×10＝90。train 10,000 / validation 1,000 / test 10,000、data_seed=0。固定平均プーリングとv5の入力正規化を継承。'],['TDT','ブロック16、K=64候補対、12,000区間、batch 128、最大1発火/区間。C8、leak=1、区間末に証拠をリセット。'],['スケール','gain=1、初期ゼロ率1/3、層スケール=1/sqrt(fan-in × (1−1/3))。S初期0.02、EMA係数0.1、下限1e-5。'],['精度・実装','三値重みをINT8で保持。復元スケール、行列積、logits、損失はFP32。逆伝播・STE・潜在浮動小数点重みなし。'],['評価・乱数','初期と500区間ごとに固定検証集合を評価。最終時点のtestを報告。batch_seed=seed+100000、CPU・threads=1。']],[88,411])
p('同じ深さ・seedの初期重みと選択機会を対応させた。閾値間・方式間で学習forward予算は同じである。ただし方式と深さで実時間・FLOPsは異なる。各runは768,000候補対、1,536,000学習forward。全108 runで82,944,000候補対を記録した。')
p('各方式4並列から開始し、A3は実行中runの完走後に6並列へ移行した。途中の学習を打ち切らず、完了runを再利用した。実行資源の変更はruntime_workers.jsonに記録している。')
section('2. 構造とA3閾値分離の定義')
manifest=json.loads((DATA/C[0]/'manifest.json').read_text())
for d in [4,8,16]:
 dims=[90,*manifest['hidden_widths'][str(d)],10]
 assert sum(a*b for a,b in zip(dims,dims[1:]))==100000
 p(f'{d}層：'+' → '.join(map(str,dims)))
p('重み数を固定するため、深くするほど隠れ幅が狭くなる。本実験は固定パラメータ予算下の深さ・幅の比較であり、幅固定の純粋な深さ効果とは区別する。')
p('A3閾値分離：各サンプル・各線形層入力xについてτ=0.5 mean(|x|)。|x|>τならq=sign(x)、それ以外はq=0。復元スケールβは選択された|x|の平均、復元値はβqとする。選択が空ならβ=1。係数0.5は固定し、精度で探索していない。')
p('A3は入力と隠れ入力の両方を量子化する。隠れ層のReLUなしは恒等関数、出力層は全条件でlogitsである。ReLUなし＋A3でも量子化による非線形性が残る。')
p('ReLUなし＋A32は、前処理後の入力からlogitsまでが線形写像となる。深さを増やしても一般的な非線形深層表現の学習を実証したことにはならない。')
section('3. RMS(h_l)と損失差の測定')
p('h_lは第l線形層の活性化後・次層の量子化前の値と定義する。最終h_Dはlogits。RMS(h_l)=sqrt(全検証例・全特徴についてのh_lの二乗平均)。初期と500区間ごとの25時点で、全層を観測した。集計はFP64で読み取り専用、学習計算はFP32のままである。')
p('量子化・復元後の線形層入力RMSも別に保存した。CSVのlayerは0始まり、本文と図は1始まり。h_lの観測25,200件と、線形層入力の観測25,200件、計50,400件をrms_h_per_seed.csvに収録する。活性化前などを含む全信号統計は各方式のsignal_metrics.csvに保存した。')
p('|y|=|L(T+)−L(T−)|。Lは同じ128例ミニバッチの平均交差エントロピーで、FP32で差を取り、絶対値を記録する。TDTのSで正規化する前の量である。各runのabs_y.npyはfloat32[12000,64]で、全候補対を保存する。')
p('損失差ゼロ率はFP32差が厳密に0の割合で、表示丸めやゼロ投票率とは異なる。|y|の平均・RMS・中央値・p90・p99・最大値・ゼロ率を集計し、500区間窓の時間推移も保存した。')
section('4. 摂動対象層別と層単独診断の区別')
table(['診断','測定と解釈'],[['学習中：摂動対象層別','各区間で選んだ16座標は複数層にまたがり得る。同一区間の64候補対は同じ座標でT+とT−が異なる。層別の選択辺数と|y|を結合し、「その層を含む摂動」の分布として集計する。'],['独立診断：層単独','初期・最終モデルの各層について、その層だけから16座標を選び64候補対を評価する。専用乱数と固定train集合からの128例ミニバッチを用い、保存重み・学習票・採否には反映しない。']],[100,399])
p('学習中の同じ候補対は複数層に重複して分類される。したがって摂動対象層別の|y|は、各層単独の因果寄与や加法分解ではない。この曖昧さを減らすため、層単独摂動を独立診断として併用した。')
p('層単独診断では初期・最終・方式間でミニバッチ、座標、乱数を対応させる。状態依存の辺選択は異なり得る。摂動辺数は16本で一定だが、層の総重み数が違うため摂動率は一定ではない。重み数と摂動率もCSVに保存した。')
p('独立診断は1,008層×runについて初期・最終を測り、計129,024候補対（258,048 forward）。学習予算とは別枠である。各層・各時点64候補対の記述統計であり、精密な裾確率推定や勾配の推定値とは扱わない。')
for c,label in zip(C,LABELS):
 section('5. E12の全条件：'+label)
 table(['層','閾値','val精度 %','test精度 %','val損失','発火回数'],[[r['depth'],r['threshold'],ms(r,'val_accuracy',100),ms(r,'test_accuracy',100),ms(r,'val_loss'),ms(r,'total_fires',1,1)] for r in agg if r['condition']==c],[30,35,112,112,106,104])
 p('平均 ± 標本標準偏差（3 seed）。評価は最終12,000区間時点。深さ・幅・初期化・学習予算を固定した範囲の測定であり、方式一般の限界を示すものではない。')
section('6. 精度の比較と深さ依存')
fig(DATA/'accuracy_comparison.png','図E12-1　全閾値の最終test精度。エラーバーは3 seedの標本標準偏差。')
p('ReLUなし＋A32は閾値1・4・8で深さ4〜16を通して高い精度を維持した。閾値8では4層88.90%、8層88.74%、16層87.31%。他方、ReLU＋A3は4層46.07%から8層10.66%へ低下し、ReLUなし＋A3も4層64.41%、8層31.40%、16層13.91%となった。')
p('ReLUを除く効果はA3でも認められるが、深さ方向の精度低下は残る。閾値16はA32および4・8層A3で精度低下が大きい。3 seedの小差から閾値の最適性を断定しない。')
section('7. 閾値8：信号強度と候補差')
rows=[]
for c,label in zip(C,SHORT):
 for d in [4,8,16]:
  ini=get(rms,c,d,step=0,layer=d-1,quantity='h_l');fin=get(rms,c,d,step=12000,layer=d-1,quantity='h_l');y=get(yagg,c,d)
  rows.append([label,d,e(ini['rms_mean']),e(fin['rms_mean']),e(y['mean_mean']),f"{100*float(y['zero_fraction_mean']):.6f}"])
table(['方式','層','初期RMS(h_D)','最終RMS(h_D)','学習mean |y|','損失差0 %'],rows,[87,30,98,98,98,88])
p('3 seed平均。RMSは検証集合、|y|は学習中の全候補対。16層でReLU＋A3は初期から強く減衰し、最終出力RMSも約0.00350にとどまる。ReLUなし＋A3は約0.1268、A32は約4.8106となった。')
p('ReLUなし＋A3の学習mean |y|は16層で約0.00719で、A32の約0.00418より大きい。それでも精度は大幅に低い。大きな候補差が、識別に有用な方向情報や効果的な学習を意味するわけではない。')
p('ReLU＋A3の16層では損失差ゼロ率が約0.498%となるが、大多数の候補対は非ゼロである。損失差が非ゼロであることだけから、高精度な学習が可能とは判断できない。')
for d in [4,8,16]:
 section(f'8. 閾値8：{d}層の層別測定値')
 p('各セルは3 seed平均。RMSは最終h_l、|y|は最終モデルの層単独摂動の平均。各層の標準偏差・初期値・他閾値は付属CSVに収録。')
 table(['層',*[s+'\nRMS' for s in SHORT],*[s+'\n|y|' for s in SHORT]],[[l+1,*[e(get(rms,c,d,step=12000,layer=l,quantity='h_l')['rms_mean']) for c in C],*[e(get(isolated,c,d,stage='final',layer=l)['mean_mean']) for c in C]] for l in range(d)],[31,78,78,78,78,78,78])
 p('RはReLUあり。「なし」はReLUなし。最終層にはReLUを置かない。層単独診断は各層16辺を摂動するため、層ごとの摂動率の違いを含む。')
for t in [1,4,8,16]:
 section(f'9. 閾値{t}：全層RMSと層単独損失差')
 fig(DATA/f'layer_rms_and_abs_y_threshold{t}.png',f'図E12-{t}a　上：RMS(h_l)、破線は初期、実線は最終。下：最終モデルの層単独mean |y|。3 seed平均、縦軸は対数。')
 p('出力側の|y|が増加する場合でも、層の重み数・相対摂動率・出力への位置が異なるため、層の重要度を直接順位付けする指標とはしない。')
 section(f'10. 閾値{t}：学習中の損失差の時間推移')
 fig(DATA/f'abs_y_training_threshold{t}.png','学習中の500区間窓mean |y|を3 seedで平均。層単独診断とは異なり、実際の複数層ブロックの候補差である。')
 p('候補差は、その時点の重み・活性分布・ミニバッチに依存する。振幅の変化を損失・精度・層別発火率と併せて解釈する。ゼロ率、中央値、p90等の時間窓統計もCSVに保存した。')
section('11. v5.1のReLU＋A32との関係')
old=read(ROOT/'tdt_mnist/results/depth-100k-20260907/aggregate.csv')
table(['層','v5.1：ReLU＋A32','E12：ReLUなし＋A32'],[[d,ms(next(r for r in old if int(r['depth'])==d and r['threshold']=='8'),'test_accuracy',100,2)+'%',ms(get(agg,C[1],d),'test_accuracy',100,2)+'%'] for d in [4,8,16]],[40,230,229])
p('表は閾値8のtest精度。左列はv5.1で報告した保存済みの測定で、今回の追加108 runには含めない。固定100k・同じ幅・予算で、ReLUなしA32は深さを増しても精度を維持した。')
p('この比較は、v5のgain=1を維持した構成でReLUの有無が深さ依存に強く関わることを示す。一方、非線形表現の獲得、最適な初期化、他データセットへの一般化を実証するものではない。')
section('12. 結論と残る問題')
p('E12では、A32でReLUを除くと16層でも高精度を維持した一方、A3閾値分離はReLUを除いても深層で大きな精度低下が残った。浅いA3で得られた閾値分離の有効性を、そのまま深層へ一般化できないことが明らかになった。')
p('RMSの減衰はReLU＋A3で特に強い。ReLUなし＋A3は振幅と候補差が大きくなるが、A32より精度が低い。RMS・|y|・発火の存在と、識別に役立つ学習の成立は区別して評価する必要がある。')
p('本実験では、深さと幅が同時に変化し、A3は入力と隠れ入力の両方を変える。各条件3 seed、固定学習予算である。層ごとのスケール補正、残差接続、学習区間延長、別の量子化設計で改善できるかは未検証である。')
p('層単独の候補差は有限摂動の感度診断であり、勾配、因果寄与の加法分解、方向情報の品質そのものではない。TDTの計算量優位性や深層収束保証を主張する結果とはしない。')
for c,label in zip(C,LABELS):
 for d in [4,8,16]:
  section(f'付録A. 個別run：{label}・{d}層')
  rs=[r for r in per if r['condition']==c and int(r['depth'])==d]
  table(['閾値','seed','val %','test %','val損失','発火回数','損失差0 %'],[[r['threshold'],r['seed'],f"{100*float(r['val_accuracy']):.2f}",f"{100*float(r['test_accuracy']):.2f}",f"{float(r['val_loss']):.6f}",r['total_fires'],f"{100*float(r['zero_difference_fraction']):.6f}"] for r in rs])
section('付録B. データ・監査・原版の収録')
p('一次結果：tdt_mnist/results/depth-activation-100k-20260907/。全108 runの保存モデル・設定・データハッシュ・ソース・学習予算・層別発火を監査済み。候補差のNPYと各区間統計の一致、ゼロ件数、有限性、摂動対象層との対応、層単独診断のFP32差も検証した。')
p('PDF生成時にはper_seed.csvから精度等の平均・標本標準偏差を再計算した。RMS、学習候補差、摂動対象層別候補差、層単独診断の集計CSVも、seed別の値との一致を確認した。')
table(['ファイル','内容'],[['rms_h_per_seed.csv / aggregate','全層・全25時点のh_lと量子化復元後入力のRMS。'],['abs_y_per_seed.csv / aggregate','各runの全学習候補差の記述統計。'],['abs_y_by_perturbed_layer*.csv','当該層を含む学習摂動に条件付けた候補差。'],['layer_isolated_abs_y*.csv','初期・最終の層単独診断。'],['layer_isolated_probes/*.csv','独立診断129,024候補対のL(T+)、L(T−)、|y|。'],['各方式のCSV・manifest・sources','信号全統計、層別発火、保存時ソース、条件とハッシュ。']],[195,304])
p('添付の追加実験データZIPにはCSV・JSON・実験時ソース・図を収録する。学習中の全82,944,000候補差のNPY、全逐区間の層ログ、モデルチェックポイントはrunsディレクトリに保存済みで、サイズの大きいこれらのrawファイルはZIPに含めない。')
p('次ページからv5.1全70ページを変更せず収録する。しおりから新旧各章に移動できる。旧版の添付実験データも継承する。\nv5.1原版 SHA-256：'+sha(ORIGINAL),'small')
main=OUT/'v5.2-update.pdf'
Doc(str(main),pagesize=A4,leftMargin=48,rightMargin=48,topMargin=48,bottomMargin=45).build(story,onFirstPage=footer,onLaterPages=footer)
(OUT/'TDT-v5.2_追加本文.txt').write_text('\n\n'.join(logs))
files=sorted(f for f in DATA.rglob('*') if f.is_file() and f.suffix in ['.csv','.json','.md','.py','.png','.svg'])
files += [Path(__file__).resolve(),OUT/'TDT-v5.2_追加本文.txt']
hashes={str(f.relative_to(ROOT)):sha(f) for f in files};(OUT/'source_hashes.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2))
archive=DOC/'TDT-v5.2_追加実験データ.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for f in files:z.write(f,str(f.relative_to(ROOT)))
 z.write(OUT/'source_hashes.json','source_hashes.json')
pdf=fitz.open(main);n=len(pdf);toc=pdf.get_toc();old=fitz.open(ORIGINAL);assert len(old)==70
pdf.insert_pdf(old);toc.append([1,'付録C：TDT-v5.1原版（全70ページ）',n+1]);toc.extend([level+1,title,page+n] for level,title,page in old.get_toc());pdf.set_toc(toc)
pdf.embfile_add('TDT-v5.2-additional-data.zip',archive.read_bytes(),filename=archive.name,desc='Audited E12 tables, layer probes, source snapshots and figures')
for i in range(old.embfile_count()):
 info=old.embfile_info(i);pdf.embfile_add(f'previous-version-data-{i+1}.zip',old.embfile_get(i),filename=info.get('filename',f'previous-{i+1}.zip'))
pdf.set_metadata({'title':'TDT-v5.2 離散状態遷移学習理論 — 活性化方式・深さ・層別RMSと損失差','author':'TDT experimental study','subject':'E12: 108 runs, 82,944,000 training candidate pairs, layer RMS and isolated-layer probes','keywords':'TDT,MNIST,A3,A32,ReLU,RMS,depth,perturbation','creator':'ReportLab / PyMuPDF'})
pdf.save(DEST,garbage=4,deflate=True);pdf.close();check=fitz.open(DEST)
for i in range(len(old)):
 assert check[n+i].get_text()==old[i].get_text()
 assert check[n+i].get_pixmap(matrix=fitz.Matrix(.5,.5)).samples==old[i].get_pixmap(matrix=fitz.Matrix(.5,.5)).samples
outside=[]
for i in range(n):
 for b in check[i].get_text('dict')['blocks']:
  for line in b.get('lines',[]):
   for s in line['spans']:
    if not check[i].rect.contains(fitz.Rect(s['bbox'])):outside.append((i+1,s['text']))
assert not outside,outside
assert check.embfile_get('TDT-v5.2-additional-data.zip')==archive.read_bytes()
for i in range(old.embfile_count()):assert check.embfile_get(f'previous-version-data-{i+1}.zip')==old.embfile_get(i)
assert all(sha(ROOT/k)==v for k,v in hashes.items())
validation={'passed':True,'new_conditions':36,'new_runs':108,'training_candidate_pairs':82944000,'independent_probe_pairs':129024,'verified_aggregate_fields':verified,'update_pages':n,'preserved_v5_1_pages':70,'total_pages':len(check),'old_text_and_pixels_preserved':True,'no_text_outside_page':True,'all_attachments_verified':True,'original_sha256':sha(ORIGINAL),'pdf_sha256':sha(DEST),'archive_sha256':sha(archive)}
(OUT/'pdf_validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2))
for page in [0,5,8,9,12,16,24,n-1]:
 if page<n:check[page].get_pixmap(matrix=fitz.Matrix(1,1)).save(OUT/f'preview-{page+1:02d}.png')
print(json.dumps(validation,ensure_ascii=False,indent=2));print(DEST)
