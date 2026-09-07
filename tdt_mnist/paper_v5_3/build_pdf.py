"""Audited E13-E16 supplement plus all 103 unchanged pages of TDT-v5.2."""
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
ORIGINAL=DOC/'TDT-v5.2_離散状態遷移学習理論.pdf';DEST=DOC/'TDT-v5.3_離散状態遷移学習理論.pdf'
R=ROOT/'tdt_mnist/results'
A=R/'a3-improvements-16layer-20260907';B=R/'backprop-a3-inference-16x79-20260907';Q=R/'qat-a3-16x79-20260907';D=R/'depth-precision-16layer-100k-20260907'
def read(p):
 with p.open() as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def get(rows,**kw):return next(r for r in rows if all(str(r[k])==str(v) for k,v in kw.items()))
def num(x):return f'{float(x):.5g}'
verified=0;hash_checks=0
for data in [A,B,Q,D]:
 assert json.loads((data/'status.json').read_text())['complete']
 audit=json.loads((data/'verification.json').read_text());assert audit['passed']
 manifest=json.loads((data/'manifest.json').read_text())
 for name,h in manifest['sources'].items():assert sha(data/'sources'/name)==h;hash_checks+=1
 for summary in audit.get('summaries',[]):
  path=data/f"seed{summary['seed']}"
  for name,h in summary['output_sha256'].items():assert sha(path/name)==h;hash_checks+=1
 for name,h in audit.get('analysis_sources',{}).items():assert sha(data/'sources'/name)==h;hash_checks+=1
 for diag in audit.get('diagnostics',[]):
  run=Path(diag['run']);assert sha(run/'model.pt')==diag['checkpoint_sha256'];hash_checks+=1
for correction in json.loads((Q/'metadata_corrections.json').read_text()):
 assert sha(Q/f"seed{correction['seed']}"/'effective_config.json')==correction['effective_config_sha256'];hash_checks+=1

def check(raw,agg,keys,fields):
 global verified
 for a in agg:
  rows=[r for r in raw if all(str(r[k])==str(a[k]) for k in keys)];assert len(rows)==3,(keys,a)
  for field in fields:
   vals=[float(r[field]) for r in rows if r[field] not in ['',None]]
   if not vals:continue
   assert abs(st.mean(vals)-float(a[field+'_mean']))<1e-9
   if len(vals)>1:assert abs(st.stdev(vals)-float(a[field+'_std']))<1e-9
   verified+=1
ap=read(A/'per_seed.csv');aa=read(A/'aggregate.csv');dp=read(D/'per_seed.csv');da=read(D/'aggregate.csv')
bp=read(B/'per_seed.csv');ba=read(B/'aggregate.csv');qp=read(Q/'comparison_per_seed.csv');qa=read(Q/'aggregate.csv')
for raw,agg in [(ap,aa),(dp,da)]:check(raw,agg,['condition'],['val_accuracy','test_accuracy','val_loss','total_fires','zero_difference_fraction','abs_y_mean'])
check(bp,ba,['split','mode'],['accuracy','loss','accuracy_delta_pp','prediction_disagreement_fraction'])
check(qp,qa,['split','method'],['accuracy','loss'])
for data in [A,D]:
 check(read(data/'signal_metrics.csv'),read(data/'signal_aggregate.csv'),['condition','step','layer','stage'],['rms','zero_fraction'])
 check(read(data/'layer_isolated_abs_y.csv'),read(data/'layer_isolated_abs_y_aggregate.csv'),['condition','stage','layer'],['mean','rms','zero_fraction'])
 check(read(data/'layer_firing.csv'),read(data/'layer_firing_aggregate.csv'),['condition','layer'],['fires','all_interval_rate','selected_interval_rate'])
 check(read(data/'activation_diagnostics.csv'),read(data/'activation_aggregate.csv'),['condition','stage','layer'],['mse','relative_squared_error','cosine_mean_valid'])
check(read(B/'signal_metrics.csv'),read(B/'signal_aggregate.csv'),['split','mode','layer','stage'],['rms','zero_fraction'])
check(read(B/'activation_metrics.csv'),read(B/'activation_aggregate.csv'),['split','mode','layer'],['mse','relative_squared_error','cosine_mean_valid'])
assert len(ap)==12 and len(dp)==12 and len(bp)==198 and len(qp)==18
for row in ap+dp:assert int(row['train_forward_calls'])==1536000
helper=(ROOT/'tdt_mnist/paper_v5_1/build_pdf.py').read_text()
helper=helper[helper.index('pdfmetrics.registerFont'):helper.index("p('TDT-v5.1\\n")]
helper=helper.replace('TDT-v5.1 | 追加実験・改訂考察','TDT-v5.3 | A3改善・FP32/PTQ・QAT・活性化精度')
helper=helper.replace('±θτβΔΣ√−→×','±θτβσΔΣ√−→×')
exec(helper)
AC=['none','rmsnorm','residual','lloyd'];AL=['A3対照','RMS正規化','残差接続','Lloyd反復']
DC=['a32','a16','a8','a4']
asig=read(A/'signal_aggregate.csv');aiso=read(A/'layer_isolated_abs_y_aggregate.csv');afire=read(A/'layer_firing_aggregate.csv')
dsig=read(D/'signal_aggregate.csv');diso=read(D/'layer_isolated_abs_y_aggregate.csv');dfire=read(D/'layer_firing_aggregate.csv')
bsig=read(B/'signal_aggregate.csv')

def tdt_results(agg,conditions,labels):
 table(['条件','val精度 %','test精度 %','val損失','発火数'],[[label,ms(get(agg,condition=c),'val_accuracy',100),ms(get(agg,condition=c),'test_accuracy',100),ms(get(agg,condition=c),'val_loss'),ms(get(agg,condition=c),'total_fires',1,1)] for c,label in zip(conditions,labels)],[91,110,110,91,97])
def tdt_layers(data,conditions,labels,kind):
 rows=[]
 for layer in range(16):
  if kind=='rms':values=[num(get(data,condition=c,step=12000,layer=layer,stage='output')['rms_mean']) for c in conditions]
  else:values=[num(get(data,condition=c,stage='final',layer=layer)['mean_mean']) for c in conditions]
  rows.append([layer+1,*values])
 table(['層',*labels],rows,[35,116,116,116,116])

p('TDT-v5.3\n離散状態遷移学習理論','title')
p('16層におけるA3改善策・FP32推論対照・STE付きQAT\nおよびA16／A8／A4のTDT学習')
p('実験追加版 · 2026年9月7日')
p('v5.2のE12以降に完了した4実験群（E13〜E16）を追加する。新規学習は8条件×3 seed＝24 run。TDTの保存済み対照6 runと、同じFP32学習済み重みによる複数の推論評価は、新規学習runに重複計上しない。')
table(['追加群','新規学習','主要結果'],[['E13：A3改善策','9 run','残差接続でtest 18.20%。対照13.91%から改善。RMS正規化・Lloyd単独は改善基準を満たさず。'],['E14：FP32→推論時A3','3 run','FP32で93.89%。入力＋全隠れ層の推論時A3化で10.29%。'],['E15：恒等STE付きQAT','3 run','同じ乱数初期値からのQATは10.59%。振幅増大とReLU全ゼロ化を観測。'],['E16：A16／A8／A4','9 run','A16 87.49%、A8 87.03%、A4 67.17%。A32対照87.31%。']],[130,55,314])
p('本追加部の値は原則3 seed平均。平均±標本標準偏差を併記する。後半にはv5.2全103ページを本文・図・ページ番号を変更せず収録し、旧版の埋め込みデータも継承する。')
section('1. 収録範囲・新規学習と再利用の区別')
table(['群','今回の学習','再利用／追加推論'],[['E13','RMS正規化・残差・Lloyd各3 seed。ReLUなし、16層、100k、A3。','v5.2のReLUなしA3・16層・閾値8の3 runを対照として再利用。'],['E14','ReLUありFP32、幅79固定・16層の3 seed。','各保存モデルで33推論条件×validation/test。計198評価行。seed 0実施後、seed 1・2を追加し、seed 0は再学習しない。'],['E15','E14と同じ乱数初期重みからA3 QATを3 seed。','E14のFP32とPTQを対照として再利用。QAT重みの量子化を外す評価も診断として保存。'],['E16','ReLUなしA16/A8/A4各3 seed。16層、100k。','v5.2のReLUなしA32・16層・閾値8の3 runを再利用。']],[40,225,234])
p('短い動作試験は学習runの集計に含めない。新規TDT学習18 runで13,824,000候補対（27,648,000学習forward）。層単独診断は既存対照を含む24モデルの初期・最終・全16層で49,152候補対。')
p('旧版にある「未検証」という記述はその版の時点の記録である。v5.3ではRMS正規化、残差、Lloyd、FP32推論対照、恒等STE、A16/A8/A4について、ここに記した限定条件の検証を追加した。')
section('2. モデル構造と学習設定')
p('TDT（E13・E16）：90 → 79 → 82 → 80 → 82 → 80 → 82 → 81 → 81 → 81 → 81 → 81 → 82 → 81 → 82 → 80 → 10。出力を含む16線形層、バイアスなし、三値重み100,000個。')
p('FP32/PTQ/QAT（E14・E15）：90 → [79を15層] → 10。出力を含む16線形層、バイアスなし、FP32重み95,274個。ユーザー指定の幅79固定であり、TDTの厳密100k構造とは層幅・重み数が異なる。')
table(['項目','TDT E13・E16','backprop E14・E15'],[['データ','MNIST train10,000 / val1,000 / test10,000、data_seed0。9×10平均プーリング、(画像/255−0.1307)/0.3081。','左と共通。'],['活性化','ReLUなし（恒等）。量子化は線形層入力。','隠れ層ReLUあり、logitsは線形。'],['更新','閾値8、block16、K64、12,000区間、batch128、最大1発火。C8、leak1、区間末証拠リセット。','Adam lr0.001、batch128、最大100 epoch。weight decay0。'],['初期化','gain1、初期ゼロ率1/3。固定層スケール1/sqrt(fan-in×2/3)。','隠れ重み正規分布std=sqrt(2/fan-in)、出力std=sqrt(1/fan-in)。'],['評価','初期と500区間ごとのval、最終12,000区間のtest。S初期0.02、EMA0.1、下限1e-5。','val損失でLRを0.5倍（patience5、下限1e-5）。30 epoch以降20 epoch改善なしで終了。val損失最小モデルを選ぶ。']],[55,222,222])
p('全条件seed0/1/2、CPU・threads1。各方式の実時間には並行実行時の資源共有が含まれ、公平な速度ベンチマークではない。testはモデル選択・条件探索に用いない。')
section('3. 共通の診断定義')
p('RMS(h_l)は各層出力の二乗平均平方根。h_lは次層の量子化前、隠れ層では指定活性化後、最終層はlogits。inputは量子化・復元後に線形層が実際に受ける値。量子化前、RMS正規化前、補正倍率などは別のstageとして保存する。')
p('量子化MSE=mean((x−復元値)^2)、相対二乗誤差=Σ(x−復元値)^2/Σx^2。コサインは両ノルムが非ゼロの例のみで平均し、未定義例数を別記する。ReLU後のA3コードは0,+1のみ。')
p('|y|=|L(T+)−L(T−)|。Lは同一128例ミニバッチの平均交差エントロピー。FP32で差・絶対値を計算し、全候補対を保存。Sによる正規化前の値である。損失差ゼロ率はFP32差が厳密に0の割合で、丸め表示やゼロ投票率と区別する。')
p('学習中の「摂動対象層別」はその層を含む16座標ブロックに条件付けた値で、複数層への重複分類を含む。層単独の寄与の加法分解ではない。別途、その層だけの16辺を摂動した64候補対を初期・最終で測定する。診断は学習へ反映しない。')
p('TDTの発火率は重み更新の頻度であり、活性の非ゼロ率ではない。「全区間発火率」は発火区間数/12,000、「選択時発火率」は発火数/当該層が選ばれた区間数。本文の層は1始まり、元CSVのlayerは原則0始まり。')
section('4. E13：A3改善策の定義')
table(['方式','操作'],[['対照','τ=0.5 mean(abs(x))、q=sign(x) if abs(x)>τ else 0。復元β=選択されたabs(x)の平均。入力・隠れ入力ともA3。'],['RMS正規化','隠れ活性を量子化・復元した後、x/max(RMS(x),1e-8)とする。目標RMS1、学習可能ゲインなし。最初の画像入力とlogits自体は正規化しない。全ゼロはゼロのまま。'],['残差接続','内部隠れ線形層2〜15で、量子化前のFP32入力をショートカットとして枝出力に加算。係数1。幅増加は右側ゼロ埋め、減少は末尾切り詰め。追加学習重みなし。'],['Lloyd反復','各例・各層の初期τ=0.6×母標準偏差。選択集合の重心β=mean(abs(x)|selected)、中点τ=β/2と再割当を最大5回反復。全線形層入力に適用。平均を引く操作はしない。']],[90,409])
p('Lloydの0.6σは初期値であり、固定の最適閾値と主張しない。有限反復・対称3値の局所的設計で、大域最適性や分類精度の最適性は保証しない。未収束率と反復回数も保存した。')
p('残差方式はFP32バイパスを持つため、全経路が三値とはいえない。RMS正規化もFP32補正である。各方式の効果はこの実装定義に対するもので、完全離散実装や同じ演算コストを実証した結果ではない。')
section('5. E13：最終精度と対応するseed差')
tdt_results(aa,AC,AL)
pa=read(A/'paired_effects.csv')
table(['介入','val差 pp：seed0/1/2','平均差 pp','改善目安'],[[label,' / '.join(f"{100*float(get(pa,condition=c,seed=s)['val_accuracy_difference']):+.2f}" for s in range(3)),f"{100*st.mean(float(r['val_accuracy_difference']) for r in pa if r['condition']==c):+.2f}",'達成' if c=='residual' else '未達'] for c,label in zip(AC[1:],AL[1:])],[95,200,94,110])
p('事前の改善目安は最終validation平均で対照比+3ポイント以上、かつ全3 seedで改善方向が一致すること。残差接続のみが満たした。testは対照13.913%に対して18.203%（+4.290ポイント）。')
p('RMS正規化はval11.0%、test12.137%で悪化。Lloydはtest14.860%だがval13.067%で対照13.8%を下回り、seed間変動も大きい。testの平均上昇だけで改善としない。')
section('6. E13：精度曲線・信号と候補差')
fig(A/'accuracy_comparison.png','E13-1　validation学習曲線と最終test精度。帯・エラーバーは3 seed標準偏差。')
table(['方式','学習mean |y|','損失差ゼロ率 %','最終RMS(h_16)'],[[label,num(get(aa,condition=c)['abs_y_mean_mean']),f"{100*float(get(aa,condition=c)['zero_difference_fraction_mean']):.6f}",num(get(asig,condition=c,step=12000,layer=15,stage='output')['rms_mean'])] for c,label in zip(AC,AL)],[115,128,128,128])
p('振幅・候補差を大きくするだけでは高精度にならない。RMS正規化は信号振幅を保つが精度改善に至らず、残差方式は精度改善と同時に大きなlogits・損失を持つ。候補差の大きさを有効な方向情報の品質と同一視しない。')
section('7. E13：層別診断の全体像')
fig(A/'layer_diagnostics.png','E13-2　初期/最終RMS、最終モデルの層単独mean |y|、量子化相対二乗誤差、学習中mean |y|。')
p('RMS正規化の量子化誤差は正規化前の復元値に対して測る。正規化後RMSが1になることは構成上の結果なので、補正前RMSと補正倍率を併せて読む。')
section('8. E13：16層の最終RMS')
tdt_layers(asig,AC,AL,'rms')
p('各層出力RMSの3 seed平均。最終層はlogits。標準偏差・初期値・全25時点・実際の線形層入力RMSは付属CSVに収録。')
section('9. E13：層単独候補差と発火率')
tdt_layers(aiso,AC,AL,'y')
p('上表は最終モデルの層単独mean |y|。各層64候補対、16辺摂動、3 seed平均。')
section('10. E13：層別発火と補正内部の診断')
table(['層',*AL],[[i+1,*[f"{100*float(get(afire,condition=c,layer=i)['all_interval_rate_mean']):.3f}" for c in AC]] for i in range(16)],[35,116,116,116,116])
p('全区間発火率（%）。選択時発火率、選択辺数、各seedの発火回数はCSVに保存。')
fig(A/'intervention_diagnostics.png','E13-3　RMS補正倍率、残差枝/バイパスRMS比、Lloyd未収束率。これらは活性自体ではなく内部診断値。')
section('11. E13：個別runの結果')
table(['方式','seed','val %','test %','val損失','発火数'],[[AL[AC.index(c)],s,f"{100*float(get(ap,condition=c,seed=s)['val_accuracy']):.2f}",f"{100*float(get(ap,condition=c,seed=s)['test_accuracy']):.2f}",num(get(ap,condition=c,seed=s)['val_loss']),get(ap,condition=c,seed=s)['total_fires']] for c in AC for s in range(3)],[100,35,80,80,102,102])
p('A3対照3 runはv5.2から再利用。他の9 runのみ今回の新規学習である。')
section('12. E14：FP32学習後、推論だけA3化')
p('目的はFP32で学習できた表現が推論時A3化にどれだけ敏感かを測ること。ReLUあり、16層・隠れ幅79固定のFP32モデルを通常のbackpropで学習し、同一重みを固定して推論だけ量子化する。量子化後の再学習・スケール校正・testによる選択は行わない。')
p('A3はE13対照と同じ閾値分離型。画像入力を除く隠れ層だけの量子化と、画像入力も含めた量子化を区別する。各例・各層でτとβを動的に計算し、logitsはFP32のまま。')
table(['推論群','条件数/モデル','説明'],[['FP32','1','量子化なし。'],['隠れ層単独','15','第i隠れ層出力だけを量子化。'],['浅い層から累積','15','第1〜i隠れ層を量子化。i=15が全隠れ層。'],['画像入力だけ','1','正規化済み90次元入力のみA3。'],['画像入力＋全隠れ層','1','全16線形層入力をA3。']],[130,85,284])
p('計33設定。単独第1層と累積第1層は同じ操作であり、予測と損失の完全一致を検証した。各seedの初期重みとミニバッチ順は乱数に依存する。3 seedはこの変動を見るための反復で、backpropに複数seedが必須という意味ではない。')
section('13. E14：FP32／推論時A3の結果')
BM=['fp32','input_only','prefix_hidden_15','all_linear_inputs'];BL=['FP32','入力のみA3','全隠れ層A3','入力＋全隠れ層A3']
table(['推論条件','val精度 %','test精度 %','test差 pp'],[[l,ms(get(ba,split='validation',mode=m),'accuracy',100),ms(get(ba,split='test',mode=m),'accuracy',100),ms(get(ba,split='test',mode=m),'accuracy_delta_pp')] for m,l in zip(BM,BL)],[139,120,120,120])
fig(B/'comparison.png','E14-1　左：隠れ層単独のA3化。中：浅い層からの累積A3化。右：線形層の実入力RMS。')
p('FP32推論93.890%から全隠れ層A3で10.433%、入力もA3で10.287%へ低下。FP32で学習できることは、そのまま深いA3推論が成立することを意味しない。')
section('14. E14：全15隠れ層の量子化感度')
table(['隠れ層i','iのみA3：test %','1〜iをA3：test %','累積のFP32差 pp'],[[i,ms(get(ba,split='test',mode=f'single_hidden_{i:02}'),'accuracy',100),ms(get(ba,split='test',mode=f'prefix_hidden_{i:02}'),'accuracy',100),ms(get(ba,split='test',mode=f'prefix_hidden_{i:02}'),'accuracy_delta_pp')] for i in range(1,16)],[49,150,150,150])
p('単独量子化では概して浅い層で影響が大きいが、単調な法則ではない。第15隠れ層の単独量子化もseed間変動が大きい。1 seedだけの傾向を一般化せず、各seedの予測変更率・正解から誤りになった件数も確認する。')
section('15. E14：全層RMSとモデル選択')
table(['層','FP32','全隠れ層A3','入力＋全隠れ層A3'],[[i+1,*[num(get(bsig,split='test',mode=m,layer=i,stage='output')['rms_mean']) for m in ['fp32','prefix_hidden_15','all_linear_inputs']]] for i in range(16)],[35,154,155,155])
p('test集合の層出力RMS、3 seed平均。全33設定の線形層実入力RMS・量子化誤差・コード分布はデータZIPに収録。')
bs=[json.loads((B/f'seed{s}/summary.json').read_text()) for s in range(3)]
table(['seed','終了epoch','選択epoch','選択val精度 %'],[[r['seed'],r['epochs'],r['best_epoch'],f"{100*r['selected_validation']['accuracy']:.2f}"] for r in bs],[50,100,100,249])
section('16. E14：個別seedの主要推論結果')
table(['推論条件','seed','val精度 %','test精度 %'],[[l,s,f"{100*float(get(bp,seed=s,split='validation',mode=m)['accuracy']):.2f}",f"{100*float(get(bp,seed=s,split='test',mode=m)['accuracy']):.2f}"] for m,l in zip(BM,BL) for s in range(3)],[189,50,130,130])
p('全33設定の各seed・全予測はper_seed.csvおよびpredictions.npzに保存。全推論で重みハッシュ不変を確認した。')
section('17. E15：A3と恒等STEによるQAT')
p('E14と同じseedの乱数初期重みから開始した。初期重みハッシュの完全一致を確認しており、保存済みFP32モデルからの微調整ではない。重みはFP32のまま、画像入力・全隠れ層を学習時からA3量子化・復元する。')
p('カスタムautograd関数の順伝播は既存A3量子化器の復元値をそのまま返す。逆伝播ではdQ/dx=1とする恒等STE。閾値・復元スケール経由の微分は行わない。線形層とReLUの逆伝播は通常どおり。クリップ付きSTEや学習可能な量子化スケールは導入していない。')
p('forwardをx+(Q(x)−x).detach()の差分算術で近似せず、量子化器との数値的一致を直接検証した。単体テストで任意の上流勾配が恒等に戻ること、16層全重みに勾配が届くことを確認した。')
p('Adam、初期学習率、データ、スケジュール、早期終了規則はE14と共通。ただしQATのvalidation選択時もA3を使う。QAT重みの量子化を外したFP32推論は補助診断で、通常FP32学習対照ではない。')
table(['方式','val精度 %','test精度 %'],[[m,ms(get(qa,split='validation',method=m),'accuracy',100),ms(get(qa,split='test',method=m),'accuracy',100)] for m in ['FP32','PTQ_A3','QAT_A3']],[130,184,185])
section('18. E15：精度と層別信号')
fig(Q/'comparison.png','E15-1　同じ量子化位置によるFP32・PTQ・QAT比較。右は3 seed平均の線形層実入力RMS。')
p('QATのtest10.587%はPTQ10.287%に近く、今回の設定では高精度学習に至らなかった。平均だけでは失敗形態を隠すため、各seedの選択モデルと最終学習epochの勾配を分けて診断した。')
qs=[json.loads((Q/f'seed{s}/summary.json').read_text()) for s in range(3)]
table(['seed','終了/選択epoch','val %','test %','val損失'],[[r['seed'],str(r['epochs'])+' / '+str(r['best_epoch']),f"{100*r['selected_validation']['accuracy']:.2f}",f"{100*float(get(qp,seed=r['seed'],split='test',method='QAT_A3')['accuracy']):.2f}",num(r['selected_validation']['loss'])] for r in qs],[45,120,90,90,154])
p('seed0・1の選択モデルはval損失が約log(10)。seed2はepoch1の崩壊前モデルが選ばれた。いずれも最終epochの全層平均勾配ノルムはゼロである。')
section('19. E15：振幅増大とReLU出力の全ゼロ化')
fig(Q/'qat_per_seed_rms.png','E15-2　選択モデルの層出力RMS。正確にゼロの点は表示のため1e-12へ置き、赤×で明示。seedごとに縦軸範囲が異なる。')
collapse=read(Q/'collapse_diagnostics.csv')
table(['seed','選択epoch','RMS：第14層','RMS：第15層','最終epoch勾配'],[[s,get(collapse,seed=s,layer=14)['selected_epoch'],num(get(collapse,seed=s,layer=14)['selected_model_output_rms']),num(get(collapse,seed=s,layer=15)['selected_model_output_rms']),'全16層で0'] for s in range(3)],[45,75,125,125,129])
p('seed0・1では第14隠れ層まで大幅に増幅し、第15隠れ層のReLU出力とlogitsがtest全例でゼロとなった。単純な減衰だけではない。選択モデルのRMSと最終epochの勾配は異なる時点の値であり、混同しない。')
p('これは恒等STE・この初期化・Adam設定での失敗であり、QAT一般の不可能性を示すものではない。学習率、STE、正規化、残差、事前学習からの微調整などの追加探索は今回行っていない。')
section('20. E16：活性化精度のTDT比較')
p('E13と同じ厳密100k・16層・ReLUなし・閾値8のTDTで、A16／A8／A4を各3 seed追加した。A32はv5.2の保存モデル・記録を再利用する。追加のRMS正規化、残差、Lloyd反復は使わない。')
table(['精度','量子化・復元'],[['A32','FP32のまま。'],['A16','FP16へキャスト後、FP32へ復元。'],['A8','各例・各層のabsmax/127でスケールを決め、[-127,127]の符号付き255値へ丸めて復元。'],['A4','各例・各層のabsmax/7でスケールを決め、[-7,7]の符号付き15値へ丸めて復元。']],[60,439])
p('入力画像と全隠れ層を線形層直前で量子化する。積和・重みスケール・logits・損失はFP32。A3は3ビットではなく3値であり、A4の15値とは区別する。')
tdt_results(da,DC,[c.upper() for c in DC])
p('test平均はA32 87.313%、A16 87.490%、A8 87.030%、A4 67.170%。A16/A8はA32に近い精度を維持し、A4では20.143ポイント低下した。3 seedの小差からA16がA32より優れると断定しない。')
section('21. E16：学習曲線と全層診断')
fig(D/'accuracy_comparison.png','E16-1　validation学習曲線と最終test精度。A32は過去の同条件3 run。')
fig(D/'layer_diagnostics.png','E16-2　初期/最終RMS、層単独候補差、量子化相対二乗誤差、学習mean |y|。')
section('22. E16：16層の最終RMS')
tdt_layers(dsig,DC,[c.upper() for c in DC],'rms')
p('最終RMS(h_16)はA32約4.811、A16約4.888、A8約4.667、A4約2.514。A4でも信号は伝搬しており、精度低下を全層の信号消失だけで説明することはできない。')
section('23. E16：層単独の候補損失差')
tdt_layers(diso,DC,[c.upper() for c in DC],'y')
p('最終モデルの各層だけを摂動したmean |y|。各層64候補対、16辺、3 seed平均。')
table(['精度','学習mean |y|','損失差ゼロ率 %'],[[c.upper(),num(get(da,condition=c)['abs_y_mean_mean']),f"{100*float(get(da,condition=c)['zero_difference_fraction_mean']):.6f}"] for c in DC],[70,215,214])
p('A4の学習mean |y|はA32より大きいが、精度は低い。非ゼロ候補差の存在や増大だけで、有効な学習方向が得られるとは判断しない。')
section('24. E16：層別発火率と個別run')
table(['層',*[c.upper() for c in DC]],[[i+1,*[f"{100*float(get(dfire,condition=c,layer=i)['all_interval_rate_mean']):.3f}" for c in DC]] for i in range(16)],[35,116,116,116,116])
p('全区間発火率（%）。選択時発火率と各seedの記録は付属CSV。')
section('25. E16：全seedの最終値')
table(['精度','seed','val %','test %','val損失','発火数'],[[c.upper(),s,f"{100*float(get(dp,condition=c,seed=s)['val_accuracy']):.2f}",f"{100*float(get(dp,condition=c,seed=s)['test_accuracy']):.2f}",num(get(dp,condition=c,seed=s)['val_loss']),get(dp,condition=c,seed=s)['total_fires']] for c in DC for s in range(3)],[60,40,90,90,110,109])
p('保存モデルの初期・最終validation再評価が元ログと完全一致した。各seed・全12,000区間の層別選択数もA32対照と一致した。')
section('26. v5.3で分かったこと・比較上の限界')
p('第一に、ReLUなし16層TDTの活性化精度はA16・A8までA32に近い精度を維持したが、A4・A3では低下する。E13のA3は閾値分離型、E16のA8/A4はabsmaxであり、A3まで含む比較をビット数だけの効果とは扱わない。')
p('第二に、A3のRMSを構造的に保つことと、精度改善は別である。今回の残差接続は改善したが、FP32バイパスを使っている。RMS正規化単独と有限Lloyd反復は事前のval改善目安を満たさなかった。')
p('第三に、FP32で高精度学習できる16層モデルでも、推論時のA3化は大きく精度を落とす。恒等STEによるQATも今回の設定では安定せず、振幅増大とReLU全ゼロ化を観測した。学習アルゴリズムだけでなく、量子化・スケール・活性化・最適化の組合せが問題になる。')
p('FP32/QATはReLUあり・95,274個の連続重み、TDTはReLUなし・100,000個の三値重みであり、最適化予算とモデル選択も異なる。両者の差を単一要因へ帰属しない。ReLUなしA32は入力前処理後からlogitsまで線形写像で、深い非線形表現の獲得を示す結果ではない。')
p('各条件3 seed、MNISTと今回の設定に限定した記述的結果であり、有意差検定、収束保証、ハードウェア効率の優位性を主張するものではない。')
section('27. backpropとTDTの計算量の違い')
table(['比較','FP32 backprop','今回のTDT'],[['1更新区間','順伝播1回＋逆伝播1回で全重みの勾配。','64候補対＝128回の順伝播で16座標を評価、最大1座標を更新。'],['学習回数','E14：35/30/37 epoch、各79ミニバッチ。2,765/2,370/2,923更新。','1 run＝12,000区間、1,536,000学習forward。'],['その他','浮動小数点の連続重みをAdamで更新。','候補生成、重みの復元、量子化、カウンタ処理。Lloydは各層に反復計算。']],[75,212,212])
p('逆伝播そのものにも計算費用があるが、全重みの勾配をまとめて得る。TDTは離散候補の損失差を測定するため、本設定では多くのforwardを必要とする。実時間差を、学習予算を揃えた比較や手法一般の速度差と読み替えない。')
p('E14は推論量子化感度の対照、E15は素朴な恒等STE設定のQAT試験であり、TDTと同一forward予算・同一FLOPsでの最適化比較は行っていない。')
section('28. データ・再現性・監査記録')
for code,path in [('E13',A),('E14',B),('E15',Q),('E16',D)]:p(code+'：'+str(path.relative_to(ROOT)),'small')
table(['保存物','内容'],[['per_seed / aggregate / paired_effects','各seedの精度・損失、平均・標本標準偏差、対応する対照との差。'],['signal / activation / firing CSV','全層RMS、量子化誤差・コサイン・コード分布、TDT発火数と分母。'],['TDT loss diagnostics','全候補差の要約、時間窓、摂動対象層別と層単独の生データ。'],['FP32/QAT seed別記録','training.csv、best_model.pt、予測NPZ、QATのgradient_metrics.csv。'],['manifest / sources / verification','実行時コード・データのハッシュ、チェックポイントと集計の監査。']],[155,344])
p('PDF生成時にもseed別CSVから平均・標本標準偏差を再計算し、精度、RMS、層単独候補差、量子化誤差、発火率の集計と一致することを確認した。FP32/QATの保存ファイルとTDT診断モデルのハッシュも検証した。')
p('QATの元config.jsonにはFP32対照から継承したquantization説明の誤記がある。実際の学習はtraining_activationsとsteに記したA3 QATであり、順伝播一致を検証済み。effective_config.jsonとmetadata_corrections.jsonに補正内容を保存し、元モデル・数値結果は変更していない。')
section('29. 添付データとv5.2原版')
p('追加データZIPには4実験群のCSV・JSON・レポート・実行時ソース・図、およびFP32/QATのチェックポイント・全予測を収録する。データ集合全体と学習run数を再現可能に追跡できるよう、ファイルごとのSHA-256を付ける。')
p('TDTの大きい逐区間rawログ・全候補差NPY・モデルはtdt_mnist/runs配下に保存されており、新しいZIPには含めない。全候補差の要約と層単独の生データ、元runの場所とチェックポイントハッシュは添付する。短い動作試験を本実験の結果と混ぜない。')
p('次ページからv5.2全103ページを原版のまま収録する。しおりで追加部と旧版を移動できる。旧版の日付・ページ番号・当時の結論は歴史的記録として保持する。')
p('v5.2原版 SHA-256：'+sha(ORIGINAL),'small')
main=OUT/'v5.3-update.pdf';Doc(str(main),pagesize=A4,leftMargin=48,rightMargin=48,topMargin=48,bottomMargin=45).build(story,onFirstPage=footer,onLaterPages=footer)
(OUT/'TDT-v5.3_追加本文.txt').write_text('\n\n'.join(logs))
files=sorted({f for data in [A,B,Q,D] for f in data.rglob('*') if f.is_file() and f.suffix in ['.csv','.json','.md','.py','.png','.svg','.pt','.npz']})
files += [Path(__file__).resolve(),OUT/'TDT-v5.3_追加本文.txt']
hashes={str(f.relative_to(ROOT)):sha(f) for f in files};(OUT/'source_hashes.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2))
archive=DOC/'TDT-v5.3_追加実験データ.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for f in files:z.write(f,str(f.relative_to(ROOT)))
 z.write(OUT/'source_hashes.json','source_hashes.json')
pdf=fitz.open(main);n=len(pdf);toc=pdf.get_toc();old=fitz.open(ORIGINAL);assert len(old)==103
pdf.insert_pdf(old);toc.append([1,'付録：TDT-v5.2原版（全103ページ）',n+1]);toc.extend([level+1,title,page+n] for level,title,page in old.get_toc());pdf.set_toc(toc)
pdf.embfile_add('TDT-v5.3-additional-data.zip',archive.read_bytes(),filename=archive.name,desc='E13-E16: 24 new training runs, audited tables, diagnostics, source snapshots and FP32/QAT checkpoints')
for i in range(old.embfile_count()):
 info=old.embfile_info(i);pdf.embfile_add(f'previous-version-data-{i+1}.zip',old.embfile_get(i),filename=info.get('filename',f'previous-{i+1}.zip'))
pdf.set_metadata({'title':'TDT-v5.3 離散状態遷移学習理論 — 16層A3改善・FP32/PTQ・QAT・活性化精度','author':'TDT experimental study','subject':'E13-E16: A3 interventions, FP32 inference quantization, identity-STE QAT, A16/A8/A4 TDT','keywords':'TDT,MNIST,A3,A4,A8,A16,FP32,QAT,STE,RMS','creator':'ReportLab / PyMuPDF'})
pdf.save(DEST,garbage=4,deflate=True);pdf.close();checkpdf=fitz.open(DEST)
for i in range(len(old)):
 assert checkpdf[n+i].get_text()==old[i].get_text()
 assert checkpdf[n+i].get_pixmap(matrix=fitz.Matrix(.5,.5)).samples==old[i].get_pixmap(matrix=fitz.Matrix(.5,.5)).samples
outside=[]
for i in range(n):
 for b in checkpdf[i].get_text('dict')['blocks']:
  for line in b.get('lines',[]):
   for span in line['spans']:
    if not checkpdf[i].rect.contains(fitz.Rect(span['bbox'])):outside.append((i+1,span['text']))
assert not outside,outside
assert checkpdf.embfile_get('TDT-v5.3-additional-data.zip')==archive.read_bytes()
for i in range(old.embfile_count()):assert checkpdf.embfile_get(f'previous-version-data-{i+1}.zip')==old.embfile_get(i)
assert all(sha(ROOT/k)==v for k,v in hashes.items())
validation={'passed':True,'new_experiment_groups':4,'new_training_conditions':8,'new_training_runs':24,'new_tdt_runs':18,'new_fp32_runs':3,'new_qat_runs':3,'reused_tdt_controls':6,
 'new_training_candidate_pairs':13824000,'independent_probe_pairs':49152,'verified_aggregate_fields':verified,'verified_source_file_hashes':hash_checks,
 'update_pages':n,'preserved_v5_2_pages':103,'total_pages':len(checkpdf),'old_text_and_pixels_preserved':True,'no_text_outside_page':True,'all_attachments_verified':True,
 'original_sha256':sha(ORIGINAL),'pdf_sha256':sha(DEST),'archive_sha256':sha(archive),'archive_files':len(files)+1}
(OUT/'pdf_validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2))
for i in range(n):checkpdf[i].get_pixmap(matrix=fitz.Matrix(.7,.7)).save(OUT/f'preview-{i+1:02}.png')
print(json.dumps(validation,ensure_ascii=False,indent=2));print(DEST)
