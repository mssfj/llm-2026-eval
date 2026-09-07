"""Summarize audited results and verify the final presentation artifacts."""
from pathlib import Path
import csv,json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'tdt_mnist/results/depth-activation-100k-20260907'
def read(name):
    with (P/name).open() as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
status=json.loads((P/'status.json').read_text());audit=json.loads((P/'verification.json').read_text())
assert status['complete'] and audit['passed'] and audit['runs']==108
for name,digest in audit['analysis_sources'].items():assert sha(P/'sources'/name)==digest
conditions=['relu-a3-threshold','identity-a32','identity-a3-threshold']
labels=['ReLU＋A3閾値分離','ReLUなし＋A32','ReLUなし＋A3閾値分離']
for c in conditions:
    m=json.loads((P/c/'manifest.json').read_text())
    assert json.loads((P/c/'verification.json').read_text())['passed']
    assert json.loads((P/c/'status.json').read_text())['complete']
    for name,digest in m['source_sha256'].items():assert sha(P/c/'sources'/name)==digest
counts={f:len(read(f)) for f in ['per_seed.csv','rms_h_per_seed.csv','abs_y_by_perturbed_layer.csv','layer_isolated_abs_y.csv']}
assert list(counts.values())==[108,50400,1008,2016]
assert sum(r['probe_pairs'] for r in audit['probe_checks'])==129024
h_count=sum(r['quantity']=='h_l' for r in read('rms_h_per_seed.csv'));assert h_count==25200
agg=read('aggregate.csv');rms=read('rms_h_aggregate.csv');ys=read('abs_y_aggregate.csv')
lines=['# 結果の要点','',
    '全108run（3方式×4/8/16層×カウンタ閾値1/4/8/16×3seed）と全層診断が完了し、監査を通過しました。100,000重み、12,000区間などはv5の基準条件です。',
    'A3はすべて閾値・復元スケール分離型（τ=0.5 mean(abs(x))）。以下は閾値8で固定した比較です。精度は最終testの3seed平均±標本標準偏差。','',
    '| 層数 | '+' | '.join(labels)+' |','| ---: | ---: | ---: | ---: |']
for depth in [4,8,16]:
    cells=[]
    for c in conditions:
        r=next(r for r in agg if r['condition']==c and int(r['depth'])==depth and r['threshold']=='8')
        cells.append(f"{100*float(r['test_accuracy_mean']):.2f} ± {100*float(r['test_accuracy_std']):.2f}%")
    lines.append(f'| {depth} | '+' | '.join(cells)+' |')
lines+=['','ReLUなし＋A32は16層でも87%台を維持しました。ReLU＋A3は8・16層で10〜11%程度となり、ReLUを外したA3も16層では約14%です。この固定予算・初期化・学習区間の範囲では、ReLUを外すだけで深いA3の精度劣化を解消できませんでした。','',
    '## 16層・閾値8の信号と候補差','',
    'RMSは検証集合全体、候補差は学習中の全候補対から集計した3seed平均です。最終層はlogits。', '',
    '| 条件 | 初期RMS(h_16) | 最終RMS(h_16) | 学習mean abs(y) | 損失差ゼロ % |',
    '| --- | ---: | ---: | ---: | ---: |']
for c,label in zip(conditions,labels):
    rs=[next(r for r in rms if r['condition']==c and r['depth']=='16' and r['threshold']=='8' and r['step']==str(step) and r['layer']=='15' and r['quantity']=='h_l') for step in [0,12000]]
    y=next(r for r in ys if r['condition']==c and r['depth']=='16' and r['threshold']=='8')
    lines.append(f"| {label} | {float(rs[0]['rms_mean']):.6g} | {float(rs[1]['rms_mean']):.6g} | {float(y['mean_mean']):.6g} | {100*float(y['zero_fraction_mean']):.6f} |")
lines += ['',
    'ReLU＋A3では初期から深さ方向の振幅減衰が強く、学習後にも出力RMSが小さい状態が残りました。ReLUなし＋A3は振幅と候補差が大きくなりますが、A32より精度が低いため、RMSやabs(y)の大きさだけでは学習の有効性を説明できません。',
    'ReLUなし＋A32では、前処理後の入力からlogitsまでが線形写像です。今回の結果は、非線形な深層表現を高精度で学習できたことを意味しません。また固定重み数では深さとともに幅も変わります。','',
    '## 記録したデータ','',
    '- 各層のRMS(h_l)：初期と500区間ごと、計25時点。活性化後・次層量子化前の25,200観測と、量子化復元後の線形層入力の25,200観測を保存。',
    '- 学習中のabs(y)：82,944,000候補対のFP32値を各runのabs_y.npyに保存。摂動対象層に条件付けた分布は1,008層×runの集計。',
    '- 層単独摂動：初期・最終の各層に64候補対、合計129,024候補対。L(T+)、L(T−)、abs(y)を保存し、2,016層×run×時点に集計。各層16辺の独立診断で、学習には使いません。',
    '- 層別発火率、量子化復元後RMS、損失差の時間窓統計・分位点・ゼロ率も保存。','',
    '学習中のブロックは複数層にまたがります。「当該層を含む摂動」の候補差は層単独の寄与ではありません。この曖昧さを避けて確認するため、層単独摂動の診断を別途保存しました。層ごとに重み数が違うため、16辺という固定本数の摂動率も異なります。','',
    '全条件の表と定義：[README.md](README.md)。図：[精度比較](accuracy_comparison.png)、[RMSと層単独損失差（閾値8）](layer_rms_and_abs_y_threshold8.png)。','']
(P/'findings.md').write_text('\n'.join(lines))
readme=P/'README.md';s=readme.read_text().replace('ReLUなし+A32はバイアスなし線形層の合成であり全体も線形写像。','ReLUなし+A32はバイアスなし線形層の合成であり、前処理後の入力からlogitsまでが線形写像。')
s=s.replace('## 最終結果','結果の要点と解釈は [findings.md](findings.md) に整理しています。\n\n## 最終結果') if '[findings.md]' not in s else s
readme.write_text(s)
progress=json.loads((P/'progress.json').read_text());progress.update(complete=True,completed=108,analysis_verified=True);(P/'progress.json').write_text(json.dumps(progress,indent=2))
shutil.copy2(__file__,P/'sources'/Path(__file__).name)
files=['README.md','findings.md','aggregate.csv','rms_h_aggregate.csv','abs_y_aggregate.csv','abs_y_by_perturbed_layer_aggregate.csv','layer_isolated_abs_y_aggregate.csv','accuracy_comparison.png','layer_rms_and_abs_y_threshold8.png']
(P/'artifact_verification.json').write_text(json.dumps({'passed':True,'rows':counts,'h_l_observations':h_count,
    'training_candidate_pairs':82944000,'independent_probe_pairs':129024,'executed_analysis_snapshots_verified':True,
    'finalizer_sha256':sha(Path(__file__)),'artifact_sha256':{f:sha(P/f) for f in files}},indent=2))
print((P/'findings.md').read_text())
