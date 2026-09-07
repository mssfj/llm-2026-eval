"""Read-only numerical findings and an explicit correction of inherited metadata."""
from pathlib import Path
import json,hashlib,csv,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from backprop_a3_inference import read,write
P=Path(__file__).resolve().parent/'results/qat-a3-16x79-20260907'
rows=[];figure,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
corrections=[]
for seed,ax in enumerate(axes):
    d=P/f'seed{seed}'
    summary=json.loads((d/'summary.json').read_text())
    for name,sha in summary['output_sha256'].items():assert hashlib.sha256((d/name).read_bytes()).hexdigest()==sha
    sig=[r for r in read(d/'signal_metrics.csv') if r['split']=='test' and r['mode']=='qat_a3' and r['stage']=='output']
    grad=read(d/'gradient_metrics.csv');last=read(d/'training.csv')[-1]
    for r in sig:
        i=int(r['layer']);g=next(t for t in grad if t['epoch']==last['epoch'] and int(t['layer'])==i)
        rows.append({'seed':seed,'selected_epoch':summary['best_epoch'],'layer':i+1,'selected_model_output_rms':float(r['rms']),
            'selected_model_output_zero_fraction':float(r['zero_fraction']),'last_training_epoch':int(last['epoch']),
            'last_epoch_mean_gradient_norm':float(g['mean_gradient_norm'])})
    rms=[float(r['rms']) for r in sig];ax.plot(range(1,17),[max(v,1e-12) for v in rms],marker='o',markersize=3)
    zero=[i+1 for i,v in enumerate(rms) if v==0]
    if zero:ax.scatter(zero,[1e-12]*len(zero),marker='x',s=55,color='red',label='Exact zero (shown at 1e-12)');ax.legend(fontsize=7)
    ax.set(yscale='log',xlabel='Layer (last = logits)',ylabel='RMS(output)',title=f'Seed {seed}, selected epoch {summary["best_epoch"]}');ax.grid(alpha=.2)
    original=json.loads((d/'config.json').read_text());effective=dict(original)
    effective['quantization']='training and inference, per-example per-layer threshold and reconstruction scale'
    (d/'effective_config.json').write_text(json.dumps(effective,indent=2))
    corrections.append({'seed':seed,'field':'quantization','original':original['quantization'],'corrected':effective['quantization'],
        'reason':'The inherited FP32 control metadata said inference only; training_activations and ste already described QAT correctly. Original files and checkpoints preserved.',
        'effective_config_sha256':hashlib.sha256((d/'effective_config.json').read_bytes()).hexdigest()})
write(P/'collapse_diagnostics.csv',rows)
for ext in ['png','svg']:figure.savefig(P/('qat_per_seed_rms.'+ext),dpi=170)
plt.close(figure)
(P/'metadata_corrections.json').write_text(json.dumps(corrections,indent=2))
text=(P/'README.md').read_text()
marker='\n## 層別診断で確認した失敗の形'
text=text.split(marker)[0]
text+=marker+'\n\n'
text+='seed 0・1の選択モデルでは、第14隠れ層のRMSが約2.06×10^7・2.94×10^6まで増大し、第15隠れ層とlogitsはtest全例でゼロ。単純な減衰だけでなく、振幅増大とReLUの全ゼロ化が観測された。\n'
text+='全3seedで最終学習epochの全層平均勾配ノルムがゼロ、train損失は約log(10)。seed 2の報告精度は崩壊前のepoch 1モデルをvalidation損失で選んだ結果である。選択モデルのRMSと最終学習epochの勾配は異なる時点なので区別する。\n'
text+='恒等STE・この初期化／Adam設定では学習が安定しなかったという結果であり、QAT一般で改善できないという結論ではない。追加の設定探索は実施していない。\n\n'
text+='collapse_diagnostics.csv、qat_per_seed_rms.pngに各seedの層別診断を保存。effective_config.jsonは設定説明を補正したもの。元のconfig.jsonにFP32対照から継承したquantization説明の誤記が残るため、metadata_corrections.jsonに明記した。元の学習記録・モデルは変更していない。\n'
(P/'README.md').write_text(text)
v=json.loads((P/'verification.json').read_text());v['metadata_corrections']=corrections;v['diagnostic_sources']={}
for name in ['summarize_qat_diagnostics.py','test_qat_a3_backprop.py']:
    src=Path(__file__).parent/name;shutil.copy2(src,P/'sources'/name);v['diagnostic_sources'][name]=hashlib.sha256(src.read_bytes()).hexdigest()
v['collapse_diagnostics_sha256']=hashlib.sha256((P/'collapse_diagnostics.csv').read_bytes()).hexdigest()
(P/'verification.json').write_text(json.dumps(v,indent=2))
print('QAT collapse diagnostics and metadata correction recorded; original training artifacts verified unchanged.')
