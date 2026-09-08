"""Final E17 figures, derived solely from audited CSVs; no model evaluation."""
import csv
import json
from pathlib import Path
import shutil
import statistics
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_residual_e17 import ROOT, sha, dump

COLORS={'E17a':'#2166ac','E17b':'#b2182b','E17c':'#168b55'}
LABELS={'E17a':'E17a: A8 + ReLU','E17b':'E17b: A8 + identity','E17c':'E17c: FP32 + ReLU'}


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def line(ax, rows, xkey, ykey, condition, factor=1., label=None):
    groups={}
    for r in rows:
        if r['condition']==condition:
            groups.setdefault(int(r[xkey]),[]).append(factor*float(r[ykey]))
    xs=sorted(groups)
    means=[statistics.mean(groups[x]) for x in xs]
    stds=[statistics.stdev(groups[x]) for x in xs]
    ax.plot(xs,means,color=COLORS[condition],label=label or LABELS[condition])
    ax.fill_between(xs,[m-s for m,s in zip(means,stds)], [m+s for m,s in zip(means,stds)],
                    color=COLORS[condition],alpha=.15)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path.with_suffix('.png'),dpi=180)
    fig.savefig(path.with_suffix('.svg'))
    plt.close(fig)


def main(root):
    assert json.loads((root/'audit.json').read_text())['passed']
    plt.rcParams.update({'font.size':9, 'axes.spines.top':False,'axes.spines.right':False})
    out=root/'figures';out.mkdir(exist_ok=True)
    curve=read(root/'aggregate/validation_curves.csv')
    scores=read(root/'per_seed/results.csv')
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for c in COLORS: line(ax[0],curve,'step','val_accuracy',c,100)
    ax[0].set(xlabel='TDT interval',ylabel='Validation accuracy (%)',title='Validation: mean +/- sample SD (3 seeds)')
    ax[0].legend(loc='lower right')
    for i,c in enumerate(COLORS):
        vals=[float(r['test_accuracy_percent']) for r in scores if r['condition']==c]
        ax[1].errorbar(i,statistics.mean(vals),yerr=statistics.stdev(vals),fmt='o',capsize=6,color=COLORS[c])
        ax[1].scatter([i-.12,i,i+.12],vals,color=COLORS[c],s=16,alpha=.6)
    ax[1].axhline(87.31,ls='--',color='gray',label='E16 A32: 87.31%')
    ax[1].axhline(90.31,ls=':',color='black',label='E17a preregistered mean cutoff: 90.31%')
    ax[1].set(xticks=range(3),xticklabels=list(COLORS),ylabel='Final test accuracy (%)',title='Final interval 12,000; no model selection')
    ax[1].legend(fontsize=8)
    save(fig,out/'accuracy')
    ratios=read(root/'signal/rms_ratios.csv')
    signals=read(root/'signal/metrics.csv')
    firing=read(root/'firing/matrices.csv')
    isolated=read(root/'signal/isolated_candidates.csv')
    activation=read(root/'activation/metrics.csv')
    fig,ax=plt.subplots(2,3,figsize=(15,8))
    for c in COLORS:
        line(ax[0,0],[r for r in ratios if r['step']=='12000'],'block','stream_rms',c)
        line(ax[0,1],[r for r in ratios if r['step']=='12000'],'block','branch_stream_rms_ratio',c)
        line(ax[0,2],[r for r in signals if r['layer']=='17' and r['stage']=='output'],'step','rms',c)
        line(ax[1,0],firing,'layer','all_interval_firing_rate',c,100)
        line(ax[1,1],[r for r in isolated if r['stage']=='final'],'layer','mean_abs_y',c)
        line(ax[1,2],[r for r in activation if r['step']=='12000'],'layer','relative_squared_error',c)
    ax[0,0].set(xlabel='Block (0-based)',ylabel='RMS',title='Final FP32 stream before branch addition')
    ax[0,1].set(xlabel='Block (0-based)',ylabel='Branch RMS / stream RMS',title='Final branch / stream ratio')
    ax[0,1].axhline(.5,color='gray',ls='--')
    ax[0,2].set(xlabel='TDT interval',ylabel='Logits RMS',title='Validation logits amplitude')
    ax[0,2].axhline(10,color='gray',ls='--')
    ax[1,0].set(xlabel='Matrix (0-based; 0=input, 17=output)',ylabel='Firing intervals / 12,000 (%)',title='Firing across all training intervals')
    ax[1,1].set(xlabel='Matrix (0-based)',ylabel='Mean |loss(+) - loss(-)|',title='Final isolated probes: 64 pairs, 16 edges')
    ax[1,2].set(xlabel='Quantization point / matrix (0-based)',ylabel='Relative squared error',title='Final per-point quantization error')
    ax[0,0].legend(fontsize=8)
    fig.suptitle('E17 diagnostics: means and sample SD across 3 seeds',y=1.02)
    save(fig,out/'diagnostics')
    shutil.copy2(Path(__file__),root/'sources'/Path(__file__).name)
    dump(out/'manifest.json',dict(matplotlib_version=matplotlib.__version__,source_sha256=sha(Path(__file__)),
         files={p.name:sha(p) for p in out.iterdir() if p.suffix in ('.png','.svg')}))
    with (root/'README.md').open('a') as f:
        f.write('\n図: [精度比較](figures/accuracy.png)、[層別診断](figures/diagnostics.png)。同名SVGも保存。帯は3seedの標本標準偏差。\n')
    dump(root/'artifacts_sha256.json',{str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*'))
                                      if p.is_file() and p.name!='artifacts_sha256.json'})


if __name__=='__main__':
    main(Path(sys.argv[1]) if len(sys.argv)>1 else ROOT)
