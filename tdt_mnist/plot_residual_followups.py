"""Publication figures from audited E18-E20 CSVs, without model evaluation."""
import csv,json,shutil,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_residual_followups import ROOT,E17
from run_residual_e17 import dump,sha


def read(p):
    with p.open() as f:return list(csv.DictReader(f))


def save(fig,path):
    fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(path.with_suffix('.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)


def main(root):
    assert json.loads((root/'audit.json').read_text())['passed']
    out=root/'figures';out.mkdir(exist_ok=True)
    rows=read(root/'aggregate/results.csv')
    agg={r['condition']:r for r in rows}
    old=[r for r in read(E17/'per_seed/results.csv') if r['condition']=='E17a']
    base=statistics.mean(float(r['test_accuracy_percent']) for r in old)
    base_sd=statistics.stdev(float(r['test_accuracy_percent']) for r in old)
    def mean(c):return float(agg[c]['test_mean_percent'])
    def sd(c):return float(agg[c]['test_sample_std_percent'])
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(1,2,figsize=(12,4.5))
    ax[0].errorbar([8,16,24,32],[base]+[mean(c) for c in ['E18a','E18b','E18c']],
        yerr=[base_sd]+[sd(c) for c in ['E18a','E18b','E18c']],marker='o',capsize=4,label='Width 76; weights grow with depth')
    ax[0].errorbar([8,16],[base,mean('E18d')],yerr=[base_sd,sd('E18d')],marker='s',capsize=4,
        label='Near 100k: width 76 -> 54 (98,712)')
    ax[0].axhline(89.637,color='gray',ls='--',label='Preregistered non-degradation cutoff')
    ax[0].set(xticks=[8,16,24,32],xlabel='Residual blocks',ylabel='Final test accuracy (%)',title='E18: depth at fixed width and near-fixed budget')
    ax[0].legend(fontsize=8)
    curves=read(root/'aggregate/validation_curves.csv')
    for c in ['E18a','E18b','E18c','E18d']:
        groups={}
        for r in curves:
            if r['condition']==c:groups.setdefault(int(r['step']),[]).append(100*float(r['val_accuracy']))
        xs=sorted(groups);means=[statistics.mean(groups[x]) for x in xs]
        ax[1].plot(xs,means,label=c)
    ax[1].set(xlabel='TDT interval',ylabel='Mean validation accuracy (%)',title='E18: all predetermined validation checkpoints')
    ax[1].legend()
    save(fig,out/'e18_depth')
    fig,ax=plt.subplots(1,2,figsize=(11,4.5))
    ax[0].bar(['E17a residual A8','E19a residual A4'],[base,mean('E19a')],yerr=[base_sd,sd('E19a')],capsize=5,color=['#2166ac','#b2182b'])
    ax[0].set(ylabel='Test accuracy (%)',title='E19: all quantization points use A4')
    ax[1].bar(['Serial E16','Residual E19'],[19.86,90.637-mean('E19a')],color=['gray','#b2182b'])
    ax[1].axhline(3,color='black',ls='--',label='Residual A4 cost cutoff: 3 pp')
    ax[1].set(ylabel='A8 minus A4 accuracy (percentage points)',title='Architecture-specific A4 cost')
    ax[1].legend(fontsize=8)
    save(fig,out/'e19_a4')
    effects=read(root/'aggregate/paired_effects.csv')
    fig,ax=plt.subplots(1,2,figsize=(12,4.5))
    cs=['E20a','E20b','E20c']
    for i,c in enumerate(cs):
        if agg[c]['test_mean_percent']:
            ax[0].errorbar(i,mean(c),yerr=sd(c),fmt='o',capsize=5)
        else:ax[0].text(i,base,'failed seed(s)',rotation=90)
    ax[0].errorbar(3,base,yerr=base_sd,fmt='o',capsize=5,color='black')
    ax[0].set(xticks=range(4),xticklabels=['E20a\nFP32/BP','E20b\nA8/BP','E20c\nW3 A8/STE','E17a\nW3 A8/TDT'],
              ylabel='Test accuracy (%)',title='E20: same residual topology')
    names=['E20a_minus_E20b_A8_cost','E20b_minus_E20c_W3_cost','E20c_minus_E17a_TDT_comparison']
    for i,name in enumerate(names):
        r=next(r for r in effects if r['comparison']==name)
        if r['mean_pp']:
            ax[1].bar(i,float(r['mean_pp']),yerr=float(r['sample_std_pp']),capsize=5)
    ax[1].axhline(0,color='black',lw=.8)
    ax[1].set(xticks=range(3),xticklabels=['A8 under BP','W3 under STE','E20c - E17a*'],
        ylabel='Accuracy difference (percentage points)',title='Paired decomposition: mean +/- sample SD')
    fig.text(.5,-.02,'*Includes initialization, W3 scaling, budget and model-selection differences; not an isolated causal learning-rule effect.',ha='center',fontsize=9)
    save(fig,out/'e20_decomposition')
    shutil.copy2(Path(__file__),root/'sources'/Path(__file__).name)
    dump(out/'manifest.json',dict(matplotlib_version=matplotlib.__version__,source_sha256=sha(Path(__file__)),
        files={p.name:sha(p) for p in out.iterdir() if p.suffix in ['.png','.svg']}))
    with (root/'README.md').open('a') as f:
        f.write('\n図: [E18深さ比較](figures/e18_depth.png)、[E19 A4コスト](figures/e19_a4.png)、[E20分解](figures/e20_decomposition.png)。同名SVGも保存。\n')
    dump(root/'artifacts_sha256.json',{str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='artifacts_sha256.json'})


if __name__=='__main__':main(ROOT)
