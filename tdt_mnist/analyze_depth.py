"""Audit depth experiments and plot per-layer firing and signal propagation."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
import statistics


def read_csv(path):
    with path.open() as f: return list(csv.DictReader(f))


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('report_dir',type=Path);args=p.parse_args();root=args.report_dir
    manifest=json.loads((root/'manifest.json').read_text());status=json.loads((root/'status.json').read_text())
    assert status['complete'] and not status['errors']
    records=json.loads((root/'summaries.json').read_text())
    expected=set(itertools.product(manifest['depths'],manifest['thresholds'],manifest['seeds']))
    assert len(records)==len(expected)==status['completed']
    assert {(r['depth'],r['threshold'],r['seed']) for r in records}==expected
    for name,digest in manifest['source_sha256'].items(): assert hashlib.sha256((root/'sources'/name).read_bytes()).hexdigest()==digest
    for name,digest in manifest['data_sha256'].items(): assert hashlib.sha256((Path(manifest['data_dir'])/'MNIST/raw'/name).read_bytes()).hexdigest()==digest
    import torch
    checks=[];curves={}
    for r in records:
        directory=Path(r['run_directory']);config=json.loads((directory/'config.json').read_text())
        checkpoint=torch.load(directory/'model.pt',map_location='cpu',weights_only=False)
        assert json.loads(json.dumps(checkpoint['config']))==config
        dims=[90,*manifest['hidden_widths'][str(r['depth'])],10]
        assert config['shapes']==[[b,a] for a,b in zip(dims,dims[1:])]
        assert sum(a*b for a,b in zip(dims,dims[1:]))==100000==r['num_params']==checkpoint['weights'].numel()
        assert set(checkpoint['weights'].unique().tolist()).issubset({-1,0,1})
        assert config['seed']==r['seed'] and config['threshold']==r['threshold']
        assert config['batch_seed']==r['seed']+100000
        for key in ('steps','measurements','block_size','batch_size','max_fires','train_size','val_size','test_size','data_seed',
                    'hidden_activation','activation_precision','gain','zero_rate','counter_bits','scale','min_scale','scale_ema','leak','oracle_every','eval_every','layer_diagnostics','device'):
            assert config[key]==manifest[key],(directory,key)
        metrics=read_csv(directory/'metrics.csv')
        steps=config['steps'];depth=r['depth'];last=[None]*depth
        assert len(metrics)==steps+1
        assert [int(x['step']) for x in metrics]==list(range(steps+1))
        assert r['train_forward_calls']==2*steps*config['measurements']==int(metrics[-1]['train_forward_calls'])
        assert sum(int(x['zero_difference_count']) for x in metrics[1:])==r['zero_difference_count']
        assert r['zero_difference_fraction']==r['zero_difference_count']/(steps*config['measurements'])
        assert sum(int(x['fires']) for x in metrics[1:])==r['total_fires']==sum(r['layer_update_counts'])
        assert float(metrics[-1]['val_accuracy'])==r['final_validation']['accuracy']
        sums=[dict(fires=0,selected_coordinates=0,selected_interval=0,fire_interval=0) for _ in range(depth)]
        # Stream layer rows: 192,000 rows per 16-layer run.
        with (directory/'layer_metrics.csv').open() as f:
            reader=csv.DictReader(f);n=0;epoch_fires=0;epoch_selected=0
            for row in reader:
                step=n//depth+1;layer=n%depth
                assert int(row['step'])==step and int(row['layer'])==layer
                assert int(row['parameters'])==dims[layer]*dims[layer+1]
                for key in sums[layer]: sums[layer][key]+=int(row[key])
                assert int(row['selected_interval'])==int(int(row['selected_coordinates'])>0)
                assert int(row['fire_interval'])==int(int(row['fires'])>0)
                assert int(row['fires'])<=int(row['selected_coordinates'])
                assert int(row['cumulative_fires'])==sums[layer]['fires']
                assert int(row['cumulative_selected_coordinates'])==sums[layer]['selected_coordinates']
                assert int(row['cumulative_selected_intervals'])==sums[layer]['selected_interval']
                assert int(row['cumulative_fire_intervals'])==sums[layer]['fire_interval']
                assert float(row['fire_interval_rate'])==sums[layer]['fire_interval']/step
                if sums[layer]['selected_interval']:
                    assert float(row['fire_given_selected_interval_rate'])==sums[layer]['fire_interval']/sums[layer]['selected_interval']
                else: assert not row['fire_given_selected_interval_rate']
                epoch_fires+=int(row['fires']);epoch_selected+=int(row['selected_coordinates'])
                if layer==depth-1:
                    assert epoch_fires==int(metrics[step]['fires']) and epoch_selected==config['block_size']
                    epoch_fires=epoch_selected=0
                last[layer]=row;n+=1
        assert n==steps*depth
        for layer in range(depth):
            for summarykey,col in (('layer_update_counts','fires'),('layer_selected_coordinates','selected_coordinates'),
                                   ('layer_selected_intervals','selected_interval'),('layer_fire_intervals','fire_interval')):
                assert r[summarykey][layer]==sums[layer][col]
            assert r['layer_fire_interval_rates'][layer]==sums[layer]['fire_interval']/steps
        signals=read_csv(directory/'signal_metrics.csv')
        evalsteps=sorted({0,steps,*range(config['eval_every'],steps+1,config['eval_every'])})
        assert len(signals)==len(evalsteps)*depth*3
        assert {(int(x['step']),int(x['layer']),x['stage']) for x in signals}==set(itertools.product(evalsteps,range(depth),('input','pre_activation','output')))
        signal_map={(int(x['step']),int(x['layer']),x['stage']):x for x in signals}
        for obs in signals:
            layer=int(obs['layer']);width=dims[layer] if obs['stage']=='input' else dims[layer+1]
            assert int(obs['values'])==config['val_size']*width and int(obs['features'])==width
            assert int(obs['nonfinite_count'])==0
            assert all(math.isfinite(float(obs[k])) for k in ('rms','std','mean','max_abs','zero_fraction'))
            if obs['stage']=='output' and layer<depth-1: assert float(obs['negative_fraction'])==0
            if obs['stage']=='input' and layer:
                prev=signal_map[(int(obs['step']),layer-1,'output')]
                for field in ('rms','mean','zero_fraction'): assert obs[field]==prev[field]
        assert all(math.isfinite(r[split][field]) for split in ('initial_validation','final_validation','test') for field in ('loss','accuracy'))
        curves[(depth,r['threshold'],r['seed'])]=[(int(x['step']),float(x['val_accuracy'])) for x in metrics if x['val_accuracy']]
        checks.append({'run':directory.name,'passed':True,'layer_rows':n,'signal_rows':len(signals),'train_forward_calls':r['train_forward_calls']})
    for depth,seed in itertools.product(manifest['depths'],manifest['seeds']):
        paired=[r for r in records if r['depth']==depth and r['seed']==seed]
        for r in paired[1:]:
            assert r['initial_validation']==paired[0]['initial_validation']
            assert r['layer_selected_coordinates']==paired[0]['layer_selected_coordinates']
            assert r['layer_selected_intervals']==paired[0]['layer_selected_intervals']
    assert len({r['total_forward_calls'] for r in records})==1
    (root/'verification.json').write_text(json.dumps({'passed':True,'runs':checks,'data_and_source_hashes_verified':True,
        'exact_100000_ternary_weights':True,'all_layer_selection_and_fire_counts_verified':True,'equal_forward_budgets':True,'paired_layer_selection_opportunities_verified':True},indent=2)+'\n')
    layer_rows=read_csv(root/'layer_firing.csv');signal_rows=read_csv(root/'signal_metrics.csv')
    def aggregate(rows,keys,fields):
        groups={}
        for row in rows:
            key=tuple(row[k] for k in keys);groups.setdefault(key,[]).append(row)
        output=[]
        for key,members in groups.items():
            row=dict(zip(keys,key));row['seeds']=len(members)
            for f in fields:
                values=[float(r[f]) for r in members if r[f]!='']
                row[f+'_mean']=statistics.mean(values) if values else None
                row[f+'_std']=statistics.stdev(values) if len(values)>1 else 0 if values else None
            output.append(row)
        return output
    layer_agg=aggregate(layer_rows,('depth','threshold','layer'),('fires','selected_coordinates','selected_intervals','fire_interval_rate',
        'fire_given_selected_interval_rate','fires_per_selected_coordinate','updates_per_parameter'))
    signal_agg=aggregate(signal_rows,('depth','threshold','step','layer','stage'),('rms','std','mean','zero_fraction','dead_feature_fraction','max_abs'))
    write_csv(root/'layer_firing_aggregate.csv',layer_agg);write_csv(root/'signal_aggregate.csv',signal_agg)
    lines=['\n## 層別の最終発火率\n','主表は「発火した区間数 / その層が選ばれた区間数」の3seed平均（%）。層は入力側から1始まりで表示。',
        '全区間を分母とする率、更新回数、選択回数、重み数あたり更新回数、標準偏差はlayer_firing_aggregate.csvとlayer_firing.csvに保存。','']
    for depth in manifest['depths']:
        lines += [f'### {depth}層','', '| 層 | '+' | '.join(f'閾値{t}' for t in manifest['thresholds'])+' |',
            '| ---: | '+' | '.join('---:' for _ in manifest['thresholds'])+' |']
        for layer in range(depth):
            vals=[next(r for r in layer_agg if (int(r['depth']),int(r['threshold']),int(r['layer']))==(depth,t,layer))['fire_given_selected_interval_rate_mean'] for t in manifest['thresholds']]
            lines.append(f'| {layer+1} | '+' | '.join(f'{100*v:.3f}' if v is not None else '未選択' for v in vals)+' |')
        lines.append('')
    marker='\n## 層別の最終発火率\n';readme=root/'README.md';readme.write_text(readme.read_text().split(marker)[0]+'\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    depths=manifest['depths'];thresholds=manifest['thresholds'];seeds=manifest['seeds']
    fig,axes=plt.subplots(1,len(depths),figsize=(5*len(depths),4),squeeze=False,layout='constrained')
    for ax,depth in zip(axes[0],depths):
        for threshold in thresholds:
            vals=np.array([[v*100 for _,v in curves[(depth,threshold,s)]] for s in seeds]);xs=[x for x,_ in curves[(depth,threshold,seeds[0])]]
            mean=vals.mean(0);sd=vals.std(0,ddof=1) if len(seeds)>1 else np.zeros_like(mean)
            ax.plot(xs,mean,label=f'Threshold {threshold}');ax.fill_between(xs,mean-sd,mean+sd,alpha=.15)
        ax.set_title(f'{depth} linear layers');ax.set_xlabel('Accumulation intervals');ax.set_ylabel('Validation accuracy (%)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    for ext in ('png','svg'): fig.savefig(root/f'learning_curves.{ext}',dpi=180)
    plt.close(fig)
    for field,title,name in [('fire_interval_rate_mean','Firing intervals / all intervals (%)','layer_fire_rates'),
            ('fire_given_selected_interval_rate_mean','Firing intervals / selected intervals (%)','layer_conditional_fire_rates')]:
        fig,axes=plt.subplots(1,len(depths),figsize=(5*len(depths),5),squeeze=False,layout='constrained')
        def rate(d,t,l):
            value=next(r for r in layer_agg if (int(r['depth']),int(r['threshold']),int(r['layer']))==(d,t,l))[field]
            return 100*value if value is not None else float('nan')
        matrices=[np.array([[rate(d,t,l) for t in thresholds] for l in range(d)]) for d in depths]
        vmax=max(float(np.nanmax(m)) for m in matrices)
        for ax,d,matrix in zip(axes[0],depths,matrices):
            im=ax.imshow(matrix,aspect='auto',vmin=0,vmax=max(vmax,1e-12),cmap='viridis')
            ax.set_xticks(range(len(thresholds)),thresholds);ax.set_yticks(range(d),range(1,d+1));ax.set_title(f'{d} layers');ax.set_xlabel('Counter threshold');ax.set_ylabel('Layer (input to output)')
            for l in range(d):
                for j in range(len(thresholds)): ax.text(j,l,f'{matrix[l,j]:.3g}' if np.isfinite(matrix[l,j]) else 'N/A',ha='center',va='center',fontsize=7,color='white' if matrix[l,j]<vmax*.5 else 'black')
        fig.colorbar(im,ax=list(axes[0]),label=title)
        for ext in ('png','svg'): fig.savefig(root/f'{name}.{ext}',dpi=180)
        plt.close(fig)
    fig,axes=plt.subplots(2,len(depths),figsize=(5*len(depths),8),squeeze=False,layout='constrained')
    for col,d in enumerate(depths):
        for t in thresholds:
            for step,linestyle in ((0,'--'),(manifest['steps'],'-')):
                rows=[next(r for r in signal_agg if (int(r['depth']),int(r['threshold']),int(r['step']),int(r['layer']),r['stage'])==(d,t,step,l,'output')) for l in range(d)]
                # Initial signals are identical across thresholds; plot once.
                if step==0 and t!=thresholds[0]: continue
                label='Initial' if step==0 else f'Final threshold {t}'
                axes[0,col].plot(range(1,d+1),[max(r['rms_mean'],1e-30) for r in rows],linestyle,label=label)
                axes[1,col].plot(range(1,d+1),[100*r['zero_fraction_mean'] for r in rows],linestyle,label=label)
        axes[0,col].set_yscale('log');axes[0,col].set_ylabel('Layer output RMS');axes[1,col].set_ylabel('Layer output zero fraction (%)')
        for ax in axes[:,col]: ax.set_title(f'{d} layers');ax.set_xlabel('Layer (last = logits)');ax.grid(alpha=.2);ax.legend(fontsize=7)
    for ext in ('png','svg'): fig.savefig(root/f'signal_propagation.{ext}',dpi=180)
    plt.close(fig)
    shutil.copy2(__file__,root/'sources'/Path(__file__).name)
    (root/'analysis_manifest.json').write_text(json.dumps({'script':Path(__file__).name,'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    print(f'Verified {len(records)} runs and generated layer firing / signal plots in {root}')


if __name__=='__main__': main()
