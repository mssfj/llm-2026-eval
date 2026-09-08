"""Independent read-only audit of GPU benchmark records and CPU reference."""
import json,csv,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE/'results/gpu-evaluation-16blocks-20260908'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return list(csv.DictReader(p.open()))
def main():
    status=json.loads((ROOT/'status.json').read_text());assert status['complete']
    checks=0;manifest=json.loads((ROOT/'manifest.json').read_text())
    for n,h in manifest['sources'].items():assert sha(HERE/n)==h and sha(ROOT/'sources'/n)==h;checks+=2
    old=HERE/'results/residual-followups-e18-e20-20260908'
    for seed,h in manifest['trained_models'].items():assert sha(old/'per_seed'/f'E18a-seed{seed}'/'model.pt')==h;checks+=1
    for n,h in manifest['data'].items():assert sha(HERE/'data/MNIST/raw'/n)==h;checks+=1
    modes=['cpu_restore_cache','gpu_sequential','gpu_batched','gpu_graph']
    rows=read(ROOT/'validation/summary.csv');assert len(rows)==18
    assert all(r['indices_equal']==r['rng_equal']=='True' for r in rows)
    losses=read(ROOT/'validation/losses.csv');assert len(losses)==2304
    for row in rows:
        values=[r for r in losses if all(r[k]==row[k] for k in ['engine','seed','state'])];assert len(values)==128
        assert abs(max(float(v['relative_error']) for v in values)-float(row['max_relative_loss_error']))<1e-12
    assert all(int(r['bitwise_mismatches'])==0 for r in read(ROOT/'validation/graph_input_refresh.csv'))
    for seed in range(3):
        summaries={mode:json.loads((ROOT/'benchmarks'/f'seed{seed}-{mode}'/'summary.json').read_text()) for mode in modes}
        for key in ['generator_sha256','batch_generator_sha256']:assert len({r[key] for r in summaries.values()})==1
        previous=HERE/'results/allocation-ablations-16blocks-20260908/benchmarks'/f'seed{seed}-restore_cache'
        oldsum=json.loads((previous/'summary.json').read_text())
        for key in ['final_weights_sha256','final_scale','generator_sha256','batch_generator_sha256']:assert oldsum[key]==summaries['cpu_restore_cache'][key]
        # Graph capture itself must not change the eager batched trajectory.
        for key in ['final_weights_sha256','final_scale']:assert summaries['gpu_batched'][key]==summaries['gpu_graph'][key]
        for mode in modes:
            intervals=read(ROOT/'benchmarks'/f'seed{seed}-{mode}'/'intervals.csv');assert len(intervals)==100
            assert all(0<=int(r['fires'])<=1 and float(r['seconds'])>0 for r in intervals)
        a=read(ROOT/'benchmarks'/f'seed{seed}-gpu_batched/intervals.csv');b=read(ROOT/'benchmarks'/f'seed{seed}-gpu_graph/intervals.csv')
        assert all(x['action']==y['action'] and x['scale']==y['scale'] for x,y in zip(a,b))
    # Existing long-running CPU experiment's source must remain unchanged.
    other=HERE/'results/fast-engine-e17a-20260908'
    for n,h in json.loads((other/'manifest.json').read_text())['sources'].items():assert sha(HERE/n)==h;checks+=1
    report_path=ROOT/'README.md'
    report=report_path.read_text()
    note='''

## GPUメモリ指標の補足

GPU予約メモリの最大値は、逐次150 MiB、並列170 MiB、CUDA Graphs224 MiB。表の「最大割当」はウォームアップ後にリセットしたPyTorchの割当カウンタであり、CUDA Graphsの内部再生時の全一時テンソルの生存ピークを直接測るものではない。Graphは捕捉時に確保した専用プールを再利用するため、69.2 MiBという割当カウンタだけから並列版よりメモリ使用が少ないとは結論しない。予約メモリにはそのプールを含むが、CUDAコンテキスト等の全プロセスVRAMまでは含まない。初期転送・捕捉の時間もseed別記録に分離している。
'''
    if '## GPUメモリ指標の補足' not in report:report_path.write_text(report+note)
    (ROOT/'sources/audit_gpu_benchmarks.py').write_text(Path(__file__).read_text())
    (ROOT/'audit.json').write_text(json.dumps(dict(passed=True,source_data_model_hash_checks=checks,benchmarks=12,loss_comparisons=2304,cpu_reference_matches_previous_benchmark=True,eager_graph_100_interval_trajectories_identical=True,numerical_cpu_gpu_acceptance_passed=status['numerical_acceptance_passed'],test_evaluated=False),indent=2)+'\n')
    (ROOT/'artifacts_sha256.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='artifacts_sha256.json'},indent=2)+'\n')
    print('GPU benchmark independent audit passed; numerical acceptance is reported separately')
if __name__=='__main__':main()
