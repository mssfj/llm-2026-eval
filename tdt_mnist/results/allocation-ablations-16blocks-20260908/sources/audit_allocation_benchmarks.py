"""Independently verify measured traces and source snapshots; no model/test evaluation."""
import csv,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE/'results/allocation-ablations-16blocks-20260908'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return list(csv.DictReader(p.open()))
def main():
    assert json.loads((ROOT/'status.json').read_text())['complete']
    manifest=json.loads((ROOT/'manifest.json').read_text());hashes=0
    for n,h in manifest['sources'].items():
        assert sha(ROOT/'sources'/n)==h and sha(HERE/n)==h,n;hashes+=2
    old=HERE/'results/residual-followups-e18-e20-20260908'
    for seed,h in manifest['trained_models'].items():assert sha(old/'per_seed'/f'E18a-seed{seed}'/'model.pt')==h;hashes+=1
    for n,h in manifest['data'].items():assert sha(HERE/'data/MNIST/raw'/n)==h;hashes+=1
    fields_checked=0
    for seed in range(3):
        naive=read(ROOT/'benchmarks'/f'seed{seed}-naive/intervals.csv')
        assert len(naive)==100
        for e in ['restore_cache','candidate_buffers']:
            candidate=read(ROOT/'benchmarks'/f'seed{seed}-{e}/intervals.csv');assert len(candidate)==100
            for n,c in zip(naive,candidate):
                assert all(n[k]==c[k] for k in ['step','fires','scale']);fields_checked+=3
                assert float(c['seconds'])>0
    validation=read(ROOT/'validation.csv');assert len(validation)==36
    for r in validation:
        assert all(int(r[k])==0 for k in ['bitwise_loss_mismatches','vote_mismatches','counter_mismatches'])
        assert all(r[k]=='True' for k in ['proposal_equal','indices_equal','stats_equal','scale_equal','rng_equal'])
    # Also ensure neither old naive/fast engine experiment source was touched.
    previous=HERE/'results/fast-engine-e17a-20260908'
    for n,h in json.loads((previous/'manifest.json').read_text())['sources'].items():assert sha(HERE/n)==h;hashes+=1
    (ROOT/'sources/audit_allocation_benchmarks.py').write_text(Path(__file__).read_text())
    (ROOT/'audit.json').write_text(json.dumps(dict(passed=True,source_data_model_hashes_checked=hashes,interval_trace_fields_checked=fields_checked,loss_comparisons=4608,test_evaluated=False,existing_running_experiment_sources_unchanged=True),indent=2)+'\n')
    (ROOT/'artifacts_sha256.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='artifacts_sha256.json'},indent=2)+'\n')
    print('Independent source and trace audit passed')
if __name__=='__main__':main()
