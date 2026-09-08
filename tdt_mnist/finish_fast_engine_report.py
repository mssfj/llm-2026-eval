"""Read-only post-training diagnostics and final Japanese report; never tunes engine."""
import csv,json,math,time,sys,subprocess
from pathlib import Path
import torch
from run_fast_engine import ROOT,OLD,HERE
from run_residual_e17 import setup,config,load_data,write_csv,dump,sha
from residual_stream import ResidualStreamModel
from fast_engine import Schedule,candidate_losses
from test_fast_engine import decide
import train

def numerical_details():
    setup();a=config(0,HERE/'data');(x,y),_,_=load_data(a,torch.device('cpu'))
    m=ResidualStreamModel();state=torch.load(OLD/'per_seed/E17a-seed0/model.pt',weights_only=False);m.weights.copy_(state['weights'] if isinstance(state,dict) else state)
    g=torch.Generator().manual_seed(913);idx=torch.randperm(m.num_params,generator=g)[:16];cases=[('global',idx)];offset=0
    for l,shape in enumerate(m.shapes):
        cases.append((f'matrix{l}',torch.randperm(math.prod(shape),generator=g)[:16]+offset));offset+=math.prod(shape)
    cases.append(('same_output_row',torch.arange(16)))
    rows=[];diffs=[]
    for name,idx in cases:
        g=torch.Generator().manual_seed(117);weights=[]
        for _ in range(64):weights.extend(train.candidate_pair(m.weights,idx,g)[:2])
        plan=Schedule(idx,torch.arange(128).repeat(64,1),torch.stack(weights))
        naive=torch.stack([train.loss(m,x[:128],y[:128],w) for w in weights]);raw,_=candidate_losses(m,x,y,plan,guard=False);guarded,meta=candidate_losses(m,x,y,plan)
        for i in range(128):rows.append(dict(case=name,candidate=i,pair=i//2,orientation='plus' if i%2==0 else 'minus',naive_loss=float(naive[i]),unguarded_loss=float(raw[i]),guarded_loss=float(guarded[i]),unguarded_relative_error=float(abs(raw[i]-naive[i])/abs(naive[i])),guarded_relative_error=float(abs(guarded[i]-naive[i])/abs(naive[i]))))
        nd=decide(m,plan,naive);rd=decide(m,plan,raw)
        for pair,coord,edge in (nd[0]!=rd[0]).nonzero().tolist():
            diffs.append(dict(case=name,kind='vote',pair=pair,coordinate=int(idx[coord]),edge=edge,naive=int(nd[0][pair,coord,edge]),unguarded=int(rd[0][pair,coord,edge])))
        for coord,edge in (nd[1]!=rd[1]).nonzero().tolist():
            diffs.append(dict(case=name,kind='final_counter',pair='',coordinate=coord,edge=edge,naive=int(nd[1][coord,edge]),unguarded=int(rd[1][coord,edge])))
    write_csv(ROOT/'numerical/candidate_losses.csv',rows)
    if diffs:write_csv(ROOT/'numerical/unguarded_decision_mismatches.csv',diffs)
    assert max(r['guarded_relative_error'] for r in rows)<1e-5


def main():
    if '--wait' in sys.argv:
        while not (ROOT/'report.json').exists() or not json.loads((ROOT/'status.json').read_text()).get('complete'):
            time.sleep(10)
    report=json.loads((ROOT/'report.json').read_text())
    for n,h in json.loads((ROOT/'manifest.json').read_text())['sources'].items():assert sha(ROOT/'sources'/n)==h and sha(HERE/n)==h,n
    numerical_details()
    benches=report['benchmarks'];runs=report['runs']
    def bench(engine,blocks,count=100):return next(r for r in benches if r['engine']==engine and r['blocks']==blocks and r['intervals']==count)
    n8,f8,n16,f16=bench('naive',8),bench('fast',8),bench('naive',16),bench('fast',16)
    full=bench('naive',8,12000);seed0=next(r for r in runs if r['seed']==0)
    speed=n8['seconds_per_interval']/f8['seconds_per_interval'];fullspeed=full['actual_12000_engine_seconds']/seed0['engine_seconds']
    rows=[dict(blocks=b,naive_seconds_per_interval=n['seconds_per_interval'],fast_seconds_per_interval=f['seconds_per_interval'],speedup=n['seconds_per_interval']/f['seconds_per_interval'],naive_peak_rss_mib=n['peak_rss']/2**20,fast_peak_rss_mib=f['peak_rss']/2**20,peak_increase_mib=(f['peak_rss']-n['peak_rss'])/2**20,naive_logical_state_mib=n['logical_total_bytes']/2**20,fast_logical_state_mib=f['logical_total_bytes']/2**20,cache_estimate_mib=f['max_cache_tensor_bytes']/2**20,guard_fraction=f['guard_fallback_mean']/128) for b,n,f in [(8,n8,f8),(16,n16,f16)]]
    write_csv(ROOT/'benchmark/comparison.csv',rows)
    lines=['# TDT候補評価エンジン最適化・E17a再現検証','', '## 結論','',f'- 数値同一性：補完付きfastの固定重みテストは合格（最大相対誤差 {report["level1"]["max_relative_error"]:.9g}、投票・カウンタ・発火不一致0件）。補完なしは最大7.64477e-4で失敗。FP32全入力に対する数学的同一性の証明ではない。',f'- 速度：8ブロック100区間ではnaive/fast = {speed:.3f}倍。1未満は低速化を意味し、高速化の目的は達成していない。',f'- メモリ：8ブロックのピークRSSはnaive {n8["peak_rss"]/2**20:.2f} MiB、fast {f8["peak_rss"]/2**20:.2f} MiB（増加 {(f8["peak_rss"]-n8["peak_rss"])/2**20:.2f} MiB）。','', '## 実装と数値上の限界','', '既存train.py等は変更せず、fast_engine.pyから元のepochコードをprivate globalsの損失関数差し替えで再利用した。候補生成、確率的投票、カウンタ、タイブレーク、S更新、乱数系列は同じコードパス。64候補対ごとの異なる128例を維持し、16辺がまたがる全行列に疎な補正を適用する。','', '依頼の「区間につきベース1回」は8192例を連結した1呼び出しで実装した。128例1回分の計算量ではなく64回分である。また後段行列も摂動されるため、後段を全て共通重みとみなすことはできない。これらを省略すると元の学習則が変わる。','', '低ランク補正は実数演算では同値だがFP32の結合順が変わる。A32対照の誤差は2.09097e-7。A8では丸め境界で小差が増幅されることと整合し、補完なし固定重み試験が失敗した。境界から1e-4以内の候補は元のnaive lossで再評価する。20固定ブロック中19では128候補全てが再評価され、速度利益を失った。閾値をtest結果で調整していない。','', '全128候補損失はnumerical/candidate_losses.csv、不一致の区間内pair・座標・edge・投票/カウンタ値はnumerical/unguarded_decision_mismatches.csv。固定重みは保存済みE17a seed0、128学習例、各行列単独・全体ブロック・同一出力行を含む20ケース。','', '## レベル2：3 seed再現','',f'test平均±標本標準偏差：{report["test_mean_percent"]:.4f} ± {report["test_sample_std_percent"]:.4f}%。最強の発火系列一致判定：{report["strong_pass"]}。90.637±0.3ptの平均精度基準：{report["fallback_accuracy_pass"]}。','', '| seed | fast test % | naive test % | 差 pt | 最初の発火分岐 | 全損失naiveの区間 | 独立naive再生区間 |','|---|---:|---:|---:|---|---:|---:|']
    for r in runs:lines.append(f'| {r["seed"]} | {r["test_percent"]:.2f} | {r["naive_test_percent"]:.2f} | {r["delta_pp"]:+.2f} | {r["first_firing_divergence"]} | {r["certified_original_loss_intervals"]} | {r["independently_replayed_intervals"]} |')
    lines+=['','各区間の発火座標・遷移先はper_seed/seed*/firing.csv。全候補損失が元のnaive関数で評価された区間は、同一状態・乱数から元の判定コードを直接実行して一致を確認。それ以外はnaiveを独立再生して最初の発火分岐を検出する。数値差が出た最初の区間と発火分岐は区別してsummary.jsonに保存。既存記録の|y|、S、カウンタ、層別選択・発火と最終重みも照合。','', '## 性能・メモリ・深さ依存','', '| blocks | naive 秒/区間 | fast 秒/区間 | 速度倍率 | naive peak MiB | fast peak MiB |','|---|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f'| {r["blocks"]} | {r["naive_seconds_per_interval"]:.6f} | {r["fast_seconds_per_interval"]:.6f} | {r["speedup"]:.3f} | {r["naive_peak_rss_mib"]:.2f} | {r["fast_peak_rss_mib"]:.2f} |')
    lines += ['',f'16/8ブロックの区間時間比：naive {n16["seconds_per_interval"]/n8["seconds_per_interval"]:.3f}倍、fast {f16["seconds_per_interval"]/f8["seconds_per_interval"]:.3f}倍。100区間ベンチマークは学習runの精度集計に含めない。',f'実測12000区間の純エンジン時間：naive {full["actual_12000_engine_seconds"]:.2f}秒、fast seed0 {seed0["engine_seconds"]:.2f}秒、倍率 {fullspeed:.3f}倍。両者は他の再現workerを起動する前に逐次実行した。100区間からの外挿値はbenchmark JSONでestimated_12000_secondsとして区別する。','','| seed | 学習全実時間 秒 | エンジン合計 秒 | 独立naive比較 秒 |','|---|---:|---:|---:|']
    for r in runs:lines.append(f'| {r["seed"]} | {r["train_elapsed_seconds"]:.2f} | {r["engine_seconds"]:.2f} | {r["reference_replay_seconds"]:.2f} |')
    lines+=['', '学習全実時間はvalidation・記録・チェックポイントを含む。seed1/2は並列で、資源共有をruntime_workers.jsonに記録。seed0以外の実時間比を公平な速度差と解釈しない。', '',f'8ブロックの論理状態：INT8重み {n8["logical_weights_bytes"]} bytes、C8証拠＋INT32訪問数 {n8["logical_counter_bytes"]} bytes、計 {n8["logical_total_bytes"]/2**20:.4f} MiB（両engine同じ）。カウンタ配列は実装上は区間ごとに確保・リセットする。一時キャッシュのテンソル参照合計は {f8["max_cache_tensor_bytes"]/2**20:.3f} MiB、候補活性1枚16384×76 FP32は4.75 MiB。キャッシュ合計は別名参照を重複計上しうる。','RSSはPython/PyTorch・データ・アロケータを含むプロセス指標。生涯ピークには学習開始前のデータ読み込みも含まれるため、rss_before_workloadとpeak_rss_before_workloadを併記した。RSS差を厳密な生存キャッシュピークと同一視しない。','',f'8ブロックの依頼式（1+128×suffix/L）平均は {f8["requested_ideal_forward_equivalent_mean"]:.3f} forward相当。実際は64バッチ分のbaseと、被摂動行列補正・suffix・naive補完があり、MAC重み付けのdense matmul換算は {f8["actual_dense_matmul_equivalent_mean"]:.3f}。後者は量子化・補正・候補生成費用を除く理論値で、実時間から推定したforward回数ではない。','', '## 監査物','', 'per_seedには設定、初期val、全区間CSV、候補差NPY、発火座標、層別診断、初期/最終プローブ、checkpoint、最終モデル、summaryとmanifest。sourcesには学習時コードと事前登録。元E17記録とソース/データのハッシュを保持。testは最終12,000区間後に各seed1回のみ評価し、エンジン調整やモデル選択に使用していない。']
    (ROOT/'README.md').write_text('\n'.join(lines)+'\n')
    (ROOT/'implementation.patch').write_text(subprocess.check_output(['git','diff','a4a69ae','0637b12','--','tdt_mnist/fast_engine.py','tdt_mnist/run_fast_engine.py','tdt_mnist/test_fast_engine.py'],text=True))
    shutil_path=ROOT/'sources/finish_fast_engine_report.py';shutil_path.write_text(Path(__file__).read_text())
    dump(ROOT/'final_report_validation.json',dict(passed=True,source_hashes_verified=True,numerical_detail_rows=2560,level1_pass=report['level1']['passed'],level2_pass=report['level2_pass'],acceleration_achieved=speed>1 and fullspeed>1,short_speedup=speed,actual_full_speedup=fullspeed))
    dump(ROOT/'artifacts_sha256.json',{str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='artifacts_sha256.json'})
    print('Final Japanese report and numerical mismatch details saved',flush=True)
if __name__=='__main__':main()
