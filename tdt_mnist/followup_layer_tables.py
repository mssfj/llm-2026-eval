"""Readable final tables from audited per-layer CSVs."""
import csv,shutil
from pathlib import Path
from residual_followup_models import TDT_CONDITIONS


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def write_tables(root):
    ratios=read(root/'signal/rms_ratios_aggregate.csv')
    signals=read(root/'signal/metrics_aggregate.csv')
    firing=read(root/'firing/matrices_aggregate.csv')
    isolated=read(root/'signal/isolated_candidates_aggregate.csv')
    lines=['# E18/E19 層別一覧','','すべて3seed平均。標本標準偏差は対応する*_aggregate.csv。行列とブロックは0始まり。',
           'W1出力RMSはReLU前の線形出力。ReLU後・量子化点ごとの詳細はsignal/metrics.csv。','']
    for c,(blocks,width,precision,count) in TDT_CONDITIONS.items():
        lines += [f'## {c}: {blocks} blocks, width {width}, {count:,} weights','',
                  '| block | 加算前stream RMS | branch RMS | branch/stream |','| --- | ---: | ---: | ---: |']
        for r in sorted([r for r in ratios if r['condition']==c and r['step']=='12000'],key=lambda r:int(r['block'])):
            lines.append(f"| {r['block']} | {float(r['stream_rms_mean']):.5f} | {float(r['branch_rms_mean']):.5f} | {float(r['branch_stream_rms_ratio_mean']):.5f} |")
        lines += ['', '| 行列 | 出力RMS | 発火数 | 全区間発火率 % | 選択時発火率 % | 初期単独mean abs(y) | 最終単独mean abs(y) |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for r in sorted([r for r in firing if r['condition']==c],key=lambda r:int(r['layer'])):
            layer=r['layer']
            rms=float(next(x['rms_mean'] for x in signals if x['condition']==c and x['layer']==layer and x['step']=='12000' and x['stage']=='output'))
            vals=[float(next(x['mean_abs_y_mean'] for x in isolated if x['condition']==c and x['layer']==layer and x['stage']==stage)) for stage in ['initial','final']]
            lines.append(f"| {r['matrix']} | {rms:.5f} | {float(r['fires_mean']):.2f} | {100*float(r['all_interval_firing_rate_mean']):.4f} | {100*float(r['selected_interval_firing_rate_mean']):.4f} | {vals[0]:.7f} | {vals[1]:.7f} |")
        lines.append('')
    lines+=['## スケール閾値への該当','',
            '| 条件 | 比率>0.5 件数/全block×seed | 比率>0.5 割合 | logits RMS>10 run数 | logits RMS最大 |',
            '| --- | ---: | ---: | ---: | ---: |']
    for r in read(root/'signal/scale_warnings.csv'):
        lines.append(f"| {r['condition']} | {r['ratio_exceed_count']}/{r['ratio_total']} | {100*float(r['ratio_exceed_fraction']):.2f}% | {r['logits_exceed_count']} | {float(r['logits_rms_max']):.5f} |")
    (root/'LAYER_TABLES.md').write_text('\n'.join(lines)+'\n')
    shutil.copy2(Path(__file__),root/'sources'/Path(__file__).name)
