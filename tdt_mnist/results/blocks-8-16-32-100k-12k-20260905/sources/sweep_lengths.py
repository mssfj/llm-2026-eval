"""Compare block sizes and training lengths using the existing TDT trainer."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lengths', nargs='+', type=int, default=[3000, 6000, 12000])
    p.add_argument('--blocks', nargs='+', type=int, default=[64, 128, 256])
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--threshold', type=int, default=8)
    p.add_argument('--workers-per-length', type=int, default=3)
    p.add_argument('--data-dir', type=Path, default=Path('/tmp/tdt-mnist-data'))
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--report-dir', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if min(args.lengths) <= 0 or len(set(args.lengths)) != len(args.lengths):
        p.error('lengths must be positive and unique')
    if args.workers_per_length <= 0:
        p.error('workers-per-length must be positive')
    source = Path(__file__).resolve().parent
    manifest = {k: str(v.resolve()) if isinstance(v, Path) else v
                for k, v in vars(args).items() if k not in ('resume', 'workers_per_length')}
    manifest.update(num_params=100000, activation_precision='a32', measurements=64,
                    batch_size=128, train_size=10000, val_size=1000, test_size=10000,
                    pool_shape=[9, 10], hidden_size=1000, max_fires=1,
                    data_seed=0, batch_seed_rule='seed + 100000')
    files = ['train.py', 'sweep.py', 'sweep_lengths.py', 'activation_quantization.py']
    manifest['source_sha256'] = {f: hashlib.sha256((source / f).read_bytes()).hexdigest() for f in files}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / 'manifest.json'
    if args.resume:
        if not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest:
            p.error('resume requires matching settings and source code')
    elif any(args.output_dir.iterdir()):
        p.error('output-dir must be empty')
    args.report_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    (args.report_dir / 'manifest.json').write_text(manifest_path.read_text())
    snapshots = args.report_dir / 'sources'
    snapshots.mkdir(exist_ok=True)
    for f in files:
        shutil.copy2(source / f, snapshots / f)

    def run(length):
        command = [sys.executable, str(source / 'sweep.py'), '--pool-shape', '9', '10',
                   '--hidden-size', '1000', '--expected-params', '100000',
                   '--blocks', *map(str, args.blocks), '--thresholds', str(args.threshold),
                   '--seeds', *map(str, args.seeds), '--steps', str(length),
                   '--measurements', '64', '--batch-size', '128', '--train-size', '10000',
                   '--val-size', '1000', '--eval-every', '500',
                   '--workers', str(args.workers_per_length), '--data-dir', str(args.data_dir.resolve()),
                   '--output-dir', str((args.output_dir / f'steps{length}').resolve()),
                   '--report-dir', str((args.report_dir / f'steps{length}').resolve())]
        if args.resume:
            command.append('--resume')
        with (args.output_dir / f'steps{length}.log').open('a' if args.resume else 'w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                           env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                                'OPENBLAS_NUM_THREADS': '1'})
        print(f'Completed length={length}', flush=True)

    with ThreadPoolExecutor(max_workers=len(args.lengths)) as pool:
        list(pool.map(run, args.lengths))
    for filename in ('per_seed.csv', 'aggregate.csv', 'counter_histograms.csv'):
        combined = []
        for length in sorted(args.lengths):
            combined.extend({'steps': length, **r} for r in read_csv(args.report_dir / f'steps{length}' / filename))
        with (args.report_dir / filename).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(combined[0]))
            writer.writeheader()
            writer.writerows(combined)
    groups = read_csv(args.report_dir / 'aggregate.csv')
    text = ['# MNIST TDT: 100k・FP32のブロックサイズと学習区間比較', '',
            f'90→1000→10、バイアスなし、三値重み100,000個、活性化・積和・損失FP32。seed={args.seeds}。',
            f'発火閾値{args.threshold}、K=64、batch=128、最大発火1重み/区間。区間ごとにカウンタをリセット。',
            '訓練10,000件、検証1,000件、テスト10,000件。データ分割seed=0。逆伝播は使わない。',
            '学習区間はカウンタ蓄積・更新の単位で、データ全体を一巡するepochではない。',
            '各長さを初期状態から独立に実行。同じseedの初期重み・バッチ乱数列を共有。',
            '同じ区間数では訓練forward回数が等しく、区間数の比較では学習量そのものが異なる。',
            '値は3seedの平均±標本標準偏差。テスト評価を更新・停止・条件選択に使わない。', '',
            '| 区間 | block | 検証精度 % | テスト精度 % | 発火数 |',
            '| ---: | ---: | ---: | ---: | ---: |']
    for r in groups:
        text.append(f"| {r['steps']} | {r['block_size']} | "
                    f"{100*float(r['val_accuracy_mean']):.2f} ± {100*float(r['val_accuracy_std']):.2f} | "
                    f"{100*float(r['test_accuracy_mean']):.2f} ± {100*float(r['test_accuracy_std']):.2f} | "
                    f"{float(r['total_fires_mean']):.1f} |")
    text.extend(['', 'カウンタの最大・平均・絶対値平均・容量・飽和率・発火数・層別更新数はper_seed.csv、',
                 '符号付き分布はcounter_histograms.csv、条件別の詳細はsteps*/README.md。',
                 'カウンタ分布は各区間末の測定済み辺だけを集計。INT8容量±127。',
                 f'詳細ログ・設定・重み: `{args.output_dir.resolve()}`', ''])
    (args.report_dir / 'README.md').write_text('\n'.join(text))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, metric, title in zip(axes, ['val_accuracy', 'test_accuracy'], ['Validation accuracy', 'Test accuracy']):
        for block in args.blocks:
            selected = [r for r in groups if int(r['block_size']) == block]
            ax.errorbar([int(r['steps']) for r in selected],
                        [100 * float(r[metric + '_mean']) for r in selected],
                        yerr=[100 * float(r[metric + '_std']) for r in selected],
                        marker='o', capsize=4, label=f'block={block}')
        ax.set(xlabel='Accumulation intervals', ylabel='Accuracy (%)', title=title)
        ax.set_xticks(sorted(args.lengths))
        ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle('TDT MNIST: 100,000 ternary weights, FP32 activations, threshold=8, K=64')
    for ext in ('png', 'svg'):
        fig.savefig(args.report_dir / f'comparison.{ext}', dpi=160)
    plt.close(fig)
    print(f'Report: {args.report_dir / "README.md"}', flush=True)


if __name__ == '__main__':
    main()
