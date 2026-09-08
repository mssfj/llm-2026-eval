> 2026-09-08：ユーザー指示で終了。実行中プロセスなし。以下は実行開始時の履歴です。確定状態はstatus.json、停止時集計はstop_audit.json、総括は../../ENGINE_OPTIMIZATION_SUMMARY.mdを参照。

# Fast engine execution and limits

Results: `tdt_mnist/results/fast-engine-e17a-20260908/`.
Preregistration: `a4a69ae`; frozen implementation and preflight amendment: `0637b12`.

`eval/.venv/bin/python tdt_mnist/run_fast_engine.py run` runs sequential100-interval naive/fast timing at8 and16 blocks, a12000-interval naive timing replay (no test), isolated fast seed0, then parallel fast seeds1/2, followed by manifest/log checks and final reporting. Do not restart into this nonempty directory. Read `benchmark_progress.json`, then per-seed `progress.json`, and final `status.json`. The process is already running.

`--engine naive` calls the existing train.epoch directly. `--engine fast` precomputes ordered candidate losses and invokes the original epoch's unchanged Python code with a private loss binding. No process-global monkey patch and no legacy source edits. Candidate generation and stochastic rounding keep their original interleaved random stream. Minibatches remain128 examples **per pair**, rather than one shared batch per interval. Global16-coordinate blocks may affect multiple matrices; all affected matrices get a sparse correction.

Level1 tests cover20 fixed blocks ×128 losses, every matrix restart, repeated output coordinates, exact fixed-signal vote/counter/action comparisons, and real legacy epoch RNG/state checks. The unguarded engine failed: maximum relative error7.64477e-4. The guarded production engine passed these tests:1.77193e-7 and zero vote/counter/fire mismatches. A32 unguarded algebra control:2.09097e-7. These are empirical checks, not proof for arbitrary FP32 inputs.

The guard marks suffix A8 inputs with normalized distance<1e-4 from any half-integer and re-evaluates that candidate through the original naive loss. The threshold is fixed before full training. All128 candidates fell back in19 of20 level1 cases; the output-matrix-only case needed0. This is a numerical safety fallback, **not a successful speed claim**. The raw requested optimizations failed the numerical criterion, and the guarded path may be slower and use more memory. Both outcomes are retained.

Peak process RSS includes loaded data, PyTorch, allocator retention and earlier data-loading peaks. Results explicitly provide baseline RSS and baseline lifetime peak so these cannot be misread as live cache bytes. Logical INT8 weights, C8 counters and INT32 visit counts are reported separately, together with cache tensor byte estimates. Evidence/count arrays are actually transient per interval despite being listed as logical counter state. Cache estimates sum references and can overcount aliases. Full-run actual engine times are separate from validation/checkpoint wall time and100-interval extrapolations.

Strong reproducibility is checked against a naive replay. When all128 losses were evaluated by the original loss function from identical weights/scale/RNG, the shared original epoch directly certifies that interval; recomputing it again would be redundant. Other intervals use an independent original epoch until the first firing divergence. Historical E17a candidate differences, counters, scale, selected matrices and final weights are checked. Report first numeric drift separately from the first fired-coordinate/target divergence. Test is evaluated once after12000 intervals and not used to change code or settings.
