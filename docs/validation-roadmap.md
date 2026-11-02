# NBA simulation validation roadmap

This is an implementation tracker, **not** a claim of predictive validity.

## Release gates

1. **Reproducibility:** Record source commit, Python environment, input snapshots, simulation seeds, and forecast creation time. Run the complete pytest suite in CI.
2. **Point-in-time correctness:** Every historical feature must carry a source availability timestamp. Reject missing timestamps, not just game dates. Audit all joins and cache refreshes. The `backtest.point_in_time` helper is an initial enforcement primitive; it is **not yet wired into all ingestion/feature builders**.
3. **Historical replay parity:** Replace `backtest.walkforward._composite_spec` synthetic five-player rosters with game-date `GameSpec` reconstruction from archived pregame inputs. Do not claim full-pipeline validation while synthetic inputs remain.
4. **Settlement:** Use recorded bet-side odds, correct spread and total pushes, and distinguish settled bets from ungraded predictions. Record both actual and assumed prices explicitly.
5. **Player props:** Store pregame player prop lines, timestamps, settled outcomes and prior-only simulation probabilities. Report calibration, log loss, Brier score, MAE/RMSE for projected stats, sample sizes, and uncertainty.
6. **Baseline comparisons:** Compare to market implied probabilities and simple historical models using fixed train/validation/holdout windows. Pre-register selection thresholds before inspecting holdout results.
7. **Minutes/injuries:** Validate 240 regulation team minutes plus overtime, player availability scenarios, usage redistribution, and distribution coverage on unseen games.
8. **Performance:** Benchmark runtime, peak memory, convergence by simulation count, and deterministic seeds. Preserve representative regression fixtures.
9. **Release:** Require green CI, no known leakage, reproducible artifacts, and an explicit review of calibration and out-of-sample results.

## Initial implementation status

- [x] Dedicated review branch
- [x] Timezone-aware point-in-time filtering primitive and tests
- [x] Python pytest CI workflow
- [ ] Integrate availability timestamps throughout all historical feature paths
- [ ] Real-player historical replay parity
- [ ] Correct and validate all push/odds settlement cases
- [ ] Historical player prop outcome archive and calibration
- [ ] Minutes and injury scenario validation
- [ ] Run CI and reproduce independent out-of-sample benchmarks

## Data requirements

Archive raw source payloads with `source`, `observed_at`, `available_at`,
`prediction_at`, `game_id`, `model_version`, and `snapshot_id`.
Do not substitute the latest corrected season aggregate for a historical pregame
snapshot. Missing provenance must be reported as unevaluable rather than filled
with hindsight.
