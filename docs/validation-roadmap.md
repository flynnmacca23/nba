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

## Validation progress (2026-11-03)

- Walk-forward team ratings are computed for the entire game date before any
  same-day final score is learned. Updates to team strength, scoring and
  home-court advantage are order-independent across games on that date.
- Availability filtering rejects timezone-naive/date-only source timestamps.
  Snapshot manifests require explicit observation, availability and prediction
  timestamps and stable identities. These checks are **not yet integrated with
  every ingestion path**.
- Historical spread/total returns are only computed when side-specific American
  odds are actually stored. Missing prices remain unpriced forecasts, not
  hypothetical -110 bets. Spread and total pushes are ungraded for binary
  calibration and cannot be treated as losses.
- Archived player props with stored probabilities/outcomes are scored only if
  source, snapshot and pregame timing metadata are present and consistent.
  This is a metadata gate, not proof that the original prediction pipeline
  was leak-free.
- The existing team-market walk-forward still constructs synthetic five-player
  teams. Its output now declares `synthetic_team_only` and
  `unverified_historical_lines`. It is **diagnostic only**, not an
  independently validated betting-performance claim.

### Required next gates

1. Persist real pregame source payloads and replay real-player `GameSpec`
   instances from those immutable snapshots; do not rebuild historical games
   from today's corrected warehouse.
2. Capture bookmaker/market/side prices with observed and available timestamps
   and evaluate actual recorded executable prices rather than assumed closes.
3. Create a settled player-prop archive with independent box-score outcomes,
   held-out calibration and forecast-error metrics.
4. Validate injury/minutes scenarios and 240-minute team totals against
   historical games; benchmark simulation speed, memory and deterministic seeds.
5. Require green CI and out-of-sample release gates before considering a merge.

## Data requirements

Archive raw source payloads with `source`, `observed_at`, `available_at`,
`prediction_at`, `game_id`, `model_version`, and `snapshot_id`.
Do not substitute the latest corrected season aggregate for a historical pregame
snapshot. Missing provenance must be reported as unevaluable rather than filled
with hindsight.
