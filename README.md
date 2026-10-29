# Stridewise — deterministic statistics engine

**Statistics decide what is true.** This repository holds the deterministic
layer of Stridewise, a hackathon project (Meta Global AI Developer
Hackathon, Oct 26 – Nov 4, 2026): the normalized data model, the 20-question
insight registry, week-block bootstrap likely ranges, evidence tiers with
density/recency caps, demotion display, and the fact-pack output that the
narrative layer (built separately) reads. This layer makes **zero API
calls** and contains **no real athlete data**.

## What it is

Stridewise never tells you what to do. It tells you what your data can
prove: every insight is a registered question with a fixed method, plain
language, no causal verbs, and every number carrying a bootstrap 95%
"likely range" and the n it rests on. Tiers (solid / emerging / no
pattern / not enough data yet) are recomputed from scratch on every
refresh, never sticky; a tier drop is displayed as a demotion with its
reason.

## Layout

- `engine/model.py` — normalized athlete model. GPS/routes are stripped
  at import and never stored; this module has no coordinate field.
- `engine/stats.py` — OLS slopes, week-block bootstrap ranges, the tier
  gate (statistical tier, density cap, recency cap, demotion detection).
- `engine/weekly.py` — weekly aggregation: mileage, ACWR, monotony, and
  the locked long-run rule (22.5% of week mileage AND 18% of the prior
  4-week average; longest qualifying non-race run).
- `engine/registry.py` — the 20 registry questions, pair builders,
  card sentences, and race-context cards.
- `engine/factpack.py` — fact-pack assembly (the only thing the
  narrative layer ever sees).
- `engine/calibrate.py` — in-app course calibration for race card
  C1 (see "Course calibration" below). The one module that touches
  the network; the statistics engine itself stays zero-API,
  zero-network.
- `adapters/garmin.py` — Garmin GDPR "Download Your Data" JSON
  summaries -> normalized JSON (activities, sleep, daily RHR).
  Reads whitelisted scalar fields only; never opens FIT archives,
  never reads coordinates, routes, or location names.
- `engine/validate.py` — fixture validation runner.
- `tests/test_engine.py` — fixture recovery, cap, and demotion tests.
- `fixtures/validation-fixture-1.json` — SYNTHETIC validation data
  (60 nights, seed 42). Never a demo profile.

## Run it

```bash
python -m engine.validate            # fixture validation report
python -m unittest discover -s tests -v
```

## Course calibration (C1)

A race entry in a normalized profile can carry a `results_url`
(TFRRS meet-results page) and a `calibration_table` (a substring of
that page's individual-results heading, e.g. `"Men 8k Champ"`).
Any such race is calibratable by the app itself:

```bash
python -m engine.calibrate <normalized.json> <race-date>
```

The run fetches the meet page, extracts the named
individual-results table (finishers = rows with a parseable TIME
column; DNF/no-time rows are excluded from the field size), then
fetches each finisher's TFRRS profile for their best OTHER
same-season mark at the race's distance — event labels are chosen
from the race entry's `distance_m` (8000 → the "8k" and "8.001k"
labels combined; 5000 → "5k"), this meet itself excluded.
Per finisher,
`pct_slower = (race_time − best_other) / best_other × 100`.
`field_size` and `matched_finishers` are always written back to
the race entry as real counts; the median pct is written back as
`field_median_slow_pct` only when the display bar is met
(matched ≥ 40 AND coverage ≥ 0.35) — the same bar the C1 card
applies when rendering. If the same-season pass fails the bar, a
fallback pass matches against pure last-season bests at the same
labels (the pools are never mixed) under the same bar; the race
entry's `reference_season` records which pool produced the shown
result, and the card's sentence says "last season's bests" when
the fallback produced it.

Fetch discipline: stdlib urllib, descriptive User-Agent,
sequential requests with a short delay, and every page cached
under `~/workspace/hackathon/cache/tfrrs/` (workspace-only, never
in the repo) keyed by URL hash and reused on re-runs. An HTTP 429
or an obvious block page stops the run immediately with nothing
written back — a partial matched count is reported as partial,
never as final. Individual non-429 failures count that athlete as
unmatched and the run continues.

## Code-only rule (from commit one)

This repo contains code + synthetic sample data ONLY. No raw exports,
no normalized real-athlete data, no contributor names or labels. Real
profiles are injected at deploy time from the workspace, never committed.

## Interpretation decisions (registry ambiguities, resolved in code)

- **Completeness** is usable pairs / collected pairs inside the window
  (share surviving sensor exclusion). A window-coverage reading was
  rejected: it would mark strong-evidence profiles insufficient (112
  paired nights in 180 days is 62% window coverage yet clears every
  other bar). Spread is handled by the density rule.
- **A7's sleep-regularity index** is the rolling 7-day SD of sleep
  midpoint (bedtime + duration/2). Scatter (A7) stays distinct from
  systematic weekend shift (A8) by construction.
- **B4's fade** needs lap/split data, which the normalized model does
  not carry: B4 reports insufficient_data in v1, shown, not dropped.
- **B5** (regularity -> fitness trend) is the most assumption-heavy
  question: fitness trend is the slope of weekly mean pace over the
  trailing 4 weeks.
- **C4's flat-equivalent range** comes from a prediction model that is
  not part of this track: C4 emits the result with
  `prediction_range = null` and an explicit reason, never a fabricated
  range.
- **Garmin adapter decisions** (verified against the export, not
  assumed): activity duration is in milliseconds and distance in
  centimeters (distance/duration reproduces stored avgSpeed on
  1151/1157 checkable records; the 6 exceptions are lap_swimming,
  where Garmin's avgSpeed uses a different basis; no run affected).
  Activity dates are local dates from startTimeLocal (local
  wall-clock packed into epoch ms). Sleep GMT timestamps convert to
  America/New_York; the converted local end date equals calendarDate
  on all 755 records, confirming calendarDate is the wake date (the
  adapter still trusts the conversion, not the label). Sleep
  duration is deep+light+REM seconds only; efficiency is null (no
  reported field, no proxy derived). Daily RHR is UDS
  restingHeartRate, the finalized day value, not
  currentDayRestingHeartRate (intraday estimate; differs on 520/655
  days by 1-3 bpm). HRV is null: no HRV field exists in UDS or
  bioMetrics_latest. Garmin's own race predictions, training
  history, and scores are deliberately never read.
- **Load -> sleep questions are OUT by design** and are never offered.
- Windows anchor at the profile's latest data date (frozen imports),
  not the wall clock.

## What is NOT here (by design)

No sleep score, no composite scores, no weight/manual-entry/
planned-workout features, no live sync, no chat, no prescriptions.
Descriptive insights only.
