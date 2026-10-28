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
- `engine/validate.py` — fixture validation runner.
- `tests/test_engine.py` — fixture recovery, cap, and demotion tests.
- `fixtures/validation-fixture-1.json` — SYNTHETIC validation data
  (60 nights, seed 42). Never a demo profile.

## Run it

```bash
python -m engine.validate            # fixture validation report
python -m unittest discover -s tests -v
```

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
- **Load -> sleep questions are OUT by design** and are never offered.
- Windows anchor at the profile's latest data date (frozen imports),
  not the wall clock.

## What is NOT here (by design)

No sleep score, no composite scores, no weight/manual-entry/
planned-workout features, no live sync, no chat, no prescriptions.
Descriptive insights only.
