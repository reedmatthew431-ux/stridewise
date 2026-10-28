"""Weekly aggregation for the training-load family.

Week = Monday-start ISO week. Mileage counts RUN activities only
(kind == "run"); races are included in mileage (the long-run rule's
denominator is "week total as displayed, races included").

Long-run rule (LOCKED 2026-10-26): a run counts toward long-run
candidacy if it is (1) at least 22.5% of its week's total mileage AND
(2) at least 18% of the athlete's average weekly mileage over the
prior 4 weeks (available-weeks average when fewer than 4 exist). The
week's long run is the LONGEST qualifying NON-RACE run; if none
qualifies, the week has no long run ("--", never forced).
Distance-based for v1.

ACWR: acute = the week's mileage; chronic = mean weekly mileage over
the prior 4 weeks. Monotony (Foster): mean daily mileage / population
SD of daily mileage over the week; undefined when SD is 0 (skipped).
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import date, timedelta

from .model import monday_of

LONG_RUN_WEEK_SHARE = 0.225
LONG_RUN_PRIOR_AVG_SHARE = 0.18


@dataclass
class Week:
    monday: date
    mileage_m: float = 0.0
    run_days: int = 0
    daily_mileage: list = field(default_factory=lambda: [0.0] * 7)
    mean_rhr: float | None = None
    mean_pace: float | None = None  # distance-weighted sec/km, runs only
    long_run_m: float | None = None
    long_run_share: float | None = None
    acwr: float | None = None
    monotony: float | None = None


def build_weeks(athlete, n_weeks=26):
    """Build weekly aggregates ending at the athlete's anchor week.

    Returns weeks sorted by Monday, oldest first. Weeks are built for
    the trailing 26-week window; a week with no runs still appears
    (mileage 0) when it falls inside the athlete's activity span, so
    rest weeks are real zeros, not missing data. Weeks outside the
    activity span are omitted.
    """
    anchor = athlete.anchor_date
    if anchor is None:
        return []
    last_monday = monday_of(anchor)
    first_monday = last_monday - timedelta(days=7 * (n_weeks - 1))

    runs = athlete.runs
    if runs:
        act_first = monday_of(min(a.date for a in runs))
        act_last = monday_of(max(a.date for a in runs))
    else:
        act_first = act_last = None

    daily_rhr = athlete.daily_by_date
    weeks = []
    monday = first_monday
    while monday <= last_monday:
        if act_first is not None and not (act_first <= monday <= act_last):
            monday += timedelta(days=7)
            continue
        wk_runs = [a for a in runs if monday_of(a.date) == monday]
        daily_mileage = [0.0] * 7
        for a in wk_runs:
            if a.distance_m:
                daily_mileage[(a.date - monday).days] += a.distance_m
        mileage = sum(daily_mileage)
        run_days = sum(1 for v in daily_mileage if v > 0)
        rhr_vals = [daily_rhr[monday + timedelta(days=i)].resting_hr_bpm
                    for i in range(7)
                    if monday + timedelta(days=i) in daily_rhr
                    and daily_rhr[monday + timedelta(days=i)].resting_hr_bpm is not None]
        mean_rhr = statistics.mean(rhr_vals) if rhr_vals else None
        total_m = sum(a.distance_m for a in wk_runs if a.distance_m)
        mean_pace = None
        if total_m > 0:
            mean_pace = sum(a.pace_sec_per_km * a.distance_m for a in wk_runs
                            if a.pace_sec_per_km and a.distance_m) / total_m
        weeks.append(Week(
            monday=monday, mileage_m=mileage, run_days=run_days,
            daily_mileage=daily_mileage, mean_rhr=mean_rhr,
            mean_pace=mean_pace,
        ))
        monday += timedelta(days=7)

    # Long runs, ACWR, monotony -- need prior weeks, so second pass.
    for i, wk in enumerate(weeks):
        prior = [w.mileage_m for w in weeks[max(0, i - 4):i]]
        prior_avg = statistics.mean(prior) if prior else None
        wk_runs = [a for a in runs if monday_of(a.date) == wk.monday
                   and not a.is_race and a.distance_m]
        candidates = []
        for a in wk_runs:
            share_ok = wk.mileage_m > 0 and \
                a.distance_m >= LONG_RUN_WEEK_SHARE * wk.mileage_m
            floor_ok = prior_avg is None or \
                a.distance_m >= LONG_RUN_PRIOR_AVG_SHARE * prior_avg
            if share_ok and floor_ok:
                candidates.append(a.distance_m)
        if candidates:
            wk.long_run_m = max(candidates)
            wk.long_run_share = wk.long_run_m / wk.mileage_m if wk.mileage_m else None
        if prior_avg:
            wk.acwr = wk.mileage_m / prior_avg if prior_avg > 0 else None
        sd = statistics.pstdev(wk.daily_mileage) if len(wk.daily_mileage) > 1 else 0.0
        mean_daily = statistics.mean(wk.daily_mileage)
        wk.monotony = (mean_daily / sd) if sd > 0 else None
    return weeks
