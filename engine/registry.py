"""Stridewise insight registry (v1, closed 2026-10-22; A9 added 2026-10-28;
A10/A11/B8/A12/A13/B9 added 2026-10-28):
27 questions.

Family A (sleep, 13): A1 bedtime -> duration; A2 bedtime variability ->
next-morning RHR; A3 duration -> next-day RHR; A4 duration -> next-day
run quality; A5 sleep consistency -> next-day run quality; A6 bedtime ->
efficiency (source-aware); A7 sleep-regularity (midpoint scatter) ->
RHR / run quality; A8 social jetlag -> Sunday-night duration / Monday
run quality; A9 awake time in bed -> next-morning RHR; A10 weekend
nights -> sleep length; A11 awake time in bed -> next-morning HRV.
Family B (training load, 9): B1 ramp (ACWR) -> next-week run quality;
B2 monotony -> next-week RHR trend; B3 mileage volatility -> following
-week run quality; B4 long-run share -> late-run fade; B5 training
regularity -> fitness trend; B6 heat sensitivity (temp -> pace-at-HR
penalty); B7 RHR early-warning (elevated morning RHR -> same-day
session quality).
Family C (race context, 5): C1 field calibration; C2 conditions
context; C3 sensor reliability; C4 result vs flat-equivalent range;
C5 pre-race sleep.

OUT BY DESIGN (never offered): load -> sleep questions ("does a hard
day ruin my sleep" is voided); weight insights; manual-entry data;
causal verbs anywhere in output copy.

Pair-construction decisions (documented, not in the registry draft):
- A2/A5/A7 rolling scatter uses the 7 nights ending on the pair night
  (inclusive), requiring >= 5 of 7 nights present.
- A7's "sleep-regularity index" is operationalized as the rolling 7-day
  SD of sleep midpoint (bedtime + duration/2). Scatter (A7) vs systematic
  weekend shift (A8) stay distinct by construction.
- A8 pairs weekly: jetlag = mean weekend midpoint minus mean weekday
  midpoint for nights with wake dates Mon-Sun of the prior week
  (weekend = wake Sat/Sun; needs >= 2 weekday + >= 1 weekend nights);
  outcomes are the Sunday night (wake date = following Monday) and
  that Monday's run quality. Pair date = that Monday.
- Run quality (v1 proxy): distance-weighted mean run pace (sec/km) for
  the day. HR-adjusted quality needs lap/split data, which the
  normalized model does not carry; the proxy is labeled as pace, not
  pace-at-HR, wherever it appears.
- B4's fade outcome needs lap/split data, so B4 is insufficient_data
  in v1 (long-run share itself is still computed for aggregates).
- B5's fitness trend is the OLS slope of weekly mean pace over the
  trailing 4 weeks (negative = improving). This is the most
  assumption-heavy question in the registry and is flagged as such.
- B6's baseline is mean pace of runs at 10-15 C (needs >= 3); the
  penalty is percent slower than that baseline, regressed on temp.
- B7's norm is the trailing-14-day median RHR (needs >= 7 prior days).
- A9's awake: Garmin reports awakeSleepSeconds in the sleep window
  (includes onset latency); Apple reports summed Awake segments
  (detected mid-sleep wakefulness; onset latency not captured in
  stages-era exports). InBed-minus-Asleep was tried for Apple and
  rejected: old InBed blocks span whole days (6-12h artifacts on 2023
  nights) and no InBed segments appear at all in 2026 windows.
- A10's weekend indicator is a binary driver (1 = wake Sat/Sun, 0 =
  weekday); the OLS slope IS the weekend-minus-weekday mean
  difference, so no separate diff statistic is needed. Week-block
  bootstrap handles the clustering.
- B8's outcome is next week's mean RHR LEVEL (not change, unlike B2's
  monotony -> trend). Driver is weekly run mileage in km (scale 10).
- A12 is the LEVEL sibling of A7's midpoint VARIABILITY question:
  driver is sleep-midpoint clock (bedtime_min_past_21 + duration/2,
  unwrapped so no midnight circularity). The three sweep-6 timing
  emergings (bedtime, wake, midpoint) are one collinear signal, so
  only midpoint is a question -- adding all three would triple-count
  one finding.
- A13 pairs each night with the previous night's duration (previous
  night must be the immediately preceding wake date). B9's driver is
  the week's qualifying long-run distance in km (scale 5); weeks
  without a qualifying long run contribute no pair.
- C4's flat-equivalent prediction range comes from the personal app's
  prediction model, which is NOT part of this track: C4 emits the
  result + calibration cross-reference with prediction_range = null
  and an explicit reason, never a fabricated range.
"""
from __future__ import annotations

import statistics
from datetime import date, timedelta

from .model import monday_of
from .stats import evaluate, ols_slope
from .weekly import build_weeks

# ---------------------------------------------------------------- helpers

def _sd(vals):
    """Sample SD; None when fewer than 2 values."""
    if len(vals) < 2:
        return None
    return statistics.stdev(vals)


def _rolling_sd(by_date, day, window=7, min_n=5):
    """SD of values over the `window` days ending on `day` (inclusive)."""
    vals = [by_date[d] for d in (day - timedelta(days=i) for i in range(window))
            if d in by_date]
    if len(vals) < min_n:
        return None
    return _sd(vals)


def _midpoint(night):
    return night.bedtime_min_past_21 + night.duration_min / 2.0


def _fmt(x, unit=""):
    if x is None:
        return "—"
    if unit == "min":
        return f"{abs(x):.0f} min"
    if unit == "bpm":
        return f"{abs(x):.1f} bpm"
    if unit == "pct":
        return f"{abs(x):.1f}%"
    if unit == "ms":
        return f"{abs(x):.1f} ms"
    if unit == "pts":
        return f"{abs(x):.1f} pts"
    if unit == "sec_km":
        return f"{abs(x):.0f} sec/km"
    return f"{x:.3g}"


def _direction(effect, outcome_kind):
    """Plain-language direction word for an effect sign."""
    if effect is None:
        return ""
    if outcome_kind == "pace":
        # Lower pace = faster. A negative effect means faster.
        return "faster" if effect < 0 else "slower"
    return "lower" if effect < 0 else "higher"


# ------------------------------------------------------------ pair builders
# Each returns (usable_pairs, collected_pairs, field_present, field_reason).
# usable: (date, x, y) with sensor-unreliable rows removed.
# collected: same pairs ignoring sensor exclusion (completeness guard).

def _sleep_pairs(athlete):
    """All sleep nights (including sensor-excluded) keyed by wake date."""
    return athlete.sleep_nights


def _nightly_base(athlete):
    sleep = athlete.sleep_by_date
    daily = athlete.daily_by_date
    return sleep, daily


def b_A1(athlete):
    usable, collected = [], []
    for n in athlete.sleep_nights:
        collected.append((n.wake_date, n.bedtime_min_past_21, n.duration_min))
        if not n.excluded_sensor_unreliable:
            usable.append((n.wake_date, n.bedtime_min_past_21, n.duration_min))
    return usable, collected, True, ""


def b_A2(athlete):
    sleep, daily = _nightly_base(athlete)
    bed = {d: n.bedtime_min_past_21 for d, n in sleep.items()}
    bed_all = {n.wake_date: n.bedtime_min_past_21 for n in athlete.sleep_nights}
    has_rhr = any(d.resting_hr_bpm is not None for d in athlete.daily)
    usable, collected = [], []
    for n in athlete.sleep_nights:
        d = n.wake_date
        sd = _rolling_sd(bed_all, d)
        rhr = daily.get(d).resting_hr_bpm if daily.get(d) else None
        if sd is not None and rhr is not None:
            collected.append((d, sd, rhr))
            if not n.excluded_sensor_unreliable:
                usable.append((d, sd, rhr))
    return usable, collected, has_rhr, \
        "your export has no resting heart rate" if not has_rhr else ""


def b_A3(athlete):
    sleep, daily = _nightly_base(athlete)
    has_rhr = any(d.resting_hr_bpm is not None for d in athlete.daily)
    usable, collected = [], []
    for n in athlete.sleep_nights:
        d = n.wake_date
        rhr = daily.get(d).resting_hr_bpm if daily.get(d) else None
        if rhr is not None:
            collected.append((d, n.duration_min, rhr))
            if not n.excluded_sensor_unreliable:
                usable.append((d, n.duration_min, rhr))
    return usable, collected, has_rhr, \
        "your export has no resting heart rate" if not has_rhr else ""


def _day_pace_pairs(athlete, driver_fn):
    """Shared helper for A4/A5/A7-quality: driver per night -> day pace."""
    usable, collected = [], []
    for n in athlete.sleep_nights:
        d = n.wake_date
        driver = driver_fn(n, d)
        pace = athlete.day_run_pace(d, include_excluded=True)
        pace_u = athlete.day_run_pace(d)
        if driver is not None and pace is not None:
            collected.append((d, driver, pace))
            if not n.excluded_sensor_unreliable and pace_u is not None:
                usable.append((d, driver, pace_u))
    has_runs = any(a.pace_sec_per_km for a in athlete.runs)
    return usable, collected, has_runs, \
        "no runs with pace data in the window" if not has_runs else ""


def b_A4(athlete):
    return _day_pace_pairs(athlete, lambda n, d: n.duration_min)


def b_A5(athlete):
    dur = {n.wake_date: n.duration_min for n in athlete.sleep_nights}
    return _day_pace_pairs(athlete, lambda n, d: _rolling_sd(dur, d))


def b_A6(athlete):
    has_eff = any(n.sleep_efficiency_pct is not None
                  for n in athlete.sleep_nights)
    usable, collected = [], []
    for n in athlete.sleep_nights:
        if n.sleep_efficiency_pct is not None:
            collected.append((n.wake_date, n.bedtime_min_past_21,
                              n.sleep_efficiency_pct))
            if not n.excluded_sensor_unreliable:
                usable.append((n.wake_date, n.bedtime_min_past_21,
                               n.sleep_efficiency_pct))
    return usable, collected, has_eff, \
        "your export has no sleep efficiency" if not has_eff else ""


def b_A7(athlete, outcome):
    sleep, daily = _nightly_base(athlete)
    mid = {n.wake_date: _midpoint(n) for n in athlete.sleep_nights}
    has_rhr = any(d.resting_hr_bpm is not None for d in athlete.daily)
    has_runs = any(a.pace_sec_per_km for a in athlete.runs)
    usable, collected = [], []
    for n in athlete.sleep_nights:
        d = n.wake_date
        sd = _rolling_sd(mid, d)
        if sd is None:
            continue
        if outcome == "rhr":
            rhr = daily.get(d).resting_hr_bpm if daily.get(d) else None
            if rhr is not None:
                collected.append((d, sd, rhr))
                if not n.excluded_sensor_unreliable:
                    usable.append((d, sd, rhr))
        else:
            pace = athlete.day_run_pace(d, include_excluded=True)
            pace_u = athlete.day_run_pace(d)
            if pace is not None:
                collected.append((d, sd, pace))
                if not n.excluded_sensor_unreliable and pace_u is not None:
                    usable.append((d, sd, pace_u))
    if outcome == "rhr":
        return usable, collected, has_rhr, \
            "your export has no resting heart rate" if not has_rhr else ""
    return usable, collected, has_runs, \
        "no runs with pace data in the window" if not has_runs else ""


def b_A8(athlete, outcome):
    """Weekly jetlag pairs. Pair date = the Monday after the week."""
    sleep = athlete.sleep_by_date
    mid = {d: _midpoint(n) for d, n in sleep.items()}
    usable, collected = [], []
    mondays = sorted({monday_of(d) for d in mid})
    for m in mondays:
        # Nights with wake dates Mon..Sun of the prior week.
        week_nights = [m - timedelta(days=i) for i in range(7, 0, -1)]
        wd = [mid[d] for d in week_nights if d in mid and d.weekday() < 5]
        we = [mid[d] for d in week_nights if d in mid and d.weekday() >= 5]
        if len(wd) < 2 or len(we) < 1:
            continue
        jetlag = statistics.mean(we) - statistics.mean(wd)
        if outcome == "duration":
            night = sleep.get(m)
            if night is not None:
                collected.append((m, jetlag, night.duration_min))
                if not night.excluded_sensor_unreliable:
                    usable.append((m, jetlag, night.duration_min))
        else:
            pace = athlete.day_run_pace(m, include_excluded=True)
            pace_u = athlete.day_run_pace(m)
            if pace is not None:
                collected.append((m, jetlag, pace))
                if pace_u is not None:
                    usable.append((m, jetlag, pace_u))
    has = True
    return usable, collected, has, ""


def b_A9(athlete):
    sleep, daily = _nightly_base(athlete)
    has_awake = any(n.awake_min is not None for n in athlete.sleep_nights)
    usable, collected = [], []
    for n in athlete.sleep_nights:
        d = n.wake_date
        rhr = daily.get(d).resting_hr_bpm if daily.get(d) else None
        if n.awake_min is not None and rhr is not None:
            collected.append((d, n.awake_min, rhr))
            if not n.excluded_sensor_unreliable:
                usable.append((d, n.awake_min, rhr))
    return usable, collected, has_awake, \
        "your export has no awake time in bed" if not has_awake else ""


def b_A10(athlete):
    usable, collected = [], []
    for n in athlete.sleep_nights:
        driver = 1.0 if n.wake_date.weekday() >= 5 else 0.0
        collected.append((n.wake_date, driver, n.duration_min))
        if not n.excluded_sensor_unreliable:
            usable.append((n.wake_date, driver, n.duration_min))
    return usable, collected, True, ""


def b_A11(athlete):
    sleep, daily = _nightly_base(athlete)
    has_hrv = any(d.hrv_ms is not None for d in athlete.daily)
    has_awake = any(n.awake_min is not None for n in athlete.sleep_nights)
    usable, collected = [], []
    for n in athlete.sleep_nights:
        d = n.wake_date
        hrv = daily.get(d).hrv_ms if daily.get(d) else None
        if n.awake_min is not None and hrv is not None:
            collected.append((d, n.awake_min, hrv))
            if not n.excluded_sensor_unreliable:
                usable.append((d, n.awake_min, hrv))
    ok = has_hrv and has_awake
    reason = ""
    if not has_awake:
        reason = "your export has no awake time in bed"
    elif not has_hrv:
        reason = "your export has no heart rate variability"
    return usable, collected, ok, reason


def b_A12(athlete):
    sleep, daily = _nightly_base(athlete)
    usable, collected = [], []
    for n in athlete.sleep_nights:
        d = n.wake_date
        rhr = daily.get(d).resting_hr_bpm if daily.get(d) else None
        if rhr is not None:
            midpoint = n.bedtime_min_past_21 + n.duration_min / 2
            collected.append((d, midpoint, rhr))
            if not n.excluded_sensor_unreliable:
                usable.append((d, midpoint, rhr))
    return usable, collected, True, ""


def b_A13(athlete):
    by_date = {n.wake_date: n for n in athlete.sleep_nights}
    usable, collected = [], []
    for n in athlete.sleep_nights:
        prev = by_date.get(n.wake_date - timedelta(days=1))
        if prev is None:
            continue
        collected.append((n.wake_date, prev.duration_min, n.duration_min))
        if not n.excluded_sensor_unreliable and not prev.excluded_sensor_unreliable:
            usable.append((n.wake_date, prev.duration_min, n.duration_min))
    return usable, collected, True, ""


def _weeks(athlete):
    return build_weeks(athlete)


def b_B1(athlete):
    weeks = _weeks(athlete)
    usable, collected = [], []
    for i, w in enumerate(weeks):
        if w.acwr is None or i + 1 >= len(weeks):
            continue
        nxt = weeks[i + 1]
        if nxt.mean_pace is None:
            continue
        collected.append((w.monday, w.acwr, nxt.mean_pace))
        usable.append((w.monday, w.acwr, nxt.mean_pace))
    has = any(w.mean_pace is not None for w in weeks)
    return usable, collected, has, \
        "no runs with pace data in the window" if not has else ""


def b_B2(athlete):
    weeks = _weeks(athlete)
    usable, collected = [], []
    for i, w in enumerate(weeks):
        if w.monotony is None or w.mean_rhr is None or i + 1 >= len(weeks):
            continue
        nxt = weeks[i + 1]
        if nxt.mean_rhr is None:
            continue
        collected.append((w.monday, w.monotony, nxt.mean_rhr - w.mean_rhr))
        usable.append((w.monday, w.monotony, nxt.mean_rhr - w.mean_rhr))
    has = any(w.mean_rhr is not None for w in weeks)
    return usable, collected, has, \
        "your export has no resting heart rate" if not has else ""


def b_B3(athlete):
    weeks = _weeks(athlete)
    usable, collected = [], []
    for i, w in enumerate(weeks):
        if i == 0 or i + 1 >= len(weeks):
            continue
        prev = weeks[i - 1]
        if prev.mileage_m <= 0:
            continue
        nxt = weeks[i + 1]
        if nxt.mean_pace is None:
            continue
        vol = abs(w.mileage_m - prev.mileage_m) / prev.mileage_m
        collected.append((w.monday, vol, nxt.mean_pace))
        usable.append((w.monday, vol, nxt.mean_pace))
    has = any(w.mean_pace is not None for w in weeks)
    return usable, collected, has, \
        "no runs with pace data in the window" if not has else ""


def b_B5(athlete):
    weeks = _weeks(athlete)
    run_days = sorted({a.date for a in athlete.runs})
    usable, collected = [], []
    for i, w in enumerate(weeks):
        if i < 3:
            continue
        win = weeks[i - 3:i + 1]
        paces = [x.mean_pace for x in win if x.mean_pace is not None]
        if len(paces) < 3:
            continue
        xs = list(range(len(win)))
        # Recompute pace trend over the weeks that actually have pace.
        trend_weeks = [x for x in win if x.mean_pace is not None]
        trend = ols_slope([j for j, x in enumerate(win)
                           if x.mean_pace is not None],
                          [x.mean_pace for x in trend_weeks])
        if trend is None:
            continue
        span_start = win[0].monday
        span_end = win[-1].monday + timedelta(days=6)
        days = [d for d in run_days if span_start <= d <= span_end]
        gaps = [(days[j + 1] - days[j]).days for j in range(len(days) - 1)]
        if len(gaps) < 3:
            continue
        gap_sd = _sd(gaps)
        if gap_sd is None:
            continue
        collected.append((w.monday, gap_sd, trend))
        usable.append((w.monday, gap_sd, trend))
    has = any(w.mean_pace is not None for w in weeks)
    return usable, collected, has, \
        "no runs with pace data in the window" if not has else ""


def b_B6(athlete):
    runs = [a for a in athlete.runs
            if a.temp_c is not None and a.pace_sec_per_km]
    has_temp = any(a.temp_c is not None for a in athlete.runs)
    if not has_temp:
        return [], [], False, "your export has no activity temperature"
    cool = [a.pace_sec_per_km for a in runs if 10 <= a.temp_c <= 15]
    if len(cool) < 3:
        return [], [], True, "fewer than 3 temperate runs for a baseline"
    baseline = statistics.mean(cool)
    usable, collected = [], []
    for a in runs:
        pen = (a.pace_sec_per_km - baseline) / baseline * 100.0
        collected.append((a.date, a.temp_c, pen))
        usable.append((a.date, a.temp_c, pen))
    return usable, collected, True, ""


def b_B7(athlete):
    daily = athlete.daily_by_date
    rhr_days = sorted(d for d, v in daily.items()
                      if v.resting_hr_bpm is not None)
    usable, collected = [], []
    for d in rhr_days:
        prior = [daily[x].resting_hr_bpm for x in rhr_days
                 if d - timedelta(days=14) <= x < d]
        if len(prior) < 7:
            continue
        dev = daily[d].resting_hr_bpm - statistics.median(prior)
        pace = athlete.day_run_pace(d, include_excluded=True)
        pace_u = athlete.day_run_pace(d)
        if pace is not None:
            collected.append((d, dev, pace))
            if pace_u is not None:
                usable.append((d, dev, pace_u))
    has_rhr = bool(rhr_days)
    has_runs = any(a.pace_sec_per_km for a in athlete.runs)
    ok = has_rhr and has_runs
    reason = ""
    if not has_rhr:
        reason = "your export has no resting heart rate"
    elif not has_runs:
        reason = "no runs with pace data in the window"
    return usable, collected, ok, reason


def b_B8(athlete):
    weeks = _weeks(athlete)
    usable, collected = [], []
    for i, w in enumerate(weeks):
        if w.mileage_m is None or w.mean_rhr is None or i + 1 >= len(weeks):
            continue
        nxt = weeks[i + 1]
        if nxt.mean_rhr is None:
            continue
        collected.append((w.monday, w.mileage_m / 1000.0, nxt.mean_rhr))
        usable.append((w.monday, w.mileage_m / 1000.0, nxt.mean_rhr))
    has_runs = any(w.mileage_m for w in weeks)
    has_rhr = any(w.mean_rhr is not None for w in weeks)
    ok = has_runs and has_rhr
    reason = ""
    if not has_runs:
        reason = "no runs with distance data in the window"
    elif not has_rhr:
        reason = "your export has no resting heart rate"
    return usable, collected, ok, reason


def b_B9(athlete):
    weeks = _weeks(athlete)
    usable, collected = [], []
    for i, w in enumerate(weeks):
        if w.long_run_m is None or i + 1 >= len(weeks):
            continue
        nxt = weeks[i + 1]
        if nxt.mean_rhr is None:
            continue
        collected.append((w.monday, w.long_run_m / 1000.0, nxt.mean_rhr))
        usable.append((w.monday, w.long_run_m / 1000.0, nxt.mean_rhr))
    ok = len(usable) >= 6
    reason = "" if ok else "not enough weeks with a long run"
    return usable, collected, ok, reason


# ------------------------------------------------------------- definitions

# floor: ("abs", value) or ("pace_pct",) meaning 1% of mean outcome.
QUESTIONS = [
    # id, family, grain, question, driver_high, driver_low, outcome,
    # outcome_kind, unit_label, window_label, scale, floor
    ("A1", "sleep", "nightly", "Later bedtimes and sleep length",
     "later-bedtime nights", "earlier-bedtime nights", "sleep length",
     "duration", "nights", "180 days", 60.0, ("abs", 15.0)),
    ("A2", "sleep", "nightly", "Bedtime consistency and morning resting heart rate",
     "more variable bedtimes", "steadier bedtimes", "morning resting HR",
     "hr", "nights", "180 days", 30.0, ("abs", 1.0)),
    ("A3", "sleep", "nightly", "Sleep length and next-day resting heart rate",
     "longer-sleep nights", "shorter-sleep nights", "next-day resting HR",
     "hr", "nights", "180 days", 60.0, ("abs", 1.0)),
    ("A4", "sleep", "nightly", "Sleep length and next-day run quality",
     "longer-sleep nights", "shorter-sleep nights", "next-day run pace",
     "pace", "nights", "180 days", 60.0, ("pace_pct",)),
    ("A5", "sleep", "nightly", "Sleep consistency and next-day run quality",
     "more variable sleep lengths", "steadier sleep lengths", "next-day run pace",
     "pace", "nights", "180 days", 30.0, ("pace_pct",)),
    ("A6", "sleep", "nightly", "Bedtime and sleep efficiency",
     "later-bedtime nights", "earlier-bedtime nights", "sleep efficiency",
     "efficiency", "nights", "180 days", 60.0, ("abs", 1.0)),
    ("A7-RHR", "sleep", "nightly", "Sleep regularity and next-day resting heart rate",
     "more irregular sleep midpoints", "steadier sleep midpoints", "next-day resting HR",
     "hr", "nights", "180 days", 30.0, ("abs", 1.0)),
    ("A7-Quality", "sleep", "nightly", "Sleep regularity and next-day run quality",
     "more irregular sleep midpoints", "steadier sleep midpoints", "next-day run pace",
     "pace", "nights", "180 days", 30.0, ("pace_pct",)),
    ("A8-Duration", "sleep", "weekly", "Weekend schedule shift and Sunday-night sleep",
     "higher weekend-shift weeks", "lower weekend-shift weeks", "Sunday-night sleep length",
     "duration", "weeks", "26 weeks", 60.0, ("abs", 15.0)),
    ("A8-Quality", "sleep", "weekly", "Weekend schedule shift and Monday run quality",
     "higher weekend-shift weeks", "lower weekend-shift weeks", "Monday run pace",
     "pace", "weeks", "26 weeks", 60.0, ("pace_pct",)),
    ("A9", "sleep", "nightly", "Awake time in bed and next-morning resting heart rate",
     "more-awake nights", "less-awake nights", "next-morning resting HR",
     "hr", "nights", "180 days", 30.0, ("abs", 1.0)),
    ("A10", "sleep", "nightly", "Weekend nights and sleep length",
     "weekend nights", "weekday nights", "sleep length",
     "duration", "nights", "180 days", 1.0, ("abs", 15.0)),
    ("A11", "sleep", "nightly", "Awake time in bed and next-morning heart rate variability",
     "more-awake nights", "less-awake nights", "next-morning HRV",
     "hrv", "nights", "180 days", 30.0, ("abs", 5.0)),
    ("A12", "sleep", "nightly", "Sleep midpoint and next-morning resting heart rate",
     "later-midpoint nights", "earlier-midpoint nights", "next-morning resting HR",
     "hr", "nights", "180 days", 60.0, ("abs", 1.0)),
    ("A13", "sleep", "nightly", "Previous night's sleep length and tonight's sleep length",
     "longer previous nights", "shorter previous nights", "tonight's sleep length",
     "duration", "nights", "180 days", 60.0, ("abs", 15.0)),
    ("B1", "load", "weekly", "Training ramp and next week's run quality",
     "higher-ramp weeks", "lower-ramp weeks", "next-week run pace",
     "pace", "weeks", "26 weeks", 1.0, ("pace_pct",)),
    ("B2", "load", "weekly", "Training monotony and resting heart rate trend",
     "more monotonous weeks", "more varied weeks", "next-week resting HR change",
     "hr", "weeks", "26 weeks", 1.0, ("abs", 1.0)),
    ("B3", "load", "weekly", "Mileage swings and the following week's run quality",
     "bigger mileage-swing weeks", "steadier mileage weeks", "next-week run pace",
     "pace", "weeks", "26 weeks", 0.10, ("pace_pct",)),
    ("B4", "load", "weekly", "Long-run share and late-run fade",
     "higher long-run-share weeks", "lower long-run-share weeks", "late-run fade",
     "pct", "weeks", "26 weeks", 1.0, ("abs", 1.0)),
    ("B5", "load", "weekly", "Training regularity and fitness trend",
     "more irregular training gaps", "steadier training gaps", "pace trend",
     "pace", "weeks", "26 weeks", 1.0, ("pace_pct",)),
    ("B6", "load", "nightly", "Heat and your pace",
     "hotter runs", "cooler runs", "pace penalty vs temperate baseline",
     "pct", "runs", "180 days", 1.0, ("abs", 1.0)),
    ("B7", "load", "nightly", "Morning resting heart rate and today's session",
     "higher-than-normal morning RHR days", "normal morning RHR days", "same-day run pace",
     "pace", "days", "180 days", 1.0, ("pace_pct",)),
    ("B8", "load", "weekly", "Weekly mileage and next week's resting heart rate",
     "higher-mileage weeks", "lower-mileage weeks", "next-week resting HR",
     "hr", "weeks", "26 weeks", 10.0, ("abs", 1.0)),
    ("B9", "load", "weekly", "Weekly long-run distance and next week's resting heart rate",
     "longer-long-run weeks", "shorter-long-run weeks", "next-week resting HR",
     "hr", "weeks", "26 weeks", 5.0, ("abs", 1.0)),
]

BUILDERS = {
    "A1": b_A1, "A2": b_A2, "A3": b_A3, "A4": b_A4, "A5": b_A5,
    "A6": b_A6, "A7-RHR": lambda a: b_A7(a, "rhr"),
    "A7-Quality": lambda a: b_A7(a, "quality"),
    "A8-Duration": lambda a: b_A8(a, "duration"),
    "A8-Quality": lambda a: b_A8(a, "quality"),
    "A9": b_A9, "A10": b_A10, "A11": b_A11, "A12": b_A12, "A13": b_A13,
    "B1": b_B1, "B2": b_B2, "B3": b_B3, "B5": b_B5, "B6": b_B6, "B7": b_B7,
    "B8": b_B8, "B9": b_B9,
}

EFFECT_UNITS = {
    "duration": "min", "hr": "bpm", "pace": "sec/km", "efficiency": "pts",
    "pct": "%", "hrv": "ms",
}

METHODS = {
    "A1": "Each night's bedtime (minutes past 9 PM, unwrapped across midnight) paired with that night's sleep length. Effect is the least-squares slope over the last 180 days, with a week-block bootstrap 95% likely range.",
    "A2": "Each night's bedtime scatter (standard deviation over the 7 nights ending that night) paired with the next morning's resting heart rate. Effect is per 30 minutes of bedtime scatter, over the last 180 days, with a week-block bootstrap 95% likely range.",
    "A3": "Each night's sleep length paired with the next morning's resting heart rate. Effect is per extra hour of sleep, over the last 180 days, with a week-block bootstrap 95% likely range.",
    "A4": "Each night's sleep length paired with the next day's run pace (distance-weighted mean pace that day). Effect is per extra hour of sleep, over the last 180 days, with a week-block bootstrap 95% likely range. Pace here is raw pace, not heart-rate-adjusted pace, because the normalized model carries no lap data.",
    "A5": "Each night's sleep-length scatter (standard deviation over the 7 nights ending that night) paired with the next day's run pace. Effect is per 30 minutes of scatter, over the last 180 days, with a week-block bootstrap 95% likely range.",
    "A6": "Each night's bedtime paired with that night's sleep efficiency, where the source reports efficiency. Effect is per hour later bedtime, over the last 180 days, with a week-block bootstrap 95% likely range.",
    "A7-RHR": "Each night's sleep-midpoint scatter (standard deviation of bedtime + sleep length / 2 over the 7 nights ending that night) paired with the next morning's resting heart rate. Effect is per 30 minutes of midpoint scatter, over the last 180 days, with a week-block bootstrap 95% likely range.",
    "A7-Quality": "Each night's sleep-midpoint scatter paired with the next day's run pace. Effect is per 30 minutes of midpoint scatter, over the last 180 days, with a week-block bootstrap 95% likely range.",
    "A8-Duration": "Each week's weekend schedule shift (mean weekend sleep midpoint minus mean weekday midpoint) paired with the following Sunday night's sleep length. Effect is per hour of weekend shift, over the last 26 weeks, with a week-block bootstrap 95% likely range.",
    "A8-Quality": "Each week's weekend schedule shift paired with the following Monday's run pace. Effect is per hour of weekend shift, over the last 26 weeks, with a week-block bootstrap 95% likely range.",
    "A9": "Each night's awake time in bed (time in bed not asleep, where the source reports it) paired with the next morning's resting heart rate. Effect is per 30 minutes awake in bed, over the last 180 days, with a week-block bootstrap 95% likely range.",
    "A10": "Each night's weekend indicator (1 for wake dates Saturday/Sunday, 0 otherwise) paired with that night's sleep length; the slope is the weekend-minus-weekday mean difference. Effect is weekend minus weekday minutes, over the last 180 days, with a week-block bootstrap 95% likely range.",
    "A11": "Each night's awake time in bed paired with the next morning's heart rate variability (SDNN, where the source reports it). Effect is per 30 minutes awake in bed, over the last 180 days, with a week-block bootstrap 95% likely range.",
    "B8": "Each week's total run mileage paired with the following week's mean resting heart rate. Effect is per 10 km of weekly mileage, over the last 26 weeks, with a week-block bootstrap 95% likely range. Level, not change: this asks whether higher-mileage weeks precede lower RHR levels, distinct from B2's monotony -> trend question.",
    "A12": "Each night's sleep midpoint (bedtime plus half the night's length, unwrapped so midnight never breaks the scale) paired with the next morning's resting heart rate. Effect is per hour later midpoint, over the last 180 days, with a week-block bootstrap 95% likely range.",
    "A13": "Each night's length paired with the immediately preceding night's length. Effect is minutes of tonight's length per hour of last night's length, over the last 180 days, with a week-block bootstrap 95% likely range.",
    "B9": "Each week's qualifying long-run distance paired with the following week's mean resting heart rate. Effect is per 5 km of long-run distance, over the last 26 weeks, with a week-block bootstrap 95% likely range. Weeks without a qualifying long run contribute no pair.",
    "B1": "Each week's ramp (that week's mileage divided by the mean of the prior 4 weeks' mileage) paired with the following week's mean run pace. Effect is per 1.0 of ramp, over the last 26 weeks, with a week-block bootstrap 95% likely range.",
    "B2": "Each week's monotony (mean daily mileage divided by its standard deviation) paired with the change in mean resting heart rate the following week. Effect is per 1.0 of monotony, over the last 26 weeks, with a week-block bootstrap 95% likely range.",
    "B3": "Each week's mileage swing (absolute change from the prior week, as a share) paired with the following week's mean run pace. Effect is per 10 percentage points of swing, over the last 26 weeks, with a week-block bootstrap 95% likely range.",
    "B5": "Each week's training-gap scatter (standard deviation of gaps between run days over the trailing 4 weeks) paired with the pace trend over the same 4 weeks (slope of weekly mean pace; negative means improving). Effect is per day of gap scatter, over the last 26 weeks, with a week-block bootstrap 95% likely range. This is the most assumption-heavy question in the registry.",
    "B6": "Each run's temperature paired with its pace penalty (percent slower than the athlete's mean pace on temperate 10-15 C runs). Effect is percent per degree C, over the last 180 days, with a week-block bootstrap 95% likely range.",
    "B7": "Each morning's resting heart rate deviation (that morning minus the trailing-14-day median) paired with the same day's run pace. Effect is per beat above normal, over the last 180 days, with a week-block bootstrap 95% likely range. Same-day pattern, not a forecast.",
}


def _sentence(q, card):
    (qid, family, grain, question, dh, dl, outcome, okind, unit_label,
     window_label, scale, floor_spec) = q
    tier = card["tier"]
    n = card["n"]
    eff = card["effect"]
    lo, hi = card["likely_range"]
    unit = EFFECT_UNITS[okind]
    if tier == "insufficient_data":
        reason = card.get("field_reason") or card["binding_rule"]
        if card.get("field_reason"):
            return (f"{question}? Not enough data yet — {reason}.")
        return (f"{question}? Not enough data yet — {card['binding_rule']}.")
    if tier == "no_pattern":
        if lo is not None and hi is not None:
            return (f"{question}? No consistent relationship in your data — "
                    f"across {n} {unit_label}, the likely range includes no "
                    f"effect ({_fmt(lo, unit)} to {_fmt(hi, unit)}).")
        return (f"{question}? No consistent relationship in your data — "
                f"across {n} {unit_label} ({card['binding_rule']}).")
    d = _direction(eff, okind)
    eff_s = _fmt(eff, unit)
    lo_s = _fmt(lo, unit)
    hi_s = _fmt(hi, unit)
    # Driver phrases that already end in the unit word ("later-bedtime
    # nights") must not get the unit appended again ("nights nights").
    dh_p = dh if dh.endswith(unit_label) else f"{dh} {unit_label}"
    dl_p = dl if dl.endswith(unit_label) else f"{dl} {unit_label}"
    if tier == "solid":
        s = (f"On {dh_p}, {outcome} runs about {eff_s} {d} "
             f"than on {dl_p} — likely range {lo_s} to {hi_s}. "
             f"Based on {n} {unit_label} in the last {window_label}.")
    else:  # emerging
        s = (f"Early evidence: on {dh_p}, {outcome} tends to run "
             f"about {eff_s} {d} — likely range {lo_s} to {hi_s}. "
             f"Not yet solid: {card['binding_rule']}.")
        if card["binding_rule"] in ("evidence too spread out",
                                    "evidence is stale"):
            s += f" Capped at emerging: {card['binding_rule']}."
    if card.get("demotion"):
        s += (f" Previously {card['demotion']['was']} — dropped at this "
              f"refresh because {card['demotion']['reason']}.")
    return s


def run_question(athlete, qid, prev_card=None):
    """Compute one registry question's card for an athlete."""
    q = next(x for x in QUESTIONS if x[0] == qid)
    (qid, family, grain, question, dh, dl, outcome, okind, unit_label,
     window_label, scale, floor_spec) = q
    if qid == "B4":
        # Long-run share is computed (see aggregates), but the fade
        # outcome needs lap/split data, which the normalized model does
        # not carry. B4 is therefore insufficient_data in v1 -- shown,
        # not silently dropped.
        weeks = _weeks(athlete)
        n_lr = sum(1 for w in weeks if w.long_run_m)
        return {
            "question": "B4", "family": "load",
            "question_text": question,
            "tier": "insufficient_data", "effect": None,
            "likely_range": [None, None], "n": n_lr,
            "binding_rule": "late-run fade needs lap/split data",
            "demotion": None,
            "methods_note": "Long-run share is computed per the locked "
            "22.5%/18% rule, but the late-run fade (final-third pace "
            "drop-off) needs lap/split data, which the normalized model "
            "does not carry. No fade is estimated from whole-run pace.",
            "sentence": f"{question}? Not enough data yet — late-run fade "
            f"needs lap/split data, which this import does not carry.",
        }
    usable, collected, field_present, field_reason = BUILDERS[qid](athlete)
    anchor = athlete.anchor_date
    card = {
        "question": qid, "family": family, "question_text": question,
        "tier": "insufficient_data", "effect": None, "likely_range": [None, None],
        "n": len(usable), "binding_rule": "", "demotion": None,
        "methods_note": METHODS[qid],
    }
    if not field_present:
        card["binding_rule"] = "required field absent"
        card["field_reason"] = field_reason
        card["sentence"] = _sentence(q, card)
        return card
    # Pace floor: 1% of the mean outcome value.
    if floor_spec[0] == "pace_pct":
        ys = [y for _, _, y in usable]
        mean_y = statistics.mean(ys) if ys else 0.0
        floor = 0.01 * mean_y
    else:
        floor = floor_spec[1]
    ev = evaluate(usable, scale, floor, grain, anchor, prev_card=prev_card,
                  collected_pairs=collected)
    card.update({k: ev[k] for k in
                 ("effect", "likely_range", "n", "tier", "binding_rule", "demotion")})
    card["density"] = ev.get("density")
    card["completeness"] = ev.get("completeness")
    card["sentence"] = _sentence(q, card)
    return card


def run_all(athlete, prev_pack=None):
    """Compute every association question. Returns cards list."""
    prev = {}
    if prev_pack:
        for c in prev_pack.get("cards", []):
            prev[c["question"]] = c
    return [run_question(athlete, q[0], prev_card=prev.get(q[0]))
            for q in QUESTIONS]


# ------------------------------------------------------------- race family

def race_cards(athlete):
    """C-family per-race cards. No races -> no cards (not insufficient)."""
    cards = []
    sleep = athlete.sleep_by_date
    run_hr = [a.avg_hr_bpm for a in athlete.runs
              if a.avg_hr_bpm is not None]
    hr_mean = statistics.mean(run_hr) if len(run_hr) >= 5 else None
    hr_sd = statistics.stdev(run_hr) if len(run_hr) >= 5 else None
    for r in sorted(athlete.races, key=lambda x: x.date):
        c1_ok = (r.field_size and r.matched_finishers
                 and r.matched_finishers >= 40
                 and r.matched_finishers / r.field_size >= 0.35)
        c1 = {
            "question": "C1", "race": r.name, "date": str(r.date),
            "shown": bool(c1_ok),
            "field_median_slow_pct": r.field_median_slow_pct if c1_ok else None,
            "matched_finishers": r.matched_finishers,
            "field_size": r.field_size,
            "sentence": (
                f"How fast was this course? Across {r.matched_finishers} of "
                f"{r.field_size} finishers matched to a season best, the field "
                f"ran a median of {r.field_median_slow_pct}% slower than those "
                f"marks." if c1_ok else
                "How fast was this course? The field hasn't been matched "
                "to season bests yet, so no adjustment is shown."
                if r.matched_finishers is None or r.field_size is None else
                f"How fast was this course? Only {r.matched_finishers} "
                f"finishers could be matched to a season best "
                f"({(r.matched_finishers / r.field_size * 100) if r.field_size else 0:.0f}% "
                f"coverage) — not enough to measure the course reliably, so "
                f"no adjustment is shown."),
        }
        c2 = {
            "question": "C2", "race": r.name, "date": str(r.date),
            "conditions_note": r.conditions_note,
            "sentence": (f"Conditions: {r.conditions_note}."
                         if r.conditions_note else
                         "Conditions: no conditions recorded for this race."),
        }
        # C3: flag when race HR is implausibly low vs run history.
        race_act = [a for a in athlete.activities
                    if a.date == r.date and a.is_race and a.avg_hr_bpm]
        race_hr = race_act[0].avg_hr_bpm if race_act else None
        flagged = (race_hr is not None and hr_mean is not None and hr_sd
                   and race_hr < hr_mean - 2 * hr_sd)
        c3 = {
            "question": "C3", "race": r.name, "date": str(r.date),
            "flagged": bool(flagged),
            "race_avg_hr": race_hr,
            "typical_avg_hr": round(hr_mean, 1) if hr_mean else None,
            "sentence": (
                f"Heart-rate data for this race looks unreliable: your watch "
                f"read {race_hr} avg against a typical {round(hr_mean, 1)} "
                f"across your runs. HR-based numbers for this day are flagged, "
                f"not trusted." if flagged else
                "Heart-rate data for this race looks consistent with your "
                "run history."),
        }
        t_str = (f"{int(r.finish_time_s // 60)}:{r.finish_time_s % 60:04.1f}"
                 if r.finish_time_s is not None else "not recorded")
        c4 = {
            "question": "C4", "race": r.name, "date": str(r.date),
            "finish_time_s": r.finish_time_s,
            "field_median_slow_pct": r.field_median_slow_pct if c1_ok else None,
            "prediction_range": None,
            "prediction_reason": "prediction model not included in this track",
            "sentence": (
                f"Your time: {t_str}. No flat-equivalent likely "
                f"range is shown: the prediction model is not part of this "
                f"build, and no range is fabricated in its place."),
        }
        # C5: pre-race sleep, T-1 = row dated race day, T-2 = race day - 1,
        # baseline = race-30..race-3 excluding featured (race T-1/T-2) nights.
        featured = set()
        for rr in athlete.races:
            featured.add(rr.date)
            featured.add(rr.date - timedelta(days=1))
        base = [sleep[d] for d in sleep
                if r.date - timedelta(days=30) <= d <= r.date - timedelta(days=3)
                and d not in featured]
        t1 = sleep.get(r.date)
        t2 = sleep.get(r.date - timedelta(days=1))

        def _delta(night, base_vals, key):
            if night is None or len(base_vals) < 14:
                return None
            base_mean = statistics.mean(base_vals)
            return round(getattr(night, key) - base_mean, 1)

        base_dur = [n.duration_min for n in base]
        base_bt = [n.bedtime_min_past_21 for n in base]
        c5 = {
            "question": "C5", "race": r.name, "date": str(r.date),
            "baseline_n": len(base),
            "baseline_duration_min": (round(statistics.mean(base_dur), 1)
                                      if len(base_dur) >= 14 else None),
            "baseline_bedtime": (round(statistics.mean(base_bt), 1)
                                 if len(base_bt) >= 14 else None),
            "t_minus_2": ({"date": str(t2.wake_date),
                           "duration_min": t2.duration_min,
                           "bedtime": t2.bedtime_min_past_21,
                           "duration_delta": _delta(t2, base_dur, "duration_min"),
                           "bedtime_delta": _delta(t2, base_bt, "bedtime_min_past_21")}
                          if t2 else None),
            "t_minus_1": ({"date": str(t1.wake_date),
                           "duration_min": t1.duration_min,
                           "bedtime": t1.bedtime_min_past_21,
                           "duration_delta": _delta(t1, base_dur, "duration_min"),
                           "bedtime_delta": _delta(t1, base_bt, "bedtime_min_past_21")}
                          if t1 else None),
        }
        if len(base_dur) < 14 or len(base_bt) < 14:
            c5["sentence"] = (
                "Pre-race sleep is shown without a reliable baseline "
                "comparison: not enough prior sleep data for a reliable "
                "28-night comparison.")
        else:
            c5["sentence"] = (
                "Pre-race sleep vs the prior 28-night baseline is shown "
                "for context; this card does not grade sleep or claim "
                "either night changed the result.")
        cards.extend([c1, c2, c3, c4, c5])
    return cards


def aggregates(athlete, weeks):
    """Board header aggregates (same weekly math as Family B).

    Sleep/RHR averages cover the last 180 days from the anchor (the
    nightly analysis window), not all-time history. Mileage covers the
    built weeks, which span the athlete's activity coverage (shorter
    than 26 weeks for short-history profiles); the key is named
    accordingly, never implying 26 weeks of data that is not there.
    """
    anchor = athlete.anchor_date
    win_start = anchor - timedelta(days=179) if anchor else None
    sleep = [n for n in athlete.sleep_nights
             if not n.excluded_sensor_unreliable
             and (win_start is None or n.wake_date >= win_start)]
    rhr = [d.resting_hr_bpm for d in athlete.daily
           if d.resting_hr_bpm is not None
           and (win_start is None or d.date >= win_start)]
    total_miles = sum(w.mileage_m for w in weeks) / 1609.34 if weeks else 0.0
    long_run_weeks = sum(1 for w in weeks if w.long_run_m)
    biggest = max((w.mileage_m for w in weeks), default=0.0) / 1609.34
    return {
        "window_anchor": str(anchor) if anchor else None,
        "total_miles_in_span": round(total_miles, 1),
        "avg_sleep_min": round(statistics.mean([n.duration_min for n in sleep]), 1)
        if sleep else None,
        "avg_rhr": round(statistics.mean(rhr), 1) if rhr else None,
        "nights_counted": len(sleep),
        "long_run_weeks": long_run_weeks,
        "biggest_week_miles": round(biggest, 1),
        "excluded_sensor_unreliable": sum(
            1 for n in athlete.sleep_nights if n.excluded_sensor_unreliable)
        + sum(1 for a in athlete.activities if a.excluded_sensor_unreliable),
    }
