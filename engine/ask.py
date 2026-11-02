"""Test-your-own-idea: structured ask-a-correlation (Matt, 2026-10-24).

The user picks a DRIVER and an OUTCOME from dropdowns and gets a
gated answer card -- never free text. In this pre-loaded build the
answers are precomputed at build time by this module (zero API
calls) and shipped in the app data; every answer runs through the
SAME evidence gate as the board (engine.stats.evaluate: same floors,
density/recency caps, week-block bootstrap likely ranges).

Pair-space rules (the honesty constraints are the feature):
- Only pairs NOT already answered by a standing registry question
  are offered. A registry pair is listed in REGISTRY_PAIRS and the
  app points at the board card instead of duplicating the answer.
- Steps are never a driver or an outcome (banned outright).
- Raw pace is never a variable. Pace appears only inside "run
  quality" -- the distance-weighted mean pace outcome the board
  already uses (next-day / next-week), exactly as in A4/B1.
- Load -> sleep questions stay out by design, as on the board:
  no ask outcome is a sleep variable for a load driver, because
  sleep outcomes only pair with sleep drivers inside the nightly
  set, and the weekly outcomes are RHR level and run quality.
- A pair whose driver and outcome are the same measurement
  (a night's sleep length against itself) is not tested; the app
  says why. The same goes for algebraically coupled pairs: sleep
  midpoint is BUILT from tonight's length (bedtime + half the
  night), and a stage share has tonight's length as its
  denominator -- testing either against tonight's length would
  partly compare a measurement with itself.
- Stage-share drivers (REM / light / deep) carry the compositional
  caveat verbatim on every answer card, the same lock as A14.

Pair construction mirrors the registry builders: usable pairs drop
sensor-unreliable rows, collected pairs keep them for the
completeness guard, nightly pairs are windowed to 180 days and
weekly pairs to 26 weeks by evaluate() itself.

One deliberate distinction: weekly monotony -> next-week mean RHR
LEVEL is offered here. The board's B2 asks monotony -> RHR CHANGE
(trend); B8's own note establishes level-vs-change as distinct
questions, so this pair does not duplicate B2.

Run: python -m engine.ask <normalized.json> [out.json]
                           [--anchor YYYY-MM-DD]
"""
from __future__ import annotations

import json
import statistics
import sys
from datetime import date, timedelta

from .factpack import _pinned_view
from .model import load_normalized, parse_date
from .registry import A14_CAVEAT, _rolling_sd, _sentence
from .stats import evaluate
from .weekly import build_weeks

# ------------------------------------------------------------ variables
# key, title (for the question text), dropdown label,
# driver-high phrase, driver-low phrase, effect scale
NIGHTLY_DRIVERS = [
    ("bedtime", "Bedtime", "Bedtime",
     "later-bedtime nights", "earlier-bedtime nights", 60.0),
    ("bedtime_variability", "Bedtime consistency", "Bedtime variability",
     "more variable bedtimes", "steadier bedtimes", 30.0),
    ("sleep_duration", "Sleep length", "Sleep length",
     "longer-sleep nights", "shorter-sleep nights", 60.0),
    ("awake_min", "Awake time in bed", "Awake time in bed",
     "more-awake nights", "less-awake nights", 30.0),
    ("rem_share", "REM sleep share", "REM share of sleep",
     "higher-REM-share nights", "lower-REM-share nights", 0.10),
    ("light_share", "Light sleep share", "Light share of sleep",
     "higher-light-share nights", "lower-light-share nights", 0.10),
    ("deep_share", "Deep sleep share", "Deep share of sleep",
     "higher-deep-share nights", "lower-deep-share nights", 0.10),
    ("weekend", "Weekend nights", "Weekend night",
     "weekend nights", "weekday nights", 1.0),
    ("midpoint", "Sleep midpoint", "Sleep midpoint",
     "later-midpoint nights", "earlier-midpoint nights", 60.0),
    ("prev_duration", "Previous night's sleep length",
     "Previous night's sleep length",
     "longer previous nights", "shorter previous nights", 60.0),
]

WEEKLY_DRIVERS = [
    ("mileage", "Weekly mileage", "Weekly mileage",
     "higher-mileage weeks", "lower-mileage weeks", 10.0),
    ("acwr", "Training ramp", "Training ramp (ACWR)",
     "higher-ramp weeks", "lower-ramp weeks", 1.0),
    ("monotony", "Training monotony", "Training monotony",
     "more monotonous weeks", "more varied weeks", 1.0),
    ("long_run", "Weekly long-run distance", "Weekly long-run distance",
     "longer-long-run weeks", "shorter-long-run weeks", 5.0),
    ("volatility", "Mileage swings", "Mileage swing from prior week",
     "bigger mileage-swing weeks", "steadier mileage weeks", 0.10),
]

# key, title, dropdown label, outcome phrase (sentence), outcome kind
NIGHTLY_OUTCOMES = [
    ("rhr", "next-morning resting heart rate",
     "Next-morning resting heart rate", "next-morning resting HR", "hr"),
    ("hrv", "next-morning heart rate variability",
     "Next-morning HRV", "next-morning HRV", "hrv"),
    ("run_quality", "next-day run quality",
     "Next-day run quality", "next-day run pace", "pace"),
    ("duration", "tonight's sleep length",
     "Tonight's sleep length", "tonight's sleep length", "duration"),
]

WEEKLY_OUTCOMES = [
    ("next_rhr", "next week's resting heart rate",
     "Next week's resting heart rate", "next-week resting HR", "hr"),
    ("next_quality", "next week's run quality",
     "Next week's run quality", "next-week run pace", "pace"),
]

STAGE_DRIVERS = {"rem_share", "light_share", "deep_share"}

# Pairs a standing board question already answers: (driver, outcome)
# -> registry question id. Never duplicated as ask answers.
REGISTRY_PAIRS = {
    ("bedtime", "duration"): "A1",
    ("bedtime_variability", "rhr"): "A2",
    ("sleep_duration", "rhr"): "A3",
    ("sleep_duration", "run_quality"): "A4",
    ("awake_min", "rhr"): "A9",
    ("weekend", "duration"): "A10",
    ("awake_min", "hrv"): "A11",
    ("midpoint", "rhr"): "A12",
    ("prev_duration", "duration"): "A13",
    ("rem_share", "rhr"): "A14",
    ("acwr", "next_quality"): "B1",
    ("volatility", "next_quality"): "B3",
    ("mileage", "next_rhr"): "B8",
    ("long_run", "next_rhr"): "B9",
}

# Driver and outcome are the same measurement: untestable by
# construction, and the app says so instead of testing it.
SAME_MEASUREMENT = {("sleep_duration", "duration")}

# Algebraically coupled pairs: the driver is built FROM the
# outcome, so a slope would partly restate the definition instead
# of measuring the athlete. Not tested; the app gives the reason.
COUPLED_PAIRS = {
    ("midpoint", "duration"):
        "Sleep midpoint is built from tonight's length (bedtime "
        "plus half the night), so this pair partly compares a "
        "measurement with itself.",
    ("rem_share", "duration"):
        "REM share has tonight's length as its denominator, so "
        "this pair partly compares a measurement with itself.",
    ("light_share", "duration"):
        "Light share has tonight's length as its denominator, so "
        "this pair partly compares a measurement with itself.",
    ("deep_share", "duration"):
        "Deep share has tonight's length as its denominator, so "
        "this pair partly compares a measurement with itself.",
}

_FLOORS = {"hr": ("abs", 1.0), "hrv": ("abs", 5.0),
           "duration": ("abs", 15.0), "pace": ("pace_pct",)}

_DRIVER_DESC = {
    "bedtime": "bedtime (minutes past 9 PM, unwrapped across midnight)",
    "bedtime_variability": "bedtime scatter (standard deviation over "
                           "the 7 nights ending that night)",
    "sleep_duration": "sleep length",
    "awake_min": "awake time in bed, where the source reports it",
    "rem_share": "REM share of the night's asleep time",
    "light_share": "light share of the night's asleep time",
    "deep_share": "deep share of the night's asleep time",
    "weekend": "a weekend indicator (1 for wake dates Saturday/Sunday)",
    "midpoint": "sleep midpoint (bedtime plus half the night's length)",
    "prev_duration": "the immediately preceding night's sleep length",
    "mileage": "weekly run mileage in km",
    "acwr": "the week's ramp (mileage over the prior 4 weeks' mean)",
    "monotony": "the week's monotony (mean daily mileage over its "
                "standard deviation)",
    "long_run": "the week's qualifying long-run distance in km",
    "volatility": "the week's mileage swing (absolute change from "
                  "the prior week, as a share)",
}
_OUTCOME_DESC = {
    "rhr": "the next morning's resting heart rate",
    "hrv": "the next morning's heart rate variability (SDNN)",
    "run_quality": "the next day's run pace (distance-weighted mean "
                   "pace that day)",
    "duration": "that night's sleep length",
    "next_rhr": "the following week's mean resting heart rate (level, "
                "not change)",
    "next_quality": "the following week's mean run pace",
}
_SCALE_DESC = {
    "bedtime": "per hour later bedtime", "bedtime_variability":
    "per 30 minutes of bedtime scatter", "sleep_duration":
    "per extra hour of sleep", "awake_min": "per 30 minutes awake "
    "in bed", "rem_share": "per 10 percentage points of REM share",
    "light_share": "per 10 percentage points of light share",
    "deep_share": "per 10 percentage points of deep share",
    "weekend": "weekend minus weekday", "midpoint":
    "per hour later midpoint", "prev_duration":
    "per hour of the previous night's length", "mileage":
    "per 10 km of weekly mileage", "acwr": "per 1.0 of ramp",
    "monotony": "per 1.0 of monotony", "long_run":
    "per 5 km of long-run distance", "volatility":
    "per 10 percentage points of swing",
}


# ------------------------------------------------------- pair builders

def _night_driver_values(athlete, key):
    """driver key -> {wake_date: value} over ALL nights (rolling
    windows and previous-night lookups see excluded nights too, as
    in the registry builders; exclusion is applied per pair)."""
    nights = athlete.sleep_nights
    if key == "bedtime":
        return {n.wake_date: n.bedtime_min_past_21 for n in nights}
    if key == "bedtime_variability":
        bed = {n.wake_date: n.bedtime_min_past_21 for n in nights}
        return {n.wake_date: _rolling_sd(bed, n.wake_date) for n in nights}
    if key == "sleep_duration":
        return {n.wake_date: n.duration_min for n in nights}
    if key == "awake_min":
        return {n.wake_date: n.awake_min for n in nights}
    if key in STAGE_DRIVERS:
        attr = {"rem_share": "rem_min", "light_share": "light_min",
                "deep_share": "deep_min"}[key]
        return {n.wake_date: (getattr(n, attr) / n.duration_min
                              if getattr(n, attr) is not None
                              and n.duration_min > 0 else None)
                for n in nights}
    if key == "weekend":
        return {n.wake_date: (1.0 if n.wake_date.weekday() >= 5 else 0.0)
                for n in nights}
    if key == "midpoint":
        return {n.wake_date: n.bedtime_min_past_21 + n.duration_min / 2.0
                for n in nights}
    if key == "prev_duration":
        by_date = {n.wake_date: n for n in nights}
        out = {}
        for n in nights:
            prev = by_date.get(n.wake_date - timedelta(days=1))
            out[n.wake_date] = prev.duration_min if prev else None
        return out
    raise KeyError(key)


def _nightly_pairs(athlete, dkey, okey):
    """(usable, collected) for one nightly ask pair, mirroring the
    registry builders' sensor-exclusion handling."""
    daily = athlete.daily_by_date
    by_date = {n.wake_date: n for n in athlete.sleep_nights}
    xs = _night_driver_values(athlete, dkey)
    usable, collected = [], []
    for n in athlete.sleep_nights:
        d = n.wake_date
        x = xs.get(d)
        if x is None:
            continue
        # Previous-night driver: the source night must also be
        # usable for the pair to count as usable (A13's rule).
        prev_blocked = False
        if dkey == "prev_duration":
            prev = by_date.get(d - timedelta(days=1))
            prev_blocked = bool(prev and prev.excluded_sensor_unreliable)
        if okey == "rhr":
            row = daily.get(d)
            y = row.resting_hr_bpm if row else None
            if y is None:
                continue
            collected.append((d, x, y))
            if not n.excluded_sensor_unreliable and not prev_blocked:
                usable.append((d, x, y))
        elif okey == "hrv":
            row = daily.get(d)
            y = row.hrv_ms if row else None
            if y is None:
                continue
            collected.append((d, x, y))
            if not n.excluded_sensor_unreliable and not prev_blocked:
                usable.append((d, x, y))
        elif okey == "run_quality":
            pace = athlete.day_run_pace(d, include_excluded=True)
            pace_u = athlete.day_run_pace(d)
            if pace is None:
                continue
            collected.append((d, x, pace))
            if (not n.excluded_sensor_unreliable and not prev_blocked
                    and pace_u is not None):
                usable.append((d, x, pace_u))
        elif okey == "duration":
            collected.append((d, x, n.duration_min))
            if not n.excluded_sensor_unreliable and not prev_blocked:
                usable.append((d, x, n.duration_min))
    return usable, collected


def _weekly_pairs(athlete, weeks, dkey, okey):
    """(usable, collected) for one weekly ask pair. Mirrors the
    B-family builders; weeks without the driver or the next week's
    outcome contribute no pair (B9's rule)."""
    usable = []
    for i, w in enumerate(weeks):
        if i + 1 >= len(weeks):
            continue
        nxt = weeks[i + 1]
        if dkey == "mileage":
            x = w.mileage_m / 1000.0
        elif dkey == "acwr":
            x = w.acwr
        elif dkey == "monotony":
            x = w.monotony
        elif dkey == "long_run":
            x = (w.long_run_m / 1000.0
                 if w.long_run_m is not None else None)
        elif dkey == "volatility":
            if i == 0:
                continue
            prev = weeks[i - 1]
            if prev.mileage_m <= 0:
                continue
            x = abs(w.mileage_m - prev.mileage_m) / prev.mileage_m
        else:
            raise KeyError(dkey)
        y = nxt.mean_rhr if okey == "next_rhr" else nxt.mean_pace
        if x is None or y is None:
            continue
        usable.append((w.monday, x, y))
    # Weekly rows carry no sensor exclusion in the model (run-level
    # exclusion already shapes the weekly aggregates), so usable and
    # collected coincide, as in the B-family builders.
    return usable, list(usable)


# ---------------------------------------------------------- availability

def _availability(athlete, weeks):
    """Which variables this profile's data actually carries. A
    variable is available when its underlying field is present --
    the dropdowns offer nothing the export does not contain."""
    nights = athlete.sleep_nights
    drivers, outcomes = [], []

    def d_add(key):
        drivers.append(key)

    if nights:
        d_add("bedtime")
        d_add("bedtime_variability")
        d_add("sleep_duration")
    if any(n.awake_min is not None for n in nights):
        d_add("awake_min")
    if any(n.rem_min is not None for n in nights):
        d_add("rem_share")
    if any(n.light_min is not None for n in nights):
        d_add("light_share")
    if any(n.deep_min is not None for n in nights):
        d_add("deep_share")
    if nights:
        d_add("weekend")
        d_add("midpoint")
        d_add("prev_duration")
    if any(w.mileage_m for w in weeks):
        d_add("mileage")
    if any(w.acwr is not None for w in weeks):
        d_add("acwr")
    if any(w.monotony is not None for w in weeks):
        d_add("monotony")
    if any(w.long_run_m is not None for w in weeks):
        d_add("long_run")
    if any(weeks[i].mileage_m > 0 for i in range(1, len(weeks))):
        d_add("volatility")

    if any(d.resting_hr_bpm is not None for d in athlete.daily):
        outcomes.append("rhr")
    if any(d.hrv_ms is not None for d in athlete.daily):
        outcomes.append("hrv")
    if any(a.pace_sec_per_km for a in athlete.runs):
        outcomes.append("run_quality")
    if nights:
        outcomes.append("duration")
    if any(w.mean_rhr is not None for w in weeks):
        outcomes.append("next_rhr")
    if any(w.mean_pace is not None for w in weeks):
        outcomes.append("next_quality")
    return drivers, outcomes


# -------------------------------------------------------------- answers

def _run_pair(athlete, weeks, grain, dspec, ospec):
    dkey, dtitle, _dlab, dh, dl, scale = dspec
    okey, otitle, _olab, ophr, okind = ospec
    if grain == "nightly":
        usable, collected = _nightly_pairs(athlete, dkey, okey)
        unit_label, window_label = "nights", "180 days"
    else:
        usable, collected = _weekly_pairs(athlete, weeks, dkey, okey)
        unit_label, window_label = "weeks", "26 weeks"
    floor_spec = _FLOORS[okind]
    if floor_spec[0] == "pace_pct":
        ys = [y for _, _, y in usable]
        floor = 0.01 * statistics.mean(ys) if ys else 0.0
    else:
        floor = floor_spec[1]
    ev = evaluate(usable, scale, floor, grain, athlete.anchor_date,
                  collected_pairs=collected)
    question_text = f"{dtitle} and {otitle}"
    q = (f"ASK-{dkey}-{okey}", "ask", grain, question_text, dh, dl,
         ophr, okind, unit_label, window_label, scale, floor_spec)
    card = {"tier": ev["tier"], "n": ev["n"], "effect": ev["effect"],
            "likely_range": ev["likely_range"],
            "binding_rule": ev["binding_rule"], "demotion": None}
    sentence = _sentence(q, card)
    caveat = None
    if dkey in STAGE_DRIVERS:
        # The compositional caveat rides on every stage-share
        # answer, whatever the tier (A14's lock, extended by the
        # ask spec to light and deep shares).
        caveat = A14_CAVEAT
        sentence += " " + A14_CAVEAT
    methods = (
        f"Each {unit_label[:-1]}'s {_DRIVER_DESC[dkey]} paired with "
        f"{_OUTCOME_DESC[okey]}. Effect is {_SCALE_DESC[dkey]}, over "
        f"the last {window_label}, with a week-block bootstrap 95% "
        f"likely range. Asked on demand: this pair is not one of the "
        f"board's standing questions, and it was tested with the same "
        f"evidence gate -- same tiers, same practical floors, same "
        f"density and recency caps.")
    return {
        "driver": dkey, "outcome": okey, "grain": grain,
        "question_text": question_text,
        "tier": ev["tier"], "effect": ev["effect"],
        "range": ev["likely_range"], "n": ev["n"],
        "binding_rule": ev["binding_rule"],
        "density": ev.get("density"),
        "completeness": ev.get("completeness"),
        "sentence": sentence, "methods_note": methods,
        "caveat": caveat, "asked_on_demand": True,
    }


def build_ask(athlete, generated_on=None):
    """Compute the full ask pack for one athlete (already pinned to
    the board anchor by the caller when an anchor is wanted)."""
    weeks = build_weeks(athlete)
    avail_d, avail_o = _availability(athlete, weeks)
    d_night = {d[0]: d for d in NIGHTLY_DRIVERS}
    d_week = {d[0]: d for d in WEEKLY_DRIVERS}
    o_night = {o[0]: o for o in NIGHTLY_OUTCOMES}
    o_week = {o[0]: o for o in WEEKLY_OUTCOMES}

    pairs = []
    for dkey in avail_d:
        for okey in avail_o:
            if (dkey, okey) in REGISTRY_PAIRS:
                continue
            if (dkey, okey) in SAME_MEASUREMENT:
                continue
            if (dkey, okey) in COUPLED_PAIRS:
                continue
            if dkey in d_night and okey in o_night:
                pairs.append(_run_pair(athlete, weeks, "nightly",
                                       d_night[dkey], o_night[okey]))
            elif dkey in d_week and okey in o_week:
                pairs.append(_run_pair(athlete, weeks, "weekly",
                                       d_week[dkey], o_week[okey]))

    def _var_list(night_specs, week_specs, avail):
        out = []
        for spec in night_specs:
            if spec[0] in avail:
                out.append({"key": spec[0], "label": spec[2],
                            "grain": "nightly"})
        for spec in week_specs:
            if spec[0] in avail:
                out.append({"key": spec[0], "label": spec[2],
                            "grain": "weekly"})
        return out

    on_board = {f"{d}|{o}": qid for (d, o), qid in REGISTRY_PAIRS.items()
                if d in avail_d and o in avail_o}
    same = [f"{d}|{o}" for (d, o) in sorted(SAME_MEASUREMENT)
            if d in avail_d and o in avail_o]
    coupled = {f"{d}|{o}": why for (d, o), why in COUPLED_PAIRS.items()
               if d in avail_d and o in avail_o}
    # Prune the dropdown lists to variables that participate in at
    # least one offered answer or explanation -- an outcome no
    # available driver can pair with (a sleep-only profile's
    # next-week RHR, say) is not offered.
    used_d = ({p["driver"] for p in pairs}
              | {k.split("|")[0] for k in on_board}
              | {k.split("|")[0] for k in same}
              | {k.split("|")[0] for k in coupled})
    used_o = ({p["outcome"] for p in pairs}
              | {k.split("|")[1] for k in on_board}
              | {k.split("|")[1] for k in same}
              | {k.split("|")[1] for k in coupled})
    avail_d = [d for d in avail_d if d in used_d]
    avail_o = [o for o in avail_o if o in used_o]
    return {
        "athlete": {"id": athlete.id,
                    "display_label": athlete.display_label,
                    "source": athlete.source},
        "generated_on": generated_on or str(date.today()),
        "window_anchor": (str(athlete.anchor_date)
                          if athlete.anchor_date else None),
        "drivers": _var_list(NIGHTLY_DRIVERS, WEEKLY_DRIVERS, avail_d),
        "outcomes": _var_list(NIGHTLY_OUTCOMES, WEEKLY_OUTCOMES, avail_o),
        "pairs": pairs,
        "on_board": on_board,
        "same_measurement": same,
        "coupled": coupled,
    }


def main(argv):
    args = list(argv[1:])
    anchor = None
    if "--anchor" in args:
        i = args.index("--anchor")
        if i + 1 >= len(args):
            print(__doc__)
            return 2
        anchor = args[i + 1]
        del args[i:i + 2]
    if len(args) < 1:
        print(__doc__)
        return 2
    athlete = load_normalized(args[0])
    if anchor is not None:
        athlete = _pinned_view(athlete, parse_date(anchor))
    pack = build_ask(athlete)
    tiers = {}
    for p in pack["pairs"]:
        tiers[p["tier"]] = tiers.get(p["tier"], 0) + 1
    if len(args) > 1:
        with open(args[1], "w") as fh:
            json.dump(pack, fh, indent=2, default=str)
        print(f"{athlete.id}: {len(pack['pairs'])} ask pairs offered "
              f"{tiers}; wrote {args[1]}")
    else:
        print(json.dumps(pack, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
