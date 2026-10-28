"""Stridewise normalized data model (v1).

One model, every source adapts INTO it. Nothing source-specific survives
past import. GPS/routes are stripped at import and never stored -- this
module has no coordinate field anywhere, by design.

All dates are local athlete dates. Sleep rows are keyed by WAKE date.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta


def parse_date(s):
    """Parse a YYYY-MM-DD string to a date. Passes dates through."""
    if s is None:
        return None
    if isinstance(s, date):
        return s
    return date.fromisoformat(str(s)[:10])


@dataclass
class SleepNight:
    wake_date: date
    duration_min: float
    bedtime_min_past_21: float  # unwrapped: 20:00 -> -60, 21:00 -> 0, 00:30 -> 210
    sleep_efficiency_pct: float | None = None
    excluded_sensor_unreliable: bool = False
    exclusion_reason: str | None = None


@dataclass
class Daily:
    date: date
    resting_hr_bpm: float | None = None
    hrv_ms: float | None = None


@dataclass
class Activity:
    date: date
    start_time: str = ""
    kind: str = "other"  # "run" | "other"
    distance_m: float | None = None
    duration_s: float | None = None
    avg_hr_bpm: float | None = None
    temp_c: float | None = None
    is_race: bool = False
    excluded_sensor_unreliable: bool = False

    @property
    def pace_sec_per_km(self) -> float | None:
        """Mean pace in seconds per km. None when distance/duration missing."""
        if self.distance_m and self.distance_m > 0 and self.duration_s:
            return self.duration_s / (self.distance_m / 1000.0)
        return None


@dataclass
class Race:
    date: date
    name: str = ""
    distance_m: float | None = None
    finish_time_s: float | None = None
    field_size: int | None = None
    matched_finishers: int | None = None
    field_median_slow_pct: float | None = None
    conditions_note: str | None = None


@dataclass
class Athlete:
    id: str
    display_label: str
    source: str
    import_date: str = ""
    window_start: date | None = None
    window_end: date | None = None
    sleep_nights: list = field(default_factory=list)
    daily: list = field(default_factory=list)
    activities: list = field(default_factory=list)
    races: list = field(default_factory=list)
    import_report: dict = field(default_factory=dict)

    # -- derived lookups -------------------------------------------------
    @property
    def sleep_by_date(self) -> dict:
        return {n.wake_date: n for n in self.sleep_nights
                if not n.excluded_sensor_unreliable}

    @property
    def daily_by_date(self) -> dict:
        return {d.date: d for d in self.daily}

    @property
    def runs(self) -> list:
        return [a for a in self.activities
                if a.kind == "run" and not a.excluded_sensor_unreliable]

    @property
    def anchor_date(self) -> date | None:
        """Window anchor: the latest date carrying any data.

        Demo profiles are frozen imports, so "last 28 days" and "last
        180 days" are relative to the data, not to the wall clock.
        """
        dates = [n.wake_date for n in self.sleep_nights]
        dates += [d.date for d in self.daily]
        dates += [a.date for a in self.activities]
        return max(dates) if dates else None

    def day_run_pace(self, day: date, include_excluded=False) -> float | None:
        """Distance-weighted mean run pace (sec/km) for one day, or None.

        include_excluded=True counts sensor-excluded runs too; used to
        count collected (pre-exclusion) pairs for the completeness guard.
        """
        pool = self.activities if include_excluded else self.runs
        runs = [a for a in pool if a.kind == "run" and a.date == day
                and a.pace_sec_per_km]
        if not runs:
            return None
        total_m = sum(a.distance_m for a in runs)
        if total_m <= 0:
            return None
        return sum(a.pace_sec_per_km * a.distance_m for a in runs) / total_m


def load_normalized(path: str) -> Athlete:
    """Load an athlete from the normalized-JSON schema (see adapters)."""
    with open(path) as fh:
        raw = json.load(fh)
    a = raw["athlete"]
    ath = Athlete(
        id=a["id"],
        display_label=a["display_label"],
        source=a["source"],
        import_date=a.get("import_date", ""),
        window_start=parse_date(a.get("window_start")),
        window_end=parse_date(a.get("window_end")),
        import_report=raw.get("import_report", {}),
    )
    for n in raw.get("sleep_nights", []):
        ath.sleep_nights.append(SleepNight(
            wake_date=parse_date(n["wake_date"]),
            duration_min=float(n["duration_min"]),
            bedtime_min_past_21=float(n["bedtime_min_past_21"]),
            sleep_efficiency_pct=(None if n.get("sleep_efficiency_pct") is None
                                  else float(n["sleep_efficiency_pct"])),
            excluded_sensor_unreliable=bool(n.get("excluded_sensor_unreliable", False)),
            exclusion_reason=n.get("exclusion_reason"),
        ))
    for d in raw.get("daily", []):
        ath.daily.append(Daily(
            date=parse_date(d["date"]),
            resting_hr_bpm=(None if d.get("resting_hr_bpm") is None
                            else float(d["resting_hr_bpm"])),
            hrv_ms=(None if d.get("hrv_ms") is None else float(d["hrv_ms"])),
        ))
    for ac in raw.get("activities", []):
        ath.activities.append(Activity(
            date=parse_date(ac["date"]),
            start_time=ac.get("start_time", ""),
            kind=ac.get("kind", "other"),
            distance_m=(None if ac.get("distance_m") is None else float(ac["distance_m"])),
            duration_s=(None if ac.get("duration_s") is None else float(ac["duration_s"])),
            avg_hr_bpm=(None if ac.get("avg_hr_bpm") is None else float(ac["avg_hr_bpm"])),
            temp_c=(None if ac.get("temp_c") is None else float(ac["temp_c"])),
            is_race=bool(ac.get("is_race", False)),
            excluded_sensor_unreliable=bool(ac.get("excluded_sensor_unreliable", False)),
        ))
    for r in raw.get("races", []):
        ath.races.append(Race(
            date=parse_date(r["date"]),
            name=r.get("name", ""),
            distance_m=(None if r.get("distance_m") is None else float(r["distance_m"])),
            finish_time_s=(None if r.get("finish_time_s") is None else float(r["finish_time_s"])),
            field_size=r.get("field_size"),
            matched_finishers=r.get("matched_finishers"),
            field_median_slow_pct=(None if r.get("field_median_slow_pct") is None
                                   else float(r["field_median_slow_pct"])),
            conditions_note=r.get("conditions_note"),
        ))
    return ath


def load_fixture(path: str) -> Athlete:
    """Load the synthetic validation fixture.

    The fixture stores nightly rows only (bedtime, duration, RHR,
    efficiency). Its RHR values are adapted into daily rows dated on the
    wake date, so the same pair builders run unchanged.
    """
    with open(path) as fh:
        raw = json.load(fh)
    a = raw["athlete"]
    ath = Athlete(
        id=a["id"],
        display_label=a["label"],
        source=a["source"],
        import_date="2026-10-28",
    )
    for n in raw["sleep_nights"]:
        wd = parse_date(n["wake_date"])
        ath.sleep_nights.append(SleepNight(
            wake_date=wd,
            duration_min=float(n["duration_min"]),
            bedtime_min_past_21=float(n["bedtime_min_past_21"]),
            sleep_efficiency_pct=(None if n.get("sleep_efficiency_pct") is None
                                  else float(n["sleep_efficiency_pct"])),
        ))
        if n.get("resting_hr_bpm") is not None:
            ath.daily.append(Daily(date=wd, resting_hr_bpm=float(n["resting_hr_bpm"])))
    return ath


def monday_of(d: date) -> date:
    return d - timedelta(days=d.weekday())
