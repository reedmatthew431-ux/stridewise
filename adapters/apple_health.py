#!/usr/bin/env python3
"""Apple Health export.xml -> Stridewise normalized JSON adapter.

Stdlib only, truly streaming (xml.parsers.expat, no tree is built), so
400-580 MB export.xml files are processed in constant memory.

GPS RULE: WorkoutRoute / FileReference elements are never read into
output. This adapter captures only Workout attributes and
WorkoutStatistics (HeartRate average); it never opens, extracts, or
stores any route/GPX file, and the normalized output contains zero
GPS/coordinate fields.

Usage:
    python3 adapter_apple_health.py <export.xml> <athlete_id> <label> <output.json> [window_start]
    window_start: YYYY-MM-DD or "null"/omitted for full history.

Rules implemented (see task spec):
  * Dates are local athlete dates AS RECORDED in the file (the date
    prefix of startDate/endDate); no timezone conversion is applied.
  * Sleep: HKCategoryTypeIdentifierSleepAnalysis records, grouped by
    wake date = local date of endDate. Duration = sum of Asleep*
    segments only (AsleepCore/Deep/REM/Unspecified, plus legacy
    Asleep). InBed and Awake segments never count toward duration.
    Bedtime = start of the earliest Asleep segment of that night,
    emitted as minutes past 21:00 unwrapped across midnight:
    clock_min = hour*60+minute; value = clock_min-1260 if
    clock_min >= 720 else clock_min+180.
    Multiple sessions sharing a wake date merge into ONE night
    (earliest bedtime, summed duration, latest wake).
  * Sleep stages (for A14): each Asleep segment is classified by
    value -- AsleepREM -> rem_min, AsleepDeep -> deep_min,
    AsleepCore -> light_min. AsleepUnspecified and legacy Asleep
    carry no stage: their minutes count toward duration only. A
    night emits stage minutes iff at least one of its segments is
    stage-classified; otherwise all three stage fields are null.
  * sleep_efficiency_pct: always null (Apple Health does not report it).
  * excluded_sensor_unreliable: false for all rows in v1.
  * Implausible rows are REJECTED and counted: sleep duration <= 0
    or > 1440 min (per merged night); a merged night of 0-60 min is a
    tracking fragment, rejected as sleep_fragment_under_60_min
    (fragment guard, approved 2026-10-30); resting HR outside
    25-220 bpm (per sample; the day's median uses the remaining
    samples).
  * Daily resting HR / HRV: median of that local date's samples
    (RestingHeartRate in count/min, HeartRateVariabilitySDNN in ms).
    Dates with no samples are absent, never zero-filled.
  * Workouts: kind "run" iff workoutActivityType is
    HKWorkoutActivityTypeRunning, else "other". distance_m converts
    totalDistance (km -> *1000, mi -> *1609.344, m -> as-is) when the
    attribute exists; these exports do NOT carry totalDistance, so
    distance falls back to WorkoutStatistics distance SUMS (scalar
    totals only, e.g. DistanceWalkingRunning sum in mi/km/m -- never
    route data). Runs prefer DistanceWalkingRunning; other kinds take
    the first available distance sum. duration_s converts duration
    (min -> *60, h -> *3600, s -> as-is). Missing
    distance/duration/avg_hr -> null. avg_hr_bpm is the
    WorkoutStatistics average for HeartRate, else null.
    temp_c is always null; is_race is always false.
  * races: always [].
  * Records nested inside a <Correlation> are skipped: the DTD notes
    they also appear as top-level records, so counting them would
    double-count.

Counting definitions (stated in import_report.notes):
  * rows_in: relevant source records examined (sleep records,
    resting-HR records, HRV records, workouts). Other HealthKit types
    are skipped without counting.
  * window_excluded: relevant source records dropped by the window
    cutoff (per segment/sample/workout; sleep judged by wake date).
  * rows_rejected: rejected units (rejected sleep nights +
    rejected resting-HR samples + records missing a usable date).
    Because sleep rejection is per merged night while rows_in counts
    per segment, rows_in != rows_rejected + emitted rows for sleep.
  * merged_duplicate_nights: wake dates whose Asleep segments split
    into 2+ sessions, where a new session starts > 60 minutes after
    the previous Asleep segment ended for that wake date.
"""

import json
import statistics
import sys
from datetime import datetime
from xml.parsers import expat

ASLEEP_VALUES = {
    "HKCategoryValueSleepAnalysisAsleep",
    "HKCategoryValueSleepAnalysisAsleepCore",
    "HKCategoryValueSleepAnalysisAsleepDeep",
    "HKCategoryValueSleepAnalysisAsleepREM",
    "HKCategoryValueSleepAnalysisAsleepUnspecified",
}
# Stage classification per Asleep segment value. Legacy "Asleep" and
# "AsleepUnspecified" carry no stage: their minutes count toward
# duration but toward no stage. A night emits stage minutes iff at
# least one of its segments is stage-classified.
STAGE_BY_VALUE = {
    "HKCategoryValueSleepAnalysisAsleepREM": "rem",
    "HKCategoryValueSleepAnalysisAsleepDeep": "deep",
    "HKCategoryValueSleepAnalysisAsleepCore": "light",
}
SLEEP_TYPE = "HKCategoryTypeIdentifierSleepAnalysis"
RHR_TYPE = "HKQuantityTypeIdentifierRestingHeartRate"
HRV_TYPE = "HKQuantityTypeIdentifierHeartRateVariabilitySDNN"
HR_TYPE = "HKQuantityTypeIdentifierHeartRate"


def parse_dt(s):
    """Parse 'YYYY-MM-DD HH:MM:SS +/-ZZZZ' -> aware datetime or None."""
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S %z")
    except ValueError:
        try:
            return datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None


def local_date(s):
    """Local date as recorded: the YYYY-MM-DD prefix, no conversion."""
    return s[:10] if s and len(s) >= 10 else None


def local_time(s):
    return s[11:19] if s and len(s) >= 19 else None


def bedtime_min_past_21(dt):
    clock_min = dt.hour * 60 + dt.minute
    if clock_min >= 720:
        return float(clock_min - 1260)
    return float(clock_min + 180)


def convert_distance(value, unit):
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if unit == "km":
        return v * 1000.0
    if unit == "mi":
        return v * 1609.344
    # "m" and any other unit are treated as metres (Apple exports use km/m).
    return v


def convert_duration(value, unit):
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if unit == "min":
        return v * 60.0
    if unit == "h":
        return v * 3600.0
    if unit == "ms":
        return v / 1000.0
    return v  # "s" and default


class Adapter:
    def __init__(self, window_start):
        self.window_start = window_start  # "YYYY-MM-DD" or None
        self.rows_in = 0
        self.window_excluded = 0
        self.rejections = {}  # reason -> count
        # Sleep: wake_date -> {"segments": [(start_dt, end_dt, stage)], "latest_wake": dt}
        self.sleep = {}
        # Awake segments per wake date, for awake_min derivation.
        self.awake = {}
        # Every wake date seen on any in-window sleep record, including
        # InBed/Awake-only dates. Dates here with no Asleep segment are
        # zero-duration nights and are rejected at build time.
        self.sleep_wake_dates = set()
        self.rhr = {}  # date -> [values]
        self.hrv = {}  # date -> [values]
        self.activities = []
        # Workout in progress (Workout elements are not self-closing).
        self.workout = None
        self.stack = []  # element-name stack, for Correlation skipping

    def reject(self, reason, n=1):
        self.rejections[reason] = self.rejections.get(reason, 0) + n

    def in_window(self, date):
        return self.window_start is None or date is None or date >= self.window_start

    # ---- expat handlers -------------------------------------------------
    def start(self, name, attrs):
        if name == "Workout":
            self.stack.append(name)
            self.workout = {
                "date": local_date(attrs.get("startDate")),
                "start_time": local_time(attrs.get("startDate")),
                "kind": "run" if attrs.get("workoutActivityType")
                == "HKWorkoutActivityTypeRunning" else "other",
                "distance_m": convert_distance(
                    attrs.get("totalDistance"), attrs.get("totalDistanceUnit")),
                "duration_s": convert_duration(
                    attrs.get("duration"), attrs.get("durationUnit")),
                "avg_hr_bpm": None,
                "distance_sums": {},
            }
            return
        if name == "WorkoutStatistics" and self.workout is not None:
            wtype = attrs.get("type") or ""
            if wtype == HR_TYPE and attrs.get("average") is not None:
                try:
                    self.workout["avg_hr_bpm"] = float(attrs["average"])
                except ValueError:
                    pass
            elif wtype.startswith("HKQuantityTypeIdentifierDistance") \
                    and attrs.get("sum") is not None:
                # Scalar distance total only -- never route data.
                d = convert_distance(attrs.get("sum"), attrs.get("unit"))
                if d is not None:
                    self.workout["distance_sums"][wtype] = d
            return
        if name in ("WorkoutRoute", "FileReference"):
            # GPS data: deliberately never captured, never stored.
            self.stack.append(name)
            return
        if name == "Correlation":
            self.stack.append(name)
            return
        if name != "Record":
            return
        # Records nested in a Correlation duplicate top-level records.
        if "Correlation" in self.stack:
            return
        rtype = attrs.get("type")
        if rtype not in (SLEEP_TYPE, RHR_TYPE, HRV_TYPE):
            return
        self.rows_in += 1
        if rtype == SLEEP_TYPE:
            self.handle_sleep(attrs)
        elif rtype == RHR_TYPE:
            self.handle_quantity(attrs, self.rhr, 25.0, 220.0,
                                 "resting_hr_out_of_range")
        else:
            self.handle_quantity(attrs, self.hrv, None, None, None)

    def handle_sleep(self, attrs):
        value = attrs.get("value")
        wake_date = local_date(attrs.get("endDate"))
        if wake_date is None:
            self.reject("missing_date")
            return
        if not self.in_window(wake_date):
            self.window_excluded += 1
            return
        self.sleep_wake_dates.add(wake_date)
        if value not in ASLEEP_VALUES:
            # Awake segments are kept to derive awake_min (summed
            # detected wakefulness per wake date). InBed / unknown
            # values are examined (rows_in) but not stored: InBed
            # blocks on old data can span whole days (6-12h artifacts
            # on 2023 nights) and no InBed segments appear at all in
            # 2026 windows, so InBed-minus-Asleep is not a usable
            # awake measure. A wake date with only non-Asleep records
            # is a zero-duration night, rejected at build time.
            if value == "HKCategoryValueSleepAnalysisAwake":
                start_dt = parse_dt(attrs.get("startDate"))
                end_dt = parse_dt(attrs.get("endDate"))
                if start_dt is None or end_dt is None:
                    self.reject("missing_date")
                    return
                self.awake.setdefault(wake_date, []).append((start_dt, end_dt))
            return
        start_dt = parse_dt(attrs.get("startDate"))
        end_dt = parse_dt(attrs.get("endDate"))
        if start_dt is None or end_dt is None:
            self.reject("missing_date")
            return
        night = self.sleep.setdefault(
            wake_date, {"segments": [], "latest_wake": end_dt})
        night["segments"].append(
            (start_dt, end_dt, STAGE_BY_VALUE.get(value)))
        if end_dt > night["latest_wake"]:
            night["latest_wake"] = end_dt

    def handle_quantity(self, attrs, store, lo, hi, reject_reason):
        date = local_date(attrs.get("startDate"))
        if date is None:
            self.reject("missing_date")
            return
        if not self.in_window(date):
            self.window_excluded += 1
            return
        try:
            v = float(attrs.get("value"))
        except (TypeError, ValueError):
            self.reject("non_numeric_value")
            return
        if lo is not None and (v < lo or v > hi):
            self.reject(reject_reason)
            return
        store.setdefault(date, []).append(v)

    def end(self, name):
        if name == "Workout" and self.workout is not None:
            w = self.workout
            self.workout = None
            if self.stack and self.stack[-1] == "Workout":
                self.stack.pop()
            self.rows_in += 1
            if w["date"] is None or w["start_time"] is None:
                self.reject("missing_date")
                return
            if not self.in_window(w["date"]):
                self.window_excluded += 1
                return
            if w["distance_m"] is None and w["distance_sums"]:
                # Runs prefer walking/running distance; other kinds
                # take the first available distance sum.
                order = ["HKQuantityTypeIdentifierDistanceWalkingRunning",
                         "HKQuantityTypeIdentifierDistanceCycling",
                         "HKQuantityTypeIdentifierDistanceSwimming"]
                for key in order + sorted(w["distance_sums"]):
                    if key in w["distance_sums"]:
                        w["distance_m"] = w["distance_sums"][key]
                        break
            self.activities.append({
                "date": w["date"],
                "start_time": w["start_time"],
                "kind": w["kind"],
                "distance_m": w["distance_m"],
                "duration_s": w["duration_s"],
                "avg_hr_bpm": w["avg_hr_bpm"],
                "temp_c": None,
                "is_race": False,
                "excluded_sensor_unreliable": False,
            })
            return
        if self.stack and self.stack[-1] == name:
            self.stack.pop()

    # ---- output ---------------------------------------------------------
    def build(self, athlete_id, label, import_date):
        sleep_nights = []
        merged_duplicate_nights = 0
        # Wake dates with sleep records but no Asleep segment are
        # zero-duration nights -> rejected (duration <= 0 rule).
        for wake_date in sorted(self.sleep_wake_dates - set(self.sleep)):
            self.reject("sleep_duration_out_of_range")
        for wake_date in sorted(self.sleep):
            segments = sorted(self.sleep[wake_date]["segments"],
                              key=lambda seg: (seg[0], seg[1]))
            if not segments:
                continue
            # Session split: new session when a segment starts > 60 min
            # after the previous segment ended.
            sessions = 1
            prev_end = segments[0][1]
            for s, e, _stage in segments[1:]:
                if (s - prev_end).total_seconds() > 3600:
                    sessions += 1
                if e > prev_end:
                    prev_end = e
            if sessions >= 2:
                merged_duplicate_nights += 1
            duration_min = sum(
                (e - s).total_seconds() for s, e, _st in segments) / 60.0
            if duration_min <= 0 or duration_min > 1440:
                self.reject("sleep_duration_out_of_range")
                continue
            if duration_min < 60:
                # Fragment guard (Matt, approved 2026-10-30): a merged
                # night under 60 minutes asleep is a tracking fragment
                # (e.g. a watch worn for part of a night), not a night
                # of sleep. Judged AFTER duplicate-night merging, so a
                # fragment merged into a real night is not rejected.
                self.reject("sleep_fragment_under_60_min")
                continue
            bedtime = min(s for s, _e, _st in segments)
            # Stage minutes: summed per stage over stage-classified
            # segments. The night carries stage fields iff at least
            # one segment is classified; unclassified minutes
            # (Unspecified / legacy Asleep) count toward duration
            # only. All three fields are null together when no
            # segment is classified.
            stage_min = {"rem": 0.0, "light": 0.0, "deep": 0.0}
            any_stage = False
            for s, e, st in segments:
                if st is not None:
                    any_stage = True
                    stage_min[st] += (e - s).total_seconds() / 60.0
            # awake_min: summed Awake segments for this wake date
            # (0.0 when the night has Asleep segments but no detected
            # awakenings). This is detected mid-sleep wakefulness;
            # sleep-onset latency is not captured in stages-era
            # exports, unlike Garmin's awake seconds in the sleep
            # window -- the per-source definitions differ, and the
            # import notes say so.
            awake_segs = self.awake.get(wake_date, [])
            awake_min = round(sum((e - s).total_seconds()
                                  for s, e in awake_segs) / 60.0, 2)
            sleep_nights.append({
                "wake_date": wake_date,
                "duration_min": round(duration_min, 2),
                "bedtime_min_past_21": bedtime_min_past_21(bedtime),
                "sleep_efficiency_pct": None,
                "awake_min": awake_min,
                "rem_min": (round(stage_min["rem"], 2)
                            if any_stage else None),
                "light_min": (round(stage_min["light"], 2)
                              if any_stage else None),
                "deep_min": (round(stage_min["deep"], 2)
                             if any_stage else None),
                "excluded_sensor_unreliable": False,
                "exclusion_reason": None,
            })

        daily_dates = sorted(set(self.rhr) | set(self.hrv))
        daily = []
        for d in daily_dates:
            row = {"date": d,
                   "resting_hr_bpm": None,
                   "hrv_ms": None}
            if d in self.rhr:
                row["resting_hr_bpm"] = round(statistics.median(self.rhr[d]), 2)
            if d in self.hrv:
                row["hrv_ms"] = round(statistics.median(self.hrv[d]), 2)
            daily.append(row)

        activities = sorted(self.activities,
                            key=lambda a: (a["date"], a["start_time"]))
        all_dates = ([n["wake_date"] for n in sleep_nights]
                     + [d["date"] for d in daily]
                     + [a["date"] for a in activities])
        window_end = max(all_dates) if all_dates else self.window_start
        rows_rejected = sum(self.rejections.values())
        return {
            "athlete": {
                "id": athlete_id,
                "display_label": label,
                "source": "apple_health",
                "import_date": import_date,
                "window_start": self.window_start,
                "window_end": window_end,
            },
            "sleep_nights": sleep_nights,
            "daily": daily,
            "activities": activities,
            "races": [],
            "import_report": {
                "rows_in": self.rows_in,
                "rows_rejected": rows_rejected,
                "rejections": [{"reason": r, "count": c}
                                for r, c in sorted(self.rejections.items())],
                "window_excluded": self.window_excluded,
                "merged_duplicate_nights": merged_duplicate_nights,
                "notes": [
                    "Dates are local athlete dates as recorded in export.xml; no timezone conversion applied.",
                    "Sleep duration sums Asleep* segments only; InBed/Awake never counted. Bedtime is the earliest Asleep segment start, as minutes past 21:00 unwrapped across midnight.",
                    "awake_min sums Awake segments per wake date (0.0 when none detected) for question A9. This is detected mid-sleep wakefulness; sleep-onset latency is not captured in stages-era exports, unlike Garmin's awake seconds in the sleep window. InBed segments are never used for awake: old InBed blocks can span whole days (6-12h artifacts on 2023 nights) and no InBed segments appear in 2026 windows.",
                    "sleep_efficiency_pct is always null: Apple Health does not report it.",
                    "Stage minutes (rem_min/light_min/deep_min, for A14): AsleepREM/AsleepDeep/AsleepCore segments summed per wake date. AsleepUnspecified and legacy Asleep minutes count toward duration but no stage. A night carries stage fields only when at least one of its segments is stage-classified; otherwise all three are null.",
                    "excluded_sensor_unreliable is false for all rows in v1; implausible rows are rejected instead.",
                    "rows_in counts relevant source records only (sleep, resting HR, HRV, workouts); other HealthKit types skipped without counting. Records nested in Correlation elements skipped (they duplicate top-level records).",
                    "window_excluded counts relevant source records dropped by the window cutoff (sleep judged by wake date, others by start date).",
                    "merged_duplicate_nights counts wake dates whose Asleep segments split into 2+ sessions (new session = segment starting >60 min after previous segment ended).",
                    "Fragment guard (approved 2026-10-30): merged nights under 60 minutes asleep are tracking fragments, rejected and counted as sleep_fragment_under_60_min. The guard runs after duplicate-night merging, so a fragment merged into a real night is kept.",
                    "Location data: none in this output; no external location file was opened.",
                    "Distance: totalDistance attribute when present, else WorkoutStatistics distance sums (scalar totals only, never route data).",
                    "temp_c is always null; is_race is always false; races is always [].",
                ],
            },
        }


def main(argv):
    if len(argv) < 5:
        print("usage: adapter_apple_health.py <export.xml> <athlete_id> "
              "<label> <output.json> [window_start| null]", file=sys.stderr)
        return 2
    src, athlete_id, label, out_path = argv[1:5]
    window_start = argv[5] if len(argv) > 5 else None
    if window_start in (None, "", "null", "None"):
        window_start = None
    adapter = Adapter(window_start)
    parser = expat.ParserCreate()
    parser.StartElementHandler = adapter.start
    parser.EndElementHandler = adapter.end
    with open(src, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            parser.Parse(chunk, False)
        parser.Parse(b"", True)
    from datetime import date as _date
    doc = adapter.build(athlete_id, label, _date.today().isoformat())
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    rep = doc["import_report"]
    print(f"{athlete_id}: sleep={len(doc['sleep_nights'])} "
          f"daily={len(doc['daily'])} activities={len(doc['activities'])} "
          f"rows_in={rep['rows_in']} rejected={rep['rows_rejected']} "
          f"window_excluded={rep['window_excluded']} "
          f"merged={rep['merged_duplicate_nights']} "
          f"window={doc['athlete']['window_start']}..{doc['athlete']['window_end']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
