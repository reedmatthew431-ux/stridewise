#!/usr/bin/env python3
"""Garmin GDPR "Download Your Data" export -> Stridewise normalized JSON.

Stdlib only. Reads the JSON summaries in DI_CONNECT; the nested FIT
archives under DI-Connect-Uploaded-Files are deliberately NOT opened
(the JSON summaries carry every field the model needs).

GPS RULE: this adapter reads a whitelist of scalar fields only
(activity type, timestamps, duration, distance, heart rate; sleep
stage seconds and GMT timestamps; daily resting HR). It never reads
latitude/longitude, location names, tracks, routes, courses, or any
FIT position data, and the normalized output contains zero
GPS/coordinate fields.

Usage:
    python3 adapter_garmin.py <export_root> <athlete_id> <label> <output.json>
    export_root: directory containing DI_CONNECT/ (the GDPR export root).

Rules implemented (see task spec):
  * Activities come from DI-Connect-Fitness/*_summarizedActivities.json
    (each file holds one list of records under
    "summarizedActivitiesExport"). kind is "run" iff sportType is
    "RUNNING" (covers running, track_running, treadmill_running);
    everything else is "other". is_race is always false: no race
    list exists for this profile, and nothing is assumed a race.
  * Activity units, verified against the data (not assumed):
    duration is in MILLISECONDS and distance in CENTIMETERS
    (elevation in cm too). Proof: distance/duration reproduces the
    stored avgSpeed to <0.005 on 1151/1157 checkable records (the 6
    exceptions are all lap_swimming, where Garmin's avgSpeed uses a
    different basis; no run is affected); cadence math agrees
    (steps / duration-minutes = recorded double cadence). So
    duration_s = duration_ms / 1000, distance_m = distance_cm / 100.
  * Activity dates are LOCAL athlete dates: startTimeLocal is local
    wall-clock time packed into epoch milliseconds (it differs from
    startTimeGmt by exactly the UTC offset: 5h in EST, 4h in EDT),
    so the local date/time is the UTC reading of startTimeLocal.
    No timezone conversion is applied to activities.
  * Sleep comes from DI-Connect-Wellness/*_sleepData.json.
    sleepStart/EndTimestampGMT are naive GMT timestamps ("...T02:53:00.0").
    They are converted to America/New_York for bedtime and for the
    wake-date check: wake_date is the LOCAL date of the converted end
    time. Verified on this export: the converted end date equals
    calendarDate on all 755 records (0 mismatches), so calendarDate
    IS the wake date, but the adapter trusts the conversion, not the
    label. Bedtime is the converted local start, as minutes past
    21:00 unwrapped across midnight (same formula as the Apple
    adapter: clock_min >= 720 -> clock_min - 1260, else clock_min
    + 180).
  * Sleep duration_min = (deep + light + REM seconds) / 60 -- asleep
    time only. Awake seconds in bed are NOT counted (same rule as
    the Apple adapter's Asleep-only duration). The timestamp span
    (end - start) is asleep + awake and is not used.
    unmeasurableSeconds is neither asleep nor an exclusion trigger
    (4 records carry it; the asleep sum stands).
  * sleep_efficiency_pct is always null: sleepData.json reports no
    efficiency field, and no proxy is derived (same source-aware
    rule as the Apple adapter).
  * Daily rows come from DI-Connect-Aggregator/UDSFile_*.json
    (calendarDate is a local date; wellnessStartTimeLocal confirms
    midnight-local day boundaries). resting_hr_bpm is
    restingHeartRate, the finalized day value -- NOT
    currentDayRestingHeartRate, the intraday estimate (the two differ
    on 520/655 days, typically by 1-3 bpm). hrv_ms is always null:
    no HRV field exists in UDS or in bioMetrics_latest (which is
    empty); the userBioMetrics file holds versioned profile settings,
    not daily values.
  * Rejected and counted: sleep nights with duration <= 0 or > 1440
    min; sleep nights of 0-60 min are tracking fragments, rejected
    as sleep_fragment_under_60_min (fragment guard, approved
    2026-10-30); activities missing startTimeLocal or with duration <= 0
    or distance < 0; daily rows with resting HR outside 25-220 bpm.
    A null distance or null avg HR is NOT a rejection: the activity
    is emitted with that field null (pace/mileage simply skip it).
    An out-of-range avg HR is nulled, not dropped (0 occurrences in
    this export).
  * Duplicate handling (same merge rules as the Apple adapter):
    two sleep records sharing a wake date merge into ONE night
    (stage seconds summed, earliest local bedtime); two UDS rows
    sharing a date keep the first. merged_duplicate_nights counts
    merged wake dates. (This export has 0 sleep duplicates and 0
    UDS duplicates; the rules exist for re-imports.)
  * excluded_sensor_unreliable is false for all rows in v1 (same as
    the Apple adapter); implausible rows are rejected instead.
  * temp_c is always null (summaries carry no activity temperature);
    races is always [].

Counting definitions (stated in import_report.notes):
  * rows_in: source records examined (activity records + sleep
    records + UDS rows). Files skipped entirely (Metrics, User,
    Hydration, Golf, Uploaded-Files FIT archives, social/profile
    images, customer/consent/events files) are not counted.
  * rows_rejected: dropped units (rejected activities + rejected
    sleep nights + rejected daily rows). Because sleep rejection is
    per merged night while rows_in counts per record, rows_in !=
    rows_rejected + emitted rows when duplicates merge.
  * window_excluded is always 0: this adapter normalizes the full
    history, no cutoff.
  * wake_date_mismatches: sleep records whose converted local end
    date differs from calendarDate (0 in this export).
"""

import glob
import json
import sys
from datetime import date as _date
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")


def bedtime_min_past_21(dt):
    """Local datetime -> minutes past 21:00, unwrapped across midnight."""
    clock_min = dt.hour * 60 + dt.minute
    if clock_min >= 720:
        return float(clock_min - 1260)
    return float(clock_min + 180)


def parse_gmt(s):
    """Parse a naive GMT timestamp ("2026-07-19T02:53:00.0") as UTC."""
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def local_from_epoch_ms(ms):
    """startTimeLocal is local wall-clock packed into epoch ms: read it
    as UTC to recover the local date/time. Returns (date_str, "HH:MM")."""
    dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    return dt.date().isoformat(), dt.strftime("%H:%M")


class Adapter:
    def __init__(self):
        self.rows_in = 0
        self.rejections = {}  # reason -> count
        self.merged_duplicate_nights = 0
        self.wake_date_mismatches = 0
        self.sleep = {}  # wake_date -> {"deep"/"light"/"rem"/"awake": s, "bedtime": dt}
        self.daily = {}  # date -> {"resting_hr_bpm": v}
        self.activities = []

    def reject(self, reason, n=1):
        self.rejections[reason] = self.rejections.get(reason, 0) + n

    # ---- activities -----------------------------------------------------
    def handle_activities(self, path):
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        for item in doc if isinstance(doc, list) else []:
            for a in item.get("summarizedActivitiesExport", []):
                self.rows_in += 1
                start_ms = a.get("startTimeLocal")
                duration_ms = a.get("duration")
                if not start_ms or not duration_ms or duration_ms <= 0:
                    self.reject("activity_missing_date_or_duration")
                    continue
                distance_cm = a.get("distance")
                if distance_cm is not None and distance_cm < 0:
                    self.reject("activity_negative_distance")
                    continue
                date_str, start_time = local_from_epoch_ms(start_ms)
                avg_hr = a.get("avgHr")
                if avg_hr is not None:
                    try:
                        avg_hr = float(avg_hr)
                    except (TypeError, ValueError):
                        avg_hr = None
                    if avg_hr is not None and (avg_hr < 25 or avg_hr > 220):
                        avg_hr = None  # implausible HR nulled, activity kept
                self.activities.append({
                    "date": date_str,
                    "start_time": start_time,
                    "kind": "run" if a.get("sportType") == "RUNNING" else "other",
                    "distance_m": (None if distance_cm is None
                                   else distance_cm / 100.0),
                    "duration_s": duration_ms / 1000.0,
                    "avg_hr_bpm": avg_hr,
                    "temp_c": None,
                    "is_race": False,
                    "excluded_sensor_unreliable": False,
                })

    # ---- sleep ----------------------------------------------------------
    def handle_sleep(self, path):
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        for r in doc if isinstance(doc, list) else []:
            self.rows_in += 1
            start_gmt = parse_gmt(r.get("sleepStartTimestampGMT"))
            end_gmt = parse_gmt(r.get("sleepEndTimestampGMT"))
            if start_gmt is None or end_gmt is None:
                self.reject("sleep_missing_timestamps")
                continue
            start_local = start_gmt.astimezone(NY)
            end_local = end_gmt.astimezone(NY)
            wake_date = end_local.date().isoformat()
            if r.get("calendarDate") is not None and wake_date != r["calendarDate"]:
                self.wake_date_mismatches += 1
            night = self.sleep.setdefault(wake_date, {
                "deep": 0.0, "light": 0.0, "rem": 0.0, "awake": 0.0,
                "bedtime": start_local, "sessions": 0,
            })
            night["deep"] += float(r.get("deepSleepSeconds") or 0)
            night["light"] += float(r.get("lightSleepSeconds") or 0)
            night["rem"] += float(r.get("remSleepSeconds") or 0)
            night["awake"] += float(r.get("awakeSleepSeconds") or 0)
            if start_local < night["bedtime"]:
                night["bedtime"] = start_local
            night["sessions"] += 1

    # ---- daily (UDS) ------------------------------------------------------
    def handle_daily(self, path):
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        for row in doc if isinstance(doc, list) else []:
            self.rows_in += 1
            date_str = row.get("calendarDate")
            if date_str is None:
                self.reject("daily_missing_date")
                continue
            if date_str in self.daily:
                continue  # windows abut; first row wins, duplicates uncounted
            rhr = row.get("restingHeartRate")
            if rhr is None:
                # A day with no resting HR still has no usable daily value.
                self.reject("daily_missing_resting_hr")
                continue
            try:
                rhr = float(rhr)
            except (TypeError, ValueError):
                self.reject("daily_non_numeric_resting_hr")
                continue
            if rhr < 25 or rhr > 220:
                self.reject("resting_hr_out_of_range")
                continue
            self.daily[date_str] = {"date": date_str, "resting_hr_bpm": rhr,
                                    "hrv_ms": None}

    # ---- output ---------------------------------------------------------
    def build(self, athlete_id, label, import_date):
        sleep_nights = []
        for wake_date in sorted(self.sleep):
            night = self.sleep[wake_date]
            if night["sessions"] >= 2:
                self.merged_duplicate_nights += 1
            duration_min = (night["deep"] + night["light"] + night["rem"]) / 60.0
            if duration_min <= 0 or duration_min > 1440:
                self.reject("sleep_duration_out_of_range")
                continue
            if duration_min < 60:
                # Fragment guard (Matt, approved 2026-10-30): a merged
                # night under 60 minutes asleep is a tracking fragment,
                # not a night of sleep. Judged AFTER duplicate-night
                # merging, so a fragment merged into a real night is
                # not rejected.
                self.reject("sleep_fragment_under_60_min")
                continue
            sleep_nights.append({
                "wake_date": wake_date,
                "duration_min": round(duration_min, 2),
                "bedtime_min_past_21": bedtime_min_past_21(night["bedtime"]),
                "sleep_efficiency_pct": None,
                "awake_min": round(night["awake"] / 60.0, 2),
                "excluded_sensor_unreliable": False,
                "exclusion_reason": None,
            })

        daily = [self.daily[d] for d in sorted(self.daily)]
        activities = sorted(self.activities,
                            key=lambda a: (a["date"], a["start_time"]))
        all_dates = ([n["wake_date"] for n in sleep_nights]
                     + [d["date"] for d in daily]
                     + [a["date"] for a in activities])
        window_start = min(all_dates) if all_dates else None
        window_end = max(all_dates) if all_dates else None
        rows_rejected = sum(self.rejections.values())
        return {
            "athlete": {
                "id": athlete_id,
                "display_label": label,
                "source": "garmin",
                "import_date": import_date,
                "window_start": window_start,
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
                "window_excluded": 0,
                "merged_duplicate_nights": self.merged_duplicate_nights,
                "wake_date_mismatches": self.wake_date_mismatches,
                "notes": [
                    "Activities: DI-Connect-Fitness summarizedActivities JSONs; kind run iff sportType == RUNNING; is_race always false (no race list exists for this profile).",
                    "Activity units verified from the data: duration in milliseconds, distance in centimeters (distance/duration reproduces stored avgSpeed on 1151/1157 checkable records; the 6 exceptions are lap_swimming, where Garmin's avgSpeed uses a different basis; no run affected). duration_s = ms/1000, distance_m = cm/100.",
                    "Activity dates are local dates from startTimeLocal (local wall-clock packed into epoch ms; differs from startTimeGmt by exactly the UTC offset). No timezone conversion applied to activities.",
                    "Sleep: DI-Connect-Wellness sleepData JSONs; sleepStart/EndTimestampGMT are naive GMT, converted to America/New_York. Wake date is the converted local end date (equals calendarDate on all records in this export; calendarDate is a cross-check, not the source of truth). Bedtime is the converted local start, as minutes past 21:00 unwrapped across midnight.",
                    "Sleep duration = (deep + light + REM seconds) / 60 (asleep only; awake seconds in bed not counted; timestamp span not used). awake_min = awakeSleepSeconds / 60, summed across merged records (time in bed not asleep, for question A9). sleep_efficiency_pct is always null: no efficiency field is reported, and no proxy is derived.",
                    "Daily: DI-Connect-Aggregator UDSFile JSONs; resting_hr_bpm is restingHeartRate (finalized day value), not currentDayRestingHeartRate (intraday estimate; the two differ on 520/655 days by 1-3 bpm). hrv_ms is always null: no HRV field exists in UDS or bioMetrics_latest.",
                    "excluded_sensor_unreliable is false for all rows in v1; implausible rows are rejected instead. A null distance or null avg HR is not a rejection (activity emitted with that field null).",
                    "Fragment guard (approved 2026-10-30): merged nights under 60 minutes asleep are tracking fragments, rejected and counted as sleep_fragment_under_60_min. The guard runs after duplicate-night merging, so a fragment merged into a real night is kept.",
                    "Deliberately skipped, never read: DI-Connect-Metrics (training history, race predictions, VO2max -- the model does not use Garmin's own predictions or scores), DI-Connect-User profile/social files, Hydration, Golf, customer/consent/events files, and the DI-Connect-Uploaded-Files FIT archives (JSON summaries carry every needed field).",
                    "GPS: none in this output. Only whitelisted scalar fields are read; latitude/longitude, location names, tracks, routes, courses, and FIT position data are never read or stored.",
                    "temp_c is always null; races is always [].",
                ],
            },
        }


def main(argv):
    if len(argv) < 5:
        print("usage: adapter_garmin.py <export_root> <athlete_id> "
              "<label> <output.json>", file=sys.stderr)
        return 2
    root, athlete_id, label, out_path = argv[1:5]
    adapter = Adapter()
    for path in sorted(glob.glob(root + "/DI_CONNECT/DI-Connect-Fitness/*_summarizedActivities.json")):
        adapter.handle_activities(path)
    for path in sorted(glob.glob(root + "/DI_CONNECT/DI-Connect-Wellness/*_sleepData.json")):
        adapter.handle_sleep(path)
    for path in sorted(glob.glob(root + "/DI_CONNECT/DI-Connect-Aggregator/UDSFile_*.json")):
        adapter.handle_daily(path)
    doc = adapter.build(athlete_id, label, _date.today().isoformat())
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    rep = doc["import_report"]
    print(f"{athlete_id}: sleep={len(doc['sleep_nights'])} "
          f"daily={len(doc['daily'])} activities={len(doc['activities'])} "
          f"runs={sum(1 for a in doc['activities'] if a['kind'] == 'run')} "
          f"rows_in={rep['rows_in']} rejected={rep['rows_rejected']} "
          f"merged={rep['merged_duplicate_nights']} "
          f"wake_mismatch={rep['wake_date_mismatches']} "
          f"window={doc['athlete']['window_start']}..{doc['athlete']['window_end']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
