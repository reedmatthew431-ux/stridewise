"""Engine tests (stdlib unittest, no third-party deps).

Run: python -m unittest discover -s tests -v
Covers: fixture recovery (planted truths), demotion display, and the
density + recency caps. The fixture itself is synthetic; these tests
use only synthetic data.
"""
import json
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.factpack import build_fact_pack
from engine.model import Activity, Athlete, Daily, Race, SleepNight, \
    load_fixture
from engine.registry import _hard_effort_hr, race_cards
from engine.stats import evaluate


def _athlete_from_nights(nights):
    """nights: list of (wake_date, bedtime_min_past_21, duration_min, rhr)."""
    ath = Athlete(id="t", display_label="test", source="synthetic")
    for wd, bt, du, hr in nights:
        ath.sleep_nights.append(SleepNight(
            wake_date=wd, duration_min=du, bedtime_min_past_21=bt,
            sleep_efficiency_pct=92.0))
        if hr is not None:
            ath.daily.append(Daily(date=wd, resting_hr_bpm=hr))
    return ath


class TestFixture(unittest.TestCase):
    def test_planted_truths(self):
        ath = load_fixture("fixtures/validation-fixture-1.json")
        pack = build_fact_pack(ath)
        by_q = {c["question"]: c for c in pack["cards"]}
        a1 = by_q["A1"]
        self.assertEqual(a1["tier"], "solid")
        self.assertTrue(-90 <= a1["effect"] <= -30)
        a3 = by_q["A3"]
        self.assertEqual(a3["tier"], "solid")
        self.assertTrue(-5 <= a3["effect"] <= -1)
        self.assertEqual(by_q["A6"]["tier"], "no_pattern")
        self.assertNotIn("A6", [c["question"] for c in pack["cards"]
                                if c["tier"] == "solid"])
        self.assertEqual(len(pack["race_cards"]), 0)


class TestCaps(unittest.TestCase):
    def _pairs(self, start, n, gap_days, slope_fn):
        """Synthetic pairs: x = day index, y = slope_fn(x) + tiny noise."""
        pairs = []
        d = start
        for i in range(n):
            x = float(i)
            pairs.append((d, x, slope_fn(x)))
            d += timedelta(days=gap_days)
        return pairs

    def test_density_cap(self):
        # 40 pairs spread over 120 days: density 40/120 < 50% -> capped.
        pairs = self._pairs(date(2026, 1, 1), 40, 3, lambda x: 500 - 2 * x)
        res = evaluate(pairs, 60.0, 15.0, "nightly", date(2026, 4, 30))
        self.assertEqual(res["tier"], "emerging")
        self.assertEqual(res["binding_rule"], "evidence too spread out")

    def test_recency_cap(self):
        # 60 dense pairs, but only 3 in the last 28 days -> stale cap.
        pairs = self._pairs(date(2026, 1, 1), 57, 1, lambda x: 500 - 2 * x)
        # Add 3 recent pairs continuing the trend.
        base = pairs[-1][0] + timedelta(days=30)
        for i in range(3):
            pairs.append((base + timedelta(days=i), 60.0 + i, 500 - 2 * (60 + i)))
        anchor = pairs[-1][0]
        res = evaluate(pairs, 60.0, 15.0, "nightly", anchor)
        self.assertEqual(res["tier"], "emerging")
        self.assertEqual(res["binding_rule"], "evidence is stale")

    def test_no_pattern_range_includes_zero(self):
        pairs = self._pairs(date(2026, 1, 1), 50, 1,
                            lambda x: 480 + (1 if x % 2 else -1))
        res = evaluate(pairs, 60.0, 15.0, "nightly", date(2026, 2, 20))
        self.assertEqual(res["tier"], "no_pattern")

    def test_insufficient_below_floor(self):
        pairs = self._pairs(date(2026, 1, 1), 10, 1, lambda x: 500 - 2 * x)
        res = evaluate(pairs, 60.0, 15.0, "nightly", date(2026, 1, 10))
        self.assertEqual(res["tier"], "insufficient_data")


class TestCompletenessWindow(unittest.TestCase):
    def test_long_history_does_not_break_completeness(self):
        # Regression: completeness must compare usable vs collected
        # INSIDE the window. An earlier version compared in-window
        # usable (110) against all-time collected (402) and wrongly
        # failed a strong-evidence profile at 27%.
        pairs = []
        d = date(2025, 1, 1)
        for i in range(292):  # old history, outside the window
            pairs.append((d, float(i), 500.0 - 2.0 * i))
            d += timedelta(days=1)
        anchor = date(2026, 10, 25)
        d = anchor - timedelta(days=109)
        for i in range(110):  # dense in-window evidence, clear slope
            pairs.append((d, float(i), 500.0 - 2.0 * i))
            d += timedelta(days=1)
        res = evaluate(pairs, 60.0, 15.0, "nightly", anchor,
                       collected_pairs=pairs)
        # 110 usable in-window pairs must NOT read as 110/402 = 27%.
        self.assertEqual(res["tier"], "solid")
        self.assertEqual(res["binding_rule"], "all gates passed")


class TestDemotion(unittest.TestCase):
    def test_solid_to_emerging_demotes(self):
        ath_full = load_fixture("fixtures/validation-fixture-1.json")
        full = build_fact_pack(ath_full)
        # Truncate to the first 30 nights: A1 n drops below the solid bar.
        ath_cut = load_fixture("fixtures/validation-fixture-1.json")
        ath_cut.sleep_nights = ath_cut.sleep_nights[:30]
        ath_cut.daily = [d for d in ath_cut.daily
                         if d.date <= ath_cut.sleep_nights[-1].wake_date]
        cut = build_fact_pack(ath_cut, prev_pack=full)
        a1 = next(c for c in cut["cards"] if c["question"] == "A1")
        self.assertEqual(a1["tier"], "emerging")
        self.assertIsNotNone(a1["demotion"])
        self.assertEqual(a1["demotion"]["was"], "solid")


class TestC3HardEfforts(unittest.TestCase):
    """C3's comparison set (2026-11-02 rebuild, narrowed to the top
    decile the same day): hard efforts only -- races plus runs whose
    avg HR is at or above the 90th percentile of run avg HRs. The
    old all-runs set was dominated by easy runs, so a real race-day
    under-read never looked unusual; the top quartile still admitted
    tempo runs, so the reference class is race-intensity efforts."""

    def _athlete(self):
        ath = Athlete(id="t", display_label="test", source="synthetic")
        d = date(2026, 8, 1)
        # 30 easy runs at 130 bpm.
        for i in range(30):
            ath.activities.append(Activity(
                date=d + timedelta(days=i), kind="run", avg_hr_bpm=130.0))
        # 10 hard workouts at 175 bpm.
        for i in range(10):
            ath.activities.append(Activity(
                date=d + timedelta(days=40 + i), kind="run",
                avg_hr_bpm=175.0))
        # 5 races: four honest efforts at 185, one under-read at 148.
        for i, hr in enumerate([185.0, 185.0, 185.0, 185.0, 148.0]):
            rd = date(2026, 9, 5) + timedelta(days=7 * i)
            ath.activities.append(Activity(
                date=rd, kind="run", avg_hr_bpm=hr, is_race=True))
            ath.races.append(Race(date=rd, name=f"Race {i}"))
        return ath

    def test_comparison_set_is_hard_efforts_only(self):
        ath = self._athlete()
        mean, sd, n = _hard_effort_hr(ath)
        # Pool P90 (90th percentile, quantiles n=10 inclusive) = 175
        # -> hard = 10 hard workouts + 4 honest races + the
        # under-read race itself (races count by definition) = 15.
        # The 30 easy runs at 130 are NOT in the set.
        self.assertEqual(n, 15)
        self.assertTrue(175.0 < mean < 176.5)

    def test_under_read_flags_and_honest_race_does_not(self):
        ath = self._athlete()
        c3 = {c["date"]: c for c in race_cards(ath)
              if c["question"] == "C3"}
        under = c3["2026-10-03"]  # the 148 bpm race (i=4)
        self.assertTrue(under["flagged"])
        self.assertEqual(under["comparison_n"], 15)
        self.assertTrue(175.0 < under["typical_avg_hr"] < 176.5)
        honest = c3["2026-09-05"]  # a 185 bpm race
        self.assertFalse(honest["flagged"])

    def test_thin_history_gives_no_verdict(self):
        # Only 4 runs carry HR in total (3 easy + the race itself):
        # below the >= 5 floor for any comparison set.
        ath = Athlete(id="t", display_label="test", source="synthetic")
        for i in range(3):
            ath.activities.append(Activity(
                date=date(2026, 9, 1) + timedelta(days=i), kind="run",
                avg_hr_bpm=150.0))
        rd = date(2026, 9, 10)
        ath.activities.append(Activity(
            date=rd, kind="run", avg_hr_bpm=100.0, is_race=True))
        ath.races.append(Race(date=rd, name="Only race"))
        mean, sd, n = _hard_effort_hr(ath)
        self.assertIsNone(mean)
        c3 = next(c for c in race_cards(ath) if c["question"] == "C3")
        self.assertFalse(c3["flagged"])
        self.assertIsNone(c3["typical_avg_hr"])

    def test_hard_set_below_five_gives_no_verdict(self):
        # Pool has >= 5 HRs but the top decile holds only 4 runs.
        ath = Athlete(id="t", display_label="test", source="synthetic")
        for i, hr in enumerate([100.0, 100.0, 100.0,
                                200.0, 200.0, 200.0, 200.0]):
            ath.activities.append(Activity(
                date=date(2026, 9, 1) + timedelta(days=i), kind="run",
                avg_hr_bpm=hr))
        mean, sd, n = _hard_effort_hr(ath)
        self.assertEqual(n, 4)
        self.assertIsNone(mean)


class TestA14(unittest.TestCase):
    """A14 (added 2026-11-02): REM share -> next-morning RHR, with
    the compositional caveat carried verbatim on the card sentence."""

    CAVEAT = ("Sleep stages add up to 100%: a higher REM share means "
              "a lower share of light or deep sleep, so this finding "
              "may reflect less light sleep rather than more REM itself.")

    def _athlete(self, with_stages, n=60):
        ath = Athlete(id="t", display_label="test", source="synthetic")
        d = date(2026, 5, 1)
        for i in range(n):
            share = 0.18 + 0.02 * (i % 6)  # 0.18 .. 0.28
            dur = 480.0
            rem = share * dur if with_stages else None
            ath.sleep_nights.append(SleepNight(
                wake_date=d + timedelta(days=i), duration_min=dur,
                bedtime_min_past_21=60.0,
                rem_min=rem,
                light_min=(dur - rem - 0.2 * dur) if with_stages else None,
                deep_min=(0.2 * dur) if with_stages else None))
            # Planted: higher REM share -> lower next-morning RHR
            # (-4 bpm per 10 percentage points of share).
            ath.daily.append(Daily(
                date=d + timedelta(days=i),
                resting_hr_bpm=round(60.0 - 40.0 * share, 2)))
        return ath

    def test_planted_rem_share_effect_and_caveat(self):
        pack = build_fact_pack(self._athlete(with_stages=True))
        a14 = next(c for c in pack["cards"] if c["question"] == "A14")
        self.assertIn(a14["tier"], ("solid", "emerging"))
        self.assertIsNotNone(a14["effect"])
        self.assertTrue(-6.0 < a14["effect"] < -2.0)
        self.assertEqual(a14["n"], 60)
        self.assertIn(self.CAVEAT, a14["sentence"])

    def test_no_stage_data_is_insufficient_with_caveat(self):
        pack = build_fact_pack(self._athlete(with_stages=False))
        a14 = next(c for c in pack["cards"] if c["question"] == "A14")
        self.assertEqual(a14["tier"], "insufficient_data")
        self.assertIn("sleep stage data", a14["sentence"])
        self.assertIn(self.CAVEAT, a14["sentence"])


class TestAnchorPin(unittest.TestCase):
    def _spanning_athlete(self):
        """Synthetic nights Aug 1 - Oct 31 2026 + races on both sides
        of the 2026-10-26 pin."""
        nights = []
        d = date(2026, 8, 1)
        i = 0
        while d <= date(2026, 10, 31):
            nights.append((d, 60.0 + (i % 5) * 10.0,
                           480.0 - 0.3 * i, 45.0))
            d += timedelta(days=1)
            i += 1
        ath = _athlete_from_nights(nights)
        ath.races.append(Race(date=date(2026, 10, 17), name="Early"))
        ath.races.append(Race(date=date(2026, 10, 31), name="Late"))
        return ath

    def test_anchor_excludes_later_rows_and_races(self):
        ath = self._spanning_athlete()
        pack = build_fact_pack(ath, anchor=date(2026, 10, 26))
        self.assertEqual(pack["window_anchor"], "2026-10-26")
        # Aug 1 - Oct 26 inclusive = 87 nights; the 5 later nights
        # (Oct 27-31) are excluded from the aggregates.
        self.assertEqual(pack["aggregates"]["nights_counted"], 87)
        # Only the pre-anchor race gets cards (5 per race).
        self.assertEqual(len(pack["race_cards"]), 5)
        self.assertEqual({c["date"] for c in pack["race_cards"]},
                         {"2026-10-17"})
        # Default behavior is unchanged: anchor is the last data
        # date and the late race is present.
        full = build_fact_pack(ath)
        self.assertEqual(full["window_anchor"], "2026-10-31")
        self.assertEqual(full["aggregates"]["nights_counted"], 92)
        self.assertEqual(len(full["race_cards"]), 10)

    def test_anchor_after_last_data_date(self):
        # Jaden's case: the shared pin (Oct 26) falls one day after
        # his last data (Oct 25). The board must compute, and the
        # pack's anchor must read the pin, not the last data date.
        nights = []
        d = date(2026, 9, 1)
        i = 0
        while d <= date(2026, 10, 25):
            nights.append((d, 60.0 + (i % 5) * 10.0,
                           480.0 - 0.3 * i, 45.0))
            d += timedelta(days=1)
            i += 1
        ath = _athlete_from_nights(nights)
        pack = build_fact_pack(ath, anchor="2026-10-26")
        self.assertEqual(pack["window_anchor"], "2026-10-26")
        self.assertEqual(len(pack["cards"]), 25)
        self.assertEqual(pack["aggregates"]["nights_counted"], 55)


if __name__ == "__main__":
    unittest.main()
