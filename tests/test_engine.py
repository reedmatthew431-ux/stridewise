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
from engine.model import Athlete, Daily, Race, SleepNight, load_fixture
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
        self.assertEqual(len(pack["cards"]), 24)
        self.assertEqual(pack["aggregates"]["nights_counted"], 55)


if __name__ == "__main__":
    unittest.main()
