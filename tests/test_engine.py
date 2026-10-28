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
from engine.model import Athlete, Daily, SleepNight, load_fixture
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


if __name__ == "__main__":
    unittest.main()
