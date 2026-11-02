"""Ask-layer tests (stdlib unittest, no third-party deps).

Run: python -m unittest discover -s tests -v
Covers: the fixture still recovers its planted truths through the
board path; a known ask pair on the synthetic fixture answers
sanely; registry pairs and coupled pairs are never ask answers;
steps and raw pace are never offered; stage-share answers carry
the compositional caveat; dropdown availability follows the data.
Synthetic data only.
"""
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.ask import (COUPLED_PAIRS, NIGHTLY_DRIVERS, NIGHTLY_OUTCOMES,
                        REGISTRY_PAIRS, SAME_MEASUREMENT, WEEKLY_DRIVERS,
                        WEEKLY_OUTCOMES, build_ask)
from engine.factpack import build_fact_pack
from engine.model import Activity, Athlete, Daily, SleepNight, load_fixture
from engine.registry import A14_CAVEAT


def _full_athlete(with_hrv=True, with_runs=True, n=70):
    """Synthetic athlete with every field the ask space can use:
    stages, awake, HRV, and frequent runs (so weekly variables and
    run quality exist). No planted ask effect except a mild
    duration -> RHR slope inherited from the RHR formula."""
    ath = Athlete(id="t", display_label="test", source="synthetic")
    d0 = date(2026, 4, 20)
    for i in range(n):
        d = d0 + timedelta(days=i)
        dur = 450.0 + (i % 7) * 8.0
        ath.sleep_nights.append(SleepNight(
            wake_date=d, duration_min=dur,
            bedtime_min_past_21=30.0 + (i % 5) * 12.0,
            awake_min=15.0 + (i % 4) * 5.0,
            rem_min=0.22 * dur, light_min=0.55 * dur,
            deep_min=0.23 * dur))
        ath.daily.append(Daily(
            date=d, resting_hr_bpm=52.0 - 0.004 * dur,
            hrv_ms=(55.0 + (i % 6)) if with_hrv else None))
        if with_runs and i % 2 == 0:
            ath.activities.append(Activity(
                date=d, kind="run", distance_m=10000.0,
                duration_s=2700.0, avg_hr_bpm=150.0))
    return ath


def _pair(pack, driver, outcome):
    for p in pack["pairs"]:
        if p["driver"] == driver and p["outcome"] == outcome:
            return p
    return None


class TestFixtureStillRecovers(unittest.TestCase):
    def test_board_truths_unchanged(self):
        ath = load_fixture("fixtures/validation-fixture-1.json")
        pack = build_fact_pack(ath)
        by_q = {c["question"]: c for c in pack["cards"]}
        self.assertEqual(by_q["A1"]["tier"], "solid")
        self.assertTrue(-90 <= by_q["A1"]["effect"] <= -30)
        self.assertEqual(by_q["A3"]["tier"], "solid")
        self.assertEqual(by_q["A6"]["tier"], "no_pattern")

    def test_known_pair_answers_sanely(self):
        # Planted chain in the fixture: later bedtime -> shorter
        # sleep (A1) and shorter sleep -> higher RHR (A3), so
        # bedtime -> next-morning RHR must come out positive and
        # gated at solid/emerging -- through the ask gate.
        ath = load_fixture("fixtures/validation-fixture-1.json")
        pack = build_ask(ath)
        p = _pair(pack, "bedtime", "rhr")
        self.assertIsNotNone(p)
        self.assertIn(p["tier"], ("solid", "emerging"))
        self.assertGreater(p["effect"], 1.0)
        self.assertEqual(p["n"], 60)
        self.assertTrue(p["asked_on_demand"])
        # An unplanted pair stays an honest non-answer.
        w = _pair(pack, "weekend", "rhr")
        self.assertIsNotNone(w)
        self.assertIn(w["tier"], ("no_pattern", "insufficient_data"))


class TestPairSpaceExclusions(unittest.TestCase):
    def test_registry_pairs_absent(self):
        pack = build_ask(_full_athlete())
        offered = {(p["driver"], p["outcome"]) for p in pack["pairs"]}
        for pair in REGISTRY_PAIRS:
            self.assertNotIn(pair, offered)
        # ... and every registry pair whose fields exist is instead
        # routed to its board card.
        for (d, o), qid in REGISTRY_PAIRS.items():
            self.assertEqual(pack["on_board"].get(f"{d}|{o}"), qid)

    def test_coupled_and_same_measurement_absent(self):
        pack = build_ask(_full_athlete())
        offered = {(p["driver"], p["outcome"]) for p in pack["pairs"]}
        for pair in COUPLED_PAIRS:
            self.assertNotIn(pair, offered)
            d, o = pair
            self.assertIn(f"{d}|{o}", pack["coupled"])
        for pair in SAME_MEASUREMENT:
            self.assertNotIn(pair, offered)
        self.assertIn("sleep_duration|duration",
                      pack["same_measurement"])

    def test_steps_and_raw_pace_never_offered(self):
        specs = (NIGHTLY_DRIVERS + WEEKLY_DRIVERS
                 + NIGHTLY_OUTCOMES + WEEKLY_OUTCOMES)
        for spec in specs:
            self.assertNotIn("step", spec[0].lower())
            self.assertNotIn("step", spec[2].lower())
            self.assertNotEqual(spec[0], "pace")
        pack = build_ask(_full_athlete())
        keys = ({d["key"] for d in pack["drivers"]}
                | {o["key"] for o in pack["outcomes"]})
        self.assertNotIn("steps", keys)
        self.assertNotIn("pace", keys)
        # Run quality is the only pace-carrying variable, as on
        # the board.
        self.assertIn("run_quality", keys)


class TestCaveatAndAvailability(unittest.TestCase):
    def test_stage_share_answers_carry_caveat(self):
        pack = build_ask(_full_athlete())
        stage = [p for p in pack["pairs"]
                 if p["driver"] in ("rem_share", "light_share",
                                    "deep_share")]
        self.assertTrue(stage)
        for p in stage:
            self.assertEqual(p["caveat"], A14_CAVEAT)
            self.assertIn(A14_CAVEAT, p["sentence"])
        non_stage = [p for p in pack["pairs"]
                     if p["driver"] == "bedtime"]
        for p in non_stage:
            self.assertIsNone(p["caveat"])

    def test_availability_follows_data(self):
        # No HRV anywhere -> the HRV outcome is not offered, and no
        # pair uses it.
        pack = build_ask(_full_athlete(with_hrv=False))
        okeys = {o["key"] for o in pack["outcomes"]}
        self.assertNotIn("hrv", okeys)
        self.assertFalse(any(p["outcome"] == "hrv"
                             for p in pack["pairs"]))
        # No runs -> no run-quality outcome and no weekly variables.
        pack = build_ask(_full_athlete(with_runs=False))
        okeys = {o["key"] for o in pack["outcomes"]}
        dkeys = {d["key"] for d in pack["drivers"]}
        self.assertNotIn("run_quality", okeys)
        self.assertNotIn("next_quality", okeys)
        self.assertFalse(dkeys & {"mileage", "acwr", "monotony",
                                  "long_run", "volatility"})
        self.assertFalse(any(p["grain"] == "weekly"
                             for p in pack["pairs"]))

    def test_fixture_availability(self):
        # The fixture carries no HRV, no awake, no stages, no runs:
        # none of those variables may be offered.
        pack = build_ask(load_fixture("fixtures/validation-fixture-1.json"))
        dkeys = {d["key"] for d in pack["drivers"]}
        okeys = {o["key"] for o in pack["outcomes"]}
        self.assertFalse(dkeys & {"awake_min", "rem_share",
                                  "light_share", "deep_share",
                                  "mileage", "acwr", "monotony",
                                  "long_run", "volatility"})
        self.assertFalse(okeys & {"hrv", "run_quality", "next_quality",
                                  "next_rhr"})


if __name__ == "__main__":
    unittest.main()
