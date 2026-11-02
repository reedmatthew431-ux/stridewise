"""Proposal-layer tests (stdlib unittest, no network, no API calls).

Covers: the transform construction mapping (lag / rolling / threshold
values built from a synthetic series), duplicate rejection through
the transform aliases (previous-night length IS length at lag1),
coupled-pair rejection, menu discipline, rationale validation, and
the gate path end to end with an injected fake Spark call -- plus
the template fallback when Spark's output is unusable twice.
Synthetic data only.

Run: python -m unittest discover -s tests -v
"""
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import ask as ask_mod
from engine.model import Activity, Athlete, Daily, SleepNight
from engine.proposals import (_transformed_nightly, classify,
                              covered_pairs, generate_proposals,
                              validate_proposals)


def _athlete(n=70):
    ath = Athlete(id="t", display_label="test", source="synthetic")
    d0 = date(2026, 4, 20)
    for i in range(n):
        d = d0 + timedelta(days=i)
        dur = 400.0 + (i % 9) * 10.0
        ath.sleep_nights.append(SleepNight(
            wake_date=d, duration_min=dur,
            bedtime_min_past_21=60.0 + (i % 5) * 15.0,
            awake_min=10.0 + (i % 4) * 5.0,
            rem_min=0.22 * dur, light_min=0.55 * dur,
            deep_min=0.23 * dur))
        ath.daily.append(Daily(date=d, resting_hr_bpm=52.0,
                               hrv_ms=55.0))
        if i % 2 == 0:
            ath.activities.append(Activity(
                date=d, kind="run", distance_m=10000.0,
                duration_s=2700.0, avg_hr_bpm=150.0))
    return ath


def _covered(ath):
    pack = ask_mod.build_ask(ath)
    covered, sources = covered_pairs(pack)
    avail_d = [d["key"] for d in pack["drivers"]]
    avail_o = [o["key"] for o in pack["outcomes"]]
    return covered, sources, avail_d, avail_o


class TestConstruction(unittest.TestCase):
    def setUp(self):
        self.ath = _athlete()
        self.d0 = date(2026, 4, 20)

    def test_lag1_shifts_by_one_night(self):
        vals, _ = _transformed_nightly(self.ath, "sleep_duration", "lag1")
        # Night i=5 has duration 450; the pair date i=6 must carry it.
        self.assertEqual(vals[self.d0 + timedelta(days=6)], 450.0)
        self.assertIsNone(vals[self.d0])

    def test_roll3_is_three_night_mean(self):
        vals, _ = _transformed_nightly(self.ath, "sleep_duration", "roll3")
        d = self.d0 + timedelta(days=2)
        expect = (400.0 + 410.0 + 420.0) / 3.0
        self.assertAlmostEqual(vals[d], expect)

    def test_thresh_binary_split(self):
        vals, _ = _transformed_nightly(self.ath, "sleep_duration", "thresh")
        # Durations run 400..480 in steps of 10: all >= 390 -> 0.0.
        self.assertTrue(all(v == 0.0 for v in vals.values()))
        vals2, _ = _transformed_nightly(self.ath, "bedtime", "thresh")
        # Bedtimes run 60..120 past 21:00: all < 180 -> 1.0? No:
        # side is "ge" at 180, so all values are 0.0 (not in group).
        self.assertTrue(all(v == 0.0 for v in vals2.values()))

    def test_sensor_exclusion_blocks_source_night(self):
        self.ath.sleep_nights[5].excluded_sensor_unreliable = True
        _vals, blocked = _transformed_nightly(
            self.ath, "sleep_duration", "lag1")
        self.assertTrue(blocked[self.d0 + timedelta(days=6)])
        self.assertFalse(blocked[self.d0 + timedelta(days=7)])


class TestClassify(unittest.TestCase):
    def setUp(self):
        self.ath = _athlete()
        self.covered, self.sources, self.ad, self.ao = _covered(self.ath)

    def _classify(self, d, t, o):
        return classify({"driver": d, "transform": t, "outcome": o},
                        self.ad, self.ao, self.covered, self.sources)

    def test_registry_identity_is_duplicate(self):
        ok, reason, _g = self._classify("sleep_duration", "identity", "rhr")
        self.assertFalse(ok)
        self.assertIn("A3", reason)

    def test_prev_duration_alias_is_duplicate(self):
        # prev_duration -> duration is A13 under its own key...
        ok, reason, _g = self._classify("prev_duration", "identity",
                                        "duration")
        self.assertFalse(ok)
        self.assertIn("A13", reason)
        # ...and sleep_duration at lag1 -> duration is the same pair.
        ok, reason, _g = self._classify("sleep_duration", "lag1",
                                        "duration")
        self.assertFalse(ok)
        self.assertIn("A13", reason)

    def test_ask_pair_under_alias_is_duplicate(self):
        # (prev_duration -> rhr) is an evaluated ask pair, i.e.
        # (sleep_duration, lag1, rhr) is already covered.
        ok, reason, _g = self._classify("sleep_duration", "lag1", "rhr")
        self.assertFalse(ok)
        self.assertIn("already covered", reason)

    def test_coupled_roll3_against_duration_rejected(self):
        ok, reason, _g = self._classify("sleep_duration", "roll3",
                                        "duration")
        self.assertFalse(ok)
        self.assertIn("coupled", reason)

    def test_weekend_takes_no_transforms(self):
        ok, reason, _g = self._classify("weekend", "lag1", "rhr")
        self.assertFalse(ok)
        self.assertIn("weekend indicator", reason)

    def test_thresh_without_cut_rejected(self):
        ok, reason, _g = self._classify("rem_share", "thresh", "rhr")
        self.assertFalse(ok)
        self.assertIn("no fixed cut", reason)

    def test_novel_candidate_accepted(self):
        ok, reason, grain = self._classify("sleep_duration", "lag2", "rhr")
        self.assertTrue(ok, reason)
        self.assertEqual(grain, "nightly")
        ok, reason, grain = self._classify("mileage", "roll2", "next_rhr")
        self.assertTrue(ok, reason)
        self.assertEqual(grain, "weekly")


class TestRationaleValidation(unittest.TestCase):
    PACK = {"aggregates": {"nights_counted": 70}}

    def test_good_rationale_passes(self):
        raw = {"candidates": [{
            "driver": "sleep_duration", "transform": "lag2",
            "outcome": "rhr",
            "rationale": "Tests whether a short night echoes two "
                         "nights later."}]}
        cands, failures = validate_proposals(raw, self.PACK)
        self.assertEqual(failures, [])
        self.assertTrue(cands[0]["rationale_ok"])

    def test_causal_rationale_fails(self):
        raw = {"candidates": [{
            "driver": "sleep_duration", "transform": "lag2",
            "outcome": "rhr",
            "rationale": "Short sleep causes a higher heart rate "
                         "two days later."}]}
        cands, _f = validate_proposals(raw, self.PACK)
        self.assertFalse(cands[0]["rationale_ok"])

    def test_invented_number_fails(self):
        raw = {"candidates": [{
            "driver": "sleep_duration", "transform": "lag2",
            "outcome": "rhr",
            "rationale": "Worth testing across all 212 nights."}]}
        cands, _f = validate_proposals(raw, self.PACK)
        self.assertFalse(cands[0]["rationale_ok"])


def _fake_call(payload_text):
    def call(system, user):
        return payload_text, {"prompt_tokens": 500,
                              "completion_tokens": 120,
                              "reasoning_tokens": 40}, "stop"
    return call


class TestGatePath(unittest.TestCase):
    def test_fake_spark_candidate_is_gated(self):
        import json as _json
        ath = _athlete()
        ask_pack = ask_mod.build_ask(ath)
        pack = {"athlete": {"id": "t"}, "generated_on": "2026-11-02",
                "window_anchor": None, "cards": [], "aggregates": {}}
        payload = _json.dumps({"candidates": [
            {"driver": "sleep_duration", "transform": "lag2",
             "outcome": "rhr",
             "rationale": "Tests whether a short night echoes two "
                          "nights later."},
            {"driver": "sleep_duration", "transform": "identity",
             "outcome": "rhr",
             "rationale": "A duplicate in disguise."},
        ]})
        props, usages = generate_proposals(
            pack, ath, ask_pack, call=_fake_call(payload))
        self.assertEqual(props["proposed_by"], "spark")
        self.assertEqual(len(usages), 1)
        by_status = {c["status"] for c in props["candidates"]}
        self.assertEqual(by_status, {"evaluated", "rejected"})
        ev = [c for c in props["candidates"]
              if c["status"] == "evaluated"][0]
        self.assertIn(ev["tier"], ("solid", "emerging", "no_pattern",
                                   "insufficient_data"))
        self.assertIsNotNone(ev["sentence"])
        rj = [c for c in props["candidates"]
              if c["status"] == "rejected"][0]
        self.assertIn("A3", rj["reject_reason"])

    def test_unusable_output_falls_back_to_template(self):
        ath = _athlete()
        ask_pack = ask_mod.build_ask(ath)
        pack = {"athlete": {"id": "t"}, "generated_on": "2026-11-02",
                "window_anchor": None, "cards": [], "aggregates": {}}
        props, usages = generate_proposals(
            pack, ath, ask_pack, call=_fake_call("not json at all"))
        self.assertEqual(props["proposed_by"], "template")
        self.assertEqual(len(usages), 2)
        self.assertTrue(any(c["status"] == "evaluated"
                            for c in props["candidates"]))


if __name__ == "__main__":
    unittest.main()
