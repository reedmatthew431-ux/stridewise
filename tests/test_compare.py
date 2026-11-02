"""Comparison-layer tests (stdlib unittest, no network, no API
calls).

Covers: comparison assembly from fact packs, the validator (number
tracing, language lint, tier fidelity, the replicated/contrasts
semantic checks, individual label + live-finding checks), the
deterministic fallback, and the retry-then-template orchestration
with an injected fake call. Synthetic packs only; profile labels
are descriptors, as in the shipped data.

Run: python -m unittest discover -s tests -v
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.compare import (assemble_comparison, generate_comparison,
                            template_comparison, validate_comparison)


def _pack(label, cards):
    return {"athlete": {"id": label, "display_label": label},
            "generated_on": "2026-11-02", "window_anchor": "2026-10-26",
            "cards": cards, "race_cards": [], "aggregates": {}}


def _card(q, text, tier, effect, lo, hi, n):
    return {"question": q, "question_text": text, "tier": tier,
            "effect": effect, "likely_range": [lo, hi], "n": n,
            "binding_rule": "x", "demotion": None, "sentence": "s",
            "methods_note": "m"}


PACKS = [
    _pack("Matt Reed", [
        _card("A1", "Later bedtimes and sleep length", "solid",
              -40.0, -48.0, -33.0, 148),
        _card("A10", "Weekend nights and sleep length", "solid",
              -22.0, -40.0, -4.0, 148),
        _card("A9", "Awake time and resting heart rate", "no_pattern",
              0.4, -0.5, 1.2, 148),
    ]),
    _pack("Runner, Garmin wearer", [
        _card("A1", "Later bedtimes and sleep length", "solid",
              -43.0, -48.0, -37.0, 175),
        _card("A10", "Weekend nights and sleep length", "no_pattern",
              3.0, -9.0, 18.0, 175),
        _card("A9", "Awake time and resting heart rate", "solid",
              -2.3, -4.8, -0.7, 175),
    ]),
    _pack("Apple Watch wearer", [
        _card("A1", "Later bedtimes and sleep length", "emerging",
              -26.0, -39.0, -4.0, 109),
        _card("A10", "Weekend nights and sleep length", "emerging",
              47.0, 6.0, 86.0, 109),
        _card("A9", "Awake time and resting heart rate",
              "insufficient_data", None, None, None, 12),
    ]),
]

GOOD = {
    "replicated": [{
        "question": "A1",
        "summary": "Later bedtimes track with shorter sleep on all "
                   "three boards: effects -40 (n 148), -43 (n 175) "
                   "and -26 (n 109), every likely range below zero."}],
    "contrasts": [{
        "question": "A10",
        "summary": "Weekend nights run about 22 lower for Matt Reed "
                   "(n 148) and about 47 higher for the Apple Watch "
                   "wearer (n 109)."}],
    "individual": [{
        "profile_label": "Runner, Garmin wearer", "question": "A9",
        "summary": "More awake time in bed tracks with a lower "
                   "next-morning resting heart rate, effect -2.3 "
                   "across 175 nights; no other board has a live "
                   "finding on this question."}],
}


class TestAssembly(unittest.TestCase):
    def test_labels_and_questions(self):
        comp = assemble_comparison(PACKS)
        self.assertEqual(comp["labels"], ["Matt Reed",
                                          "Runner, Garmin wearer",
                                          "Apple Watch wearer"])
        a1 = comp["questions"]["A1"]["profiles"]
        self.assertEqual(a1["Matt Reed"]["tier"], "solid")
        self.assertEqual(a1["Apple Watch wearer"]["n"], 109)


class TestValidate(unittest.TestCase):
    def setUp(self):
        self.comp = assemble_comparison(PACKS)

    def test_good_output_passes(self):
        self.assertEqual(validate_comparison(GOOD, self.comp), [])

    def test_invented_number_fails(self):
        bad = json.loads(json.dumps(GOOD))
        bad["replicated"][0]["summary"] = "Effects reach -91 minutes."
        fails = validate_comparison(bad, self.comp)
        self.assertTrue(any("does not trace" in f for f in fails))

    def test_causal_language_fails(self):
        bad = json.loads(json.dumps(GOOD))
        bad["contrasts"][0]["summary"] = (
            "Weekend sleep differs because schedules change.")
        fails = validate_comparison(bad, self.comp)
        self.assertTrue(any("causal" in f for f in fails))

    def test_replicated_needs_same_direction(self):
        bad = json.loads(json.dumps(GOOD))
        bad["replicated"] = [{
            "question": "A10",
            "summary": "Weekend effects -22 (n 148) and 47 (n 109) "
                       "point the same way."}]
        fails = validate_comparison(bad, self.comp)
        self.assertTrue(any("replicate" in f for f in fails))

    def test_individual_needs_live_finding(self):
        bad = json.loads(json.dumps(GOOD))
        bad["individual"] = [{
            "profile_label": "Matt Reed", "question": "A9",
            "summary": "An individual finding with effect 0.4."}]
        fails = validate_comparison(bad, self.comp)
        self.assertTrue(any("no live finding" in f for f in fails))

    def test_tier_fidelity(self):
        bad = json.loads(json.dumps(GOOD))
        bad["individual"][0]["summary"] = (
            "An emerging finding: effect -2.3 across 175 nights.")
        fails = validate_comparison(bad, self.comp)
        self.assertTrue(any("tier" in f for f in fails))


class TestFallback(unittest.TestCase):
    def test_template_lists(self):
        comp = assemble_comparison(PACKS)
        out = template_comparison(comp)
        self.assertEqual([e["question"] for e in out["replicated"]],
                         ["A1"])
        self.assertEqual([e["question"] for e in out["contrasts"]],
                         ["A10"])
        ind = {(e["profile_label"], e["question"])
               for e in out["individual"]}
        self.assertIn(("Runner, Garmin wearer", "A9"), ind)
        # Every number in a fallback summary traces by construction.
        self.assertEqual(validate_comparison(out, comp), [])


def _fake_call(text):
    def call(system, user):
        return text, {"prompt_tokens": 900, "completion_tokens": 300,
                      "reasoning_tokens": 100}, "stop"
    return call


class TestOrchestration(unittest.TestCase):
    def test_spark_path(self):
        comp, usages = generate_comparison(
            PACKS, call=_fake_call(json.dumps(GOOD)))
        self.assertEqual(comp["source"], "spark")
        self.assertEqual(len(usages), 1)
        self.assertEqual(len(comp["replicated"]), 1)

    def test_unusable_output_falls_back(self):
        comp, usages = generate_comparison(
            PACKS, call=_fake_call("no json here"))
        self.assertEqual(comp["source"], "template")
        self.assertEqual(len(usages), 2)
        self.assertEqual([e["question"] for e in comp["replicated"]],
                         ["A1"])


if __name__ == "__main__":
    unittest.main()
