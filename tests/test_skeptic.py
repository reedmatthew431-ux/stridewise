"""Skeptic-layer tests (stdlib unittest, no network, no API calls).

Covers the note validator (number tracing, language lint, tier
fidelity, sentence count) and the retry-then-none orchestration with
an injected fake call. Run: python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.skeptic import generate_note, validate_note

PACK = {
    "athlete": {"id": "t", "display_label": "Test Athlete"},
    "aggregates": {"nights_counted": 148, "avg_rhr": 42.7},
}
CARD = {"question": "A14", "question_text": "REM share and RHR",
        "tier": "solid", "effect": -1.24, "likely_range": [-1.66, -0.65],
        "n": 148, "binding_rule": "all gates passed", "demotion": None,
        "sentence": "On higher-REM-share nights, RHR runs lower.",
        "methods_note": "Paired nightly."}

GOOD = ("One innocent reading is the calendar: higher-REM-share nights "
        "may cluster on weekends, when the rest of the day also looks "
        "different. A check restricted to weekday nights only, over "
        "the same 148 nights, would settle whether the pattern "
        "survives away from the weekend.")


def fake_call(text):
    def call(system, user):
        return text, {"prompt_tokens": 100, "completion_tokens": 50,
                      "reasoning_tokens": 10}, "stop"
    return call


class TestValidateNote(unittest.TestCase):
    def test_good_note_passes(self):
        self.assertEqual(validate_note(GOOD, PACK, CARD), [])

    def test_invented_number_fails(self):
        bad = GOOD.replace("148 nights", "212 nights")
        fails = validate_note(bad, PACK, CARD)
        self.assertTrue(any("does not trace" in f for f in fails))

    def test_causal_language_fails(self):
        bad = ("The pattern shows up because weekends differ. "
               "More weekday nights would settle it.")
        fails = validate_note(bad, PACK, CARD)
        self.assertTrue(any("causal" in f for f in fails))

    def test_due_to_fails(self):
        bad = ("The pattern may be due to weekend clustering. "
               "Weekday-only nights would settle it.")
        fails = validate_note(bad, PACK, CARD)
        self.assertTrue(any("causal" in f for f in fails))

    def test_prescription_fails(self):
        bad = ("You should collect more weekday nights. "
               "That would settle the question.")
        fails = validate_note(bad, PACK, CARD)
        self.assertTrue(any("prescriptive" in f for f in fails))

    def test_tier_upgrade_fails(self):
        bad = ("This emerging-grade pattern may reflect weekends. "
               "More nights would settle it.")
        fails = validate_note(bad, PACK, CARD)
        self.assertTrue(any("tier" in f for f in fails))

    def test_sentence_count_fails(self):
        fails = validate_note("Only one sentence here.", PACK, CARD)
        self.assertTrue(any("sentence" in f for f in fails))


class TestGenerateNote(unittest.TestCase):
    def test_first_try_success(self):
        rec, usages = generate_note(PACK, CARD, fake_call(GOOD))
        self.assertEqual(rec["status"], "ok")
        self.assertEqual(rec["attempts"], 1)
        self.assertEqual(len(usages), 1)

    def test_retry_recovers(self):
        calls = {"n": 0}

        def flaky(system, user):
            calls["n"] += 1
            text = "Bad because reasons. More data needed." \
                if calls["n"] == 1 else GOOD
            return text, {"prompt_tokens": 100,
                          "completion_tokens": 50,
                          "reasoning_tokens": 10}, "stop"

        rec, usages = generate_note(PACK, CARD, flaky)
        self.assertEqual(rec["status"], "ok")
        self.assertEqual(rec["attempts"], 2)
        self.assertEqual(len(usages), 2)

    def test_two_failures_record_no_note(self):
        rec, usages = generate_note(
            PACK, CARD, fake_call("Bad because reasons. Nope."))
        self.assertEqual(rec["status"], "no_note")
        self.assertIsNone(rec["note"])
        self.assertEqual(len(usages), 2)

    def test_empty_completion_is_a_failure_not_exception(self):
        def empty(system, user):
            return "", {"prompt_tokens": 100, "completion_tokens": 900,
                        "reasoning_tokens": 900}, "length"

        rec, usages = generate_note(PACK, CARD, empty)
        self.assertEqual(rec["status"], "no_note")
        self.assertEqual(len(usages), 2)


if __name__ == "__main__":
    unittest.main()
