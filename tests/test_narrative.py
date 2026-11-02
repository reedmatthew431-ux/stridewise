"""Narrative-layer tests (stdlib unittest, no network, no API calls).

Covers the deterministic validator (number tracing, language lint,
tagging, tier fidelity), the retry-then-fallback orchestration with an
injected fake call, the contributor-tier ban, and spend-log appends.
Run: python -m unittest discover -s tests -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import narrative
from engine.narrative import (append_spend_log, build_index, call_spark,
                              generate_report, validate_narrative)

PACK = {
    "athlete": {"id": "t", "display_label": "Test Athlete"},
    "generated_on": "2026-11-02",
    "window_anchor": "2026-10-31",
    "cards": [
        {"question": "A1", "tier": "solid", "effect": -40.5,
         "likely_range": [-48.1, -33.6], "n": 153, "demotion": None,
         "binding_rule": "all gates passed",
         "sentence": "On later-bedtime nights, sleep length runs about "
                     "41 min lower."},
        {"question": "A6", "tier": "no_pattern", "effect": 0.1,
         "likely_range": [-0.8, 0.5], "n": 60, "demotion": None,
         "binding_rule": "range includes 0",
         "sentence": "No pattern showed up."},
        {"question": "B4", "tier": "insufficient_data", "effect": None,
         "likely_range": None, "n": 3, "demotion": None,
         "binding_rule": "needs lap data",
         "sentence": "Not enough data yet."},
        {"question": "A9", "tier": "emerging", "effect": 8.3,
         "likely_range": [2.7, 21.3], "n": 37,
         "demotion": {"from": "solid", "to": "emerging"},
         "binding_rule": "below solid bar (37 of 40 needed)",
         "sentence": "Awake time tracks with next-morning HRV."},
    ],
    "race_cards": [
        {"question": "C1", "race": "Test Race", "date": "2026-10-17",
         "field_median_slow_pct": 9.5, "matched_finishers": 97,
         "field_size": 115,
         "sentence": "Field median: 9.5% slower than season bests "
                     "(97 of 115 matched)."},
    ],
    "aggregates": {"total_miles_in_span": 822.7, "avg_sleep_min": 396.2,
                   "nights_counted": 153},
}


def _report(headline=None, ruled=None, needs=None):
    return {
        "headline_findings": headline if headline is not None else [
            {"text": "One solid pattern: on later-bedtime nights, "
                     "sleep length runs about 41 min lower, based on "
                     "153 nights.", "cites": ["A1"]}],
        "ruled_out": ruled if ruled is not None else [
            {"text": "Bedtime timing and sleep efficiency showed no "
                     "pattern across 60 nights.", "cites": ["A6"]}],
        "evidence_needs": needs if needs is not None else [
            {"text": "The late-run fade question needs lap data; only "
                     "3 paired observations exist.", "cites": ["B4"]}],
    }


class TestValidator(unittest.TestCase):
    def setUp(self):
        self.index = build_index(PACK)

    def test_valid_report_passes(self):
        self.assertEqual(validate_narrative(_report(), self.index), [])

    def test_invented_number_rejected(self):
        bad = _report(headline=[
            {"text": "Sleep length runs about 37 min lower on "
                     "later-bedtime nights.", "cites": ["A1"]}])
        fails = validate_narrative(bad, self.index)
        self.assertTrue(any("does not trace" in f for f in fails), fails)

    def test_number_from_uncited_fact_rejected(self):
        bad = _report(headline=[
            {"text": "A solid pattern: sleep length runs about 41 min "
                     "lower.", "cites": ["A6"]}])
        fails = validate_narrative(bad, self.index)
        self.assertTrue(any("does not trace" in f for f in fails), fails)

    def test_race_and_aggregate_numbers_trace(self):
        ok = _report(headline=[
            {"text": "The field ran a median of 9.5% slower than "
                     "season bests, with 97 of 115 matched.",
             "cites": ["C1@2026-10-17"]},
            {"text": "Across the window the athlete covered 822.7 "
                     "miles and averaged 6.6 hours of sleep.",
             "cites": ["AGG"]}])
        self.assertEqual(validate_narrative(ok, self.index), [])

    def test_causal_language_rejected(self):
        bad = _report(headline=[
            {"text": "Later bedtimes cause shorter sleep across 153 "
                     "nights.", "cites": ["A1"]}])
        fails = validate_narrative(bad, self.index)
        self.assertTrue(any("causal" in f for f in fails), fails)

    def test_medical_language_rejected(self):
        bad = _report(headline=[
            {"text": "This pattern across 153 nights may point to "
                     "sleep apnea.", "cites": ["A1"]}])
        fails = validate_narrative(bad, self.index)
        self.assertTrue(any("medical" in f for f in fails), fails)

    def test_untagged_and_unknown_cites_rejected(self):
        bad = _report(headline=[{"text": "Sleep matters.", "cites": []}])
        self.assertTrue(any("untagged" in f for f in
                            validate_narrative(bad, self.index)))
        bad2 = _report(headline=[
            {"text": "Sleep matters.", "cites": ["Z9"]}])
        self.assertTrue(any("unknown" in f for f in
                            validate_narrative(bad2, self.index)))

    def test_tier_upgrade_rejected(self):
        bad = _report(headline=[
            {"text": "A solid pattern links bedtime and efficiency "
                     "across 60 nights.", "cites": ["A6"]}])
        fails = validate_narrative(bad, self.index)
        self.assertTrue(any("tier" in f for f in fails), fails)

    def test_headline_cannot_rest_on_no_pattern(self):
        bad = _report(headline=[
            {"text": "Efficiency held steady across 60 nights.",
             "cites": ["A6"]}])
        fails = validate_narrative(bad, self.index)
        self.assertTrue(any("headline" in f for f in fails), fails)

    def test_prediction_rejected(self):
        bad = _report(needs=[
            {"text": "With more data like this, you will run faster "
                     "next month.", "cites": ["A1"]}])
        fails = validate_narrative(bad, self.index)
        self.assertTrue(any("prediction" in f for f in fails), fails)

    def test_prescriptive_rejected(self):
        bad = _report(needs=[
            {"text": "You should keep bedtime steady for 153 nights.",
             "cites": ["A1"]}])
        fails = validate_narrative(bad, self.index)
        self.assertTrue(any("prescriptive" in f for f in fails), fails)

    def test_demotion_must_be_acknowledged(self):
        bad = _report(headline=[
            {"text": "Awake time tracks with next-morning HRV, about "
                     "8 ms across 37 nights.",
             "cites": ["A9"]}])
        fails = validate_narrative(bad, self.index)
        self.assertTrue(any("demot" in f for f in fails), fails)
        good = _report(headline=[
            {"text": "Awake time tracks with next-morning HRV, about "
                     "8 ms across 37 nights. This card "
                     "dropped from solid to emerging.",
             "cites": ["A9"]}])
        self.assertEqual(validate_narrative(good, self.index), [])


class TestOrchestration(unittest.TestCase):
    def test_retry_then_success(self):
        calls = []

        def fake(system, user):
            calls.append(user)
            if len(calls) == 1:
                return (json.dumps(_report(headline=[
                    {"text": "Later bedtimes cause shorter sleep.",
                     "cites": ["A1"]}])), {"prompt_tokens": 10,
                                           "completion_tokens": 5},
                        "stop")
            return (json.dumps(_report()), {"prompt_tokens": 12,
                                            "completion_tokens": 6},
                    "stop")

        out, usages = generate_report(PACK, call=fake)
        self.assertEqual(out["source"], "spark")
        self.assertEqual(out["attempts"], 2)
        self.assertEqual(len(usages), 2)
        self.assertIn("rejected by the validator", calls[1])

    def test_double_failure_falls_back_to_template(self):
        def fake(system, user):
            return ("not json at all", {"prompt_tokens": 10,
                                        "completion_tokens": 5}, "stop")

        out, usages = generate_report(PACK, call=fake)
        self.assertEqual(out["source"], "template")
        self.assertEqual(out["attempts"], 2)
        self.assertEqual(len(usages), 2)
        texts = [c["text"] for c in out["sections"]["headline_findings"]]
        self.assertIn(PACK["cards"][0]["sentence"], texts)

    def test_empty_content_retries_then_falls_back(self):
        # Measured live behavior: reasoning tokens can consume the
        # whole budget and content comes back empty. That must count
        # as a failed attempt, never a crash.
        def fake(system, user):
            return (None, {"prompt_tokens": 10,
                           "completion_tokens": 6000}, "length")

        out, usages = generate_report(PACK, call=fake)
        self.assertEqual(out["source"], "template")
        self.assertEqual(len(usages), 2)
        self.assertTrue(any("finish_reason=length" in f for f in
                            out["validation"]["final_failures"]))

    def test_empty_completion_is_a_named_failure_and_retries(self):
        # The hardened path: an empty completion must surface as a
        # validation failure literally named "empty completion",
        # feed the retry prompt, and never raise.
        calls = []

        def fake(system, user):
            calls.append(user)
            if len(calls) == 1:
                return ("", {"prompt_tokens": 10,
                             "completion_tokens": 6000}, "length")
            return (json.dumps(_report()), {"prompt_tokens": 12,
                                            "completion_tokens": 6},
                    "stop")

        out, usages = generate_report(PACK, call=fake)
        self.assertEqual(out["source"], "spark")
        self.assertEqual(out["attempts"], 2)
        self.assertIn("empty completion", calls[1])

    def test_empty_completion_usage_is_recorded(self):
        # Usage from an empty, length-truncated response must reach
        # the caller (and so the spend log) even though no draft was
        # produced: the tokens were billed.
        def fake(system, user):
            return (None, {"prompt_tokens": 10224,
                           "completion_tokens": 6000,
                           "reasoning_tokens": 6000}, "length")

        out, usages = generate_report(PACK, call=fake)
        self.assertEqual(out["source"], "template")
        self.assertEqual(len(usages), 2)
        for u in usages:
            self.assertEqual(u["prompt_tokens"], 10224)
            self.assertEqual(u["completion_tokens"], 6000)
            self.assertEqual(u["reasoning_tokens"], 6000)
        self.assertTrue(any("empty completion" in f for f in
                            out["validation"]["final_failures"]))

    def test_contributor_tier_banned(self):
        with self.assertRaises(ValueError):
            call_spark("s", "u", model="muse-spark-1.3-contributor")


class TestSpendLog(unittest.TestCase):
    def test_append_continues_numbering_and_totals(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "spend-log.md"
            path.write_text(
                "| # | Date (ET) | Purpose | Model | Tokens in | "
                "Tokens out | Est. cost |\n"
                "|---|---|---|---|---|---|---|\n"
                "| 1 | 2026-11-02 | ping | muse-spark-1.3 | 15 | 5 | "
                "~$0.0000 |\n")
            n, tokens, cost = append_spend_log(path, [{
                "purpose": "analyst report attempt 1",
                "model": "muse-spark-1.3", "prompt_tokens": 1000,
                "completion_tokens": 500}])
            self.assertEqual(n, 2)
            self.assertEqual(tokens, 20 + 1500)
            text = path.read_text()
            self.assertIn("| 2 |", text)
            self.assertIn("Running totals after #2", text)


if __name__ == "__main__":
    unittest.main()
