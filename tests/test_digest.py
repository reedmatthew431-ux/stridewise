"""Digest tests (stdlib unittest, deterministic, no API calls).

Every digest line must correspond to an actual diff entry between a
previous and a current fact pack. Run: python -m unittest discover
-s tests -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.digest import build_digest


def card(q, tier, **kw):
    base = {"question": q, "question_text": f"Question {q}",
            "tier": tier, "effect": -1.0, "n": 50, "demotion": None,
            "sentence": f"Sentence for {q}."}
    base.update(kw)
    return base


def pack(cards, race_cards=None, agg=None):
    return {"athlete": {"id": "t", "display_label": "T"},
            "generated_on": "2026-11-02", "window_anchor": "2026-10-26",
            "cards": cards, "race_cards": race_cards or [],
            "aggregates": agg or {}}


class TestDigest(unittest.TestCase):
    def test_identical_packs_have_no_lines(self):
        p = pack([card("A1", "solid")])
        d = build_digest(p, p)
        self.assertEqual(d["lines"], [])

    def test_new_card_line_uses_card_sentence(self):
        prev = pack([card("A1", "solid")])
        cur = pack([card("A1", "solid"),
                    card("A14", "no_pattern",
                         sentence="REM share? No consistent "
                                  "relationship.")])
        d = build_digest(prev, cur)
        self.assertEqual(len(d["lines"]), 1)
        self.assertTrue(d["lines"][0].startswith("New card: REM share?"))
        self.assertEqual(d["entries"][0]["kind"], "new_card")

    def test_tier_change(self):
        prev = pack([card("A1", "emerging")])
        cur = pack([card("A1", "solid")])
        d = build_digest(prev, cur)
        self.assertEqual(len(d["lines"]), 1)
        self.assertIn("moved from emerging to solid", d["lines"][0])

    def test_demotion_cleared(self):
        prev = pack([card("A11", "emerging",
                          demotion={"was": "solid",
                                    "reason": "more data moved it"})])
        cur = pack([card("A11", "emerging")])
        d = build_digest(prev, cur)
        self.assertEqual(len(d["lines"]), 1)
        self.assertIn("no longer carries", d["lines"][0])
        self.assertIn("dropped from solid", d["lines"][0])

    def test_demotion_appeared(self):
        prev = pack([card("A3", "solid")])
        cur = pack([card("A3", "emerging",
                         demotion={"was": "solid",
                                   "reason": "range widened"})])
        d = build_digest(prev, cur)
        kinds = [e["kind"] for e in d["entries"]]
        self.assertIn("demotion_appeared", kinds)
        self.assertIn("tier_change", kinds)

    def test_estimate_update_and_noise_floor(self):
        prev = pack([card("B6", "no_pattern", effect=-0.215)])
        cur = pack([card("B6", "no_pattern", effect=-0.230)])
        d = build_digest(prev, cur)
        self.assertEqual(len(d["lines"]), 1)
        self.assertIn("-0.215 → -0.23", d["lines"][0])
        tiny_prev = pack([card("B6", "no_pattern", effect=1.0001)])
        tiny_cur = pack([card("B6", "no_pattern", effect=1.0002, n=50)])
        self.assertEqual(build_digest(tiny_prev, tiny_cur)["lines"], [])

    def test_c3_grouped_line(self):
        def rc(date, typical):
            return {"question": "C3", "race": "R", "date": date,
                    "flagged": False, "race_avg_hr": 170.0,
                    "typical_avg_hr": typical, "comparison_n": 43,
                    "sentence": "s"}
        prev = pack([], [dict(rc("2026-09-12", 142.4), comparison_n=None),
                         dict(rc("2026-10-17", 142.4), comparison_n=None)])
        cur = pack([], [rc("2026-09-12", 159.0), rc("2026-10-17", 159.0)])
        d = build_digest(prev, cur)
        self.assertEqual(len(d["lines"]), 1)
        self.assertIn("races and hardest efforts", d["lines"][0])
        self.assertIn("159 bpm across 43 hard efforts", d["lines"][0])
        self.assertIn("previously 142.4 bpm", d["lines"][0])

    def test_race_flag_change(self):
        def rc(flag):
            return {"question": "C3", "race": "R", "date": "2026-10-17",
                    "flagged": flag, "race_avg_hr": 146.0,
                    "typical_avg_hr": 159.0, "comparison_n": 43,
                    "sentence": "s"}
        d = build_digest(pack([], [rc(False)]), pack([], [rc(True)]))
        self.assertEqual(len(d["lines"]), 1)
        self.assertIn("now flags", d["lines"][0])

    def test_aggregate_change_skips_anchor(self):
        prev = pack([], agg={"window_anchor": "2026-10-26",
                             "avg_rhr": 42.0})
        cur = pack([], agg={"window_anchor": "2026-10-26",
                            "avg_rhr": 43.5})
        d = build_digest(prev, cur)
        self.assertEqual(len(d["lines"]), 1)
        self.assertIn("average resting HR 42 → 43.5", d["lines"][0])


if __name__ == "__main__":
    unittest.main()
