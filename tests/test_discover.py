"""Tests for engine/discover.py — race discovery parsers.

All fixtures are SYNTHETIC: invented athlete names, schools, meet
names and times, structured the way the platforms' pages are
structured. No real page HTML and no real names live in the repo
(the code-only rule); live behavior is verified separately in the
workspace, never from fixtures.
"""
import unittest

from engine import discover as D


TFRRS_PROFILE = """
<html><head><title>TFRRS | Jamie Sample - Track and Field
Results &amp; Statistics</title></head><body>
<h3 class="panel-title">JAMIE SAMPLE
(FR-1)</h3>
<h3>EXAMPLE COLLEGE</h3>
<h3>2026

  XC</h3>
<div id="meet-results">
<table class="table">
<thead><tr><th><a href="https://tfrrs.org/results/xc/90002/Example_Open">
Example Open
        &nbsp;&nbsp;&nbsp;Oct  3, 2026</a></th></tr></thead>
<tbody>
<tr><td>8k</td><td>26:40.0</td><td>12th</td></tr>
</tbody></table>
<table class="table">
<thead><tr><th><a href="https://tfrrs.org/results/xc/90001/Example_Invite">
Example Invite
        &nbsp;&nbsp;&nbsp;Sep 12, 2026</a></th></tr></thead>
<tbody>
<tr><td>5k</td><td>16:20.5</td><td>3rd</td></tr>
</tbody></table>
<table class="table">
<thead><tr><th><a href="https://tfrrs.org/results/90003/Example_Track_Classic">
Example Track Classic
        &nbsp;&nbsp;&nbsp;Apr 10, 2026</a></th></tr></thead>
<tbody>
<tr><td>3000</td><td>9:05.0</td><td>5th</td></tr>
<tr><td>1500</td><td>4:10.0</td><td>8th</td></tr>
</tbody></table>
</div>
<div id="event-history"></div>
</body></html>
"""

# A meet page with two individual-results tables; the athlete's
# row (profile id 700001, 26:40.0) is in the SECOND table.
MEET_PAGE = """
<html><body>
<h3>Women 8k Open Individual Results (8k) Top</h3>
<table>
<tr><th>PL</th><th>NAME</th><th>TEAM</th><th>TIME</th></tr>
<tr><td>1</td><td><a href="https://www.tfrrs.org/athletes/700099/X/Robin_Other.html">Robin Other</a></td><td>X</td><td>24:00.0</td></tr>
</table>
<h3>Men 8k Open Individual Results (8k) Top</h3>
<table>
<tr><th>PL</th><th>NAME</th><th>TEAM</th><th>TIME</th></tr>
<tr><td>12</td><td><a href="https://www.tfrrs.org/athletes/700001/Example/Jamie_Sample.html">Jamie Sample</a></td><td>Example College</td><td>26:40.0</td></tr>
<tr><td>13</td><td><a href="https://www.tfrrs.org/athletes/700002/Example/Pat_Teammate.html">Pat Teammate</a></td><td>Example College</td><td>26:41.0</td></tr>
</table>
</body></html>
"""

# Paul-Short-style heading: the short hint must win over the full
# prefix when it verifies.
MEET_PAGE_DASHED = """
<html><body>
<h3>Women's Open Race - 6000 Meters Individual Results (6k) Top</h3>
<table>
<tr><th>PL</th><th>NAME</th><th>TEAM</th><th>TIME</th></tr>
<tr><td>1</td><td><a href="https://www.tfrrs.org/athletes/700099/X/Robin_Other.html">Robin Other</a></td><td>X</td><td>21:00.0</td></tr>
</table>
<h3>Men's Open Race - 8000 Meters Individual Results (8k) Top</h3>
<table>
<tr><th>PL</th><th>NAME</th><th>TEAM</th><th>TIME</th></tr>
<tr><td>7</td><td><a href="https://www.tfrrs.org/athletes/700001/Example/Jamie_Sample.html">Jamie Sample</a></td><td>Example College</td><td>26:40.0</td></tr>
</table>
</body></html>
"""

ATHLETICNET_SHELL = """
<html><head><title>
\tJamie Sample - NJ Cross Country Bio
</title>
<script src="https://www.googletagmanager.com/gtag/js?id=G-XXXX"></script>
<script src="https://angular.athletic.net/app/site-app/main-XXXX.js"></script>
</head><body><app-root></app-root></body></html>
"""

ATHLETICNET_RENDERED = """Jamie Sample - NJ Cross Country Bio
Rankings
2026
5,000 Meters - 18:50.2
3 Miles - 19:40.0
Season Records
2026
5000 Meters
26 18:50.2PB Sep 5 Example Classic (Varsity)
31 19:05.0 Sep 19 Example Invitational (Varsity)
2024 Sample High School 9th Grade
"""


class TestTfrrsParse(unittest.TestCase):
    def test_identity(self):
        p = D.parse_tfrrs_profile(TFRRS_PROFILE)
        self.assertEqual(p["athlete_name"], "Jamie Sample")
        self.assertEqual(p["team"], "Example College")

    def test_results_rows(self):
        p = D.parse_tfrrs_profile(TFRRS_PROFILE)
        self.assertEqual(len(p["results"]), 4)
        by_label = {(r["event_label"], r["meet_name"]): r
                    for r in p["results"]}
        xc = by_label[("8k", "Example Open")]
        self.assertEqual(xc["date"], "2026-10-03")
        self.assertEqual(xc["finish_time_s"], 26 * 60 + 40.0)
        self.assertEqual(xc["place"], 12)
        tr = by_label[("3000", "Example Track Classic")]
        self.assertEqual(tr["date"], "2026-04-10")
        self.assertEqual(tr["finish_time_s"], 9 * 60 + 5.0)

    def test_kind_by_url(self):
        # discover_tfrrs marks /results/xc/ as xc, /results/ as track;
        # the parse layer keeps the raw URL for that decision.
        p = D.parse_tfrrs_profile(TFRRS_PROFILE)
        kinds = {r["meet_name"]: ("/results/xc/" in r["meet_url"])
                 for r in p["results"]}
        self.assertTrue(kinds["Example Open"])
        self.assertFalse(kinds["Example Track Classic"])


class TestResolveTable(unittest.TestCase):
    def test_picks_table_containing_athlete(self):
        hint = D.resolve_calibration_table(
            MEET_PAGE, "700001", "Jamie Sample", 26 * 60 + 40.0)
        self.assertEqual(hint, "Men 8k Open")

    def test_short_hint_preferred_when_it_verifies(self):
        hint = D.resolve_calibration_table(
            MEET_PAGE_DASHED, "700001", "Jamie Sample", 26 * 60 + 40.0)
        self.assertEqual(hint, "Men's Open Race")

    def test_absent_athlete_yields_none(self):
        hint = D.resolve_calibration_table(
            MEET_PAGE, "700555", "Nobody Here", 20 * 60.0)
        self.assertIsNone(hint)

    def test_name_time_fallback_without_profile_id(self):
        hint = D.resolve_calibration_table(
            MEET_PAGE, None, "Jamie Sample", 26 * 60 + 40.0)
        self.assertEqual(hint, "Men 8k Open")


class TestDistanceLabels(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(D.distance_for_label("8k"), 8000.0)
        self.assertEqual(D.distance_for_label("8.001k"), 8000.0)
        self.assertEqual(D.distance_for_label("5k"), 5000.0)
        self.assertEqual(D.distance_for_label("3000"), 3000.0)
        self.assertEqual(D.distance_for_label("mile"), 1609.3)
        self.assertAlmostEqual(D.distance_for_label("3.11 mile"),
                               5005.1, places=0)
        self.assertIsNone(D.distance_for_label("4x400"))
        self.assertIsNone(D.distance_for_label("HJ"))


class TestAthleticNet(unittest.TestCase):
    def test_title_name(self):
        self.assertEqual(D.athleticnet_title_name(ATHLETICNET_SHELL),
                         "Jamie Sample")

    def test_shell_detection(self):
        self.assertTrue(D.is_athleticnet_shell(ATHLETICNET_SHELL))
        self.assertFalse(D.is_athleticnet_shell(
            "<html><body>Season Records</body></html>"))

    def test_rendered_parse(self):
        p = D.parse_athleticnet_rendered(ATHLETICNET_RENDERED,
                                         "Jamie Sample")
        self.assertEqual(p["athlete_name"], "Jamie Sample")
        # Team-history line format is '<year> <School> <Level>
        # <grade>': the school token is 'Sample', the level is
        # 'High School'.
        self.assertEqual(p["team"], "Sample")
        self.assertEqual(p["bests"]["5,000 Meters"], 18 * 60 + 50.2)
        self.assertEqual(len(p["races"]), 2)
        r = p["races"][0]
        self.assertEqual(r["date"], "2026-09-05")
        self.assertEqual(r["place"], 26)
        self.assertEqual(r["finish_time_s"], 18 * 60 + 50.2)
        self.assertEqual(r["distance_m"], 5000.0)
        self.assertIsNone(r["calibration_table"])

    def test_shell_page_yields_no_races(self):
        # The shell has a title and nothing else: the parser must
        # not invent results from it.
        visible = "Jamie Sample - NJ Cross Country Bio"
        p = D.parse_athleticnet_rendered(visible, "Jamie Sample")
        self.assertEqual(p["races"], [])
        self.assertEqual(p["bests"], {})


if __name__ == "__main__":
    unittest.main()
