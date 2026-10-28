"""Fixture validation runner (private harness, not shipped as a feature).

Usage: python -m engine.validate [path/to/validation-fixture-1.json]

Pass bar (from the registry): recover planted effects at the expected
tier or one below, and produce ZERO solid patterns on planted-null
data. For this fixture the planted truths are:
  A1 bedtime -> duration: -60 min per hour late (planted, strong)
  A3 duration -> RHR: -3 bpm per extra hour (planted, moderate)
  A6 bedtime -> efficiency: planted null (must return no_pattern)
The fixture has no activities and no races, so Family B and Family C
are expected to report insufficient_data (missing fields), not
fabricated results.
"""
from __future__ import annotations

import sys

from .factpack import build_fact_pack, header_line
from .model import load_fixture

FIXTURE = "fixtures/validation-fixture-1.json"


def main(path=FIXTURE):
    athlete = load_fixture(path)
    pack = build_fact_pack(athlete)
    print(header_line(pack))
    print()
    by_q = {c["question"]: c for c in pack["cards"]}
    for qid in ("A1", "A2", "A3", "A4", "A5", "A6", "A7-RHR", "A7-Quality",
                "A8-Duration", "A8-Quality"):
        c = by_q[qid]
        print(f"{qid:11s} tier={c['tier']:17s} effect={c['effect']} "
              f"range={c['likely_range']} n={c['n']} "
              f"binding={c['binding_rule']}")
    print()
    for qid in ("B1", "B2", "B3", "B4", "B5", "B6", "B7"):
        c = by_q[qid]
        print(f"{qid:11s} tier={c['tier']:17s} n={c['n']} "
              f"binding={c['binding_rule']}")
    print()
    print(f"race_cards: {len(pack['race_cards'])} (expected 0: no races)")
    print()
    # Pass/fail checks against planted truth.
    checks = []
    a1 = by_q["A1"]
    checks.append(("A1 recovers ~-60 min/hr",
                   a1["effect"] is not None and -90 <= a1["effect"] <= -30
                   and a1["tier"] in ("solid", "emerging")))
    a3 = by_q["A3"]
    checks.append(("A3 recovers ~-3 bpm/hr",
                   a3["effect"] is not None and -5 <= a3["effect"] <= -1
                   and a3["tier"] in ("solid", "emerging")))
    a6 = by_q["A6"]
    checks.append(("A6 planted null -> no_pattern",
                   a6["tier"] == "no_pattern"))
    solids = [c["question"] for c in pack["cards"] if c["tier"] == "solid"]
    checks.append(("zero solid on planted-null efficiency question",
                   "A6" not in solids))
    ok = True
    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'}  {name}")
        ok = ok and passed
    print()
    print("VALIDATION " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else FIXTURE))
