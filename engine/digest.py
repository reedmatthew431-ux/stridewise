"""What-changed digest: a deterministic diff between the previous
fact pack and the current one, written as plain lines.

No model is involved. Every line corresponds to at least one actual
diff entry (the machine-readable entries are returned alongside the
lines, so any line can be audited back to the field that changed).
Covers: new and removed cards, tier changes, demotions appearing and
clearing, estimate/n updates, race-card changes (the C3 sensor check
is grouped into one line when its comparison typical moved the same
way on every race), and aggregate changes.

Run: python -m engine.digest <prev.json> <cur.json> <out.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TIER_LABEL = {
    "solid": "solid",
    "emerging": "emerging",
    "no_pattern": "no pattern",
    "insufficient_data": "not enough data",
}

_AGG_LABEL = {
    "total_miles_in_span": "total miles in span",
    "avg_sleep_min": "average sleep (min)",
    "avg_rhr": "average resting HR",
    "nights_counted": "nights counted",
    "long_run_weeks": "long-run weeks",
    "biggest_week_miles": "biggest week (miles)",
    "excluded_sensor_unreliable": "sensor-unreliable exclusions",
}


def _fmt(v):
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _dem_was(dem):
    if not isinstance(dem, dict):
        return None
    return dem.get("was") or dem.get("from")


def build_digest(prev, cur):
    """Returns {"athlete_id", "lines": [...], "entries": [...]}."""
    lines = []
    entries = []

    def emit(kind, line, **fields):
        lines.append(line)
        entries.append({"kind": kind, "line": line, **fields})

    pc = {c["question"]: c for c in prev.get("cards", [])}
    cc = {c["question"]: c for c in cur.get("cards", [])}

    for q in sorted(set(cc) - set(pc)):
        c = cc[q]
        emit("new_card",
             f"New card: {c['sentence']}",
             question=q, tier=c.get("tier"))
    for q in sorted(set(pc) - set(cc)):
        emit("removed_card",
             f"Removed card: {pc[q].get('question_text', q)} ({q}).",
             question=q)

    for q in sorted(set(pc) & set(cc)):
        a, b = pc[q], cc[q]
        name = b.get("question_text", q)
        if a.get("tier") != b.get("tier"):
            emit("tier_change",
                 f"{name} ({q}) moved from "
                 f"{TIER_LABEL.get(a.get('tier'), a.get('tier'))} to "
                 f"{TIER_LABEL.get(b.get('tier'), b.get('tier'))}.",
                 question=q, **{"from": a.get("tier"), "to": b.get("tier")})
        aw, bw = _dem_was(a.get("demotion")), _dem_was(b.get("demotion"))
        if a.get("demotion") and not b.get("demotion"):
            emit("demotion_cleared",
                 f"{name} ({q}) no longer carries the \"dropped from "
                 f"{aw}\" marker — that drop was recorded on the "
                 f"previous refresh; the card now simply stands at "
                 f"{TIER_LABEL.get(b.get('tier'), b.get('tier'))}.",
                 question=q, was=aw)
        elif b.get("demotion") and not a.get("demotion"):
            emit("demotion_appeared",
                 f"{name} ({q}) dropped from {bw} — "
                 f"{b['demotion'].get('reason', '')} — and the card "
                 f"now shows that drop.",
                 question=q, was=bw)
        elif a.get("demotion") != b.get("demotion"):
            emit("demotion_changed",
                 f"{name} ({q}): the recorded drop changed "
                 f"({a.get('demotion')} → {b.get('demotion')}).",
                 question=q)
        ae, be = a.get("effect"), b.get("effect")
        an, bn = a.get("n"), b.get("n")
        effect_moved = (ae is not None and be is not None
                        and abs(be - ae) >= 0.005)
        if a.get("tier") == b.get("tier") and (effect_moved or an != bn):
            bits = []
            if effect_moved:
                bits.append(f"effect {_fmt(ae)} → {_fmt(be)}")
            if an != bn:
                bits.append(f"paired observations {_fmt(an)} → {_fmt(bn)}")
            emit("estimate_update",
                 f"Estimate updated on {name} ({q}): "
                 + "; ".join(bits)
                 + f"; tier unchanged "
                   f"({TIER_LABEL.get(b.get('tier'), b.get('tier'))}).",
                 question=q)

    prc = {(r["question"], r["date"]): r for r in prev.get("race_cards", [])}
    crc = {(r["question"], r["date"]): r for r in cur.get("race_cards", [])}
    for k in sorted(set(crc) - set(prc)):
        emit("new_race_card",
             f"New race card: {crc[k].get('race', '')} "
             f"({k[1]}), check {k[0]}.",
             question=k[0], date=k[1])
    for k in sorted(set(prc) - set(crc)):
        emit("removed_race_card",
             f"Removed race card: {prc[k].get('race', '')} "
             f"({k[1]}), check {k[0]}.",
             question=k[0], date=k[1])

    shared = sorted(set(prc) & set(crc))
    # Grouped C3 line: the sensor check's comparison typical is a
    # property of the profile, not of one race, so when every C3 card
    # moved from the same old typical to the same new typical it is
    # one change, stated once.
    c3 = [k for k in shared if k[0] == "C3"]
    c3_grouped = False
    if len(c3) >= 2:
        olds = {prc[k].get("typical_avg_hr") for k in c3}
        news = {crc[k].get("typical_avg_hr") for k in c3}
        nsets = {crc[k].get("comparison_n") for k in c3}
        if len(olds) == 1 and len(news) == 1 and olds != news \
                and len(nsets) == 1 and None not in nsets:
            emit("c3_comparison_change",
                 f"Sensor check now compares race heart rate against "
                 f"hard efforts only (races and top-quartile runs): "
                 f"the typical hard-effort average is "
                 f"{_fmt(news.pop())} bpm across {nsets.pop()} hard "
                 f"efforts, previously {_fmt(olds.pop())} bpm across "
                 f"all runs.",
                 races=len(c3))
            c3_grouped = True
    for k in shared:
        a, b = prc[k], crc[k]
        if a.get("flagged") != b.get("flagged"):
            state = ("now flags" if b.get("flagged")
                     else "no longer flags")
            emit("race_flag_change",
                 f"Sensor check on {b.get('race', '')} ({k[1]}) {state} "
                 f"the heart-rate reading: race average "
                 f"{_fmt(b.get('race_avg_hr'))} bpm against a typical "
                 f"{_fmt(b.get('typical_avg_hr'))} bpm.",
                 question=k[0], date=k[1])
        skip = {"flagged", "sentence"}
        if c3_grouped and k[0] == "C3":
            skip |= {"typical_avg_hr", "comparison_n"}
        for f in sorted((set(a) | set(b)) - skip):
            if a.get(f) != b.get(f):
                emit("race_field_change",
                     f"Race card {k[0]} ({b.get('race', '')}, {k[1]}): "
                     f"{f} {_fmt(a.get(f))} → {_fmt(b.get(f))}.",
                     question=k[0], date=k[1], field=f)

    pa, ca = prev.get("aggregates", {}), cur.get("aggregates", {})
    for k in sorted(set(pa) | set(ca)):
        if k == "window_anchor":
            continue
        if pa.get(k) != ca.get(k):
            emit("aggregate_change",
                 f"Board totals: {_AGG_LABEL.get(k, k)} "
                 f"{_fmt(pa.get(k))} → {_fmt(ca.get(k))}.",
                 field=k)

    return {
        "athlete_id": (cur.get("athlete") or {}).get("id"),
        "display_label": (cur.get("athlete") or {}).get("display_label"),
        "from_generated_on": prev.get("generated_on"),
        "to_generated_on": cur.get("generated_on"),
        "window_anchor": cur.get("window_anchor"),
        "lines": lines,
        "entries": entries,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stridewise what-changed digest")
    ap.add_argument("prev")
    ap.add_argument("cur")
    ap.add_argument("out")
    args = ap.parse_args(argv)
    prev = json.loads(Path(args.prev).read_text())
    cur = json.loads(Path(args.cur).read_text())
    digest = build_digest(prev, cur)
    Path(args.out).write_text(json.dumps(digest, indent=2))
    print(json.dumps({"lines": len(digest["lines"]), "out": args.out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
