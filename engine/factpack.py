"""Fact-pack assembly: the single structured output per refresh.

The fact pack is the ONLY thing the (later) narrative layer ever sees --
never raw rows. It carries per-question cards, per-race cards,
aggregates, and the import report, plus a header count line in the
shape the card templates specify for the report view.
"""
from __future__ import annotations

import json
from datetime import date

from .registry import aggregates, race_cards, run_all
from .weekly import build_weeks


def build_fact_pack(athlete, prev_pack=None, generated_on=None):
    weeks = build_weeks(athlete)
    cards = run_all(athlete, prev_pack=prev_pack)
    rcards = race_cards(athlete)
    aggs = aggregates(athlete, weeks)
    counts = {
        "solid": sum(1 for c in cards if c["tier"] == "solid"),
        "emerging": sum(1 for c in cards if c["tier"] == "emerging"),
        "no_pattern": sum(1 for c in cards if c["tier"] == "no_pattern"),
        "insufficient_data": sum(1 for c in cards
                                 if c["tier"] == "insufficient_data"),
        "demoted": sum(1 for c in cards if c.get("demotion")),
    }
    pack = {
        "athlete": {
            "id": athlete.id,
            "display_label": athlete.display_label,
            "source": athlete.source,
            "import_date": athlete.import_date,
        },
        "generated_on": generated_on or str(date.today()),
        "window_anchor": aggs["window_anchor"],
        "header_counts": counts,
        "cards": cards,
        "race_cards": rcards,
        "aggregates": aggs,
        "import_report": athlete.import_report,
    }
    return pack


def header_line(pack):
    c = pack["header_counts"]
    return (
        f"Board refreshed {pack['generated_on']} on data through "
        f"{pack['window_anchor']}: {c['solid']} solid · {c['emerging']} "
        f"emerging · {c['no_pattern']} no pattern · "
        f"{c['insufficient_data']} not enough data yet · "
        f"{c['demoted']} changed since last refresh."
    )


def pack_to_json(pack, path):
    with open(path, "w") as fh:
        json.dump(pack, fh, indent=2, default=str)
