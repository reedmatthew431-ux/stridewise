"""Run the engine on one normalized profile and emit a fact pack.

Usage:
  python -m engine.run <normalized.json> [prev_factpack.json] [out.json]

- normalized.json: workspace-only profile file (never committed).
- prev_factpack.json (optional): previous refresh's fact pack, used
  only for demotion detection.
- out.json (optional): where to write the fact pack. Defaults to
  stdout. Fact packs are workspace-only outputs, never committed.

Zero API calls. Deterministic: same input -> same fact pack.
"""
from __future__ import annotations

import json
import sys

from .factpack import build_fact_pack, header_line, pack_to_json
from .model import load_normalized


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    athlete = load_normalized(argv[1])
    prev = None
    if len(argv) > 2 and argv[2] != "-":
        with open(argv[2]) as fh:
            prev = json.load(fh)
    pack = build_fact_pack(athlete, prev_pack=prev)
    if len(argv) > 3:
        pack_to_json(pack, argv[3])
        print(header_line(pack))
        print(f"wrote {argv[3]}")
    else:
        print(json.dumps(pack, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
