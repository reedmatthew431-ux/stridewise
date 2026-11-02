"""Run the engine on one normalized profile and emit a fact pack.

Usage:
  python -m engine.run <normalized.json> [prev_factpack.json] [out.json]
                       [--anchor YYYY-MM-DD]

- normalized.json: workspace-only profile file (never committed).
- prev_factpack.json (optional): previous refresh's fact pack, used
  only for demotion detection.
- out.json (optional): where to write the fact pack. Defaults to
  stdout. Fact packs are workspace-only outputs, never committed.
- --anchor (optional): board-anchor pin. When given, the board is
  computed AS OF that date: lookback windows end there, rows and
  races after it are excluded, and the pack's window_anchor is the
  pin. Without it, the anchor is the latest data date, as before.

Zero API calls. Deterministic: same input -> same fact pack.
"""
from __future__ import annotations

import json
import sys

from .factpack import build_fact_pack, header_line, pack_to_json
from .model import load_normalized


def main(argv):
    args = list(argv[1:])
    anchor = None
    if "--anchor" in args:
        i = args.index("--anchor")
        if i + 1 >= len(args):
            print(__doc__)
            return 2
        anchor = args[i + 1]
        del args[i:i + 2]
    if len(args) < 1:
        print(__doc__)
        return 2
    athlete = load_normalized(args[0])
    prev = None
    if len(args) > 1 and args[1] != "-":
        with open(args[1]) as fh:
            prev = json.load(fh)
    pack = build_fact_pack(athlete, prev_pack=prev, anchor=anchor)
    if len(args) > 2:
        pack_to_json(pack, args[2])
        print(header_line(pack))
        print(f"wrote {args[2]}")
    else:
        print(json.dumps(pack, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
