"""In-app course calibration (race card C1) from TFRRS results.

Usage:
  python -m engine.calibrate <normalized.json> <race-date>

Any race entry that carries a `results_url` (a TFRRS meet-results
page) and a `calibration_table` (a substring of that page's
individual-results heading, e.g. "Men 8k Champ") can be calibrated by
the app itself:

1. Fetch the meet page and extract the named individual-results
   table. Finishers are rows with a parseable TIME (the TIME column
   only, never the split columns). DNF / no-time rows are excluded
   from the field size. Names link to athlete profiles.
2. Fetch each finisher's TFRRS profile and find their best OTHER
   2026-season 8K, combining the "8k" and "8.001k" event labels and
   excluding this meet itself. Season = the race's calendar year.
3. pct_slower = (race_time - best_other) / best_other * 100.
   matched = finishers with a profile AND a best found.
   coverage = matched / field_size.
   Display bar: matched >= 40 AND coverage >= 0.35. Only then is the
   median pct written back as `field_median_slow_pct`; `field_size`
   and `matched_finishers` are always written back as real counts.

Fetch discipline: stdlib urllib with a descriptive User-Agent,
sequential requests with a short delay, and every fetched page
cached under ~/workspace/hackathon/cache/tfrrs/ keyed by URL hash
(never in the repo) and reused on re-runs. An HTTP 429 or an obvious
block page STOPS the run immediately: nothing is written back and
the partial state is reported as partial. Individual non-429
failures count as unmatched and the run continues.

Writes only the one race entry's calibration fields in the given
normalized.json (a workspace data file, never committed). Prints a
JSON summary to stdout; progress goes to stderr.
"""
from __future__ import annotations

import hashlib
import html as html_mod
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import date

CACHE_DIR = os.path.expanduser("~/workspace/hackathon/cache/tfrrs")
USER_AGENT = ("Stridewise/1.0 (in-app course calibration; "
              "+https://stridewise.pages.dev)")
FETCH_DELAY_S = 0.15
EVENT_LABELS = {"8k", "8.001k"}
MIN_MATCHED = 40
MIN_COVERAGE = 0.35
SPOT_SURNAMES = ("Hoyer", "Shibles")
BLOCK_MARKERS = ("attention required", "are you a robot",
                 "verify you are human", "access denied")


class Blocked(Exception):
    """TFRRS (or the egress path) appears to be blocking us. Stop."""


def _cache_path(url: str) -> str:
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, key + ".html")


def fetch(url: str, stats: dict) -> str | None:
    """Fetch a page with cache + block discipline.

    Returns page text, or None on an individual non-429 failure.
    Raises Blocked on HTTP 429 or an obvious block page.
    """
    path = _cache_path(url)
    if os.path.exists(path):
        stats["from_cache"] += 1
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise Blocked(f"HTTP 429 for {url}")
        stats["failures"] += 1
        print(f"  fetch failed ({exc.code}): {url}", file=sys.stderr)
        return None
    except Exception as exc:  # timeout, DNS, connection reset, ...
        stats["failures"] += 1
        print(f"  fetch failed ({exc.__class__.__name__}): {url}",
              file=sys.stderr)
        return None
    low = body.lower()
    title = low[:low.find("</title>") + 8] if "</title>" in low else low[:512]
    if len(body) < 1500 or any(m in title for m in BLOCK_MARKERS):
        raise Blocked(f"block page suspected for {url} "
                      f"({len(body)} bytes)")
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body)
    stats["fetched_live"] += 1
    time.sleep(FETCH_DELAY_S)
    return body


def parse_time(text: str) -> float | None:
    """'28:59.0' -> 1739.0; '1:02:03.4' -> 3723.4; else None."""
    text = text.strip()
    m = re.fullmatch(r"(\d+):(\d{2})\.(\d+)", text)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2)) \
            + int(m.group(3)) / (10 ** len(m.group(3)))
    m = re.fullmatch(r"(\d+):(\d{2}):(\d{2})\.(\d+)", text)
    if m:
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 \
            + int(m.group(3)) + int(m.group(4)) / (10 ** len(m.group(4)))
    return None


def _strip(fragment: str) -> str:
    return html_mod.unescape(re.sub(r"<[^>]+>", " ", fragment)).strip()


def parse_results_table(meet_html: str, table_hint: str) -> list[dict]:
    """Finishers of the named individual-results table.

    Returns [{name, profile_url (or None), race_time_s}] for rows
    whose TIME column parses. The TIME column is located by header
    name, never by position among the split columns.
    """
    heading_at = None
    for m in re.finditer(r"<h3[^>]*>(.*?)</h3>", meet_html, re.S):
        text = _strip(m.group(1))
        if table_hint in text and "Individual" in text:
            heading_at = m.end()
            break
    if heading_at is None:
        raise ValueError(f"no individual-results heading contains "
                         f"{table_hint!r}")
    tbl_start = meet_html.find("<table", heading_at)
    tbl_end = meet_html.find("</table>", tbl_start)
    if tbl_start < 0 or tbl_end < 0:
        raise ValueError("results table not found after heading")
    tbl = meet_html[tbl_start:tbl_end]
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", tbl, re.S)
    if not rows:
        raise ValueError("results table has no rows")
    headers = [_strip(c).upper()
               for c in re.findall(r"<th[^>]*>(.*?)</th>", rows[0], re.S)]
    if "TIME" not in headers or "NAME" not in headers:
        raise ValueError(f"unexpected results-table headers: {headers}")
    time_idx, name_idx = headers.index("TIME"), headers.index("NAME")
    finishers = []
    for row in rows[1:]:
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
        if len(cells) <= max(time_idx, name_idx):
            continue
        race_time = parse_time(_strip(cells[time_idx]))
        if race_time is None:
            continue  # DNF / DNS / no-time row: not a finisher
        name_cell = cells[name_idx]
        link = re.search(r'href="([^"]*/athletes/[^"]*)"', name_cell)
        finishers.append({
            "name": _strip(name_cell),
            "profile_url": link.group(1) if link else None,
            "race_time_s": race_time,
        })
    return finishers


_MONTHS = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}


def _parse_profile_date(text: str) -> date | None:
    m = re.search(r"([A-Z][a-z]{2})[a-z]*\s+(\d{1,2}),\s+(\d{4})", text)
    if not m or m.group(1) not in _MONTHS:
        return None
    return date(int(m.group(3)), _MONTHS[m.group(1)], int(m.group(2)))


def _results_id(url: str | None) -> str | None:
    m = re.search(r"/results/(?:xc/)?(\d+)", url or "")
    return m.group(1) if m else None


def best_other_8k(profile_html: str, race_date: date,
                  results_url: str) -> float | None:
    """Best other same-season 8K/8.001K from a profile's meet results.

    Excludes this meet (same results-page id, or same date). Returns
    seconds, or None if no qualifying mark exists.
    """
    start = profile_html.find('id="meet-results"')
    if start < 0:
        return None
    end = profile_html.find('id="event-history"', start)
    pane = profile_html[start:end if end >= 0 else len(profile_html)]
    this_id = _results_id(results_url)
    best = None
    for tbl in re.findall(r"<table.*?</table>", pane, re.S):
        head = re.search(
            r'<a[^>]*href="([^"]*/results/[^"]*)"[^>]*>(.*?)</a>', tbl, re.S)
        if not head:
            continue
        meet_url = head.group(1)
        meet_date = _parse_profile_date(_strip(tbl[:tbl.find("</thead>")])
                                        if "</thead>" in tbl else tbl[:600])
        if _results_id(meet_url) == this_id and this_id is not None:
            continue  # this meet itself
        if meet_date is not None and meet_date == race_date:
            continue  # same-day mark: this meet under another posting
        if meet_date is not None and meet_date.year != race_date.year:
            continue  # not this season
        for row in re.findall(r"<tr[^>]*>(.*?)</tr>", tbl, re.S):
            cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
            if len(cells) < 2:
                continue
            if _strip(cells[0]).lower() not in EVENT_LABELS:
                continue
            t = parse_time(_strip(cells[1]))
            if t is not None and (best is None or t < best):
                best = t
    return best


def calibrate(normalized_path: str, race_date_s: str) -> int:
    with open(normalized_path) as fh:
        data = json.load(fh)
    race = next((r for r in data.get("races", [])
                 if r.get("date") == race_date_s), None)
    if race is None:
        print(json.dumps({"error": f"no race entry dated {race_date_s}"}))
        return 2
    results_url = race.get("results_url")
    table_hint = race.get("calibration_table")
    if not results_url or not table_hint:
        print(json.dumps({"error": "race entry lacks results_url / "
                                   "calibration_table"}))
        return 2
    race_date = date.fromisoformat(race_date_s)
    stats = {"fetched_live": 0, "from_cache": 0, "failures": 0}

    meet_html = fetch(results_url, stats)
    if meet_html is None:
        print(json.dumps({"error": "meet page fetch failed",
                          "fetch_failures": stats["failures"]}))
        return 2
    finishers = parse_results_table(meet_html, table_hint)
    field_size = len(finishers)
    print(f"field: {field_size} finishers; "
          f"{sum(1 for f in finishers if f['profile_url'])} with profiles",
          file=sys.stderr)

    pcts: list[float] = []
    per_finisher: dict[str, dict] = {}
    attempted = 0
    try:
        for i, fin in enumerate(finishers, 1):
            if fin["profile_url"] is None:
                per_finisher[fin["name"]] = {"matched": False}
                continue
            attempted += 1
            page = fetch(fin["profile_url"], stats)
            best = (best_other_8k(page, race_date, results_url)
                    if page is not None else None)
            if best is None:
                per_finisher[fin["name"]] = {"matched": False}
            else:
                pct = (fin["race_time_s"] - best) / best * 100.0
                pcts.append(pct)
                per_finisher[fin["name"]] = {
                    "matched": True, "race_time_s": fin["race_time_s"],
                    "best_other_s": best, "pct_slower": round(pct, 2)}
            if i % 25 == 0:
                print(f"  profiles {i}/{field_size} "
                      f"(matched so far: {len(pcts)})", file=sys.stderr)
    except Blocked as exc:
        print(json.dumps({
            "blocked": True, "reason": str(exc),
            "field_size": field_size,
            "profiles_attempted": attempted,
            "profiles_total": sum(1 for f in finishers
                                  if f["profile_url"]),
            "matched_so_far_partial": len(pcts),
            "fetch_failures": stats["failures"],
            "note": "partial state only; nothing written back",
        }, indent=2))
        return 3

    matched = len(pcts)
    coverage = matched / field_size if field_size else 0.0
    bar_met = matched >= MIN_MATCHED and coverage >= MIN_COVERAGE
    median = round(statistics.median(pcts), 1) if pcts else None

    race["field_size"] = field_size
    race["matched_finishers"] = matched
    if bar_met:
        race["field_median_slow_pct"] = median
    with open(normalized_path, "w") as fh:
        json.dump(data, fh, indent=1)

    spot = {}
    for surname in SPOT_SURNAMES:
        hits = [{"name": n, **v} for n, v in per_finisher.items()
                if surname.lower() in n.lower() and v.get("matched")]
        if len(hits) == 1:
            spot[surname.lower()] = hits[0]
        elif hits:
            spot[surname.lower()] = hits
        else:
            spot[surname.lower()] = None
    print(json.dumps({
        "race_date": race_date_s, "race": race.get("name", ""),
        "field_size": field_size, "matched": matched,
        "coverage": round(coverage, 4),
        "median_slow_pct": median if bar_met else None,
        "bar_met": bar_met,
        "fetch_failures": stats["failures"],
        "profiles_fetched_live": stats["fetched_live"],
        "profiles_from_cache": stats["from_cache"],
        "spot": spot,
    }, indent=2))
    return 0


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    try:
        return calibrate(argv[1], argv[2])
    except Blocked as exc:  # meet-page fetch blocked before the loop
        print(json.dumps({"blocked": True, "reason": str(exc),
                          "note": "stopped before any profile fetch; "
                                  "nothing written back"}, indent=2))
        return 3


if __name__ == "__main__":
    sys.exit(main(sys.argv))
