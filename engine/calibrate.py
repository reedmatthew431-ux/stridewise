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
   same-season mark at the race's distance, excluding this meet
   itself. Event labels are chosen from the race entry's
   `distance_m`: 8000 -> "8k" + "8.001k"; 5000 -> "5k".
   Season = the race's calendar year.
3. pct_slower = (race_time - best_other) / best_other * 100.
   matched = finishers with a profile AND a best found.
   coverage = matched / field_size.
   Display bar: matched >= 40 AND coverage >= 0.35. Only then is the
   median pct written back as `field_median_slow_pct`; `field_size`
   and `matched_finishers` are always written back as real counts.
4. Last-season fallback: if the same-season pass fails the bar, a
   second pass matches against PURE last-season (year - 1) bests at
   the same labels, under the same bar. The two pools are never
   mixed — a shown median comes entirely from one pass.
   `reference_season` records which pool produced the written
   result (race year for same-season, year - 1 for the fallback);
   if neither pass clears the bar, the same-season matched count
   and reference stay the record.

Fetch discipline: stdlib urllib with a descriptive User-Agent,
sequential requests with a short delay, and every fetched page
cached under ~/workspace/hackathon/cache/tfrrs/ keyed by URL hash
(never in the repo) and reused on re-runs. An HTTP 429 or an obvious
block page STOPS the run immediately: nothing is written back and
the partial state is reported as partial. Individual non-429
failures count as unmatched and the run continues.

The same run also captures the profile athlete's own result
context from the meet page: his row in the individual table is
identified by an exact TIME match against the entry's
`finish_time_s` (tolerance 0.05s) — exactly one match yields
`place` and `team_name`; zero or multiple matches store nothing
(never a guess). His team's standing comes from the matching
team-results table (`team_place` / `team_field`, counting placed
teams only — unplaced/DNF team rows are not part of the field,
mirroring the individual DNF rule), and `team_position` is his
rank among his own team's finishers.

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
LABELS_BY_DISTANCE_M = {8000: {"8k", "8.001k"}, 5000: {"5k"}}
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


def parse_results_table_full(meet_html: str,
                               table_hint: str) -> tuple[list[dict], str]:
    """Like parse_results_table, plus PL / TEAM per finisher.

    Returns (finishers, heading_text) where finishers carry the
    extra keys "place" (int or None) and "team" (str or None), and
    heading_text is the whitespace-normalized individual-results
    heading (used to locate the matching team-results table).
    """
    heading_at = None
    heading_text = ""
    for m in re.finditer(r"<h3[^>]*>(.*?)</h3>", meet_html, re.S):
        text = _strip(m.group(1))
        if table_hint in text and "Individual" in text:
            heading_at = m.end()
            heading_text = " ".join(text.split())
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
    pl_idx = headers.index("PL") if "PL" in headers else None
    team_idx = headers.index("TEAM") if "TEAM" in headers else None
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
        place = None
        if pl_idx is not None and len(cells) > pl_idx:
            pl_txt = _strip(cells[pl_idx])
            if re.fullmatch(r"\d+", pl_txt):
                place = int(pl_txt)
        team = None
        if team_idx is not None and len(cells) > team_idx:
            team = _strip(cells[team_idx]) or None
        finishers.append({
            "name": _strip(name_cell),
            "profile_url": link.group(1) if link else None,
            "race_time_s": race_time,
            "place": place,
            "team": team,
        })
    return finishers, heading_text


def team_standing(meet_html: str, indiv_heading_text: str,
                  team_name: str) -> dict:
    """His team's standing in the matching team-results table.

    The team table is the one whose heading starts with the
    individual heading's text up to "Individual" (e.g. "Men 8k
    Champ") and names Team (not Individual) results. Only rows
    with a parseable PL count — unplaced/DNF team rows are not
    part of the field, mirroring the individual DNF rule.
    Returns {"status", "team_place", "team_field"} with status
    "ok" | "no_team_table" | "team_not_listed".
    """
    out = {"status": "no_team_table", "team_place": None,
           "team_field": None}
    probe = indiv_heading_text.split("Individual")[0].strip()
    if not probe:
        return out
    for m in re.finditer(r"<h3[^>]*>(.*?)</h3>", meet_html, re.S):
        text = " ".join(_strip(m.group(1)).split())
        if (not text.startswith(probe) or "Team" not in text
                or "Individual" in text):
            continue
        tbl_start = meet_html.find("<table", m.end())
        tbl_end = meet_html.find("</table>", tbl_start)
        if tbl_start < 0 or tbl_end < 0:
            return out
        tbl = meet_html[tbl_start:tbl_end]
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", tbl, re.S)
        if not rows:
            return out
        headers = [_strip(c).upper()
                   for c in re.findall(r"<th[^>]*>(.*?)</th>",
                                       rows[0], re.S)]
        if "TEAM" not in headers:
            return out
        team_idx = headers.index("TEAM")
        pl_idx = headers.index("PL") if "PL" in headers else 0
        standings = []
        for row in rows[1:]:
            cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
            if len(cells) <= max(team_idx, pl_idx):
                continue
            pl_txt = _strip(cells[pl_idx])
            if not re.fullmatch(r"\d+", pl_txt):
                continue  # unplaced (DNF) team row: not in the field
            standings.append((int(pl_txt), _strip(cells[team_idx])))
        out["team_field"] = len(standings)
        for pl, tn in standings:
            if tn == team_name:
                out["status"] = "ok"
                out["team_place"] = pl
                return out
        out["status"] = "team_not_listed"
        return out
    return out


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


def labels_for_distance(distance_m) -> set[str] | None:
    """TFRRS event labels for a race distance, or None if unsupported."""
    if distance_m is None:
        return None
    return LABELS_BY_DISTANCE_M.get(int(round(distance_m)))


def _best_mark(profile_html: str, race_date: date, results_url: str,
               labels: set[str], season_year: int,
               strict_season: bool) -> float | None:
    """Shared scan of a profile's meet-results pane for a best mark.

    strict_season=False (the primary, same-season pass): exclude this
    meet (same results-page id, or same date) and skip dated rows
    outside season_year. strict_season=True (the last-season
    fallback): the pool is PURE season_year — a meet table counts
    only when its date parses and falls in season_year; this-meet
    exclusion is irrelevant (different year) and not applied.
    Returns seconds, or None if no qualifying mark exists.
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
        if strict_season:
            if meet_date is None or meet_date.year != season_year:
                continue  # fallback pool is purely season_year rows
        else:
            if _results_id(meet_url) == this_id and this_id is not None:
                continue  # this meet itself
            if meet_date is not None and meet_date == race_date:
                continue  # same-day mark: this meet under another posting
            if meet_date is not None and meet_date.year != season_year:
                continue  # not the target season
        for row in re.findall(r"<tr[^>]*>(.*?)</tr>", tbl, re.S):
            cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
            if len(cells) < 2:
                continue
            if _strip(cells[0]).lower() not in labels:
                continue
            t = parse_time(_strip(cells[1]))
            if t is not None and (best is None or t < best):
                best = t
    return best


def best_other_mark(profile_html: str, race_date: date,
                    results_url: str, labels: set[str]) -> float | None:
    """Best other same-season mark at the race's distance.

    `labels` are the TFRRS event labels for that distance (see
    labels_for_distance). Excludes this meet (same results-page id,
    or same date). Returns seconds, or None if no qualifying mark
    exists.
    """
    return _best_mark(profile_html, race_date, results_url, labels,
                      race_date.year, strict_season=False)


def best_last_season_mark(profile_html: str, race_date: date,
                          results_url: str,
                          labels: set[str]) -> float | None:
    """Best last-season mark at the same labels (fallback pass only).

    Pure last-season pool: every reference is a mark from
    race_date.year - 1. Never mixed with same-season references.
    """
    return _best_mark(profile_html, race_date, results_url, labels,
                      race_date.year - 1, strict_season=True)


def match_pass(finishers: list[dict], stats: dict, best_fn,
               progress: dict) -> None:
    """One matching pass over the field against one reference pool.

    Fills progress["pcts"], progress["per_finisher"] and
    progress["attempted"] in place (so a Blocked stop still leaves
    the partial state readable). Raises Blocked.
    """
    field_size = len(finishers)
    for i, fin in enumerate(finishers, 1):
        if fin["profile_url"] is None:
            progress["per_finisher"][fin["name"]] = {"matched": False}
            continue
        progress["attempted"] += 1
        page = fetch(fin["profile_url"], stats)
        best = best_fn(page) if page is not None else None
        if best is None:
            progress["per_finisher"][fin["name"]] = {"matched": False}
        else:
            pct = (fin["race_time_s"] - best) / best * 100.0
            progress["pcts"].append(pct)
            progress["per_finisher"][fin["name"]] = {
                "matched": True, "race_time_s": fin["race_time_s"],
                "best_other_s": best, "pct_slower": round(pct, 2)}
        if i % 25 == 0:
            print(f"  profiles {i}/{field_size} "
                  f"(matched so far: {len(progress['pcts'])})",
                  file=sys.stderr)


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
    labels = labels_for_distance(race.get("distance_m"))
    if labels is None:
        print(json.dumps({"error": "unsupported race distance for "
                                   f"calibration: {race.get('distance_m')!r}"}))
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

    # Place / team capture: his row is the one whose TIME equals
    # the entry's official finish_time_s (tolerance 0.05s). Exactly
    # one match identifies him; zero or multiple matches store
    # nothing — never a guess.
    detailed, heading_text = parse_results_table_full(meet_html,
                                                      table_hint)
    capture = {"status": "no_finish_time", "matches": 0}
    if race.get("finish_time_s") is not None:
        target = float(race["finish_time_s"])
        hits = [f for f in detailed
                if abs(f["race_time_s"] - target) <= 0.05]
        capture["matches"] = len(hits)
        if len(hits) == 1:
            his = hits[0]
            capture.update(status="ok", place=his["place"],
                           team_name=his["team"])
            if his["place"] is not None and his["team"]:
                capture["team_position"] = 1 + sum(
                    1 for f in detailed
                    if f["team"] == his["team"]
                    and f["place"] is not None
                    and f["place"] < his["place"])
                st = team_standing(meet_html, heading_text, his["team"])
                capture["team_table"] = st["status"]
                capture["team_place"] = st["team_place"]
                capture["team_field"] = st["team_field"]
        elif not hits:
            capture["status"] = "no_match"
        else:
            capture["status"] = "ambiguous"
            capture["candidates"] = [
                {"place": f["place"], "name": f["name"],
                 "team": f["team"]} for f in hits]
        print(f"place capture: {capture['status']} "
              f"({capture['matches']} TIME match/es)", file=sys.stderr)

    def fresh_progress() -> dict:
        return {"pcts": [], "per_finisher": {}, "attempted": 0}

    def pass_stats(prog: dict) -> dict:
        m = len(prog["pcts"])
        cov = m / field_size if field_size else 0.0
        return {"matched": m, "coverage": round(cov, 4),
                "median": (round(statistics.median(prog["pcts"]), 1)
                           if prog["pcts"] else None),
                "bar_met": m >= MIN_MATCHED and cov >= MIN_COVERAGE}

    def blocked_payload(exc: Blocked, prog: dict, pass_name: str) -> dict:
        return {
            "blocked": True, "reason": str(exc), "pass": pass_name,
            "field_size": field_size,
            "profiles_attempted": prog["attempted"],
            "profiles_total": sum(1 for f in finishers
                                  if f["profile_url"]),
            "matched_so_far_partial": len(prog["pcts"]),
            "fetch_failures": stats["failures"],
            "note": "partial state only; nothing written back",
        }

    # Pass 1 (primary): same-season references, exactly as built.
    prog = fresh_progress()
    try:
        match_pass(finishers, stats,
                   lambda page: best_other_mark(page, race_date,
                                                results_url, labels),
                   prog)
    except Blocked as exc:
        print(json.dumps(blocked_payload(exc, prog, "same_season"),
                         indent=2))
        return 3
    primary = pass_stats(prog)
    pcts, per_finisher = prog["pcts"], prog["per_finisher"]
    reference_season = race_date.year

    # Pass 2 (fallback, only if the primary bar failed): pure
    # last-season references. The pools are never mixed — a shown
    # median comes entirely from one pass.
    fallback = None
    if not primary["bar_met"]:
        print("primary bar not met; running last-season fallback pass",
              file=sys.stderr)
        prog2 = fresh_progress()
        try:
            match_pass(finishers, stats,
                       lambda page: best_last_season_mark(
                           page, race_date, results_url, labels),
                       prog2)
        except Blocked as exc:
            payload = blocked_payload(exc, prog2, "last_season_fallback")
            payload["same_season"] = primary
            print(json.dumps(payload, indent=2))
            return 3
        fallback = pass_stats(prog2)
        if fallback["bar_met"]:
            pcts, per_finisher = prog2["pcts"], prog2["per_finisher"]
            reference_season = race_date.year - 1

    matched = len(pcts)
    coverage = matched / field_size if field_size else 0.0
    bar_met = matched >= MIN_MATCHED and coverage >= MIN_COVERAGE
    median = round(statistics.median(pcts), 1) if pcts else None

    race["field_size"] = field_size
    race["matched_finishers"] = matched
    if bar_met:
        race["field_median_slow_pct"] = median
    race["reference_season"] = reference_season
    if capture["status"] == "ok":
        race["place"] = capture["place"]
        race["team_name"] = capture["team_name"]
        race["team_place"] = capture.get("team_place")
        race["team_field"] = capture.get("team_field")
        race["team_position"] = capture.get("team_position")
    else:
        for key in ("place", "team_name", "team_place", "team_field",
                    "team_position"):
            race.pop(key, None)
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
        "distance_m": race.get("distance_m"),
        "labels": sorted(labels),
        "field_size": field_size, "matched": matched,
        "coverage": round(coverage, 4),
        "median_slow_pct": median if bar_met else None,
        "bar_met": bar_met,
        "reference_season": reference_season,
        "same_season": primary,
        "last_season_fallback": fallback,
        "fetch_failures": stats["failures"],
        "profiles_fetched_live": stats["fetched_live"],
        "profiles_from_cache": stats["from_cache"],
        "place_capture": capture,
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
