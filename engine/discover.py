"""Race discovery: from a linked public results profile to a race list.

Usage:
  python -m engine.discover <profile-url> [out.json] [--rendered <file>]

Why this exists: a wearable export is anonymous by design — the
adapters deliberately strip identity, so the app cannot find who
an athlete is, or how their races went, from an export alone.
Discovery closes that gap with one deliberate act: the athlete
links a PUBLIC results profile (TFRRS for college runners,
Athletic.net for high school). The app reads that profile and
proposes the athlete's race list — meet, date, event, official
time, place. The athlete confirms the list before anything is
used; conditions notes are always the athlete's own words, never
discovered. Each proposed XC entry from TFRRS also carries the
`results_url` + `calibration_table` pair engine.calibrate
consumes unchanged, so a confirmed race can be calibrated by the
app itself (see "Course calibration" in the README).

Two platform adapters sit behind one interface:

  discover(url) -> {
      "platform": "tfrrs" | "athleticnet",
      "profile_url": url,
      "athlete_name": str | None,
      "team": str | None,
      "status": "ok" | "partial" | "blocked",
      "races": [ {date, meet_name, event_label, kind,
                  distance_m, finish_time_s, place, masked,
                  results_url, calibration_table} ],
      "bests": {event_label: seconds},        # athleticnet only
      "bests_source": "rankings_panel",       # athleticnet only
      "results_found" / "results_visible" / "results_masked",
      ... plus platform notes / block evidence when relevant
  }

TFRRS: server-rendered; fully working. The profile's own page is
fetched LIVE (a profile gains races over a season) and written
through to the shared calibration cache; meet pages go through
engine.calibrate.fetch (cached, reused). The meet-results pane
TFRRS serves by default shows the current season — discovery
proposes what that pane shows, and the confirmation step is
where the athlete sees exactly what was found. The calibration
table for an XC entry is resolved by locating the athlete's own
row among the meet page's individual-results tables (matched by
profile id, else by name + time) and is then verified by parsing
that table through calibrate's own parser before it is emitted.

Athletic.net: the public athlete page is a JavaScript app —
a plain fetch returns only the Angular shell (verified
2026-11-02: ~10 KB whose only athlete-specific content is the
<title>; the legacy .aspx athlete URLs redirect to the
homepage). Discovery is therefore a two-step, agent-run flow,
like the rest of this build's refreshes: (1) the operator
renders the page in their browser at refresh time and saves
the rendered page; (2) discovery parses that rendered copy,
supplied with --rendered <file> (or the rendered_path
argument). Given no rendered copy, the plain-fetch path still
reports status "blocked" (reason "js_wall") with the evidence
— the wall is reported, never papered over. A live-browser
proof (2026-11-02) confirmed the rendered page carries the
athlete's identity, rankings/bests, and results — and also
that logged-out viewing MASKS part of the history: some
results render as "Hidden Meet" entries with time and place
withheld, and some season-record values render as "########",
behind a "Log In or Create Free Account to View All Stats"
banner. Masked entries are represented explicitly (masked =
True, withheld fields null) and counted (results_found /
results_visible / results_masked); any mask makes the status
"partial" with the reason stated. A masked value is never
invented or inferred — a best visible in the Rankings panel
is reported as a best (bests_source "rankings_panel") and the
masked result it came from stays masked. Field calibration
is out of scope for Athletic.net in this build: its entries
carry calibration_table = None and a platform note — no
field match is faked.

Fetch discipline is calibrate's: descriptive User-Agent,
sequential requests with a short delay, every fetched page
cached under ~/workspace/hackathon/cache/ (never in the repo).
An HTTP 429 or an obvious block page raises calibrate.Blocked
and stops the run; partial state is reported as partial.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

from . import calibrate as C

ATHLETICNET_CACHE_DIR = os.path.expanduser(
    "~/workspace/hackathon/cache/athleticnet")


# ------------------------------------------------------------------
# fetching
# ------------------------------------------------------------------

def _looks_blocked(body: str) -> bool:
    low = body.lower()
    title = low[:low.find("</title>") + 8] if "</title>" in low \
        else low[:512]
    return len(body) < 1500 or any(m in title for m in C.BLOCK_MARKERS)


def fetch_live(url: str, cache_dir: str, stats: dict) -> str | None:
    """Fetch a page live and write it through to the cache.

    Used for athlete profile pages, which gain races over a
    season: a cached copy would silently freeze the athlete's
    history. Returns page text, or None on an individual non-429
    failure. Raises C.Blocked on HTTP 429 or a block page.
    """
    req = urllib.request.Request(url, headers={"User-Agent": C.USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise C.Blocked(f"HTTP 429 for {url}")
        stats["failures"] += 1
        print(f"  fetch failed ({exc.code}): {url}", file=sys.stderr)
        return None
    except Exception as exc:
        stats["failures"] += 1
        print(f"  fetch failed ({exc.__class__.__name__}): {url}",
              file=sys.stderr)
        return None
    if _looks_blocked(body):
        raise C.Blocked(f"block page suspected for {url} "
                        f"({len(body)} bytes)")
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(
            cache_dir,
            hashlib.sha256(url.encode("utf-8")).hexdigest() + ".html"),
            "w", encoding="utf-8") as fh:
        fh.write(body)
    stats["fetched_live"] += 1
    time.sleep(C.FETCH_DELAY_S)
    return body


def fetch_cached(url: str, cache_dir: str, stats: dict) -> str | None:
    """calibrate.fetch with a selectable cache directory."""
    if os.path.abspath(cache_dir) == os.path.abspath(C.CACHE_DIR):
        return C.fetch(url, stats)
    path = os.path.join(
        cache_dir,
        hashlib.sha256(url.encode("utf-8")).hexdigest() + ".html")
    if os.path.exists(path):
        stats["from_cache"] += 1
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    body = fetch_live(url, cache_dir, stats)
    return body


# ------------------------------------------------------------------
# shared helpers
# ------------------------------------------------------------------

def _smart_case(text: str) -> str:
    """'MATT REED' -> 'Matt Reed'; leaves mixed case untouched."""
    text = " ".join(text.split())
    if text and text == text.upper():
        return text.title()
    return text


def _ordinal(text: str) -> int | None:
    m = re.fullmatch(r"(\d+)(?:st|nd|rd|th)", text.strip().lower())
    return int(m.group(1)) if m else None


_LABEL_KM = {"8k": 8000.0, "8.001k": 8000.0, "5k": 5000.0,
             "6k": 6000.0, "7k": 7000.0, "10k": 10000.0,
             "4k": 4000.0, "3k": 3000.0, "2k": 2000.0}


def distance_for_label(label: str) -> float | None:
    """Best-effort distance in meters for a results event label.

    Handles the labels the platforms print: '8k' / '8.001k' /
    '5k', '3.11 mile', 'mile', plain-meter track labels ('3000',
    '5000'), and steeple variants ('3000s'). Unknown -> None.
    """
    lab = label.strip().lower().replace(" ", "").replace("(xc)", "")
    if lab in _LABEL_KM:
        return _LABEL_KM[lab]
    m = re.fullmatch(r"(\d+(?:\.\d+)?)k", lab)
    if m:
        return float(m.group(1)) * 1000.0
    m = re.fullmatch(r"(\d+(?:\.\d+)?)miles?", lab)
    if m:
        return round(float(m.group(1)) * 1609.344, 1)
    if lab in ("mile", "1mile"):
        return 1609.3
    if lab in ("3000s", "3000sc", "2000s", "2000sc"):
        return float(lab[:4])
    m = re.fullmatch(r"(\d+(?:,\d{3})*(?:\.\d+)?)meters?", lab)
    if m:
        return float(m.group(1).replace(",", ""))
    m = re.fullmatch(r"(\d+(?:,\d{3})*(?:\.\d+)?)m", lab)
    if m:  # Athletic.net short form: '5000m'
        return float(m.group(1).replace(",", ""))
    if re.fullmatch(r"\d{3,5}", lab):
        return float(lab)
    return None


# ------------------------------------------------------------------
# TFRRS adapter
# ------------------------------------------------------------------

def parse_tfrrs_profile(profile_html: str) -> dict:
    """Identity + raw results from a TFRRS athlete profile page.

    Returns {"athlete_name", "team", "results": [...]} where each
    result is {meet_url, meet_name, date (ISO), event_label,
    finish_time_s, place}. Pure function of the page text.
    """
    h3s = [" ".join(C._strip(m.group(1)).split())
           for m in re.finditer(r"<h3[^>]*>(.*?)</h3>",
                                profile_html, re.S)]
    name = None
    team = None
    if h3s:
        raw_name = re.sub(r"\s*\([^)]*\)\s*$", "", h3s[0])
        name = _smart_case(raw_name) or None
        for cand in h3s[1:]:
            if re.search(r"\b(19|20)\d{2}\b", cand):
                break  # season header ('2026 XC'): team came before
            team = _smart_case(cand) or None
            break
    results = []
    start = profile_html.find('id="meet-results"')
    if start >= 0:
        end = profile_html.find('id="event-history"', start)
        pane = profile_html[start:end if end >= 0
                            else len(profile_html)]
        for tbl in re.findall(r"<table.*?</table>", pane, re.S):
            head = re.search(
                r'<a[^>]*href="([^"]*/results/[^"]*)"[^>]*>(.*?)</a>',
                tbl, re.S)
            if not head:
                continue
            meet_url = head.group(1)
            head_text = tbl[:tbl.find("</thead>")] \
                if "</thead>" in tbl else tbl[:600]
            meet_date = C._parse_profile_date(C._strip(head_text))
            # The anchor text carries name + date ('Example Open
            # Oct 3, 2026'); the meet name is what precedes the date.
            link_text = " ".join(C._strip(head.group(2)).split())
            dm = re.search(r"[A-Z][a-z]{2}[a-z]*\s+\d{1,2},\s+\d{4}",
                           link_text)
            meet_name = link_text[:dm.start()].strip() \
                if dm else link_text
            for row in re.findall(r"<tr[^>]*>(.*?)</tr>", tbl, re.S):
                cells = [C._strip(c) for c in
                         re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)]
                if len(cells) < 2:
                    continue
                label = cells[0]
                t = None
                place = None
                for cell in cells[1:]:
                    if t is None:
                        t = C.parse_time(cell)
                        if t is not None:
                            continue
                    if place is None:
                        place = _ordinal(cell)
                if t is None:
                    continue
                results.append({
                    "meet_url": meet_url,
                    "meet_name": meet_name,
                    "date": meet_date.isoformat() if meet_date else None,
                    "event_label": label,
                    "finish_time_s": t,
                    "place": place,
                })
    return {"athlete_name": name, "team": team, "results": results}


def _canonical_results_url(url: str) -> str:
    """TFRRS serves the same meet page under tfrrs.org and
    www.tfrrs.org, and profile pages link one form or the other.
    Canonicalize to the bare host so discovered entries match
    the form the rest of the system already carries (and share
    one cache entry per meet)."""
    return re.sub(r"^(https?://)www\.(tfrrs\.org)", r"\1\2", url)


def _athlete_id(profile_url: str) -> str | None:
    m = re.search(r"/athletes/(\d+)", profile_url or "")
    return m.group(1) if m else None


def _row_is_athlete(fin: dict, athlete_id: str | None,
                    name: str | None, finish_time_s: float) -> bool:
    if athlete_id and fin.get("profile_url") \
            and f"/athletes/{athlete_id}/" in fin["profile_url"]:
        return True
    if name and fin.get("name"):
        same_name = " ".join(fin["name"].split()).upper() \
            == " ".join(name.split()).upper()
        if same_name and abs(fin["race_time_s"] - finish_time_s) <= 0.05:
            return True
    return False


def resolve_calibration_table(meet_html: str, athlete_id: str | None,
                              name: str | None,
                              finish_time_s: float) -> str | None:
    """The calibration_table hint for the athlete's own XC table.

    Scans the meet page's individual-results headings; for each,
    derives candidate hints (the full heading prefix before
    'Individual', and its shortened form before ' - ' when one
    exists — the form the hand-wired entries use) and verifies a
    candidate by parsing through calibrate's own table parser
    and finding the athlete's row in it. Returns the first
    verified hint, or None.
    """
    for m in re.finditer(r"<h3[^>]*>(.*?)</h3>", meet_html, re.S):
        text = " ".join(C._strip(m.group(1)).split())
        if "Individual" not in text:
            continue
        prefix = text.split("Individual")[0].strip().rstrip("-").strip()
        if not prefix:
            continue
        candidates = []
        if " - " in prefix:
            candidates.append(prefix.split(" - ")[0].strip())
        candidates.append(prefix)
        for cand in dict.fromkeys(candidates):
            if not cand:
                continue
            try:
                fins = C.parse_results_table(meet_html, cand)
            except ValueError:
                continue
            if any(_row_is_athlete(f, athlete_id, name, finish_time_s)
                   for f in fins):
                return cand
    return None


def discover_tfrrs(profile_url: str, stats: dict | None = None) -> dict:
    """Full TFRRS discovery for one athlete profile URL."""
    stats = stats if stats is not None else {
        "fetched_live": 0, "from_cache": 0, "failures": 0}
    out = {"platform": "tfrrs", "profile_url": profile_url,
           "athlete_name": None, "team": None, "status": "ok",
           "races": [], "bests": {}}
    page = fetch_live(profile_url, C.CACHE_DIR, stats)
    if page is None:
        out["status"] = "blocked"
        out["reason"] = "profile_fetch_failed"
        return out
    parsed = parse_tfrrs_profile(page)
    out["athlete_name"] = parsed["athlete_name"]
    out["team"] = parsed["team"]
    athlete_id = _athlete_id(profile_url)
    for res in parsed["results"]:
        kind = "xc" if "/results/xc/" in res["meet_url"] else "track"
        entry = {
            "date": res["date"],
            "meet_name": res["meet_name"],
            "event_label": res["event_label"],
            "kind": kind,
            "distance_m": distance_for_label(res["event_label"]),
            "finish_time_s": res["finish_time_s"],
            "place": res["place"],
            "results_url": _canonical_results_url(res["meet_url"]),
            "calibration_table": None,
        }
        if kind == "xc":
            meet_html = C.fetch(entry["results_url"], stats)
            if meet_html is not None:
                entry["calibration_table"] = resolve_calibration_table(
                    meet_html, athlete_id, parsed["athlete_name"],
                    res["finish_time_s"])
        out["races"].append(entry)
    out["races"].sort(key=lambda r: (r["date"] or "", r["meet_name"]))
    out["fetch_stats"] = dict(stats)
    return out


# ------------------------------------------------------------------
# Athletic.net adapter
# ------------------------------------------------------------------

def athleticnet_title_name(page_html: str) -> str | None:
    """'Jamie Sample - NJ Cross Country Bio' -> 'Jamie Sample'."""
    m = re.search(r"<title>(.*?)</title>", page_html, re.S)
    if not m:
        return None
    title = " ".join(C._strip(m.group(1)).split())
    m = re.match(r"(.+?)\s+-\s+[A-Z]{2}\s+(?:Cross Country|Track "
                 r"and Field|Track & Field)\s+Bio", title)
    return m.group(1) if m else None


def is_athleticnet_shell(page_html: str) -> bool:
    """True when the fetched document is the Angular app shell:
    the app bundle is referenced and no results content is in the
    document (the live-verified shape of every Athletic.net page
    this stack can fetch)."""
    return "angular.athletic.net" in page_html \
        and "Season Records" not in page_html


_RANKING_RE = re.compile(
    r"^(.+?)\s+-\s+(\d+:\d{2}\.\d+|\d+\.\d+)\s*$")
_RACE_ROW_RE = re.compile(
    r"^(\d+)\s+(\d+:\d{2}\.\d+|\d+\.\d+)(?:PB|SB)?\s+"
    r"([A-Z][a-z]{2}\s+\d{1,2})\s+(.+?)(?:\s+\(([^)]*)\))?\s*$")
_EVENT_HDR_RE = re.compile(
    r"^\d+(?:,\d{3})?\s*(?:Meters|m)$|^\d+(?:\.\d+)?\s+Miles?$",
    re.IGNORECASE)
_DATE_TOKEN_RE = re.compile(r"([A-Z][a-z]{2}\s+\d{1,2})\b")
_MASK_RUN_RE = re.compile(r"#{3,}")


def _season_date(mon_day: str, season_year: int | None) -> str | None:
    if season_year is None:
        return None
    md = C._parse_profile_date(f"{mon_day}, {season_year}")
    return md.isoformat() if md else None


def _masked_race(line: str, event: str | None, kind: str | None,
                 season_year: int | None) -> dict:
    """A result the page withholds from logged-out viewing.

    Two rendered forms: a 'Hidden Meet' entry (meet, time and
    place all hidden) and a row whose values render as
    '########'. Whatever the line still shows — an event
    section, a date, a place, a meet name — is kept; every
    withheld field stays None. Nothing is inferred, including
    from the Rankings panel.
    """
    hidden_meet = "hidden meet" in line.lower()
    date = None
    dm = _DATE_TOKEN_RE.search(line)
    if dm:
        date = _season_date(dm.group(1), season_year)
    place = None
    meet = None
    if hidden_meet:
        meet = "Hidden Meet"
    else:
        pm = re.match(r"^(\d+)\s", line)
        if pm:
            place = int(pm.group(1))
        rest = line
        if dm:
            rest = line[dm.end():]
        rest = _MASK_RUN_RE.sub(" ", rest)
        rest = re.sub(r"^[\d\s]+", "", rest).strip(" -")
        rest = re.sub(r"\s*\([^)]*\)\s*$", "", rest).strip()
        meet = rest or None
    return {
        "date": date,
        "meet_name": meet,
        "event_label": event,
        "kind": kind,
        "distance_m": distance_for_label(event) if event else None,
        "finish_time_s": None,
        "place": place,
        "results_url": None,
        "calibration_table": None,
        "masked": True,
        "partial": date is None,
    }


def parse_athleticnet_rendered(text: str, title_name: str | None) -> dict:
    """Best-effort parse of an Athletic.net athlete page's
    RENDERED text (tags stripped): identity, bests from the
    Rankings block, and race rows from the Season Records block.

    The parser targets the structure the platform renders —
    per-event season sections whose rows read '<place>
    <time>[PB|SB] <Mon D> <Meet> (<division>)', under a
    season-year line, with team history lines of the form
    '<year> <School> ...'. Logged-out viewing masks part of
    the history: 'Hidden Meet' entries and '########' values
    become masked race entries (see _masked_race), never
    filled in. A race row whose year cannot be attributed
    keeps date=None and is flagged partial. Pure function of
    the text.
    """
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    bests: dict[str, float] = {}
    races: list[dict] = []
    team = None
    # Team history lines: '2023 Lasell Collegiate 18 Freshman'
    for ln in lines:
        m = re.match(r"^(?:19|20)\d{2}\s+(.+?)\s+(?:Collegiate|High "
                     r"School|Middle School)\b", ln)
        if m:
            team = m.group(1)
    if team is None:  # header school line: 'Lincoln-Way West High School'
        for ln in lines:
            m = re.fullmatch(r"(.+?)\s+High School", ln)
            if m:
                team = m.group(1)
                break
    kind = None
    for ln in lines[:8]:  # the page's own title declares the sport
        if re.search(r"-\s+[A-Z]{2}\s+Cross Country\s+Bio", ln):
            kind = "xc"
            break
    if kind is None and "cross country" in text.lower()[:300]:
        kind = "xc"
    in_rankings = False
    in_records = False
    season_year: int | None = None
    event: str | None = None
    for ln in lines:
        if ln == "Rankings":
            in_rankings, in_records = True, False
            continue
        if ln == "Season Records":
            in_records, in_rankings = True, False
            continue
        if in_rankings:
            m = _RANKING_RE.match(ln)
            if m:
                t = C.parse_time(m.group(2)) \
                    if ":" in m.group(2) else None
                if t is None:
                    try:
                        t = float(m.group(2))
                    except ValueError:
                        t = None
                if t is not None:
                    bests[m.group(1)] = t
            continue
        if not in_records:
            continue
        ym = re.fullmatch(r"(?:19|20)\d{2}", ln)
        if ym:
            season_year = int(ln)
            continue
        if _EVENT_HDR_RE.match(ln):
            event = ln
            continue
        if "hidden meet" in ln.lower() or _MASK_RUN_RE.search(ln):
            if event:
                races.append(_masked_race(ln, event, kind,
                                          season_year))
            continue
        rm = _RACE_ROW_RE.match(ln)
        if rm and event:
            d = _season_date(rm.group(3), season_year)
            races.append({
                "date": d,
                "meet_name": rm.group(4),
                "event_label": event,
                "kind": kind,
                "distance_m": distance_for_label(event),
                "finish_time_s": C.parse_time(rm.group(2)),
                "place": int(rm.group(1)),
                "results_url": None,
                "calibration_table": None,
                "masked": False,
                "partial": d is None,
            })
    return {"athlete_name": title_name, "team": team,
            "bests": bests, "races": races}


def _visible_text(page_html: str) -> str:
    """A page's visible text: tags stripped line-wise so the
    rendered layout's line structure survives."""
    visible = re.sub(r"<script.*?</script>", " ", page_html, flags=re.S)
    visible = re.sub(r"<style.*?</style>", " ", visible, flags=re.S)
    visible = re.sub(r"<[^>]+>", "\n", visible)
    return "\n".join(" ".join(ln.split())
                     for ln in visible.splitlines())


def athleticnet_title_from_text(text: str) -> str | None:
    """The title-derived name from rendered TEXT (no <title>
    tag): the first lines carry 'Name - IL Cross Country Bio'."""
    for ln in text.splitlines()[:8]:
        m = re.match(r"(.+?)\s+-\s+[A-Z]{2}\s+(?:Cross Country|Track "
                     r"and Field|Track & Field)\s+Bio", ln.strip())
        if m:
            return m.group(1)
    return None


def athleticnet_output(profile_url: str, parsed: dict) -> dict:
    """The standard discovery output for a parsed Athletic.net
    page, with visible/masked counts and the status they imply:
    no results at all -> blocked; any masked result -> partial
    (reason stated); otherwise ok."""
    races = parsed["races"]
    masked_n = sum(1 for r in races if r.get("masked"))
    out = {"platform": "athleticnet", "profile_url": profile_url,
           "athlete_name": parsed["athlete_name"],
           "team": parsed["team"], "status": "ok",
           "races": races, "bests": parsed["bests"],
           "bests_source": "rankings_panel" if parsed["bests"]
                           else None,
           "results_found": len(races),
           "results_visible": len(races) - masked_n,
           "results_masked": masked_n,
           "platform_note": "Field calibration runs on TFRRS only "
                            "in this build; Athletic.net entries "
                            "carry calibration_table = null."}
    if not races:
        out["status"] = "blocked"
        out["reason"] = "no_results_found"
    elif masked_n:
        out["status"] = "partial"
        out["reason"] = "some results are hidden for logged-out viewing"
    return out


def _athleticnet_blocked(profile_url: str, name: str | None,
                         detail: str, nbytes: int) -> dict:
    out = athleticnet_output(profile_url, {
        "athlete_name": name, "team": None, "bests": {}, "races": []})
    out["reason"] = "js_wall"
    out["evidence"] = {"bytes": nbytes, "title_only_content": True,
                       "detail": detail}
    return out


def _parse_athleticnet_content(content: str) -> dict:
    """Parse Athletic.net page content, rendered HTML or text:
    HTML is reduced to its visible text first; the athlete name
    comes from the <title> or the rendered title line."""
    low = content.lower()
    if "<html" in low or "<body" in low or "<div" in low:
        name = athleticnet_title_name(content) \
            or athleticnet_title_from_text(_visible_text(content))
        return parse_athleticnet_rendered(_visible_text(content), name)
    name = athleticnet_title_from_text(content)
    return parse_athleticnet_rendered(content, name)


def discover_athleticnet(profile_url: str,
                         stats: dict | None = None,
                         cache_dir: str = ATHLETICNET_CACHE_DIR,
                         rendered_path: str | None = None) -> dict:
    """Athletic.net discovery for one athlete profile URL.

    Two-step, like the rest of this build's refreshes: the
    operator renders the page in their browser and saves it;
    discovery parses that rendered copy when one is supplied
    (rendered_path), writing it through to the cache as
    '<sha256>.rendered.html'. Without a rendered copy, the
    plain fetch is tried — and when the fetched document is
    the app shell, the JS wall is reported honestly: status
    'blocked', reason 'js_wall', evidence attached, and only
    the title-derived name carried.
    """
    stats = stats if stats is not None else {
        "fetched_live": 0, "from_cache": 0, "failures": 0}
    if rendered_path is not None:
        with open(rendered_path, encoding="utf-8",
                  errors="replace") as fh:
            content = fh.read()
        os.makedirs(cache_dir, exist_ok=True)
        cached = os.path.join(
            cache_dir,
            hashlib.sha256(profile_url.encode("utf-8")).hexdigest()
            + ".rendered.html")
        if os.path.abspath(rendered_path) != os.path.abspath(cached):
            with open(cached, "w", encoding="utf-8") as fh:
                fh.write(content)
        if is_athleticnet_shell(content):
            out = _athleticnet_blocked(
                profile_url, athleticnet_title_name(content)
                or athleticnet_title_from_text(content),
                "The supplied rendered copy is still the Angular "
                "app shell — the page had not finished rendering "
                "when it was saved.", len(content))
            out["source"] = "rendered"
            return out
        out = athleticnet_output(
            profile_url, _parse_athleticnet_content(content))
        out["source"] = "rendered"
        return out
    page = fetch_cached(profile_url, cache_dir, stats)
    if page is None:
        out = athleticnet_output(profile_url, {
            "athlete_name": None, "team": None, "bests": {},
            "races": []})
        out["reason"] = "profile_fetch_failed"
        out["fetch_stats"] = dict(stats)
        return out
    if is_athleticnet_shell(page):
        out = _athleticnet_blocked(
            profile_url, athleticnet_title_name(page),
            "Fetched document is the Angular app shell "
            "(angular.athletic.net bundle); the athlete's "
            "results, school and marks load client-side and are "
            "absent from the document. Supply a rendered copy "
            "with --rendered.", len(page))
        out["fetch_stats"] = dict(stats)
        return out
    out = athleticnet_output(
        profile_url, _parse_athleticnet_content(page))
    out["fetch_stats"] = dict(stats)
    return out


# ------------------------------------------------------------------
# dispatch + CLI
# ------------------------------------------------------------------

def discover(profile_url: str,
             rendered_path: str | None = None) -> dict:
    """Dispatch on the profile URL's platform. rendered_path is
    the operator-rendered page copy Athletic.net needs (see
    discover_athleticnet); TFRRS is server-rendered and ignores
    it."""
    if "athletic.net" in profile_url:
        return discover_athleticnet(profile_url,
                                    rendered_path=rendered_path)
    if "tfrrs.org" in profile_url:
        return discover_tfrrs(profile_url)
    raise ValueError(f"unsupported results platform: {profile_url}")


def main(argv):
    args = list(argv[1:])
    rendered = None
    if "--rendered" in args:
        i = args.index("--rendered")
        if i + 1 >= len(args):
            print(__doc__)
            return 2
        rendered = args[i + 1]
        del args[i:i + 2]
    if len(args) not in (1, 2):
        print(__doc__)
        return 2
    try:
        result = discover(args[0], rendered_path=rendered)
    except C.Blocked as exc:
        print(json.dumps({"blocked": True, "reason": str(exc),
                          "note": "stopped; partial state is not a "
                                  "race list"}, indent=2))
        return 3
    if len(args) == 2:
        with open(args[1], "w") as fh:
            json.dump(result, fh, indent=1)
    summary = {k: result.get(k) for k in
               ("platform", "athlete_name", "team", "status", "reason",
                "results_found", "results_visible", "results_masked")
               if result.get(k) is not None}
    summary["races_found"] = len(result.get("races", []))
    summary["races"] = result.get("races", [])
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
