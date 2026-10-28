"""Deterministic statistics for Stridewise: OLS slopes, week-block
bootstrap likely ranges, and the evidence-tier gate.

Design notes (see insight-registry-draft.md):
- Every association is an OLS slope with a bootstrap 95% "likely range".
- Bootstrap resamples by WEEK blocks, not single observations, so
  night-to-night autocorrelation cannot fake a tighter range than the
  data earns. Weekly-grain questions have one pair per week, so their
  blocks are the pairs themselves (a pairs bootstrap).
- Tiers are recomputed from scratch on every refresh -- never sticky.
- Final tier = min(statistical tier, density cap, recency cap), and the
  card always names the binding rule.

Interpretation decision (registry ambiguity, documented here): the
registry lists "completeness < 70%" as an insufficient_data trigger
alongside the density rule (>= 50% coverage of first-to-last span).
Completeness is implemented as usable_pairs / collected_pairs inside
the window, i.e. the share of collected pairs surviving sensor
exclusion. A window-coverage reading was rejected: it would mark
strong-evidence profiles insufficient (e.g. 112 paired nights in a
180-day window is 62% window coverage, yet comfortably clears every
other bar). Spread is handled by the density rule, not by completeness.
"""
from __future__ import annotations

import random
from datetime import date, timedelta

# Tier ranks: higher = stronger evidence.
TIER_RANK = {"insufficient_data": 0, "no_pattern": 1, "emerging": 2, "solid": 3}

NIGHTLY_MIN_PAIRS = 20
NIGHTLY_SOLID_PAIRS = 40
WEEKLY_MIN_WEEKS = 6
WEEKLY_SOLID_WEEKS = 10
DENSITY_FLOOR = 0.50
COMPLETENESS_FLOOR = 0.70
RECENCY_NIGHTLY_MIN = 8      # paired obs in last 28 days for solid
RECENCY_WEEKLY_MIN = 3       # usable weeks of last 4 for solid
BOOT_REPS = 1000
BOOT_SEED = 20261028


def ols_slope(xs, ys):
    """Least-squares slope, or None when it cannot be computed."""
    n = len(xs)
    if n < 2:
        return None
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0:
        return None
    return (n * sxy - sx * sy) / denom


def _week_key(d: date):
    iso = d.isocalendar()
    return (iso[0], iso[1])


def bootstrap_range(pairs, seed=BOOT_SEED, reps=BOOT_REPS):
    """Week-block bootstrap 95% interval for the OLS slope.

    pairs: list of (date, x, y). Returns (lo, hi) or (None, None) when
    the interval cannot be computed (too few blocks / degenerate draws).
    Deterministic: fixed seed, so refreshes reproduce exactly.
    """
    if len(pairs) < 2:
        return (None, None)
    groups = {}
    for d, x, y in pairs:
        groups.setdefault(_week_key(d), []).append((x, y))
    blocks = list(groups.values())
    if len(blocks) < 2:
        return (None, None)
    # Per-block sums let each resample stay O(#blocks) instead of O(n).
    sums = []
    for b in blocks:
        n = len(b)
        sums.append((n, sum(x for x, _ in b), sum(y for _, y in b),
                     sum(x * x for x, _ in b),
                     sum(x * y for x, y in b)))
    rng = random.Random(seed)
    slopes = []
    nb = len(blocks)
    for _ in range(reps):
        n = sx = sy = sxx = sxy = 0
        for _ in range(nb):
            cn, csx, csy, csxx, csxy = sums[rng.randrange(nb)]
            n += cn
            sx += csx
            sy += csy
            sxx += csxx
            sxy += csxy
        denom = n * sxx - sx * sx
        if n >= 2 and denom != 0:
            slopes.append((n * sxy - sx * sy) / denom)
    if len(slopes) < 50:
        return (None, None)
    slopes.sort()
    lo = slopes[int(0.025 * len(slopes))]
    hi = slopes[int(0.975 * len(slopes))]
    return (lo, hi)


def _sign(v):
    if v is None:
        return 0
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


def _half_slopes(pairs):
    """OLS slopes of the first and second halves (by date order)."""
    ordered = sorted(pairs)
    n = len(ordered)
    if n < 6:
        return (None, None)
    mid = n // 2
    first = ordered[:mid]
    second = ordered[mid:]
    s1 = ols_slope([x for _, x, _ in first], [y for _, _, y in first])
    s2 = ols_slope([x for _, x, _ in second], [y for _, _, y in second])
    return (s1, s2)


def evaluate(pairs, effect_scale, floor, grain, anchor, prev_card=None,
             collected_pairs=None):
    """Run the full evidence gate for one association question.

    pairs: usable (date, x, y) tuples, already sensor-filtered.
    effect_scale: multiply the raw slope by this to get effect units
        (e.g. 60.0 turns a per-minute slope into a per-hour effect).
    floor: practical floor in effect units (below = not worth attention).
    grain: "nightly" or "weekly".
    anchor: latest data date; windows end here.
    prev_card: previous refresh's card dict for this question, or None.
    collected_pairs: pairs counted before sensor exclusion (for the
        completeness guard); defaults to len(pairs).

    Returns a dict with effect, likely_range, n, tier, binding_rule,
    and demotion (None or {was, reason}).
    """
    if grain == "nightly":
        min_n, solid_n = NIGHTLY_MIN_PAIRS, NIGHTLY_SOLID_PAIRS
        window_start = anchor - timedelta(days=179) if anchor else None
        recent_start = anchor - timedelta(days=27) if anchor else None
        recent_need = RECENCY_NIGHTLY_MIN
    else:
        min_n, solid_n = WEEKLY_MIN_WEEKS, WEEKLY_SOLID_WEEKS
        # Weekly pairs carry Monday dates; the window is the last 26
        # Mondays and recency counts the last 4 Mondays.
        window_start = None
        recent_start = None
        if anchor is not None:
            from .model import monday_of
            window_start = monday_of(anchor) - timedelta(days=25 * 7)
            recent_start = monday_of(anchor) - timedelta(days=21)
        recent_need = RECENCY_WEEKLY_MIN

    in_window = [(d, x, y) for (d, x, y) in pairs
                 if window_start is None or d >= window_start]
    n = len(in_window)
    collected = collected_pairs if collected_pairs is not None else n
    completeness = (n / collected) if collected else 1.0

    result = {
        "effect": None, "likely_range": [None, None], "n": n,
        "tier": "insufficient_data", "binding_rule": "", "demotion": None,
        "completeness": round(completeness, 3),
    }

    if n < min_n:
        result["binding_rule"] = (
            f"n below compute floor ({n} of {min_n} needed)")
        return _with_demotin(result, prev_card)

    if completeness < COMPLETENESS_FLOOR:
        result["binding_rule"] = (
            f"completeness below floor ({completeness:.0%} usable after "
            f"sensor exclusion)")
        return _with_demotin(result, prev_card)

    xs = [x for _, x, _ in in_window]
    ys = [y for _, _, y in in_window]
    slope = ols_slope(xs, ys)
    if slope is None:
        result["binding_rule"] = "no variation in driver"
        result["tier"] = "no_pattern"
        return _with_demotin(result, prev_card)
    effect = slope * effect_scale
    lo, hi = bootstrap_range(in_window)
    lo_e = None if lo is None else lo * effect_scale
    hi_e = None if hi is None else hi * effect_scale
    # Keep lo <= hi even for negative effects (scale is positive).
    if lo_e is not None and hi_e is not None and lo_e > hi_e:
        lo_e, hi_e = hi_e, lo_e
    result["effect"] = round(effect, 3)
    result["likely_range"] = [
        None if lo_e is None else round(lo_e, 3),
        None if hi_e is None else round(hi_e, 3),
    ]

    # Statistical tier.
    if lo_e is not None and hi_e is not None and lo_e <= 0 <= hi_e:
        result["tier"] = "no_pattern"
        result["binding_rule"] = "likely range includes zero"
        return _with_demotin(result, prev_card)
    if abs(effect) < floor:
        result["tier"] = "no_pattern"
        result["binding_rule"] = (
            f"effect below practical floor (|{effect:.3g}| < {floor:g})")
        return _with_demotin(result, prev_card)
    if lo_e is None:
        # Range unavailable: cannot confirm an effect, but the point
        # estimate clears the floor, so this is not "no pattern" by
        # range -- it is unconfirmed evidence.
        result["tier"] = "emerging"
        result["binding_rule"] = "likely range unavailable (too few blocks)"
        return _with_demotin(result, prev_card)

    # Range excludes zero and effect clears the floor: emerging at least.
    s1, s2 = _half_slopes(in_window)
    halves_ok = (_sign(s1) == _sign(effect) and _sign(s2) == _sign(effect)
                 and s1 is not None and s2 is not None)
    recent_pairs = [(d, x, y) for (d, x, y) in in_window
                    if recent_start is None or d >= recent_start]
    recent_slope = ols_slope([x for _, x, _ in recent_pairs],
                             [y for _, _, y in recent_pairs])
    recent_ok = (_sign(recent_slope) == _sign(effect)
                 and recent_slope is not None)

    stat_tier = "emerging"
    stat_reason = ""
    if n < solid_n:
        stat_reason = f"below solid bar ({n} of {solid_n} needed)"
    elif not halves_ok:
        stat_reason = "direction inconsistent across halves"
    elif not recent_ok:
        stat_reason = "direction inconsistent in recent data"
    else:
        stat_tier = "solid"

    # Caps (apply after the statistical tier).
    if grain == "nightly":
        span = (max(d for d, _, _ in in_window)
                - min(d for d, _, _ in in_window)).days + 1
        density = n / span if span else 0.0
        recent_n = len(recent_pairs)
    else:
        # Weekly pairs are one per week by construction, but weeks with
        # no data are absent -- density is usable weeks / span weeks.
        mondays = sorted({d for d, _, _ in in_window})
        span = ((mondays[-1] - mondays[0]).days // 7 + 1) if mondays else 0
        density = n / span if span else 0.0
        recent_weeks = {_week_key(d) for d, _, _ in recent_pairs}
        recent_n = len(recent_weeks)
    result["density"] = round(density, 3)

    if density < DENSITY_FLOOR:
        result["tier"] = "emerging"
        result["binding_rule"] = "evidence too spread out"
    elif recent_n < recent_need:
        result["tier"] = "emerging"
        result["binding_rule"] = "evidence is stale"
    else:
        result["tier"] = stat_tier
        result["binding_rule"] = ("all gates passed" if stat_tier == "solid"
                                  else stat_reason)
    return _with_demotin(result, prev_card)


def _with_demotin(result, prev_card):
    """Attach a demotion line when a solid/emerging card dropped."""
    if not prev_card:
        return result
    prev_tier = prev_card.get("tier")
    if prev_tier in ("solid", "emerging") and \
            TIER_RANK[result["tier"]] < TIER_RANK[prev_tier]:
        prev_effect = prev_card.get("effect")
        prev_range = prev_card.get("likely_range") or [None, None]
        cur_range = result["likely_range"]
        if (prev_range[0] is not None and prev_range[1] is not None
                and cur_range[0] is not None and cur_range[1] is not None
                and not (prev_range[0] <= 0 <= prev_range[1])
                and cur_range[0] <= 0 <= cur_range[1]):
            reason = "the likely range widened past zero on more data"
        elif (prev_effect is not None and result["effect"] is not None
                and _sign(prev_effect) != _sign(result["effect"])):
            reason = "the direction flipped in recent data"
        else:
            reason = "more data moved the estimate"
        result["demotion"] = {"was": prev_tier, "reason": reason}
    return result
