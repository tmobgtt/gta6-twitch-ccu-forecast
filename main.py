"""Forecast a launch day's Twitch viewers, hour by hour, from its first hour.

Entry for make.fun's gta6-twitch-ccu-forecast-01 (Python 3.13, standard
library only, no network). The grader runs this file once per case: it
reads every <id>.in in the input directory and writes <id>.out holding,
for each of hours 2 to 24 of the release day, five quantiles (5%, 25%,
50%, 75%, 95%) of that hour's mean concurrent viewers in the game's
Twitch category.

Method: an empirical ratio model over the case's own comparables. Every
past big launch in the case contributes the shape of its release day as
log-ratios to its own first hour, ln(viewers_h / viewers_1). A launch is
weighted by how much it resembles this one:

  - scale: launches whose first hour is the same size as this case's
    first hour trace the most similar days (a Gaussian kernel on the
    log of hour 1, sigma 0.9);
  - maturity: launches whose first hour sits at the same height above
    the previous day's peak as this one's (a Gaussian kernel on
    ln(hour 1 / previous-day peak), sigma 1.2);
  - unlock shape: a launch that unlocked the way this one did (rolling
    midnight local, or one simultaneous global unlock) counts in full,
    a different shape counts a quarter.

Each hour's five quantiles are the weighted quantiles of those
log-ratios; the median curve is smoothed over neighbouring hours and
the other quantiles keep their distance from it, so the spread is the
spread the comparable launches actually showed. The two tail quantiles
are blended halfway toward a lognormal tail fitted to the same
samples, so no single comparable can own an extreme quantile. Hours
with too little comparable evidence fall back to the persistence shape
(flat at hour 1, spread widening like a random walk's). Everything is
a deterministic function of the case file, so two runs write the same
bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import NormalDist

# The grader's -P leaves /bot off sys.path: put it back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Kernel widths and the mismatched-unlock weight (tuned by leave-one-out
# backtests over the practice cases' comparables; see README).
SIGMA_SCALE = 0.9
SIGMA_MATURITY = 1.2
CLASS_MISMATCH = 0.25
MIN_SAMPLES = 5
# Log-ratio quantiles are clipped to this band: no hour of a launch day
# is forecast below e^-3 or above e^3.5 times its first hour.
LOG_LO, LOG_HI = -3.0, 3.5

_NORM = NormalDist()


def parse_ts(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def unlock_class(text):
    """'rolling' for midnight-local rollouts, 'global' for one simultaneous
    unlock, 'other' when the text says neither."""
    t = (text or "").lower()
    if "midnight local" in t or "rolling" in t:
        return "rolling"
    if "global" in t or "simultaneous" in t:
        return "global"
    return "other"


def hourly_map(points, key):
    out = {}
    for p in points or []:
        try:
            out[parse_ts(p[key])] = p.get("viewers")
        except (KeyError, TypeError, ValueError):
            continue
    return out


def day_curve(points_by_time, day_start):
    """The 24 hourly values of the release day and the previous day's peak."""
    day = [points_by_time.get(day_start + timedelta(hours=k)) for k in range(24)]
    pre = [points_by_time.get(day_start - timedelta(hours=k)) for k in range(1, 25)]
    pre = [v for v in pre if v]
    return day, (max(pre) if pre else None)


def comparable_features(comp):
    """A comparable's release-day curve and features, or None if its day
    is not recorded well enough to learn a shape from."""
    try:
        ds = parse_ts(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    day, prepeak = day_curve(hourly_map(comp.get("hourly"), "from"), ds)
    if not day[0] or day[0] <= 0:
        return None
    if sum(1 for v in day if v) < 20:
        return None
    maturity = math.log(day[0] / prepeak) if prepeak else 0.0
    return {
        "name": comp.get("name", ""),
        "day": day,
        "v1": float(day[0]),
        "maturity": maturity,
        "cls": unlock_class(comp.get("unlock")),
    }


def target_features(case):
    """This case's first-hour print, maturity and unlock shape."""
    first = next((k for k in case.get("known", []) if k.get("hour") == 1), None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    ds = parse_ts(case["day_start"])
    _, prepeak = day_curve(hourly_map(case.get("history_hourly"), "from"), ds)
    if not prepeak:
        # No whole-hour means before the day: fall back to the raw
        # history points, averaged into hours, for the previous day.
        by_hour = {}
        for p in case.get("history", []):
            try:
                t = parse_ts(p["at"])
            except (KeyError, TypeError, ValueError):
                continue
            if ds - timedelta(hours=24) <= t < ds and p.get("viewers"):
                by_hour.setdefault(t.replace(minute=0, second=0, microsecond=0), []).append(p["viewers"])
        if by_hour:
            prepeak = max(sum(v) / len(v) for v in by_hour.values())
    maturity = math.log(p1 / prepeak) if prepeak else 0.0
    return p1, maturity, unlock_class(case.get("notes"))


def weight(comp, p1, maturity, cls):
    d_scale = (math.log(comp["v1"]) - math.log(p1)) / SIGMA_SCALE
    d_mat = (comp["maturity"] - maturity) / SIGMA_MATURITY
    w = math.exp(-0.5 * d_scale * d_scale) * math.exp(-0.5 * d_mat * d_mat)
    if comp["cls"] != cls:
        w *= CLASS_MISMATCH
    return w


def weighted_quantile(samples, tau):
    """samples: list of (value, weight); linear-interpolated weighted quantile."""
    samples = sorted(samples)
    total = sum(w for _, w in samples)
    cum, prev_v, prev_c = 0.0, samples[0][0], 0.0
    for v, w in samples:
        cum += w
        c = cum / total
        if c >= tau:
            if c == prev_c:
                return v
            return prev_v + (tau - prev_c) / (c - prev_c) * (v - prev_v)
        prev_v, prev_c = v, c
    return samples[-1][0]


def baseline_row(p1, h, levels):
    """The persistence shape: flat at hour 1, log-spread 0.15 * sqrt(h - 1)."""
    return [p1 * math.exp(_NORM.inv_cdf(t) * 0.15 * math.sqrt(h - 1)) for t in levels]


def round2(x):
    """Round to 2 decimals, half up; whole numbers come back as ints."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if float(value).is_integer() else value


def forecast_case(case):
    levels = case.get("levels") or LEVELS
    p1, maturity, cls = target_features(case)
    pool = []
    for comp in sorted(case.get("comparables", []), key=lambda c: c.get("name", "")):
        f = comparable_features(comp)
        if f is not None:
            f["w"] = weight(f, p1, maturity, cls)
            pool.append(f)
    # Per-hour weighted quantiles of the comparables' log-ratios.
    raw = {}
    for row in case["forecast"]:
        h = row["hour"]
        samples = []
        for c in pool:
            v = c["day"][h - 1]
            if v and v > 0 and c["w"] > 1e-9:
                samples.append((math.log(v) - math.log(c["v1"]), c["w"]))
        if len(samples) < MIN_SAMPLES:
            raw[h] = None
            continue
        qs = [weighted_quantile(samples, t) for t in levels]
        # Blend the two tail quantiles halfway toward a lognormal tail
        # fitted to the same samples: with a few dozen comparables one
        # heavy launch can otherwise own the 95% quantile outright.
        total = sum(w for _, w in samples)
        mu = sum(v * w for v, w in samples) / total
        var = sum(w * (v - mu) ** 2 for v, w in samples) / total
        sd = math.sqrt(var)
        for qi in (0, len(levels) - 1):
            ln_q = mu + _NORM.inv_cdf(levels[qi]) * sd
            qs[qi] = 0.5 * qs[qi] + 0.5 * ln_q
        raw[h] = sorted(qs)
    # Smooth the median curve over neighbouring hours; the other
    # quantiles keep their distance from the raw median. The median is
    # the middle level (the schema's five levels centre on 0.5).
    mid = len(levels) // 2
    med = {h: r[mid] for h, r in raw.items() if r}
    smooth = {}
    for h in med:
        num = den = 0.0
        for k, kw in ((h - 1, 0.25), (h, 0.5), (h + 1, 0.25)):
            if k in med:
                num += kw * med[k]
                den += kw
        smooth[h] = num / den
    rows = []
    for row in case["forecast"]:
        h = row["hour"]
        if raw[h] is None:
            qs = baseline_row(p1, h, levels)
        else:
            logs = [smooth[h] + (raw[h][i] - med[h]) for i in range(len(levels))]
            logs = sorted(min(max(x, LOG_LO), LOG_HI) for x in logs)
            qs = [p1 * math.exp(x) for x in logs]
        vals = []
        for q in qs:
            v = round2(min(max(q, 0.01), 1e9))
            if vals and v < vals[-1]:
                v = vals[-1]
            vals.append(v)
        rows.append({"hour": h, "q": vals})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
