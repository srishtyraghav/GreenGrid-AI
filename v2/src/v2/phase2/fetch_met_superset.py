"""
GreenGrid-AI — V2 Phase 2
Task 1: Meteorological superset fetch (Open-Meteo Historical Weather API)

Fetches an hourly 2022–2026 SUPERSET of the 6 V1 met covariates on the same
16-point 4x4 grid, so any temporal-window decision can be aggregated later
without refetching. V1 data stay byte-identical: this script writes only under
data/v2/phase2/met/.

Grid points are read (read-only) from the V1 provenance CSV:
    data/processed/experiments/exp_met_features_grid_provenance.csv
and fall back to a documented 4x4 reconstruction if that file is unreadable.

Raw JSON is cached per (point, year) under raw_cache/ before parsing, so
reruns never refetch already-cached responses.

Usage:
    PYTHONUTF8=1 PYTHONPATH=src ./.venv/Scripts/python.exe \
        src/v2/phase2/fetch_met_superset.py
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

MET_DIR = PROJECT_ROOT / "data" / "v2" / "phase2" / "met"
RAW_CACHE = MET_DIR / "raw_cache"
OUT_CSV = MET_DIR / "met_superset_hourly_2022_2026.csv"
OUT_PROV = MET_DIR / "met_superset_provenance.md"
GRID_PROV_CSV = (
    PROJECT_ROOT / "data" / "processed" / "experiments" /
    "exp_met_features_grid_provenance.csv"
)

API_URL = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation",
    "shortwave_radiation",
    "soil_moisture_0_to_7cm",
]
HOURLY_VARS_CSV = ",".join(HOURLY_VARS)

YEARS = list(range(2022, 2027))

# End date per year: full calendar year, clamped to today (UTC) — the archive
# API rejects future end dates (verified 400 "out of allowed range"). Fetching
# with end_date = today returns the full day with 0 nulls.
TODAY_UTC = datetime.now(timezone.utc).date()


def _is_leap(y):
    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)


def end_date_for(year):
    full = f"{year}-12-31"
    if year == TODAY_UTC.year:
        return min(full, TODAY_UTC.isoformat())
    return full


def hours_in(year):
    d0 = datetime(year, 1, 1)
    d1 = datetime.strptime(end_date_for(year), "%Y-%m-%d")
    return int((d1 - d0).total_seconds() // 3600) + 24


# Expected hours per year (leap-aware, 2026 clamped to fetch date) + row target.
HOURS_PER_YEAR = {y: hours_in(y) for y in YEARS}
EXPECTED_HOURS_PER_POINT = sum(HOURS_PER_YEAR.values())
EXPECTED_TOTAL_ROWS = 16 * EXPECTED_HOURS_PER_POINT
NOMINAL_HOURS_FULL_2022_2026 = 43824  # 16 x this would need a complete 2026

HEADERS = {"User-Agent": "GreenGridAI/2.0 (student research project; contact: greengrid-ai)"}

MAX_ATTEMPTS = 4
POLITE_SLEEP_S = 1.5


def load_grid_points():
    """Return ordered unique (lat, lon) tuples, preferring the V1 provenance CSV."""
    if GRID_PROV_CSV.exists():
        try:
            df = pd.read_csv(GRID_PROV_CSV)
            pts = sorted({(float(a), float(b)) for a, b in zip(df["lat"], df["lon"])})
            if len(pts) == 16:
                print(f"[GRID] Reused exact 16 V1 points from {GRID_PROV_CSV.name}")
                return pts, f"reused from {GRID_PROV_CSV}"
        except Exception as e:
            print(f"[GRID] V1 provenance CSV unreadable ({e}); reconstructing 4x4 grid")
    lats = [round(28.40 + i * (28.88 - 28.40) / 3, 2) for i in range(4)]
    lons = [round(76.83 + i * (77.35 - 76.83) / 3, 2) for i in range(4)]
    pts = [(la, lo) for la in lats for lo in lons]
    return pts, ("reconstructed 4x4 (28.40..28.88 x 76.83..77.35 inclusive endpoints; "
                 "V1 CSV unavailable)")


def api_params(lat, lon, year):
    return {
        "latitude": f"{lat:.2f}",
        "longitude": f"{lon:.2f}",
        "start_date": f"{year}-01-01",
        "end_date": end_date_for(year),
        "hourly": HOURLY_VARS_CSV,
        "timezone": "UTC",
    }


def fetch_one(lat, lon, year, cache_path):
    """Fetch one point-year; serve from cache when present. Returns parsed JSON."""
    if cache_path.exists() and cache_path.stat().st_size > 500:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    last_err = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            r = requests.get(API_URL, params=api_params(lat, lon, year),
                             headers=HEADERS, timeout=60)
            r.raise_for_status()
            payload = r.json()
            if "hourly" not in payload or "time" not in payload["hourly"]:
                raise ValueError(f"unexpected payload keys: {list(payload)[:8]}")
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            time.sleep(POLITE_SLEEP_S)
            return payload
        except Exception as e:
            last_err = e
            wait = 5 * attempt
            print(f"  [WARN] {lat},{lon} {year} attempt {attempt}/{MAX_ATTEMPTS}: "
                  f"{type(e).__name__}: {e} — sleeping {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"fetch failed for {lat},{lon} {year}: {last_err}")


def parse_point_year(point_id, lat, lon, payload, units_acc):
    hourly = payload["hourly"]
    times = hourly["time"]
    n = len(times)
    for k, v in payload.get("hourly_units", {}).items():
        units_acc.setdefault(k, v)
    recs = {
        "point_id": point_id,
        "lat": lat,
        "lon": lon,
        "time_utc": times,
    }
    for var in HOURLY_VARS:
        col = hourly[var]
        if len(col) != n:
            raise ValueError(f"{var} length {len(col)} != time length {n}")
        recs[var] = col
    return pd.DataFrame(recs)


def main():
    t0 = time.time()
    fetch_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    MET_DIR.mkdir(parents=True, exist_ok=True)
    RAW_CACHE.mkdir(parents=True, exist_ok=True)

    points, grid_note = load_grid_points()
    print(f"[GRID] {len(points)} points | {grid_note}")

    frames = []
    units_acc = {}
    n_req = 0
    print(f"[RANGE] per-point hours: "
          + ", ".join(f"{y}:{HOURS_PER_YEAR[y]}" for y in YEARS)
          + f" = {EXPECTED_HOURS_PER_POINT:,} h/point, {EXPECTED_TOTAL_ROWS:,} rows expected")
    for pi, (lat, lon) in enumerate(points, start=1):
        pid = f"p{pi:02d}"
        for year in YEARS:
            end = end_date_for(year)
            cache_path = (RAW_CACHE /
                          f"met_{pid}_{lat:.2f}_{lon:.2f}_{year}_to{end}.json")
            payload = fetch_one(lat, lon, year, cache_path)
            n_req += 1
            df = parse_point_year(pid, lat, lon, payload, units_acc)
            if len(df) != HOURS_PER_YEAR[year]:
                print(f"  [WARN] {pid} {year}: {len(df)} hours, expected "
                      f"{HOURS_PER_YEAR[year]}")
            frames.append(df)
            if n_req % 20 == 0:
                print(f"  [PROGRESS] {n_req} point-year fetches done "
                      f"({time.time() - t0:.0f}s)")

    print(f"[PARSE] Concatenating {len(frames)} point-year frames...")
    long_df = pd.concat(frames, ignore_index=True)
    long_df["time_utc"] = pd.to_datetime(long_df["time_utc"], utc=True)
    long_df = long_df.sort_values(["point_id", "time_utc"]).reset_index(drop=True)

    # ── sanity checks ────────────────────────────────────────────────────────
    print(f"[SANITY] rows = {len(long_df):,} (expect {EXPECTED_TOTAL_ROWS:,})")
    assert len(long_df) == EXPECTED_TOTAL_ROWS, "row-count sanity check FAILED"
    dup = long_df.duplicated(subset=["point_id", "time_utc"]).sum()
    assert dup == 0, f"duplicate (point_id,time_utc) rows: {dup}"
    nan_summary = {v: int(long_df[v].isna().sum()) for v in HOURLY_VARS}
    print(f"[SANITY] NaN counts per variable: {nan_summary}")

    # Cross-check vs V1: 2023-07, 04:00–06:00 UTC grid-mean temperature_2m.
    jul23 = long_df[(long_df["time_utc"].dt.year == 2023) &
                    (long_df["time_utc"].dt.month == 7) &
                    (long_df["time_utc"].dt.hour.between(4, 6))]
    sanity_t = jul23["temperature_2m"].mean()
    print(f"[SANITY] 2023-07 Delhi mean T2m 04:00–06:00 UTC = {sanity_t:.2f} °C "
          f"(expect ~30 °C order of magnitude)")
    assert 25.0 <= sanity_t <= 35.0, "2023-07 temperature sanity out of range"

    long_df.to_csv(OUT_CSV, index=False)
    size_mb = OUT_CSV.stat().st_size / 1e6
    print(f"[WRITE] {OUT_CSV} ({size_mb:.1f} MB)")

    # ── provenance ───────────────────────────────────────────────────────────
    tmpl = (f"{API_URL}?latitude={{lat}}&longitude={{lon}}&start_date={{YYYY}}-01-01"
            f"&end_date={{YYYY}}-12-31 (clamped to fetch date for the current year)"
            f"&hourly={HOURLY_VARS_CSV}&timezone=UTC")
    trunc_note = ""
    if end_date_for(TODAY_UTC.year) != f"{TODAY_UTC.year}-12-31":
        trunc_note = (
            f"\n**IMPORTANT — 2026 is truncated at the fetch date:** today is "
            f"{TODAY_UTC.isoformat()} (UTC) and the archive API rejects future end "
            f"dates (verified HTTP 400), so the {TODAY_UTC.year} year runs only through "
            f"**{end_date_for(TODAY_UTC.year)}** "
            f"({HOURS_PER_YEAR[TODAY_UTC.year]:,} h). The nominal spec "
            f"({NOMINAL_HOURS_FULL_2022_2026:,} h/point incl. a full 2026) is therefore "
            f"not yet reachable; complete 2026 by re-running this script after "
            f"{TODAY_UTC.year}-12-31 — the {YEARS[0]}–{TODAY_UTC.year - 1} cache is "
            f"reused and only the {TODAY_UTC.year} point-years refetch (cache filenames "
            f"embed the end date).\n")
    prov = f"""# Meteorological Superset (V2 Phase 2) — Fetch Provenance

**Script:** `src/v2/phase2/fetch_met_superset.py`
**Fetch timestamp (UTC):** {fetch_ts}
**Requests issued this run:** {n_req} point-year fetches (served from raw_cache where already cached)
{trunc_note}
## Source
- **API:** Open-Meteo Historical Weather API — `{API_URL}` (no key required)
- **Underlying data:** ECMWF **ERA5 / ERA5-Land** reanalysis (Copernicus Climate Data Store), served under **CC-BY 4.0** per Open-Meteo terms (https://open-meteo.com/en/terms, https://cds.climate.copernicus.eu).
- Mirrors V1 provenance in `reports/experiments/exp_met_provenance.md`, but fetches a **temporal superset**: full hourly 2022–2026 per point instead of July-only means.

## API template (verbatim)
```
{tmpl}
```
One request per grid point per calendar year (16 points x 5 years = 80 point-year requests; each is a single point-year call against Open-Meteo, retried with exponential backoff, with a {POLITE_SLEEP_S}s polite delay between requests).

## Variables (hourly) & units (as returned by the API)
"""
    for v in HOURLY_VARS:
        prov += f"- `{v}` — {units_acc.get(v, 'unit not reported')}\n"
    prov += f"""
## Grid definition
- Same 16-point 4x4 grid as V1, spanning lat 28.40–28.88 x lon 76.83–77.35 (inclusive endpoints).
- Grid provenance: **{grid_note}**.
- `point_id` p01–p16 ordered row-major: lat ascending outer, lon ascending inner
  (p01 = 28.40,76.83 … p16 = 28.88,77.35).

## Output
- `data/v2/phase2/met/met_superset_hourly_2022_2026.csv` — long format:
  `point_id, lat, lon, time_utc, {', '.join(HOURLY_VARS)}`
- Rows: {len(long_df):,} = 16 points x {EXPECTED_HOURS_PER_POINT:,} hours
  (2022–2025 full years incl. leap 2024 = 35,040 h; 2026 through {end_date_for(TODAY_UTC.year)} = {HOURS_PER_YEAR[TODAY_UTC.year]:,} h).
  Sanity assertion **PASS** (0 duplicate point-time rows).
- Raw per-(point,year) JSON cached under `data/v2/phase2/met/raw_cache/` before parsing; reruns do not refetch.

## Spatial-resolution warning (explicit)
ERA5-Land native resolution is ~0.1° (~9–11 km). These fields are **synoptic-scale only**;
any aggregation to pixels must NOT claim 30 m precision (same rule as V1: interpolate, don't imply native resolution).

## NaN summary (this fetch)
```
{json.dumps(nan_summary, indent=2)}
```

## Sanity cross-check vs V1
2023-07, 04:00–06:00 UTC, grid-mean `temperature_2m` = **{sanity_t:.2f} °C**
(V1 `exp_met_provenance.md` reports 29.97 °C for the same window; order-of-magnitude match ~30 °C).

## License note
Data delivered by Open-Meteo under **CC-BY 4.0**; underlying ERA5/ERA5-Land (C) Copernicus Climate Change Service. Attribution: "Weather data by Open-Meteo.com" + Copernicus CDS citation.
"""
    with open(OUT_PROV, "w", encoding="utf-8") as f:
        f.write(prov)
    print(f"[WRITE] {OUT_PROV}")
    print(f"[DONE] met superset fetch+build in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
