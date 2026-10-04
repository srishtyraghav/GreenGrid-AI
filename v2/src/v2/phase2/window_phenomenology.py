# GreenGrid-AI V2 Phase 2 — Delhi phenomenology from Open-Meteo archive (no key).
# 4x4 point grid over the study bbox, daily 2022-01-01..2026-12-31.
# Outputs phenomenology.csv and delhi_tmax_precip_2022_2026.png under
# data/v2/phase2/window_investigation/.

import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd

ROOT = "C:/GreenGrid-AI"
OUT = os.path.join(ROOT, "data/v2/phase2/window_investigation")
CACHE = os.path.join(OUT, "met_cache")
os.makedirs(CACHE, exist_ok=True)

LAT_MIN, LAT_MAX = 28.40, 28.88
LON_MIN, LON_MAX = 76.83, 77.35
YEARS = [2022, 2023, 2024, 2025, 2026]
VARS = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum",
        "shortwave_radiation_sum"]
ONSET_THRESH = 2.5  # mm/day, 10-day rolling mean precip persisting after Jun 1


def fetch_point_year(lat, lon, year, retries=5):
    key = f"met_{lat:.2f}_{lon:.2f}_{year}.json"
    path = os.path.join(CACHE, key)
    if os.path.exists(path):
        return json.load(open(path))
    import datetime as _dt
    end = min(f"{year}-12-31", _dt.date.today().isoformat())  # archive rejects future dates
    url = ("https://archive-api.open-meteo.com/v1/archive"
           f"?latitude={lat:.4f}&longitude={lon:.4f}"
           f"&start_date={year}-01-01&end_date={end}"
           "&daily=" + ",".join(VARS) + "&timezone=Asia%2FKolkata")
    delay = 5.0
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = json.loads(r.read().decode("utf-8"))
            json.dump(data, open(path, "w"))
            return data
        except Exception as e:
            print(f"retry {attempt+1}/{retries} {key}: {e}", flush=True)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"failed to fetch {key}")


def monsoon_onset(daily_idx, precip, persist_days=15):
    """First day d >= Jun 1 such that the 10-day rolling mean precip is >= 2.5
    mm/day at d and stays >= 2.5 for the following `persist_days` days."""
    s = pd.Series(precip, index=daily_idx).rolling(10, min_periods=10).mean()
    after_jun1 = s[s.index >= pd.Timestamp(s.index[0].year, 6, 1)]
    vals = after_jun1.values
    ok = vals >= ONSET_THRESH
    for i in range(len(vals) - persist_days + 1):
        if ok[i:i + persist_days].all():
            return after_jun1.index[i]
    return pd.NaT


def hottest_30d(daily_idx, tmax):
    s = pd.Series(tmax, index=daily_idx).rolling(30, min_periods=30).mean()
    if s.empty:
        return pd.NaT, np.nan
    d = s.idxmax()
    return d, float(s.max())


def main():
    lats = np.linspace(LAT_MIN, LAT_MAX, 4)
    lons = np.linspace(LON_MIN, LON_MAX, 4)
    frames = []
    for lat in lats:
        for lon in lons:
            for year in YEARS:
                d = fetch_point_year(lat, lon, year)
                df = pd.DataFrame(d["daily"])
                df["lat"], df["lon"] = lat, lon
                frames.append(df)
                print(f"fetched {lat:.2f},{lon:.2f} {year}", flush=True)
    allp = pd.concat(frames, ignore_index=True)
    allp["date"] = pd.to_datetime(allp["time"])
    daily = allp.groupby("date")[VARS].mean()  # Delhi-mean daily series
    daily.to_csv(os.path.join(OUT, "delhi_daily_met_2022_2026.csv"))

    rows = []
    for year in YEARS:
        y = daily[daily.index.year == year]
        rec = {"year": year}
        for m in range(1, 13):
            mm = y[y.index.month == m]["temperature_2m_max"].mean()
            rec[f"tmax_mean_month_{m:02d}_C"] = round(float(mm), 2)
        onset = monsoon_onset(y.index, y["precipitation_sum"].values)
        hstart, hval = hottest_30d(y.index, y["temperature_2m_max"].values)
        rec["monsoon_onset_date"] = onset.strftime("%Y-%m-%d") if pd.notna(onset) else ""
        rec["hottest_30d_start"] = hstart.strftime("%Y-%m-%d") if pd.notna(hstart) else ""
        rec["hottest_30d_mean_tmax_C"] = round(hval, 2)
        rec["annual_precip_mm"] = round(float(y["precipitation_sum"].sum()), 1)
        rows.append(rec)
    pheno = pd.DataFrame(rows)
    pheno.to_csv(os.path.join(OUT, "phenomenology.csv"), index=False)
    print(pheno.to_string(), flush=True)

    # PNG: Tmax line per year + precip bars, candidate windows shaded
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(YEARS)))
    for c, year in zip(colors, YEARS):
        y = daily[daily.index.year == year]
        doy = y.index.dayofyear
        ax.plot(doy, y["temperature_2m_max"], color=c, lw=1.2, label=f"Tmax {year}")
        ax.bar(doy, y["precipitation_sum"] * 3, width=1.0, color=c, alpha=0.15)
    # shade candidate windows in 2026 doy space (leap year shifts Jul by 1; draw by month-day)
    spans = [("W0 Jul1-30", "2026-07-01", "2026-07-30", "red"),
             ("W1 Apr1-Jun30", "2026-04-01", "2026-06-30", "green"),
             ("W2 Mar1-Jun30", "2026-03-01", "2026-06-30", "blue"),
             ("W3 Apr1-Jul31", "2026-04-01", "2026-07-31", "orange"),
             ("W4 May1-Jun30", "2026-05-01", "2026-06-30", "purple")]
    for i, (name, s, e, col) in enumerate(spans):
        sd = pd.Timestamp(s).dayofyear
        ed = pd.Timestamp(e).dayofyear
        ax.axvspan(sd, ed, color=col, alpha=0.06)
        ax.text((sd + ed) / 2, ax.get_ylim()[1] * 0.98, name, ha="center",
                va="top", fontsize=8, color=col, rotation=90)
    ax.set_xlabel("day of year (2026 reference)")
    ax.set_ylabel("Tmax (°C) / precip×3 (mm)")
    ax.legend(ncol=5, fontsize=8, loc="lower left")
    ax.set_title("Delhi daily Tmax (lines) and precipitation (bars) 2022-2026 — candidate windows shaded")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "delhi_tmax_precip_2022_2026.png"), dpi=130)
    print("wrote phenomenology.csv and delhi_tmax_precip_2022_2026.png", flush=True)


if __name__ == "__main__":
    main()
