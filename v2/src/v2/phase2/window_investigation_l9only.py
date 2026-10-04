# GreenGrid-AI V2 Phase 2 — L9-only sensitivity re-run (user decision 2026-10-04:
# production dataset is Landsat 9 ONLY, exact sensor parity with V1).
# Re-reads ONLY the LC09 scenes already in scene_list.csv (same Mar 1-Jul 31
# union span, same common grid, same V1 masking recipe) and re-accumulates
# per-pixel metrics for all 5 windows x 5 years.
# Output: window_metrics_l9only.csv (landsat rows only, same schema as
# window_metrics.csv). S2 rows are unchanged and already in window_metrics.csv.

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS

from pystac_client import Client
import planetary_computer as pc

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import window_investigation as wi  # grid, masks, accumulators, metrics helpers

OUT = wi.OUT
WINDOWS = wi.WINDOWS
YEARS = wi.YEARS


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    sl = pd.read_csv(os.path.join(OUT, "scene_list.csv"))
    l9_ids = set(sl[(sl.collection == "landsat") & (sl.platform == "LC09")]["id"])
    log(f"L9 scene ids from scene_list: {len(l9_ids)}")

    cat = Client.open(wi.STAC_URL)
    rows = []
    failures = []
    processed = set()
    for year in YEARS:
        t0 = time.time()
        dt_range = f"{year}-03-01T00:00:00Z/{year}-07-31T23:59:59Z"
        items = list(cat.search(collections=["landsat-c2-l2"], bbox=wi.SEARCH_BBOX,
                                datetime=dt_range, limit=1000).items())
        items = [it for it in items if it.id in l9_ids]
        log(f"{year}: {len(items)} L9 items to re-read (query {time.time()-t0:.0f}s)")
        accums = {w: wi.LandsatAccum() for w in WINDOWS}
        n = 0
        sanity = 0
        for item in items:
            dt = wi.parse_dt(item)
            wins = wi.window_membership(dt)
            item_id = item.id
            try:
                href_qa = pc.sign(item.assets["qa_pixel"]).href
                href_st = pc.sign(item.assets["lwir11"]).href
                with rasterio.open(href_qa) as qa_src, rasterio.open(href_st) as st_src:
                    crs = qa_src.crs
                    qa, qa_tr = wi.windowed_read(qa_src, wi.SEARCH_BBOX, CRS.from_epsg(4326))
                    st, st_tr = wi.windowed_read(st_src, wi.SEARCH_BBOX, CRS.from_epsg(4326))
                valid_sr = ((qa & wi.QA_CLOUD_BITS) == 0).astype(np.uint8)
                raw = st.astype(np.float32)
                st_k = raw * 0.00341802 + 149.0
                valid_st = ((valid_sr == 1) & np.isfinite(raw) & (raw != 0)
                            & (st_k >= 250.0) & (st_k <= 350.0)).astype(np.uint8)
                st_c = st_k - 273.15
                g_sr = wi.reproj_to_grid(valid_sr, qa_tr, crs, np.uint8, wi.Resampling.nearest)
                g_st = wi.reproj_to_grid(valid_st, qa_tr, crs, np.uint8, wi.Resampling.nearest)
                st_masked = np.where(valid_st == 1, st_c, np.nan).astype(np.float32)
                g_stc = wi.reproj_to_grid(st_masked, st_tr, crs, np.float32, wi.Resampling.nearest)
                g_stc[~np.isfinite(g_stc)] = 0.0
                for w in wins:
                    accums[w].add(g_sr, g_st, g_stc)
                if year == 2024 and sanity < 3:
                    frac = float((g_sr[wi.STUDY_MASK] == 1).mean())
                    log(f"  SANITY(L9) {item_id}: study valid-SR fraction={frac:.3f} "
                        f"scene eo:cloud_cover={item.properties.get('eo:cloud_cover')}")
                    sanity += 1
                processed.add(item_id)
                n += 1
                if n % 10 == 0:
                    log(f"  l9 {year}: {n}/{len(items)}")
            except Exception as e:
                failures.append(dict(year=year, id=item_id, error=str(e)[:300]))
        log(f"{year}: processed {n} L9 scenes, failures={len(failures)}")

        for w in WINDOWS:
            i = WINDOWS.index(w)
            members = sl[(sl.id.isin([it.id for it in items]))
                         & (sl[f"w{i}"] == 1)]
            d = members.copy()
            d["doy"] = pd.to_datetime(d["datetime"]).dt.dayofyear
            rows.append(wi.landsat_metrics(year, w, accums[w], d))
        del accums

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "window_metrics_l9only.csv"), index=False)
    if failures:
        pd.DataFrame(failures).to_csv(os.path.join(OUT, "read_failures_l9only.csv"),
                                      index=False)
    missing = l9_ids - processed
    log(f"DONE: rows={len(df)} processed={len(processed)} expected={len(l9_ids)} "
        f"missing={len(missing)} failures={len(failures)}")


if __name__ == "__main__":
    main()
