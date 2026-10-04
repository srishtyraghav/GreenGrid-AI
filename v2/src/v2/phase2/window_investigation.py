# GreenGrid-AI V2 Phase 2 — temporal-window investigation.
# Reads REAL per-pixel evidence from Planetary Computer STAC
# (landsat-c2-l2 = LC08+LC09, sentinel-2-l2a) for 2022-2026, Mar 1 - Jul 31,
# and evaluates five candidate acquisition/compositing windows against the
# V1 baseline (Jul 1-30). Masking recipe cloned from gee/export_data_2023_2025.js.
#
# Writes ONLY under data/v2/phase2/window_investigation/ (V1 untouched).

import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.features import geometry_mask
from rasterio.warp import reproject, transform, Resampling
from rasterio.windows import from_bounds
from rasterio.transform import array_bounds

from pystac_client import Client
import planetary_computer as pc

ROOT = "C:/GreenGrid-AI"
OUT = os.path.join(ROOT, "data/v2/phase2/window_investigation")
os.makedirs(os.path.join(OUT, "maps/tif"), exist_ok=True)

GRID_TIF = os.path.join(ROOT, "data/raw/landsat9/2026_07/landsat9_2026_07_composite_30m.tif")
STUDY_GEOJSON = os.path.join(ROOT, "data/raw/gis/study_area/study_area.geojson")
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

YEARS = [int(y) for y in os.environ.get("WINDOW_YEARS", "2022,2023,2024,2025,2026").split(",")]
WINDOWS = ["W0", "W1", "W2", "W3", "W4"]


def window_membership(dt):
    """W0 Jul1-30, W1 Apr1-Jun30, W2 Mar1-Jun30, W3 Apr1-Jul31, W4 May1-Jun30.
    Month/day based, leap-year safe."""
    m, day = dt.month, dt.day
    w = set()
    if m == 7 and day <= 30:
        w.add("W0")
    if 4 <= m <= 6:
        w.add("W1")
    if 3 <= m <= 6:
        w.add("W2")
    if 4 <= m <= 7:
        w.add("W3")
    if m in (5, 6):
        w.add("W4")
    return w


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- grid setup
with rasterio.open(GRID_TIF) as d:
    GRID_TRANSFORM = d.transform
    GRID_CRS = d.crs
    GRID_SHAPE = d.shape  # (height, width)

geoj = json.load(open(STUDY_GEOJSON))
GEOMS = [f["geometry"] for f in geoj["features"]]
STUDY_MASK = geometry_mask(GEOMS, out_shape=GRID_SHAPE, transform=GRID_TRANSFORM,
                           invert=True)  # True inside Delhi NCT
STUDY_CELLS = int(STUDY_MASK.sum())
log(f"common grid: {GRID_CRS.to_string()} shape={GRID_SHAPE} study_cells={STUDY_CELLS}")

SEARCH_BBOX = (76.8388351, 28.4046285, 77.3453379, 28.8834464)  # study_area.geojson bbox

read_failures = []
scene_rows = []


def fail_row(year, coll, item_id, stage, err):
    read_failures.append(dict(year=year, collection=coll, id=item_id, stage=stage,
                              error=str(err)[:300]))


def windowed_read(src, bounds, crs):
    """Read src band 1 over the window covering `bounds` (in `crs`), plus a small
    buffer of pixels. Returns (array, window_transform)."""
    xs, ys = transform(crs, src.crs, [bounds[0], bounds[2]], [bounds[1], bounds[3]])
    win = from_bounds(min(xs), min(ys), max(xs), max(ys), transform=src.transform)
    pad = 2
    win = win.round_offsets().round_lengths()
    col_start = max(0, int(np.floor(win.col_off)) - pad)
    row_start = max(0, int(np.floor(win.row_off)) - pad)
    col_stop = min(src.width, int(np.ceil(win.col_off + win.width)) + pad)
    row_stop = min(src.height, int(np.ceil(win.row_off + win.height)) + pad)
    win = rasterio.windows.Window(col_start, row_start, col_stop - col_start,
                                  row_stop - row_start)
    return src.read(1, window=win), src.window_transform(win)


def reproj_to_grid(src_arr, src_transform, src_crs, dst_dtype, resampling):
    dst = np.zeros(GRID_SHAPE, dtype=dst_dtype)
    reproject(
        source=src_arr,
        src_crs=src_crs, src_transform=src_transform,
        destination=dst, dst_crs=GRID_CRS, dst_transform=GRID_TRANSFORM,
        resampling=resampling,
    )
    return dst


class LandsatAccum:
    def __init__(self):
        self.sr_count = np.zeros(GRID_SHAPE, np.uint16)
        self.st_count = np.zeros(GRID_SHAPE, np.uint16)
        self.st_sum = np.zeros(GRID_SHAPE, np.float64)
        self.st_sumsq = np.zeros(GRID_SHAPE, np.float64)
        self.st_min = np.full(GRID_SHAPE, np.inf)
        self.st_max = np.full(GRID_SHAPE, -np.inf)

    def add(self, valid_sr, valid_st, st_c):
        self.sr_count += valid_sr
        if (valid_st == 1).any():
            m = valid_st == 1
            self.st_count[m] += 1
            self.st_sum[m] += st_c[m]
            self.st_sumsq[m] += st_c[m] ** 2
            np.minimum(self.st_min, np.where(m, st_c, np.inf), out=self.st_min)
            np.maximum(self.st_max, np.where(m, st_c, -np.inf), out=self.st_max)


class S2Accum:
    def __init__(self):
        self.count = np.zeros(GRID_SHAPE, np.uint16)
        self.frac_sum = np.zeros(GRID_SHAPE, np.float32)


# V1 GEE recipe constants (gee/export_data_2023_2025.js):
#   Landsat pixel mask: qa.bitwiseAnd(1<<4).eq(0).and(qa.bitwiseAnd(1<<3).eq(0))
#     -> ONLY bits 3 (cloud) and 4 (cloud shadow); dilated cloud (1), cirrus (2),
#        snow (5) and fill (0) are NOT masked.
#   Landsat scene filter: none on CLOUD_COVER; collection T1 only (LC09 in V1).
#   S2 pixel mask: SCL neq 3,8,9,10 (shadow, med cloud, high cloud, cirrus);
#     classes 0 (no data) and 1 (saturated/defective) are NOT masked.
#   S2 scene filter: CLOUDY_PIXEL_PERCENTAGE < 60.
#   ST_B10 scale 0.00341802 / offset 149.0 is NOT applied in the V1 script
#     (raw DN exported); we apply it here to get physical Kelvin for ST metrics.
QA_CLOUD_BITS = (1 << 3) | (1 << 4)
SCL_INVALID = (3, 8, 9, 10)
S2_MAX_CLOUD = 60.0


def parse_dt(item):
    return datetime.fromisoformat(item.properties["datetime"].replace("Z", "+00:00"))


def process_landsat(year, items):
    accums = {w: LandsatAccum() for w in WINDOWS}
    n = 0
    sanity_logged = 0
    for item in items:
        if not (item.id.startswith("LC08") or item.id.startswith("LC09")):
            continue  # V1/Phase-2 scope: Landsat 8+9 only (LE07 lacks ST_B10 asset)
        if item.properties.get("landsat:collection_category") != "T1":
            continue  # V1 used LANDSAT/LC09/C02/T1_L2 (T1 only)
        dt = parse_dt(item)
        wins = window_membership(dt)
        if not wins:
            continue
        item_id = item.id
        try:
            href_qa = pc.sign(item.assets["qa_pixel"]).href
            href_st = pc.sign(item.assets["lwir11"]).href
            with rasterio.open(href_qa) as qa_src, rasterio.open(href_st) as st_src:
                crs = qa_src.crs
                qa, qa_tr = windowed_read(qa_src, SEARCH_BBOX, CRS.from_epsg(4326))
                st, st_tr = windowed_read(st_src, SEARCH_BBOX, CRS.from_epsg(4326))
            valid_sr = ((qa & QA_CLOUD_BITS) == 0).astype(np.uint8)
            raw = st.astype(np.float32)
            st_k = raw * 0.00341802 + 149.0
            valid_st = ((valid_sr == 1) & np.isfinite(raw) & (raw != 0)
                        & (st_k >= 250.0) & (st_k <= 350.0)).astype(np.uint8)
            st_c = st_k - 273.15
            g_sr = reproj_to_grid(valid_sr, qa_tr, crs, np.uint8, Resampling.nearest)
            g_st = reproj_to_grid(valid_st, qa_tr, crs, np.uint8, Resampling.nearest)
            st_masked = np.where(valid_st == 1, st_c, np.nan).astype(np.float32)
            g_stc = reproj_to_grid(st_masked, st_tr, crs, np.float32, Resampling.nearest)
            g_stc[~np.isfinite(g_stc)] = 0.0
            for w in wins:
                accums[w].add(g_sr, g_st, g_stc)
            # sanity check: per-pixel valid fraction vs scene-level cloud cover
            if year == 2024 and sanity_logged < 3:
                frac = float((g_sr[STUDY_MASK] == 1).mean())
                log(f"  SANITY {item_id}: study valid-SR fraction={frac:.3f} "
                    f"scene eo:cloud_cover={item.properties.get('eo:cloud_cover')}")
                sanity_logged += 1
            props = item.properties
            scene_rows.append(dict(
                year=year, collection="landsat", id=item_id,
                platform="LC09" if item_id.startswith("LC09") else "LC08",
                datetime=dt.isoformat(), doy=dt.timetuple().tm_yday,
                cloud=props.get("eo:cloud_cover"),
                wrs_path=props.get("landsat:wrs_path"),
                wrs_row=props.get("landsat:wrs_row"),
                mgrs_tile="", s2_native_valid_frac=np.nan,
                **{f"w{i}": int(f"W{i}" in wins) for i in range(5)}))
            n += 1
            if n % 20 == 0:
                log(f"  landsat {year}: {n} scenes processed")
        except Exception as e:
            fail_row(year, "landsat", item_id, "read", e)
    return accums, n


def process_s2(year, items):
    accums = {w: S2Accum() for w in WINDOWS}
    best = {}
    for it in items:
        dt = parse_dt(it)
        if not window_membership(dt):
            continue
        cloud = it.properties.get("eo:cloud_cover")
        if cloud is None or cloud >= S2_MAX_CLOUD:  # V1: CLOUDY_PIXEL_PERCENTAGE < 60
            continue
        key = (it.properties.get("s2:mgrs_tile") or it.id.split("_")[5][:6],
               dt.strftime("%Y-%m-%d"))
        if key not in best or cloud < best[key][0]:
            best[key] = (cloud, it)
    n = 0
    for key, (cloud, item) in sorted(best.items()):
        dt = parse_dt(item)
        wins = window_membership(dt)
        item_id = item.id
        try:
            href = pc.sign(item.assets["SCL"]).href
            with rasterio.open(href) as scl_src:
                crs = scl_src.crs
                scl, scl_tr = windowed_read(scl_src, SEARCH_BBOX, CRS.from_epsg(4326))
            valid = np.isin(scl, SCL_INVALID, invert=True).astype(np.uint8)
            native_valid_frac = float(valid.mean())
            g_ct = reproj_to_grid(valid, scl_tr, crs, np.uint8, Resampling.nearest)
            g_fr = reproj_to_grid(valid.astype(np.float32), scl_tr, crs, np.float32,
                                  Resampling.average)
            for w in wins:
                accums[w].count += g_ct
                accums[w].frac_sum += g_fr
            props = item.properties
            scene_rows.append(dict(
                year=year, collection="s2", id=item_id,
                platform=props.get("platform", ""),
                datetime=dt.isoformat(), doy=dt.timetuple().tm_yday,
                cloud=props.get("eo:cloud_cover"),
                wrs_path="", wrs_row="",
                mgrs_tile=props.get("s2:mgrs_tile", ""),
                s2_native_valid_frac=round(native_valid_frac, 4),
                **{f"w{i}": int(f"W{i}" in wins) for i in range(5)}))
            n += 1
            if n % 25 == 0:
                log(f"  s2 {year}: {n} scenes processed")
        except Exception as e:
            fail_row(year, "s2", item_id, "read", e)
    return accums, n


def pct(cells):
    return round(100.0 * cells / STUDY_CELLS, 2)


def landsat_metrics(year, w, acc, df):
    sm = STUDY_MASK
    n_scenes = len(df)
    platforms = df["platform"].value_counts() if n_scenes else pd.Series(dtype=int)
    doys = df["doy"].values if n_scenes else np.array([])
    cloud = df["cloud"].dropna() if n_scenes else pd.Series(dtype=float)
    mean_st = np.full(GRID_SHAPE, np.nan)
    np.divide(acc.st_sum, acc.st_count, out=mean_st, where=acc.st_count > 0)
    cell_mean = mean_st[sm]
    ok = np.isfinite(cell_mean)
    cov1 = pct(int((acc.st_count[sm] >= 1).sum()))
    cov2 = pct(int((acc.st_count[sm] >= 2).sum()))
    if ok.any():
        vals = cell_mean[ok]
        cnt = acc.st_count[sm][ok]
        var = acc.st_sumsq[sm][ok] / cnt - vals ** 2
        var = np.clip(var, 0, None)
        inter_std = float(np.sqrt(var).mean())
        st_mean, st_std = float(np.mean(vals)), float(np.std(vals))
        p5, p95 = (float(x) for x in np.percentile(vals, [5, 95]))
    else:
        inter_std = st_mean = st_std = p5 = p95 = np.nan
    return dict(
        window=w, year=year, collection="landsat",
        n_scenes=n_scenes,
        n_scenes_L8=int(platforms.get("LC08", 0)),
        n_scenes_L9=int(platforms.get("LC09", 0)),
        n_tiles_S2=np.nan,
        first_date=df["datetime"].min()[:10] if n_scenes else "",
        last_date=df["datetime"].max()[:10] if n_scenes else "",
        doy_std=float(np.std(doys)) if len(doys) else np.nan,
        scene_cloud_min=float(cloud.min()) if len(cloud) else np.nan,
        scene_cloud_median=float(cloud.median()) if len(cloud) else np.nan,
        scene_cloud_max=float(cloud.max()) if len(cloud) else np.nan,
        valid_coverage_pct=pct(int((acc.sr_count[sm] >= 1).sum())),
        valid_coverage_2plus_pct=pct(int((acc.sr_count[sm] >= 2).sum())),
        st_coverage_pct=cov1,
        st_coverage_2plus_pct=cov2,
        st_mean_C=st_mean, st_std_C=st_std,
        st_p5_C=p5, st_p95_C=p95,
        st_interscene_std_C=inter_std,
        obs_count_median_per_cell=float(np.median(acc.sr_count[sm])),
        s2_native10m_coverage_pct=np.nan,
    )


def s2_metrics(year, w, acc, df):
    sm = STUDY_MASK
    n_scenes = len(df)
    doys = df["doy"].values if n_scenes else np.array([])
    cloud = df["cloud"].dropna() if n_scenes else pd.Series(dtype=float)
    return dict(
        window=w, year=year, collection="s2",
        n_scenes=n_scenes, n_scenes_L8=np.nan, n_scenes_L9=np.nan,
        n_tiles_S2=int(df["mgrs_tile"].nunique()) if n_scenes else 0,
        first_date=df["datetime"].min()[:10] if n_scenes else "",
        last_date=df["datetime"].max()[:10] if n_scenes else "",
        doy_std=float(np.std(doys)) if len(doys) else np.nan,
        scene_cloud_min=float(cloud.min()) if len(cloud) else np.nan,
        scene_cloud_median=float(cloud.median()) if len(cloud) else np.nan,
        scene_cloud_max=float(cloud.max()) if len(cloud) else np.nan,
        valid_coverage_pct=pct(int((acc.count[sm] >= 1).sum())),
        valid_coverage_2plus_pct=pct(int((acc.count[sm] >= 2).sum())),
        st_coverage_pct=np.nan, st_coverage_2plus_pct=np.nan,
        st_mean_C=np.nan, st_std_C=np.nan, st_p5_C=np.nan, st_p95_C=np.nan,
        st_interscene_std_C=np.nan,
        obs_count_median_per_cell=float(np.median(acc.count[sm])),
        s2_native10m_coverage_pct=pct(int((acc.frac_sum[sm] >= 1.0).sum())),
    )


_MPL = None


def write_maps(w, year, acc):
    global _MPL
    if _MPL is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _MPL = plt
    plt = _MPL
    disp = np.where(STUDY_MASK, acc.st_count, 0).astype(np.uint8)
    tif_path = os.path.join(OUT, "maps/tif", f"st_valid_{w}_{year}.tif")
    with rasterio.open(tif_path, "w", driver="GTiff", height=GRID_SHAPE[0],
                       width=GRID_SHAPE[1], count=1, dtype="uint8",
                       crs=GRID_CRS, transform=GRID_TRANSFORM, nodata=255) as d:
        d.write(np.where(STUDY_MASK, np.clip(disp, 0, 10), 255).astype(np.uint8), 1)
    fig, ax = plt.subplots(figsize=(5, 4.4))
    show = np.where(STUDY_MASK, acc.st_count, np.nan).astype(np.float32)
    vmax = float(np.nanmax(show)) if np.isfinite(show).any() else 5
    im = ax.imshow(show, cmap="viridis", vmin=0, vmax=max(5, vmax))
    ax.set_title(f"{w} {year} — valid-ST count (Landsat)")
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.8, label="valid ST obs per cell (cap 10)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "maps", f"st_valid_{w}_{year}.png"), dpi=110)
    plt.close(fig)


def main():
    cat = Client.open(STAC_URL)
    all_metrics = []
    for year in YEARS:
        t0 = time.time()
        log(f"=== YEAR {year}: STAC query Mar 1 - Jul 31 ===")
        dt_range = f"{year}-03-01T00:00:00Z/{year}-07-31T23:59:59Z"
        l_items = list(cat.search(collections=["landsat-c2-l2"], bbox=SEARCH_BBOX,
                                  datetime=dt_range, limit=1000).items())
        s_items = list(cat.search(collections=["sentinel-2-l2a"], bbox=SEARCH_BBOX,
                                  datetime=dt_range, limit=1000).items())
        log(f"{year}: {len(l_items)} landsat items, {len(s_items)} s2 items "
            f"(query {time.time()-t0:.0f}s)")

        l_acc, ln = process_landsat(year, l_items)
        s_acc, sn = process_s2(year, s_items)
        log(f"{year}: processed landsat={ln} s2={sn} failures_so_far={len(read_failures)}")

        sr = pd.DataFrame([r for r in scene_rows if r["year"] == year])
        for w in WINDOWS:
            i = WINDOWS.index(w)
            ldf = sr[(sr["collection"] == "landsat") & (sr[f"w{i}"] == 1)]
            sdf = sr[(sr["collection"] == "s2") & (sr[f"w{i}"] == 1)]
            all_metrics.append(landsat_metrics(year, w, l_acc[w], ldf))
            all_metrics.append(s2_metrics(year, w, s_acc[w], sdf))
            write_maps(w, year, l_acc[w])
        del l_acc, s_acc

    df = pd.DataFrame(all_metrics)
    df.to_csv(os.path.join(OUT, "window_metrics.csv"), index=False)
    pd.DataFrame(scene_rows).to_csv(os.path.join(OUT, "scene_list.csv"), index=False)
    fdf = pd.DataFrame(read_failures, columns=["year", "collection", "id", "stage", "error"])
    fdf.to_csv(os.path.join(OUT, "read_failures.csv"), index=False)
    log(f"DONE: metrics={len(df)} scenes={len(scene_rows)} failures={len(fdf)}")


if __name__ == "__main__":
    main()
