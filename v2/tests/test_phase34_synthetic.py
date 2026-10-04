"""Synthetic unit tests for V2 Phase 3 (preprocessing) and Phase 4 (features).

All fixtures are generated here — tiny 60x80 grids with known transforms,
constant/checkerboard/NaN patterns, a half-pixel-shifted S2-20m grid, a B2
zero-fill artifact stripe, a fabricated hourly met slice and fake constraint
GeoJSON/Overpass-JSON squares. Nothing reads real V1/V2 rasters.

Run from the project root:
    PYTHONUTF8=1 PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/v2/test_phase34_synthetic.py -q
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from affine import Affine
from rasterio.enums import Resampling

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # the v2/ tree
SRC = PROJECT_ROOT / "src"
for p in (str(PROJECT_ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from v2.common import (  # noqa: E402
    GATES,
    GRID_FILE_DEFAULT,
    HOLDOUT_BLOCKS,
    PINNED_BLOCKS_SOURCE,
    PINNED_BLOCKS_SOURCE_SHA256,
    PINNED_COL_BINS,
    PINNED_ROW_BINS,
    Grid,
    block_bands_for_indices,
    compute_block_raster,
    load_frozen_schema,
    lst_from_dn,
    norm_diff,
    reproject_to_grid,
    schema_hash,
    sha256_file,
    valid_l9_mask,
    valid_lulc_mask,
    valid_s2_10m_mask,
    valid_s2_20m_mask,
    vegetation_cover_from_ndvi,
)

V1_ROOT = PROJECT_ROOT.parent / "v1"
BLOCKS_MANIFEST = PROJECT_ROOT / "data" / "phase2" / "spatial_blocks_manifest.json"
W4_INPUT_AUDIT = PROJECT_ROOT / "data" / "phase2" / "w4_input_audit.json"
from v2.phase3 import build_core, verify_inputs  # noqa: E402
from v2.phase4 import assemble_features, verify_features  # noqa: E402
from v2.phase4._spatial import (  # noqa: E402
    aggregate_met_window,
    focal_mean_block_aware,
    idw_weights,
    met_fields_for_year,
    nan_window_stats,
)

RES = 0.00026949458523585647  # V1 GEE 30 m export resolution (degrees)
ORIGIN_X, ORIGIN_Y = 76.83, 28.92
H, W = 60, 80
GRID_TRANSFORM = Affine(RES, 0.0, ORIGIN_X, 0.0, -RES, ORIGIN_Y)
YEARS_FIX = (2022, 2023)
FIXTURES = PROJECT_ROOT / "data" / "_smoketest" / "p34" / "fixtures"
TMP = PROJECT_ROOT / "data" / "_smoketest" / "p34" / "tmp"

LST_DN = 45000.0
LST_EXPECTED = LST_DN * 0.00341802 + 149.0 - 273.15  # 29.6609 °C


# ---------------------------------------------------------------------------
# Fixture generation
# ---------------------------------------------------------------------------
def _write_raster(path: Path, arr: np.ndarray, transform: Affine, crs: str = "EPSG:4326",
                  dtype: str = "float64") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 1 if arr.ndim == 2 else arr.shape[0]
    profile = dict(driver="GTiff", height=arr.shape[-2], width=arr.shape[-1],
                   count=count, dtype=dtype, crs=crs, transform=transform,
                   compress="lzw")
    with rasterio.open(path, "w", **profile) as ds:
        if arr.ndim == 2:
            ds.write(arr, 1)
        else:
            ds.write(arr)


def _square(x0, y0, x1, y1) -> dict:
    return {
        "type": "Polygon",
        "coordinates": [[(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]],
    }


def _geojson(path: Path, geoms: list[dict], props: list[dict] | None = None) -> None:
    if props is None:
        props = [{} for _ in geoms]
    fc = {"type": "FeatureCollection",
          "features": [{"type": "Feature", "properties": pr, "geometry": g}
                       for pr, g in zip(props, geoms)]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fc))


def _overpass_json(path: Path, nodes: dict[int, tuple[float, float]],
                   ways: list[tuple[list[int], dict]]) -> None:
    elements = [{"type": "node", "id": i, "lon": c[0], "lat": c[1]}
                for i, c in nodes.items()]
    elements += [{"type": "way", "id": 100 + k, "nodes": n, "tags": t}
                 for k, (n, t) in enumerate(ways)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 0.6, "elements": elements}))


def make_study_area(path: Path) -> None:
    _geojson(path, [_square(ORIGIN_X + 2 * RES, ORIGIN_Y - (H - 2) * RES,
                            ORIGIN_X + (W - 2) * RES, ORIGIN_Y - 2 * RES)])


def make_raw_tree(root: Path) -> None:
    """Synthetic L9 / S2 / LULC W4 exports for YEARS_FIX."""
    grid = Grid(GRID_TRANSFORM, H, W, "EPSG:4326")
    for year in YEARS_FIX:
        yoff = 0.001 * (year - 2022)
        # L9: 9 bands on the authoritative grid
        l9 = np.zeros((9, H, W), dtype=np.float64)
        vals = {"SR_B2": 0.12 + yoff, "SR_B3": 0.14, "SR_B4": 0.10, "SR_B5": 0.22,
                "SR_B6": 0.20, "SR_B7": 0.18}
        for i, v in enumerate(vals.values()):
            l9[i] = v
        l9[6] = LST_DN
        l9[7] = 21824.0
        l9[8] = 4.0
        l9[8, 30:33, 40:43] = 0.0                     # QA_CLEAR_COUNT == 0 patch
        l9[6, 20:30, 75:77] = np.nan                  # ST NaN stripe
        l9[0, 50:52, 60:62] = np.nan                  # SR_B2 NaN patch
        _write_raster(root / "landsat9" / f"{year}_05_06" / f"landsat9_{year}_05_06_composite_30m.tif", l9, GRID_TRANSFORM)

        # S2 10 m: aligned 3x lattice, 180x240, B2-zero artifact stripe
        res10 = RES / 3.0
        t10 = Affine(res10, 0.0, ORIGIN_X, 0.0, -res10, ORIGIN_Y)
        h10, w10 = 3 * H, 3 * W
        s2 = np.zeros((5, h10, w10), dtype=np.float64)
        s2[0] = 0.12   # B2
        s2[1] = 0.13   # B3
        s2[2] = 0.10   # B4
        s2[3] = 0.30   # B8 -> NDVI = 0.5
        s2[4] = 3.0    # VALID_COUNT
        # B2 zero-fill artifact stripe: 5 native rows (30 m cell 6 fully
        # covered, cell 7 mixed 2 artifact + 1 valid) with DISTINCT band
        # values so aggregation leaks are detectable (stripe NDVI ~0.8947).
        s2[0, 18:23, :] = 0.0
        s2[2, 18:23, :] = 0.05
        s2[3, 18:23, :] = 0.90
        s2[4, 50:52, 100:102] = 0.0                    # VALID_COUNT 0 patch
        s2[3, 159:162, 198:201] = np.nan               # B8 NaN 3x3 block -> whole 30 m cell NaN
        _write_raster(root / "sentinel2" / f"{year}_05_06" / f"sentinel2_{year}_05_06_10m_composite.tif", s2, t10)

        # S2 20 m: HALF-PIXEL y-offset (V1 F4 defect emulation), constant fields
        res20 = RES * 2.0 / 3.0
        t20 = Affine(res20, 0.0, ORIGIN_X, 0.0, -res20, ORIGIN_Y - res20 / 2.0)
        h20, w20 = 90, 120
        s220 = np.zeros((8, h20, w20), dtype=np.float32)
        for i, v in enumerate((0.15, 0.16, 0.17, 0.28, 0.25, 0.24)):
            s220[i] = v
        s220[6] = 4.0   # SCL
        s220[7] = 3.0   # VALID_COUNT
        _write_raster(root / "sentinel2" / f"{year}_05_06" / f"sentinel2_{year}_05_06_20m_composite.tif", s220, t20)

        # LULC 10 m: (r+c)%9 labels + NaN band
        lab = (np.add.outer(np.arange(h10), np.arange(w10)) % 9).astype(np.float64)
        lab[100:110, :] = np.nan
        _write_raster(root / "lulc" / f"{year}_05_06" / f"lulc_{year}_05_06_10m.tif", lab, t10)


def make_vectors(vdir: Path) -> dict:
    vdir.mkdir(parents=True, exist_ok=True)
    study = vdir / "study_area.geojson"
    make_study_area(study)

    # Grid interior: x 76.8305..76.8512 (cols 2..77), y 28.9039..28.9186
    # (rows 2..57). All vector squares must lie inside the grid.
    #
    # Context layers are V2-style GeoJSON with OSM tag properties. Cases:
    #  - roads: primary segment (kept by the V1-parity filter) + residential
    #    segment (dropped -> must NOT burn)
    #  - vegetation: natural=wood polygon (kept) + leisure=park polygon (kept)
    #  - landuse: four landuse-tagged squares (kept, coded) + one natural-only
    #    square (dropped by the V1-parity filter)
    roads = vdir / "roads.geojson"
    _geojson(
        roads,
        [{"type": "LineString", "coordinates": [(76.8310, 28.9100), (76.8355, 28.9100)]},
         {"type": "LineString", "coordinates": [(76.8360, 28.9100), (76.8385, 28.9100)]}],
        [{"highway": "primary"}, {"highway": "residential"}],
    )
    veg = vdir / "vegetation.geojson"
    _geojson(
        veg,
        [_square(76.836, 28.906, 76.838, 28.908),
         _square(76.8385, 28.906, 76.8405, 28.908)],
        [{"natural": "wood"}, {"leisure": "park"}],
    )
    landuse = vdir / "landuse.geojson"
    _geojson(
        landuse,
        [_square(76.831, 28.914, 76.833, 28.916),      # A residential -> 6
         _square(76.836, 28.914, 76.838, 28.916),      # B park -> 1
         _square(76.842, 28.914, 76.844, 28.916),      # C forest -> 2
         _square(76.831, 28.905, 76.833, 28.907),      # D farmland -> 8
         _square(76.836, 28.905, 76.838, 28.907)],     # E natural-only -> DROPPED
        [{"landuse": "residential"}, {"landuse": "park"}, {"landuse": "forest"},
         {"landuse": "farmland"}, {"natural": "wood"}],
    )
    # Legacy Overpass-JSON fixture: the tolerant parser is retained for
    # provenance/audit but is no longer used for features.
    legacy = vdir / "legacy_overpass.json"
    _overpass_json(
        legacy,
        {1: (76.84, 28.911), 2: (76.841, 28.911), 3: (76.841, 28.912), 4: (76.84, 28.912)},
        [([1, 2, 3, 4, 1], {"landuse": "residential"})],
    )
    buildings = vdir / "buildings.geojson"
    _geojson(buildings, [_square(76.840, 28.911, 76.841, 28.912),
                         _square(76.845, 28.916, 76.846, 28.917)])
    water = vdir / "water.geojson"
    _geojson(water, [_square(76.846, 28.904, 76.848, 28.906)])
    road_surfaces = vdir / "road_surfaces.geojson"
    _geojson(road_surfaces, [_square(76.8335, 28.9095, 76.8345, 28.9105)])
    return {"study_area": study, "roads": roads, "vegetation": veg,
            "landuse": landuse, "legacy_overpass": legacy,
            "buildings": buildings, "water": water,
            "road_surfaces": road_surfaces}


def make_met_csv(path: Path) -> pd.DataFrame:
    """16-point grid; in-window value = point index k, out-of-window = 999."""
    lats = [28.40, 28.56, 28.72, 28.88]
    lons = [76.83, 77.00, 77.17, 77.35]
    rows = []
    k = 0
    for lat in lats:
        for lon in lons:
            k += 1
            pid = f"p{k:02d}"
            for year in YEARS_FIX:
                for month, day in ((5, 1), (5, 15), (6, 15), (6, 30)):
                    for hour in (4, 5, 6):
                        rows.append((pid, lat, lon, f"{year}-{month:02d}-{day:02d}T{hour:02d}:00",
                                     float(k), 999.0, 999.0, 999.0, 999.0, 999.0))
                # decoys: wrong hour, wrong month, out-of-window edges
                rows.append((pid, lat, lon, f"{year}-05-02T10:00", 999.0, 999.0, 999.0, 999.0, 999.0, 999.0))
                rows.append((pid, lat, lon, f"{year}-07-01T05:00", 999.0, 999.0, 999.0, 999.0, 999.0, 999.0))
                rows.append((pid, lat, lon, f"{year}-04-30T06:00", 999.0, 999.0, 999.0, 999.0, 999.0, 999.0))
    df = pd.DataFrame(rows, columns=["point_id", "lat", "lon", "time_utc",
                                     "temperature_2m", "relative_humidity_2m",
                                     "wind_speed_10m", "precipitation",
                                     "shortwave_radiation", "soil_moisture_0_to_7cm"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


@pytest.fixture(scope="session")
def synth():
    """Build the full synthetic fixture tree once per test session."""
    if FIXTURES.exists():
        shutil.rmtree(FIXTURES)
    FIXTURES.mkdir(parents=True)
    raw = FIXTURES / "raw_v2"
    make_raw_tree(raw)
    vectors = make_vectors(FIXTURES / "vectors")
    met_csv = FIXTURES / "met_superset.csv"
    met_df = make_met_csv(met_csv)
    TMP.mkdir(parents=True, exist_ok=True)
    return {"raw": raw, "vectors": vectors, "met_csv": met_csv, "met_df": met_df,
            "grid_file": raw / "landsat9" / "2023_05_06" / "landsat9_2023_05_06_composite_30m.tif"}


def _build_core_args(raw: Path, out: Path, vectors: dict, grid_file: Path,
                     years="2022,2023", roads_full: bool = False):
    args = ["--data-root", str(raw), "--out-root", str(out),
            "--years", years, "--grid-file", str(grid_file),
            "--study-area", str(vectors["study_area"]),
            "--roads-geojson", str(vectors["roads"]),
            "--vegetation-geojson", str(vectors["vegetation"]),
            "--landuse-geojson", str(vectors["landuse"]),
            "--buildings-geojson", str(vectors["buildings"]),
            "--water-geojson", str(vectors["water"]),
            "--road-surfaces-geojson", str(vectors["road_surfaces"])]
    if roads_full:
        args.append("--roads-full")
    return args


@pytest.fixture(scope="session")
def phase3_tree(synth):
    out = TMP / "phase3_out"
    if out.exists():
        shutil.rmtree(out)
    rc = build_core.main(_build_core_args(synth["raw"], out, synth["vectors"], synth["grid_file"]))
    assert rc == 0
    return out


@pytest.fixture(scope="session")
def phase4_tree(synth, phase3_tree):
    out = TMP / "phase4_out"
    if out.exists():
        shutil.rmtree(out)
    rc = assemble_features.main([
        "--phase3-root", str(phase3_tree), "--out-root", str(out),
        "--years", "2022,2023", "--grid-file", str(synth["grid_file"]),
        "--met-csv", str(synth["met_csv"]),
        "--study-area", str(synth["vectors"]["study_area"]),
    ])
    assert rc == 0
    return out


# ---------------------------------------------------------------------------
# Formula unit tests
# ---------------------------------------------------------------------------
def test_lst_scaling_exact():
    dn = np.array([0.0, 10000.0, 45000.0, np.nan])
    out = lst_from_dn(dn)
    assert out[0] == pytest.approx(0 * 0.00341802 + 149.0 - 273.15, abs=1e-9)
    assert out[2] == pytest.approx(LST_EXPECTED, abs=1e-9)
    assert np.isnan(out[3])


def test_index_formulas():
    a = np.array([[0.30, 0.10], [np.nan, 0.0]])
    b = np.array([[0.10, 0.10], [0.20, 0.0]])
    nd = norm_diff(a, b)
    assert nd[0, 0] == pytest.approx(0.5)       # NDVI (B8,B4)
    assert nd[0, 1] == pytest.approx(0.0)
    assert np.isnan(nd[1, 0])                   # NaN preserved
    assert np.isnan(nd[1, 1])                   # 0/0 -> NaN


def test_ndbi_ndre_band_usage():
    """V1 convention: S2 NDBI uses B11/B8A; NDRE uses B8A/B5 (20 m bands)."""
    b11, b8a, b5 = 0.25, 0.28, 0.15
    ndbi = norm_diff(b11, b8a)
    ndre = norm_diff(b8a, b5)
    assert ndbi == pytest.approx((0.25 - 0.28) / 0.53)
    assert ndre == pytest.approx((0.28 - 0.15) / 0.43)


def test_pvc_formula():
    ndvi = np.array([-0.2, 0.05, 0.425, 0.8, 1.2, np.nan])
    pvc = vegetation_cover_from_ndvi(ndvi)
    assert pvc[0] == 0.0 and pvc[1] == 0.0
    assert pvc[2] == pytest.approx(0.5)
    assert pvc[3] == 1.0 and pvc[4] == 1.0      # clamped
    assert np.isnan(pvc[5])


def test_valid_mask_rules():
    good = np.ones((3, 3))
    nan_arr = np.full((3, 3), np.nan)
    zero = np.zeros((3, 3))
    cc0 = np.zeros((3, 3))

    assert valid_l9_mask(good, good, good).all()
    assert not valid_l9_mask(nan_arr, good, good).any()
    assert not valid_l9_mask(good, good, cc0).any()

    assert valid_s2_10m_mask(good, good, good, good, good).all()
    assert not valid_s2_10m_mask(zero, good, good, good, good).any()   # B2>0 guard
    assert not valid_s2_10m_mask(good, good, good, good, cc0).any()    # count>=1
    assert not valid_s2_10m_mask(good, nan_arr, good, good, good).any()

    assert valid_s2_20m_mask(good, good, good, good, good, good).all()
    assert not valid_s2_20m_mask(good, nan_arr, good, good, good, good).any()
    # 0 is a REAL value for the 20 m bands (the B2>0 guard applies only to
    # the known S2-10m B2 zero-fill artifact)
    assert valid_s2_20m_mask(zero, good, good, good, good, good).all()

    assert valid_lulc_mask(zero).all()          # 0 (water) is a REAL class
    assert not valid_lulc_mask(nan_arr).any()


# ---------------------------------------------------------------------------
# Aggregation / alignment tests
# ---------------------------------------------------------------------------
def test_s2_10m_mean_of_3x3_when_aligned():
    """Aligned 10 m -> 30 m average aggregation equals the exact 3x3 mean."""
    grid = Grid(GRID_TRANSFORM, 4, 6, "EPSG:4326")
    res10 = RES / 3.0
    t10 = Affine(res10, 0.0, ORIGIN_X, 0.0, -res10, ORIGIN_Y)
    # distinct values in every 10 m pixel
    src = (np.arange(12)[:, None] * 100 + np.arange(18)[None, :]).astype(np.float64)
    dst = reproject_to_grid(src, t10, "EPSG:4326", grid, Resampling.average)
    blocks = src.reshape(4, 3, 6, 3).transpose(0, 2, 1, 3).reshape(4, 6, 9)
    np.testing.assert_allclose(dst, blocks.mean(axis=2), atol=1e-12)
    # NaN-safe: a fully-NaN 3x3 block stays NaN, partial NaN averages the rest
    src2 = src.copy()
    src2[0:3, 0:3] = np.nan
    dst2 = reproject_to_grid(src2, t10, "EPSG:4326", grid, Resampling.average)
    assert np.isnan(dst2[0, 0])
    src3 = src.copy()
    src3[0, 0] = np.nan
    dst3 = reproject_to_grid(src3, t10, "EPSG:4326", grid, Resampling.average)
    assert dst3[0, 0] == pytest.approx(np.nanmean(src3[0:3, 0:3]))


def test_s2_20m_half_pixel_shift_lands_on_lattice():
    """The V1 F4 defect (half-pixel y-offset) is fixed by explicit reprojection.

    Hand-rolled area-weighted reference, independent of GDAL: every 30 m cell
    must equal the area-weighted mean of the 20 m pixels it overlaps.
    """
    grid = Grid(GRID_TRANSFORM, 6, 8, "EPSG:4326")
    res20 = RES * 2.0 / 3.0
    t20 = Affine(res20, 0.0, ORIGIN_X, 0.0, -res20, ORIGIN_Y - res20 / 2.0)
    res30 = RES

    def expected_1d(src_vals, src_edges, cell_edges):
        out = np.zeros(len(cell_edges) - 1)
        for i in range(len(cell_edges) - 1):
            num = den = 0.0
            for v, e0, e1 in zip(src_vals, src_edges[:-1], src_edges[1:]):
                ov = min(cell_edges[i + 1], e1) - max(cell_edges[i], e0)
                if ov > 0:
                    num += ov * v
                    den += ov
            out[i] = num / den
        return out

    # Downward-positive frame: position p increases from the origin going
    # south (row index direction). Source row j occupies
    # [res20/2 + j*res20, res20/2 + (j+1)*res20] (the half-pixel shift pushes
    # the whole source lattice half a pixel DOWN); grid row r occupies
    # [r*res30, (r+1)*res30].
    x20 = np.arange(25) * res20                     # 24 px span the 8 cells
    x30 = np.arange(9) * res30
    y20 = res20 / 2.0 + np.arange(13) * res20
    y30 = np.arange(7) * res30

    # x-gradient field (constant along y)
    grad_x = np.tile(np.arange(24, dtype=np.float64), (18, 1))
    dst = reproject_to_grid(grad_x, t20, "EPSG:4326", grid, Resampling.average)
    ref_rows = np.array([expected_1d(grad_x[0], x20, x30)] * 6)
    np.testing.assert_allclose(dst, ref_rows, rtol=0, atol=1e-9)

    # y-gradient field (constant along x): the half-pixel shift drives the
    # weights. 12 source rows cover the full 6-cell grid height (8R - half px).
    grad_y = np.repeat(np.arange(12, dtype=np.float64)[:, None], 24, axis=1)
    dst_y = reproject_to_grid(grad_y, t20, "EPSG:4326", grid, Resampling.average)
    ref_cols = np.array([expected_1d(grad_y[:, 0], y20, y30)] * 8).T
    np.testing.assert_allclose(dst_y, ref_cols, rtol=0, atol=1e-9)

    # constant field lands exactly on the lattice with the exact transform
    const = np.full((18, 24), 0.7, dtype=np.float64)
    dst_c = reproject_to_grid(const, t20, "EPSG:4326", grid, Resampling.average)
    assert np.isfinite(dst_c).all()
    np.testing.assert_allclose(dst_c, 0.7, atol=1e-12)


def test_20m_nearest_label_aggregation():
    grid = Grid(GRID_TRANSFORM, 2, 3, "EPSG:4326")
    res20 = RES * 2.0 / 3.0
    t20 = Affine(res20, 0.0, ORIGIN_X, 0.0, -res20, ORIGIN_Y - res20 / 2.0)
    lab = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]] * 6)
    dst = reproject_to_grid(lab, t20, "EPSG:4326", grid, Resampling.nearest)
    assert np.isfinite(dst).all()
    assert set(np.unique(dst)).issubset({1, 2, 3, 4, 5, 6})


def test_distance_raster_semantics():
    binary = np.zeros((7, 7), dtype=np.uint8)
    binary[0, 0] = 1
    dist = build_core.distance_raster(binary)
    assert dist[0, 0] == 0.0                       # feature pixel == 0
    assert dist[1, 0] == 30.0 and dist[0, 1] == 30.0
    assert dist[3, 4] == pytest.approx(150.0)      # 3-4-5 triangle * 30 m
    empty = build_core.distance_raster(np.zeros((3, 3), dtype=np.uint8))
    assert np.isnan(empty).all()


# ---------------------------------------------------------------------------
# Met tests
# ---------------------------------------------------------------------------
def _tiny_met_df():
    rows = [
        ("p01", 28.40, 76.83, "2022-05-01T04:00", 10.0),
        ("p01", 28.40, 76.83, "2022-05-01T05:00", 20.0),
        ("p01", 28.40, 76.83, "2022-05-01T06:00", 30.0),
        ("p01", 28.40, 76.83, "2022-05-01T07:00", 999.0),   # wrong hour
        ("p01", 28.40, 76.83, "2022-04-30T06:00", 999.0),   # before window
        ("p01", 28.40, 76.83, "2022-07-01T05:00", 999.0),   # after window
        ("p01", 28.40, 76.83, "2023-05-01T04:00", 40.0),    # other year
    ]
    df = pd.DataFrame(rows, columns=["point_id", "lat", "lon", "time_utc",
                                     "temperature_2m"])
    return df


def test_met_w4_aggregation_window():
    df = _tiny_met_df()
    agg22 = aggregate_met_window(df, 2022)
    assert len(agg22) == 1
    assert agg22.loc[0, "temperature_2m"] == pytest.approx(20.0)   # mean of 10,20,30
    agg23 = aggregate_met_window(df, 2023)
    assert agg23.loc[0, "temperature_2m"] == pytest.approx(40.0)


def test_met_window_boundaries():
    rows = [("p01", 28.40, 76.83, "2022-05-01T04:00", 2.0),    # first valid hour
            ("p01", 28.40, 76.83, "2022-06-30T06:00", 4.0),    # last valid hour
            ("p01", 28.40, 76.83, "2022-06-30T07:00", 8.0),    # wrong hour
            ("p01", 28.40, 76.83, "2022-04-30T06:00", 16.0)]   # before window
    df = pd.DataFrame(rows, columns=["point_id", "lat", "lon", "time_utc", "temperature_2m"])
    agg = aggregate_met_window(df, 2022)
    assert agg.loc[0, "temperature_2m"] == pytest.approx(3.0)  # mean of 2.0 and 4.0


def test_idw_hand_computed():
    g_lon = np.array([77.00, 77.17])
    g_lat = np.array([28.56, 28.56])
    vals = np.array([[10.0], [20.0]])
    # midpoint: equal weights -> 15
    px_lon = np.array([77.085])
    px_lat = np.array([28.56])
    w, wsum = idw_weights(px_lon, px_lat, g_lon, g_lat)
    pred = (w @ vals) / wsum[:, None]
    assert pred[0, 0] == pytest.approx(15.0, abs=1e-9)
    # exact point location: dominated by that point (1e-6 epsilon)
    px_lon2 = np.array([77.00])
    w2, wsum2 = idw_weights(px_lon2, px_lat, g_lon, g_lat)
    pred2 = (w2 @ vals) / wsum2[:, None]
    assert pred2[0, 0] == pytest.approx(10.0, abs=1e-3)


def test_idw_longitude_scaling():
    """dlon is scaled by cos(28.64 deg): 1 deg lon != 1 deg lat in weight."""
    g_lon = np.array([77.00, 77.00])
    g_lat = np.array([28.56, 28.57])
    px_lon = np.array([77.00])
    px_lat = np.array([28.565])
    w, _ = idw_weights(px_lon, px_lat, g_lon, g_lat)  # dlat = +/-0.005
    # same numeric offset purely in longitude must give a LARGER weight (cos<1)
    g_lon2 = np.array([77.00 - 0.005, 77.00 + 0.005])
    g_lat2 = np.array([28.565, 28.565])
    w2, _ = idw_weights(px_lon, px_lat, g_lon2, g_lat2)
    assert w2[0, 0] > w[0, 0]


def test_met_fields_match_hand_idw(synth):
    """End-to-end met grid: fields equal a hand-rolled IDW of point means."""
    met_df = synth["met_df"]
    grid = Grid(GRID_TRANSFORM, H, W, "EPSG:4326")
    px_lon, px_lat = grid.pixel_centers()
    fields = met_fields_for_year(met_df, 2022, px_lon, px_lat)

    agg = aggregate_met_window(met_df, 2022)
    g_lon, g_lat = agg["lon"].to_numpy(), agg["lat"].to_numpy()
    k = agg["temperature_2m"].to_numpy()           # in-window mean == point index
    cos_s = np.cos(np.deg2rad(28.64))
    # check three pixels across the grid
    for r, c in ((0, 0), (H // 2, W // 2), (H - 1, W - 1)):
        flat = r * W + c
        d2 = (px_lat[flat] - g_lat) ** 2 + (cos_s * (px_lon[flat] - g_lon)) ** 2 + 1e-6
        expected = float((k / d2).sum() / (1.0 / d2).sum())
        assert fields["met_t2m_c"][flat] == pytest.approx(expected, abs=1e-9)
    # decoys (999) must not leak: all values bounded by the point-index range
    assert fields["met_t2m_c"].max() <= 16.0 and fields["met_t2m_c"].min() >= 1.0


# ---------------------------------------------------------------------------
# Spatial helpers
# ---------------------------------------------------------------------------
def test_block_tiling_pinned_semantics():
    """Pinned bins: small grids fall entirely into band 0; real-grid band
    edges come from PINNED_ROW/COL_BINS only (never from the data extent)."""
    height, width = 60, 80
    block = compute_block_raster(height, width)
    assert set(np.unique(block).tolist()) == {0}     # all below first edge
    # real-grid semantics: band edges at the pinned bin boundaries
    big = compute_block_raster(1768, 1874)
    assert big[0, 0] == 0
    assert big[1, 0] == 0 and big[354, 0] == 0
    assert big[355, 0] == 5                          # row band 1, col band 0
    assert big[355, 375] == 6                        # row band 1, col band 1
    assert big[1767, 1873] == 24
    assert big[0, 750] == 2                          # block 2 = (rb0, cb2)
    assert big[750, 0] == 10                         # block 10 = (rb2, cb0)
    assert big[355, 1499] == 9                       # block 9 = (rb1, cb4)
    assert big[1061, 0] == 15                        # block 15 = (rb3, cb0)
    assert big[1414, 1124] == 23                     # block 23 = (rb4, cb3)


def test_focal_mean_respects_block_boundary():
    arr = np.zeros((10, 10))
    arr[:, 5:] = 100.0
    blocks = np.zeros((10, 10), dtype=np.int64)
    blocks[:, 5:] = 1
    mean = focal_mean_block_aware(arr, blocks, 3)
    # pixel just left of the boundary: window must not see the other block
    assert mean[5, 4] == pytest.approx(0.0)
    assert mean[5, 5] == pytest.approx(100.0)
    # NaN handling
    arr2 = arr.copy()
    arr2[4, 4] = np.nan
    mean2 = focal_mean_block_aware(arr2, blocks, 3)
    assert mean2[4, 4] == pytest.approx((8 * 0.0) / 8.0)


def test_nan_window_stats_values():
    arr = np.arange(25, dtype=np.float64).reshape(5, 5)
    arr[2, 2] = np.nan
    mean, std, rng = nan_window_stats(arr, 3)
    assert mean[0, 0] == pytest.approx(np.nanmean(arr[0:2, 0:2]))   # border-correct
    assert std[2, 2] == pytest.approx(np.nanstd(arr[1:4, 1:4]))
    assert rng[2, 2] == pytest.approx(np.nanmax(arr[1:4, 1:4]) - np.nanmin(arr[1:4, 1:4]))
    allnan = np.full((3, 3), np.nan)
    m2, s2, r2 = nan_window_stats(allnan, 3)
    assert np.isnan(m2).all() and np.isnan(s2).all() and np.isnan(r2).all()


# ---------------------------------------------------------------------------
# Preflight tests
# ---------------------------------------------------------------------------
def _preflight(raw: Path, grid_file: Path, vectors: dict, out: Path):
    return verify_inputs.main([
        "--data-root", str(raw), "--grid-file", str(grid_file),
        "--years", "2022,2023", "--study-area", str(vectors["study_area"]),
        "--out", str(out),
    ])


def test_preflight_pass_on_synthetic(synth):
    out = TMP / "audit_pass.json"
    rc = _preflight(synth["raw"], synth["grid_file"], synth["vectors"], out)
    assert rc == 0
    audit = json.loads(out.read_text())
    assert audit["status"] == "PASS"
    for y in ("2022", "2023"):
        prod = audit["checks"][YEARS_FIX.index(int(y))]["products"]
        assert prod["landsat9"]["coverage_st"] >= GATES["l9_st"]
        assert prod["sentinel2_10m"]["coverage"] >= GATES["s2_10m"]
        assert prod["lulc"]["coverage"] >= GATES["lulc"]


def _corrupt_tree(synth, name: str) -> Path:
    raw = TMP / f"raw_{name}"
    if raw.exists():
        shutil.rmtree(raw)
    shutil.copytree(synth["raw"], raw)
    return raw


def test_preflight_fail_missing_file(synth):
    raw = _corrupt_tree(synth, "missing")
    (raw / "sentinel2" / "2023_05_06" / "sentinel2_2023_05_06_10m_composite.tif").unlink()
    out = TMP / "audit_missing.json"
    rc = _preflight(raw, synth["grid_file"], synth["vectors"], out)
    assert rc == 1
    audit = json.loads(out.read_text())
    assert any("MISSING" in f for f in audit["failures"])


def test_preflight_fail_wrong_band_count(synth):
    raw = _corrupt_tree(synth, "bands")
    p = raw / "landsat9" / "2022_05_06" / "landsat9_2022_05_06_composite_30m.tif"
    with rasterio.open(p) as ds:
        arr = ds.read()
        prof = dict(ds.profile)
    prof.update(count=8)
    with rasterio.open(p, "w", **prof) as ds:
        ds.write(arr[:8])
    out = TMP / "audit_bands.json"
    rc = _preflight(raw, synth["grid_file"], synth["vectors"], out)
    assert rc == 1
    audit = json.loads(out.read_text())
    assert any("bands" in f for f in audit["failures"])


def test_preflight_fail_wrong_transform(synth):
    raw = _corrupt_tree(synth, "transform")
    p = raw / "landsat9" / "2022_05_06" / "landsat9_2022_05_06_composite_30m.tif"
    with rasterio.open(p) as ds:
        arr = ds.read()
        prof = dict(ds.profile)
    prof.update(transform=Affine(RES, 0.0, ORIGIN_X + RES, 0.0, -RES, ORIGIN_Y))
    with rasterio.open(p, "w", **prof) as ds:
        ds.write(arr)
    out = TMP / "audit_transform.json"
    rc = _preflight(raw, synth["grid_file"], synth["vectors"], out)
    assert rc == 1
    audit = json.loads(out.read_text())
    assert any("transform mismatch" in f for f in audit["failures"])


def test_preflight_fail_coverage_gate(synth):
    raw = _corrupt_tree(synth, "coverage")
    p = raw / "lulc" / "2022_05_06" / "lulc_2022_05_06_10m.tif"
    with rasterio.open(p) as ds:
        arr = ds.read(1, masked=True).astype(np.float64)
        prof = dict(ds.profile)
    arr[:, arr.shape[1] // 2:] = np.nan          # ~50% NaN inside the study area
    with rasterio.open(p, "w", **prof) as ds:
        ds.write(arr, 1)
    out = TMP / "audit_coverage.json"
    rc = _preflight(raw, synth["grid_file"], synth["vectors"], out)
    assert rc == 1
    audit = json.loads(out.read_text())
    assert any("LULC coverage" in f for f in audit["failures"])


# ---------------------------------------------------------------------------
# build_core product tests (on the synthetic phase-3 tree)
# ---------------------------------------------------------------------------
def _read(path: Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        return ds.read(1, masked=True).astype(np.float64).filled(np.nan)


def _read_raw(path: Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        return ds.read(1)


def test_core_lst_and_l9_indices(phase3_tree):
    lst = _read(phase3_tree / "2022" / "lst_30m.tif")
    r, c = 5, 5
    assert lst[r, c] == pytest.approx(LST_EXPECTED, abs=1e-9)
    ndvi_l9 = _read(phase3_tree / "2022" / "l9_ndvi_30m.tif")
    ndbi_l9 = _read(phase3_tree / "2022" / "l9_ndbi_30m.tif")
    ndmi = _read(phase3_tree / "2022" / "ndmi_30m.tif")
    mndwi = _read(phase3_tree / "2022" / "mndwi_30m.tif")
    bsi = _read(phase3_tree / "2022" / "bsi_30m.tif")
    assert ndvi_l9[r, c] == pytest.approx((0.22 - 0.10) / 0.32)
    assert ndbi_l9[r, c] == pytest.approx((0.20 - 0.22) / 0.42)
    assert ndmi[r, c] == pytest.approx((0.22 - 0.20) / 0.42)
    assert mndwi[r, c] == pytest.approx((0.14 - 0.20) / 0.34)
    assert bsi[r, c] == pytest.approx(((0.20 + 0.10) - (0.22 + 0.12)) / 0.64)
    # ST NaN stripe propagates to LST
    assert np.isnan(lst[25, 76])


def test_core_s2_indices_and_pvc(phase3_tree):
    ndvi = _read(phase3_tree / "2022" / "ndvi_30m.tif")
    ndbi = _read(phase3_tree / "2022" / "ndbi_30m.tif")
    ndre = _read(phase3_tree / "2022" / "ndre_30m.tif")
    pvc = _read(phase3_tree / "2022" / "vegetation_cover_30m.tif")
    r, c = 5, 5
    assert ndvi[r, c] == pytest.approx(0.5, abs=1e-9)      # (0.30-0.10)/(0.40)
    assert ndbi[r, c] == pytest.approx((0.25 - 0.28) / 0.53, abs=1e-6)
    assert ndre[r, c] == pytest.approx((0.28 - 0.15) / 0.43, abs=1e-6)
    assert pvc[r, c] == pytest.approx((0.5 - 0.05) / 0.75, abs=1e-9)
    # B8 NaN 3x3 block (10 m rows 159-161 / cols 198-200) -> its 30 m cell NaN
    assert np.isnan(ndvi[53, 66])
    assert np.isnan(pvc[53, 66])


def test_core_grid_alignment_of_outputs(phase3_tree):
    """Every 30 m product sits EXACTLY on the authoritative lattice."""
    for p in sorted((phase3_tree / "2022").glob("*.tif")):
        with rasterio.open(p) as ds:
            assert ds.transform == GRID_TRANSFORM, p.name
            assert (ds.height, ds.width) == (H, W), p.name


def test_core_valid_masks(phase3_tree, synth):
    m_l9 = _read_raw(phase3_tree / "2022" / "valid_l9_30m.tif")
    m_s2 = _read_raw(phase3_tree / "2022" / "valid_s2_10m_30m.tif")
    m_s220 = _read_raw(phase3_tree / "2022" / "valid_s2_20m_30m.tif")
    m_lulc = _read_raw(phase3_tree / "2022" / "valid_lulc_30m.tif")
    assert set(np.unique(m_l9).tolist()) <= {0, 1, 255}
    assert m_l9[0, 0] == 255                      # border outside study
    assert m_l9[5, 5] == 1
    assert m_l9[25, 76] == 0                      # ST NaN stripe
    assert m_l9[31, 41] == 0                      # QA_CLEAR_COUNT == 0 patch
    assert m_l9[51, 61] == 0                      # SR_B2 NaN patch
    assert m_s2[5, 5] == 1
    assert m_s220[5, 5] == 1
    assert m_lulc[5, 5] == 1
    assert (m_lulc == 0).sum() > 0                # NaN band leaves invalid cells


def test_core_lulc_nearest_label(phase3_tree):
    lulc = _read(phase3_tree / "2022" / "lulc_30m.tif")
    vals = lulc[np.isfinite(lulc)]
    assert set(np.unique(vals)).issubset(set(range(9)))
    # label(r,c) = (r+c) % 9 at 10 m; nearest-30 m centre is 10 m px (3r+1, 3c+1)
    assert lulc[5, 5] == ((3 * 5 + 1) + (3 * 5 + 1)) % 9


def test_core_static_layers(phase3_tree):
    sdir = phase3_tree / "static"
    lu = _read(sdir / "landuse_raster_30m.tif")
    assert lu[18, 7] == 6          # residential square (rows 15-21, cols 4-10)
    assert lu[18, 25] == 1         # park square (rows 15-21, cols 22-29)
    assert lu[18, 48] == 2         # forest square (cols 45-51)
    assert lu[52, 7] == 8          # farmland square (rows 48-55, cols 4-10)
    assert lu[52, 25] == 0         # natural-only square DROPPED by V1-parity filter
    assert lu[5, 70] == 0          # background
    bdist = _read(sdir / "buildings_distance_30m.tif")
    assert bdist[31, 38] == 0.0    # building square 1 (rows 29-33, cols 37-40)
    assert bdist[13, 57] == 0.0    # building square 2 (rows 11-14, cols 55-59)
    assert bdist[52, 7] > 0.0
    rdist = _read(sdir / "roads_distance_30m.tif")
    assert rdist[37, 10] == 0.0    # primary segment burned (row 37, cols 4-20)
    assert rdist[37, 25] > 0.0     # residential segment NOT burned (V1-parity filter)
    assert rdist[5, 70] > 0.0
    vdist = _read(sdir / "vegetation_distance_30m.tif")
    assert vdist[47, 25] == 0.0    # natural=wood square (rows 44-51, cols 22-29)
    assert vdist[47, 34] == 0.0    # leisure=park square (rows 44-51, cols 31-38)
    assert vdist[5, 70] > 0.0
    water = _read(sdir / "water_presence_30m.tif")
    assert water[54, 62] == 1      # water square (rows 52-59, cols 59-66)
    assert water[5, 10] == 0
    assert set(np.unique(water).tolist()).issubset({0, 1})
    rs = _read(sdir / "road_surfaces_presence_30m.tif")
    assert rs[37, 14] == 1         # road-surface square (rows 35-39, cols 13-16)
    assert set(np.unique(rs).tolist()).issubset({0, 1})
    meta = json.loads((sdir / "static_manifest.json").read_text())
    # stored-vs-burned bookkeeping: full network stored, V1 subset burned
    assert meta["layers"]["roads_distance"]["n_features_stored_in_source"] == 2
    assert meta["layers"]["roads_distance"]["n_features_burned"] == 1
    assert "motorway" in meta["layers"]["roads_distance"]["feature_filter"]
    assert meta["layers"]["vegetation_distance"]["n_features_burned"] == 2
    assert meta["layers"]["landuse_raster"]["n_classified_features"] == 4
    assert meta["roads_full_network_burned"] is False
    assert meta["layers"]["buildings_distance"]["n_features_burned"] == 2


def test_core_static_layers_roads_full(synth):
    """--roads-full burns the entire network (opt-in, non-default)."""
    out = TMP / "phase3_out_roadsfull"
    if out.exists():
        shutil.rmtree(out)
    rc = build_core.main(
        _build_core_args(synth["raw"], out, synth["vectors"], synth["grid_file"],
                         roads_full=True))
    assert rc == 0
    rdist = _read(out / "static" / "roads_distance_30m.tif")
    assert rdist[37, 25] == 0.0    # residential segment now burned too
    meta = json.loads((out / "static" / "static_manifest.json").read_text())
    assert meta["layers"]["roads_distance"]["n_features_burned"] == 2
    assert meta["roads_full_network_burned"] is True


def test_v1_parity_filters():
    """Unit: the V1-parity filters select exactly V1's query definitions."""
    import geopandas as gpd
    from shapely.geometry import LineString, Point

    roads = gpd.GeoDataFrame(
        {"highway": ["motorway", "trunk", "primary", "secondary", "tertiary",
                     "primary_link", "residential", None, "PRIMARY"]},
        geometry=[Point(0, 0)] * 9, crs="EPSG:4326")
    kept = build_core.filter_v1_road_classes(roads)
    # exact V1 query semantics: motorway|trunk|primary, anchored, case-insensitive
    assert kept["highway"].tolist() == ["motorway", "trunk", "primary", "PRIMARY"]

    veg = gpd.GeoDataFrame(
        {"leisure": ["park", None, None, None, "golf_course"],
         "landuse": [None, "forest", "meadow", "recreation_ground", "grass"],
         "natural": [None, None, "scrub", None, None]},
        geometry=[Point(0, 0)] * 5, crs="EPSG:4326")
    kept_v = build_core.filter_v1_vegetation(veg)
    # rows 0,1,2,4 match V1's union; recreation_ground (V2 extra) dropped
    assert kept_v.index.tolist() == [0, 1, 2, 4]

    lu = gpd.GeoDataFrame(
        {"landuse": ["residential", "park", "grass", "construction", None],
         "natural": [None, None, None, None, "wood"]},
        geometry=[Point(0, 0)] * 5, crs="EPSG:4326")
    kept_l = build_core.filter_v1_landuse(lu)
    # only V1's 7 landuse values; natural-only rows dropped
    assert kept_l["landuse"].tolist() == ["residential", "park"]


def test_core_landuse_code_semantics():
    """Codes come from the 8-class priority table; natural-only rows never
    reach classify_landuse in the feature path (V1-parity filter first)."""
    import geopandas as gpd
    from shapely.geometry import Point

    gdf = gpd.GeoDataFrame(
        {"landuse": ["residential", "park", "forest", "farmland"]},
        geometry=[Point(0, 0)] * 4, crs="EPSG:4326")
    cls = build_core.classify_landuse(gdf)
    codes = dict(zip(cls["class"], cls["code"]))
    assert codes == {"residential": 6, "park": 1, "forest": 2, "farmland": 8}
    # fallbacks still exist in classify_landuse but are unreachable post-filter
    fb = gpd.GeoDataFrame(
        {"leisure": ["pitch", None], "natural": [None, "scrub"]},
        geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:4326")
    cls_fb = build_core.classify_landuse(fb)
    assert cls_fb["class"].tolist() == ["grass", "grass"]
    assert cls_fb["code"].tolist() == [3, 3]


def test_overpass_parser_retained_but_unused(synth):
    """The tolerant Overpass-JSON parser is kept for provenance only; the
    feature path reads V2 GeoJSON (asserted by the other static-layer tests)."""
    gdf = build_core.parse_overpass_json(synth["vectors"]["legacy_overpass"])
    assert len(gdf) == 1
    assert gdf.iloc[0]["landuse"] == "residential"
    assert gdf.crs is not None


# ---------------------------------------------------------------------------
# Phase 4 end-to-end tests (assembly + verification gates)
# ---------------------------------------------------------------------------
def test_assemble_178_schema_order_and_nan_free(phase4_tree):
    import pyarrow.parquet as pqt

    schema = load_frozen_schema()
    for year in YEARS_FIX:
        df = pd.read_parquet(phase4_tree / f"features_{year}.parquet")
        assert list(df.columns[:4]) == ["row", "col", "spatial_block_id", "lst_C"]
        predictors = list(df.columns[4:])
        assert predictors == schema, f"{year}: predictor order != frozen 178"
        arr = df.iloc[:, 4:].to_numpy(dtype=np.float64)
        assert np.isfinite(arr).all(), f"{year}: NaN/Inf in predictors"
        banned = {"lst_C", "lon", "lat", "row", "col", "spatial_block_id"}
        assert not banned.intersection(predictors)
        # distance columns kept float64 (V1 parity)
        assert str(df["dist_road_m"].dtype) == "float64"
        assert str(df["ndvi"].dtype) == "float32"
        # metadata sanity
        assert (df["year"] == year).all()
        assert df["lst_C"].iloc[0] == pytest.approx(LST_EXPECTED, abs=1e-3)
    meta = json.loads((phase4_tree / "v2_feature_manifest.json").read_text())
    assert meta["n_features"] == 178
    assert meta["schema_sha256"] == schema_hash(schema)
    assert set(meta["per_year"].keys()) == {"2022", "2023"}


def test_assemble_block_tiling_matches_v1_semantics(phase4_tree):
    """spatial_block_id in the assembled table follows the PINNED V1 bins."""
    df_all = pd.concat([pd.read_parquet(phase4_tree / f"features_{y}.parquet")
                        for y in YEARS_FIX], ignore_index=True)
    rb, cb = block_bands_for_indices(df_all["row"].to_numpy(),
                                     df_all["col"].to_numpy())
    expected = rb * 5 + cb
    assert np.array_equal(expected, df_all["spatial_block_id"].to_numpy())
    # sampling independence on real-grid coordinates: ids from a subset of
    # rows equal the full-raster values at those coordinates
    raster = compute_block_raster(1768, 1874)
    rng = np.random.default_rng(0)
    rs = rng.integers(0, 1768, 400)
    cs = rng.integers(0, 1874, 400)
    rb2, cb2 = block_bands_for_indices(rs, cs)
    assert np.array_equal(rb2 * 5 + cb2, raster[rs, cs])


def test_assemble_met_columns_in_table(phase4_tree, synth):
    df = pd.read_parquet(phase4_tree / "features_2022.parquet")
    grid = Grid(GRID_TRANSFORM, H, W, "EPSG:4326")
    px_lon, px_lat = grid.pixel_centers()
    met_df = synth["met_df"]
    agg = aggregate_met_window(met_df, 2022)
    g_lon, g_lat = agg["lon"].to_numpy(), agg["lat"].to_numpy()
    k = agg["temperature_2m"].to_numpy()
    cos_s = np.cos(np.deg2rad(28.64))
    for i in range(0, len(df), max(1, len(df) // 7)):
        r, c = int(df["row"].iloc[i]), int(df["col"].iloc[i])
        flat = r * W + c
        d2 = (px_lat[flat] - g_lat) ** 2 + (cos_s * (px_lon[flat] - g_lon)) ** 2 + 1e-6
        assert df["met_t2m_c"].iloc[i] == pytest.approx(float((k / d2).sum() / (1 / d2).sum()), abs=1e-4)


def test_verify_features_passes(phase4_tree, phase3_tree):
    out = TMP / "verify_pass.json"
    rc = verify_features.main([
        "--features-dir", str(phase4_tree), "--phase3-root", str(phase3_tree),
        "--out", str(out),
    ])
    assert rc == 0
    report = json.loads(out.read_text())
    assert report["status"] == "PASS"
    for entry in report["parquets"]:
        assert entry["rows"] > 0
        assert entry["coverage"]["valid_l9"] >= GATES["l9_st"]
        assert entry["coverage"]["valid_s2_10m"] >= GATES["s2_10m"]
        assert entry["coverage"]["valid_lulc"] >= GATES["lulc"]


def test_verify_features_fails_on_leakage(phase4_tree, phase3_tree):
    df = pd.read_parquet(phase4_tree / "features_2022.parquet")
    df["lon"] = 0.0   # inject a banned column among the predictors
    bad = TMP / "phase4_bad"
    bad.mkdir(exist_ok=True)
    df.to_parquet(bad / "features_2022.parquet", index=False)
    out = TMP / "verify_leak.json"
    rc = verify_features.main([
        "--features-dir", str(bad), "--phase3-root", str(phase3_tree), "--out", str(out),
    ])
    assert rc == 1
    report = json.loads(out.read_text())
    assert any("LEAKAGE" in f for f in report["failures"])


def test_verify_features_fails_on_nan(phase4_tree, phase3_tree):
    df = pd.read_parquet(phase4_tree / "features_2022.parquet")
    df.loc[0, "ndvi"] = np.nan
    bad = TMP / "phase4_nan"
    bad.mkdir(exist_ok=True)
    df.to_parquet(bad / "features_2022.parquet", index=False)
    out = TMP / "verify_nan.json"
    rc = verify_features.main([
        "--features-dir", str(bad), "--phase3-root", str(phase3_tree), "--out", str(out),
    ])
    assert rc == 1
    report = json.loads(out.read_text())
    assert any("NaN" in f for f in report["failures"])


# ---------------------------------------------------------------------------
# Change 1: pinned V1 block geometry (verified against the read-only V1 table)
# ---------------------------------------------------------------------------
def test_pinned_bins_reproduce_v1_formula():
    """(a) The pinned constants equal linspace(min, max+1, 6) recomputed in-test
    from the read-only V1 production table, and the source hash matches."""
    src = PROJECT_ROOT.parent / PINNED_BLOCKS_SOURCE
    assert src.exists(), f"V1 production table not found at {src}"
    assert sha256_file(src) == PINNED_BLOCKS_SOURCE_SHA256

    df = pd.read_csv(src, usecols=["row", "col"])
    rbins = np.linspace(int(df.row.min()), int(df.row.max()) + 1, 6)
    cbins = np.linspace(int(df.col.min()), int(df.col.max()) + 1, 6)
    np.testing.assert_array_equal(rbins, np.array(PINNED_ROW_BINS))
    np.testing.assert_array_equal(cbins, np.array(PINNED_COL_BINS))


def test_holdout_geography():
    """(b) Holdout blocks {2,9,15,23} are pairwise non-adjacent and their
    manifest geographic bboxes match the pinned formula on the identical
    grid (concrete lon/lat ranges asserted)."""
    import itertools

    assert sorted(HOLDOUT_BLOCKS) == [2, 9, 15, 23]

    def band(bid):
        return divmod(bid, 5)

    for a, b in itertools.combinations(HOLDOUT_BLOCKS, 2):
        ra, ca = band(a)
        rb_, cb_ = band(b)
        assert not (abs(ra - rb_) <= 1 and abs(ca - cb_) <= 1), \
            f"holdout blocks {a} and {b} are adjacent"

    grid = Grid.from_file(GRID_FILE_DEFAULT)
    res = grid.transform.a
    ox, oy = grid.transform.c, grid.transform.f
    manifest = json.loads(BLOCKS_MANIFEST.read_text())
    records = {b["block_id"]: b for b in manifest["blocks"]}
    for bid in HOLDOUT_BLOCKS:
        rb_i, cb_i = band(bid)
        rec = records[bid]
        # band 0 includes the lower clip zone (indices below the first edge),
        # band 4 the upper clip zone (indices >= last edge, incl. grid edge)
        r0 = 0 if rb_i == 0 else int(np.ceil(PINNED_ROW_BINS[rb_i]))
        r1 = (grid.height - 1) if rb_i == 4 else int(np.ceil(PINNED_ROW_BINS[rb_i + 1])) - 1
        c0 = 0 if cb_i == 0 else int(np.ceil(PINNED_COL_BINS[cb_i]))
        c1 = (grid.width - 1) if cb_i == 4 else int(np.ceil(PINNED_COL_BINS[cb_i + 1])) - 1
        assert (rec["row_min"], rec["row_max"]) == (r0, r1), (bid, rec)
        assert (rec["col_min"], rec["col_max"]) == (c0, c1), (bid, rec)
        lon_min, lon_max = ox + (c0 + 0.5) * res, ox + (c1 + 0.5) * res
        lat_max, lat_min = oy - (r0 + 0.5) * res, oy - (r1 + 0.5) * res
        assert rec["lon_min"] == pytest.approx(lon_min, abs=1e-9)
        assert rec["lon_max"] == pytest.approx(lon_max, abs=1e-9)
        assert rec["lat_min"] == pytest.approx(lat_min, abs=1e-9)
        assert rec["lat_max"] == pytest.approx(lat_max, abs=1e-9)
    # concrete geographic anchors on the verified V1/V2 grid
    assert records[2]["lon_min"] == pytest.approx(77.0351619, abs=1e-6)
    assert records[2]["lat_max"] == pytest.approx(28.8845644, abs=1e-6)
    assert records[23]["lat_min"] == pytest.approx(28.4083675, abs=1e-6)


def test_sampling_independence_of_block_ids():
    """(c) Block ids for sampled rows are identical whether computed from the
    full grid or any subset of rows (pinned bins never depend on the data)."""
    full = compute_block_raster(1768, 1874)
    rng = np.random.default_rng(7)
    for _ in range(3):
        rs = rng.integers(0, 1768, 5000)
        cs = rng.integers(0, 1874, 5000)
        rb, cb = block_bands_for_indices(rs, cs)
        assert np.array_equal(rb * 5 + cb, full[rs, cs])
    for rs, cs in [(np.zeros(10, dtype=int), np.arange(10)),
                   (np.arange(10), np.zeros(10, dtype=int)),
                   (np.array([0]), np.array([0]))]:
        rb, cb = block_bands_for_indices(rs, cs)
        assert np.array_equal(rb * 5 + cb, full[rs, cs])


def test_all_25_blocks_nonempty_on_full_grid():
    """(d) Every one of the 25 pinned blocks contains pixels on the full grid."""
    raster = compute_block_raster(1768, 1874)
    counts = np.bincount(raster.ravel(), minlength=25)
    assert (counts > 0).all()
    assert counts.sum() == 1768 * 1874


def test_spatial_blocks_manifest_contents():
    """The generated manifest is consistent with the pinned constants and
    carries the honest formula-reproduction verification."""
    assert BLOCKS_MANIFEST.exists()
    m = json.loads(BLOCKS_MANIFEST.read_text())
    pc = m["pinned_constants"]
    assert tuple(pc["PINNED_ROW_BINS"]) == PINNED_ROW_BINS
    assert tuple(pc["PINNED_COL_BINS"]) == PINNED_COL_BINS
    assert tuple(pc["HOLDOUT_BLOCKS"]) == HOLDOUT_BLOCKS
    assert m["provenance"]["source_table_sha256"] == PINNED_BLOCKS_SOURCE_SHA256
    v = m["verification"]
    assert v["formula_reproduction_vs_column_total"] == 749998
    # the V1 column is per-year Phase-3 output; documented, not hidden
    assert v["formula_reproduction_vs_column_mismatches"] > 0
    assert v["all_25_blocks_non_empty_on_grid"] is True
    for b in HOLDOUT_BLOCKS:
        assert v["locked_holdout_preservation"][str(b)]["fraction"] > 0.9


# ---------------------------------------------------------------------------
# Change 2: S2 B2>0 hard invariant
# ---------------------------------------------------------------------------
def test_s2_b2_zero_excluded_from_values(phase3_tree):
    """B2==0 native pixels never enter the value stack: a fully-artifact 30 m
    cell is NaN; a mixed cell aggregates ONLY the valid native pixels."""
    ndvi = _read(phase3_tree / "2022" / "ndvi_30m.tif")
    pvc = _read(phase3_tree / "2022" / "vegetation_cover_30m.tif")
    stripe_ndvi = (0.90 - 0.05) / (0.90 + 0.05)      # what a leak would produce
    assert np.isnan(ndvi[6, 10])                     # cell row 6: fully artifact
    assert np.isnan(pvc[6, 10])
    assert ndvi[7, 10] == pytest.approx(0.5, abs=1e-9)          # valid only
    assert ndvi[7, 10] != pytest.approx(stripe_ndvi, abs=1e-3)  # not the leak
    assert ndvi[8, 10] == pytest.approx(0.5, abs=1e-9)
    m = _read_raw(phase3_tree / "2022" / "valid_s2_10m_30m.tif")
    assert m[6, 10] == 0
    assert m[7, 10] == 1


def test_core_b2_zero_guard_reported(phase3_tree):
    """Phase-3 build reports the B2==0 exclusion accounting per year."""
    man = json.loads((phase3_tree / "2022" / "build_manifest.json").read_text())
    g = man["stats"]["s2_10m_b2_zero_guard"]
    # stripe: 5 native rows x 240 cols = 1200 B2==0 px per year
    assert g["b2_zero_px"] == 1200
    assert g["finite_px"] > g["valid_excluding_b2_zero"]
    assert g["valid_excluding_b2_zero"] == g["finite_px"] - g["b2_zero_px"]
    assert g["identity_holds"] is True


def test_preflight_b2_gate_and_crosscheck(synth):
    """The B2 accounting is a hard gate; counts crosscheck against a prior
    audit JSON exactly (both structured and legacy warning formats)."""
    out_a = TMP / "audit_b2_a.json"
    rc = verify_inputs.main([
        "--data-root", str(synth["raw"]), "--grid-file", str(synth["grid_file"]),
        "--years", "2022,2023", "--study-area", str(synth["vectors"]["study_area"]),
        "--out", str(out_a)])
    assert rc == 0
    a = json.loads(out_a.read_text())
    for chk in a["checks"]:
        g = chk["products"]["sentinel2_10m"]["b2_zero_guard"]
        assert g["identity_holds"] is True
        assert g["b2_zero_px"] == 1200
    # crosscheck against this very audit -> PASS
    out_b = TMP / "audit_b2_b.json"
    rc = verify_inputs.main([
        "--data-root", str(synth["raw"]), "--grid-file", str(synth["grid_file"]),
        "--years", "2022,2023", "--study-area", str(synth["vectors"]["study_area"]),
        "--out", str(out_b), "--crosscheck-audit", str(out_a)])
    assert rc == 0
    b = json.loads(out_b.read_text())
    assert len(b["crosschecks"]) == 2 and all(c["match"] for c in b["crosschecks"])
    # corrupt the prior count -> FAIL
    a_bad = json.loads(out_a.read_text())
    a_bad["checks"][0]["products"]["sentinel2_10m"]["b2_zero_guard"]["b2_zero_px"] += 1
    bad_path = TMP / "audit_b2_bad.json"
    bad_path.write_text(json.dumps(a_bad))
    out_c = TMP / "audit_b2_c.json"
    rc = verify_inputs.main([
        "--data-root", str(synth["raw"]), "--grid-file", str(synth["grid_file"]),
        "--years", "2022,2023", "--study-area", str(synth["vectors"]["study_area"]),
        "--out", str(out_c), "--crosscheck-audit", str(bad_path)])
    assert rc == 1
    c = json.loads(out_c.read_text())
    assert any("crosscheck" in f for f in c["failures"])
    # legacy warning-string format is also accepted
    legacy = json.loads(out_a.read_text())
    for chk in legacy["checks"]:
        chk["products"]["sentinel2_10m"].pop("b2_zero_guard", None)
    legacy["warnings"] = [
        f"[{chk['year']}] S2-10m: B2==0 zero-fill pattern present (1200 px); B2>0 guard active"
        for chk in legacy["checks"]]
    leg_path = TMP / "audit_b2_legacy.json"
    leg_path.write_text(json.dumps(legacy))
    out_d = TMP / "audit_b2_d.json"
    rc = verify_inputs.main([
        "--data-root", str(synth["raw"]), "--grid-file", str(synth["grid_file"]),
        "--years", "2022,2023", "--study-area", str(synth["vectors"]["study_area"]),
        "--out", str(out_d), "--crosscheck-audit", str(leg_path)])
    assert rc == 0


def test_real_audit_b2_counts_parseable():
    """The real w4_input_audit.json (read-only) exposes per-year B2==0 counts;
    the user-quoted figures (2022: 84,909 / 2026: 7,867) are exactly what the
    JSON stores. Later audit regenerations record the b2_zero_guard block for
    every year (0 for artifact-free years), so assert the quoted figures as a
    subset rather than exact dict equality."""
    assert W4_INPUT_AUDIT.exists()
    audit = json.loads(W4_INPUT_AUDIT.read_text())
    counts = {}
    for chk in audit["checks"]:
        g = chk.get("products", {}).get("sentinel2_10m", {}).get("b2_zero_guard")
        if g is not None:
            counts[chk["year"]] = g["b2_zero_px"]
    if not counts:   # legacy format: parse the warning strings
        import re
        for w in audit.get("warnings", []):
            m = re.search(r"^\[(\d{4})\] S2-10m: B2==0.*\((\d+) px\)", w)
            if m:
                counts[int(m.group(1))] = int(m.group(2))
    assert counts.get(2022) == 84909 and counts.get(2026) == 7867, counts


def test_verify_features_b2_gate_pass_and_fail(phase4_tree, phase3_tree):
    """Gate 5b: rows sit on cells with >=1 valid native B2>0 pixel; a
    doctored mask (leak) must fail."""
    out = TMP / "verify_b2_pass.json"
    rc = verify_features.main([
        "--features-dir", str(phase4_tree), "--phase3-root", str(phase3_tree),
        "--out", str(out)])
    assert rc == 0

    df = pd.read_parquet(phase4_tree / "features_2022.parquet")
    bad3 = TMP / "phase3_b2leak"
    if bad3.exists():
        shutil.rmtree(bad3)
    shutil.copytree(phase3_tree, bad3)
    r, c = int(df["row"].iloc[0]), int(df["col"].iloc[0])
    mask_path = bad3 / "2022" / "valid_s2_10m_30m.tif"
    with rasterio.open(mask_path, "r+") as ds:
        m = ds.read(1)
        m[r, c] = 0
        ds.write(m, 1)
    out_bad = TMP / "verify_b2_fail.json"
    rc = verify_features.main([
        "--features-dir", str(phase4_tree), "--phase3-root", str(bad3),
        "--out", str(out_bad)])
    assert rc == 1
    rep = json.loads(out_bad.read_text())
    assert any("B2==0 LEAKAGE" in f for f in rep["failures"])


def test_all_modules_compile():
    import py_compile

    mods = [
        "src/v2/__init__.py", "src/v2/common.py",
        "src/v2/phase3/__init__.py", "src/v2/phase3/verify_inputs.py",
        "src/v2/phase3/build_core.py",
        "src/v2/phase4/__init__.py", "src/v2/phase4/_spatial.py",
        "src/v2/phase4/assemble_features.py", "src/v2/phase4/verify_features.py",
    ]
    for m in mods:
        py_compile.compile(str(PROJECT_ROOT / m), doraise=True)
