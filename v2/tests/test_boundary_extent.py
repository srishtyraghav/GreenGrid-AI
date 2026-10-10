"""Focused tests for the data-extent study-area boundary (Phase 2 erratum).

Verifies that study_area_data_extent.geojson is a faithful vectorization of
the persistent valid-data mask and that the disclosure documents agree.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.features import geometry_mask

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
for p in (str(PROJECT_ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

GIS = PROJECT_ROOT / "data" / "gis" / "study_area"
EXTENT = GIS / "study_area_data_extent.geojson"
OSM = GIS / "study_area.geojson"
PROVENANCE = PROJECT_ROOT / "data" / "phase2" / "provenance_table.csv"
PHASE2_REPORT = PROJECT_ROOT / "reports" / "phase2_dataset_report.md"

YEARS = (2022, 2023, 2024, 2025, 2026)
HA_PER_PX = 0.09


def _union_mask() -> tuple[np.ndarray, rasterio.Affine]:
    union = None
    transform = None
    for year in YEARS:
        for path in (
            PROJECT_ROOT / "data" / "phase6" / "rasters" / f"severity_{year}.tif",
            PROJECT_ROOT / "data" / "phase3" / str(year) / "lst_30m.tif",
        ):
            if not path.exists():
                pytest.skip(f"raster absent: {path}")
            with rasterio.open(path) as ds:
                arr = ds.read(1)
                valid = np.isfinite(arr) & (arr != ds.nodata) if ds.nodata is not None else np.isfinite(arr)
                if union is None:
                    union = np.zeros(valid.shape, dtype=bool)
                    transform = ds.transform
                union |= valid
    return union, transform


@pytest.fixture(scope="module")
def extent_gdf():
    if not EXTENT.exists():
        pytest.skip("data-extent geojson not built")
    return gpd.read_file(EXTENT)


def test_extent_file_exists_and_is_valid_geojson(extent_gdf):
    assert len(extent_gdf) >= 1
    assert extent_gdf.crs is not None


def test_extent_covers_union_mask(extent_gdf):
    """≥99.5% of the union-mask pixels must fall inside the extent polygons
    (small loss is expected from the documented simplify tolerance)."""
    union, transform = _union_mask()
    geoms = [g.__geo_interface__ for g in extent_gdf.geometry]
    inside = geometry_mask(geoms, union.shape, transform, invert=True)
    covered = float((union & inside).sum()) / float(union.sum())
    assert covered >= 0.995, f"only {covered:.4%} of union mask covered"


def test_extent_does_not_grossly_exceed_mask(extent_gdf):
    """The simplified polygons must not balloon beyond the mask they describe."""
    union, transform = _union_mask()
    geoms = [g.__geo_interface__ for g in extent_gdf.geometry]
    inside = geometry_mask(geoms, union.shape, transform, invert=True)
    excess = float((~union & inside).sum()) / float(union.sum())
    assert excess <= 0.02, f"extent exceeds mask by {excess:.2%}"


def test_extent_regions_match_connected_components(extent_gdf):
    union, _ = _union_mask()
    n_components = sum(len(g.geoms) if g.geom_type == "MultiPolygon" else 1
                       for g in extent_gdf.geometry)
    expected = 0
    from scipy import ndimage
    _, expected = ndimage.label(union, structure=np.ones((3, 3)))
    assert n_components == expected, (n_components, expected)


def test_extent_bounds_within_grid(extent_gdf):
    with rasterio.open(PROJECT_ROOT / "data" / "phase6" / "rasters" / "severity_2026.tif") as ds:
        b = ds.bounds
    xmin, ymin, xmax, ymax = extent_gdf.total_bounds
    tol = 0.005  # degrees; simplify tolerance is ~0.0003
    assert xmin >= b.left - tol and xmax <= b.right + tol
    assert ymin >= b.bottom - tol and ymax <= b.top + tol


def test_extent_metadata_discloses_gaul(extent_gdf):
    raw = json.loads(EXTENT.read_text())
    meta = raw.get("metadata", {})
    assert "GAUL" in meta.get("operational_clip_geometry", "")
    assert meta.get("n_regions") == len(extent_gdf)


def test_osm_reference_boundary_retained():
    assert OSM.exists()
    g = json.loads(OSM.read_text())
    props = g["features"][0]["properties"]
    assert props.get("role") == "reference_boundary"
    assert "GAUL" in props.get("purpose", "")


def test_provenance_table_discloses_operational_clip():
    df = pd.read_csv(PROVENANCE, header=0)
    col = "layer" if "layer" in df.columns else df.columns[0]
    names = " | ".join(df[col].astype(str))
    assert "operational GEE clip" in names
    blob = " ".join(str(v) for v in df.fillna("").to_numpy(dtype=str).ravel())
    assert "GAUL" in blob


def test_phase2_report_erratum_present():
    text = PHASE2_REPORT.read_text(encoding="utf-8")
    assert "Erratum" in text
    assert "FAO/GAUL/2015" in text
    assert "data-extent boundary" in text
