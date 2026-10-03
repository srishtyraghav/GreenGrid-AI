"""Build leakage-safe spatial context features for exp_spatial_context.

Adds nan-safe neighborhood statistics of NON-TARGET environmental rasters at
3x3, 5x5 and 11x11 windows (30 m grid), sampled at the exact pixels of the
modeling table:

  std  (heterogeneity): ndvi, ndbi, ndre, ndmi, mndwi, bsi, vegetation_cover
  mean (missing bands): mndwi, ndre          (ndvi/ndbi/veg/ndmi/bsi means
                                               already exist in the dataset)
  range (max-min):      ndvi, ndbi

33 new columns. NO LST-derived statistics are included (target-derived
predictors are forbidden by the standing leakage rule; decided with the user
on 2026-10-03). All statistics are pure functions of the environmental
rasters, constructed identically for every pixel regardless of spatial-block
membership (train/validation/LOBO/locked identical).

Source rasters verified numerically against the modeling table columns
(max_abs_err = 0.0 on 2022 samples):
  ndvi,ndbi,ndre -> data/processed/phase3/aligned/s2_{year}_*_30m.tif
  ndmi,mndwi,bsi -> data/processed/phase3/aligned/l9_{year}_composite_*_30m.tif
  vegetation_cover -> data/processed/phase4/features/vegetation_cover_{year}_30m.tif

Output: data/processed/experiments/exp_spatial_context_features.csv
        (row, col, year + 33 new columns, merged with the 6 met covariates)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import maximum_filter, minimum_filter, uniform_filter

PROJECT = Path(__file__).resolve().parent.parent
ALIGNED = PROJECT / "data" / "processed" / "phase3" / "aligned"
PHASE4 = PROJECT / "data" / "processed" / "phase4" / "features"
DATASET = PROJECT / "data" / "processed" / "lulc_outputs" / "combined_urban_environmental_dataset.csv"
MET_CSV = PROJECT / "data" / "processed" / "experiments" / "exp_met_features.csv"
OUT_CSV = PROJECT / "data" / "processed" / "experiments" / "exp_spatial_context_features.csv"

WINDOWS = (3, 5, 11)
YEARS = (2022, 2023, 2024, 2025, 2026)

STD_BANDS = ("ndvi", "ndbi", "ndre", "ndmi", "mndwi", "bsi", "vegetation_cover")
MEAN_BANDS = ("mndwi", "ndre")
RANGE_BANDS = ("ndvi", "ndbi")


def band_path(band: str, year: int) -> Path:
    if band in ("ndvi", "ndbi", "ndre"):
        return ALIGNED / f"s2_{year}_{band}_30m.tif"
    if band in ("ndmi", "mndwi", "bsi"):
        return ALIGNED / f"l9_{year}_composite_{band}_30m.tif"
    if band == "vegetation_cover":
        return PHASE4 / f"vegetation_cover_{year}_30m.tif"
    raise ValueError(band)


def nan_window_stats(arr: np.ndarray, w: int):
    """Border-correct nan-safe window mean/std/range via summed-area filters.

    Returns (mean, std, range); a pixel is valid iff >=1 valid pixel lies in
    its window. mode='constant', cval=0 so window mass outside the image is
    never counted (no padding contamination at the study-area border).
    """
    valid = np.isfinite(arr)
    a = np.where(valid, arr, 0.0).astype(np.float32)
    w2 = float(w * w)
    c = uniform_filter(valid.astype(np.float32), size=w, mode="constant", cval=0) * w2
    safe_c = np.where(c > 0, c, np.nan)
    s = uniform_filter(a, size=w, mode="constant", cval=0) * w2
    s2 = uniform_filter(a * a, size=w, mode="constant", cval=0) * w2
    mean = s / safe_c
    var = np.clip(s2 / safe_c - mean ** 2, 0, None)
    mx = maximum_filter(np.where(valid, arr, -np.inf), size=w, mode="constant", cval=-np.inf)
    mn = minimum_filter(np.where(valid, arr, np.inf), size=w, mode="constant", cval=np.inf)
    rng = np.where(c > 0, mx - mn, np.nan)
    return mean, np.sqrt(var), rng


def main() -> int:
    df = pd.read_csv(DATASET, usecols=["row", "col", "year"])
    print(f"[BUILD] sampled pixels: {len(df)}")
    out = df.copy()

    for year in YEARS:
        rows = df.loc[df.year == year, "row"].values
        cols = df.loc[df.year == year, "col"].values
        n = len(rows)
        stats = {}
        for band in dict.fromkeys(STD_BANDS + MEAN_BANDS + RANGE_BANDS):
            with rasterio.open(band_path(band, year)) as ds:
                arr = ds.read(1).astype(np.float32)
            for w in WINDOWS:
                mean, std, rng = nan_window_stats(arr, w)
                if band in STD_BANDS:
                    stats[f"{band}_std{w}"] = std[rows, cols]
                if band in MEAN_BANDS:
                    stats[f"{band}_mean{w}"] = mean[rows, cols]
                if band in RANGE_BANDS:
                    stats[f"{band}_range{w}"] = rng[rows, cols]
                print(f"[BUILD] {year} {band} w={w} done", flush=True)
            del arr
        for k, v in stats.items():
            out.loc[df.year == year, k] = v
        # sanity: report nan fraction in the new columns for this year
        newcols = [c for c in out.columns if c not in ("row", "col", "year")]
        nanfrac = out.loc[df.year == year, newcols].isna().mean().mean()
        print(f"[BUILD] {year}: n={n}, new-col NaN fraction={nanfrac:.6f}", flush=True)

    newcols = [c for c in out.columns if c not in ("row", "col", "year")]
    assert len(newcols) == 33, f"expected 33 new columns, got {len(newcols)}"

    met = pd.read_csv(MET_CSV)
    merged = out.merge(met, on=["row", "col", "year"], how="left")
    assert len(merged) == len(out), "merge changed row count"
    assert not merged[newcols].isna().any().any(), "unexpected NaN in new features"
    merged.to_csv(OUT_CSV, index=False)
    print(f"[BUILD] wrote {OUT_CSV} rows={len(merged)} cols={len(merged.columns)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
