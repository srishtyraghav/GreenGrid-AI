"""Shared helpers for the V2 Phase-2 data audit (read-only on V1 data)."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
OUT_ROOT = PROJECT_ROOT / "data" / "v2" / "phase2"
REFERENCE_RASTER = (
    RAW_ROOT / "landsat9" / "2026_07" / "landsat9_2026_07_composite_30m.tif"
)
STUDY_AREA = RAW_ROOT / "gis" / "study_area" / "study_area.geojson"
# V1 Phase-4 combined table: defines the exact row/col extent that V1 used to
# derive spatial-block bins (see src/models/spatial_features.py).
V1_COMBINED_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phase4"
    / "tables"
    / "combined_urban_environmental_dataset.csv"
)
N_BLOCKS = 5  # V1 default in _compute_spatial_block_raster
YEARS = [2022, 2023, 2024, 2025, 2026]

CHUNK = 1 << 20  # 1 MiB


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


_YEAR_RE = re.compile(r"(20\d{2})_07")


def year_from_path(path: Path) -> int | None:
    m = _YEAR_RE.search(path.as_posix())
    return int(m.group(1)) if m else None


def category_from_path(path: Path) -> str:
    return path.relative_to(RAW_ROOT).parts[0]


class BandStats:
    """Exact streaming per-band statistics accumulated over row windows."""

    __slots__ = (
        "n", "finite", "nan", "posinf", "neginf", "zeros",
        "min", "max", "sum", "sumsq",
    )

    def __init__(self) -> None:
        self.n = 0
        self.finite = 0
        self.nan = 0
        self.posinf = 0
        self.neginf = 0
        self.zeros = 0
        self.min = math.inf
        self.max = -math.inf
        self.sum = 0.0
        self.sumsq = 0.0

    def update(self, win: np.ndarray) -> None:
        self.n += win.size
        f = np.isfinite(win)
        self.finite += int(f.sum())
        n_nan = int(np.isnan(win).sum())
        self.nan += n_nan
        inf = np.isinf(win)
        self.posinf += int((inf & (win > 0)).sum())
        self.neginf += int((inf & (win < 0)).sum())
        fin = win[f]
        if fin.size:
            self.zeros += int((fin == 0).sum())
            self.min = min(self.min, float(fin.min()))
            self.max = max(self.max, float(fin.max()))
            self.sum += float(fin.sum())
            self.sumsq += float((fin.astype(np.float64) ** 2).sum())

    def mean_std(self) -> tuple[float, float]:
        if not self.finite:
            return (math.nan, math.nan)
        mean = self.sum / self.finite
        var = max(self.sumsq / self.finite - mean * mean, 0.0)
        return (mean, math.sqrt(var))

    def as_dict(self) -> dict:
        mean, std = self.mean_std()
        return {
            "n": self.n,
            "finite": self.finite,
            "nan": self.nan,
            "posinf": self.posinf,
            "neginf": self.neginf,
            "zeros": self.zeros,
            "pct_finite": 100.0 * self.finite / self.n if self.n else math.nan,
            "min": self.min if self.finite else math.nan,
            "max": self.max if self.finite else math.nan,
            "mean": mean,
            "std": std,
        }


def band_stats_windowed(src: rasterio.DatasetReader, window_rows: int = 256) -> dict[int, BandStats]:
    stats = {i: BandStats() for i in range(1, src.count + 1)}
    for row0 in range(0, src.height, window_rows):
        h = min(window_rows, src.height - row0)
        w = rasterio.windows.Window(0, row0, src.width, h)
        for i in range(1, src.count + 1):
            arr = src.read(i, window=w)
            stats[i].update(arr)
    return stats


def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def dump_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(obj), f, indent=2)
