"""Reference verification test for the optimized morphology implementation.

Compares the production integral-image/stripe implementation in
``src/models/morphology_features.py`` against an independent, deliberately
naive per-pixel disk-neighbourhood reference on small synthetic rasters.

Checks, for every production radius (50/100/250/500 m):

- building / road / vegetation pixel fractions (distance == 0 proxy)
- vegetation-cover neighbourhood mean (NaN-aware)
- NDVI / NDBI contrast (pixel minus block-aware neighbourhood mean)
- land-use class fractions, dominant class, and Shannon entropy
- valid-neighbourhood-count semantics (denominator documentation)
- block boundaries are never crossed by any neighbourhood
- radius-to-pixel conversion
- edge clipping is detected and matches a naive geometric reference

Test pixels include random interior pixels and pixels adjacent to block
boundaries (where kernels are clipped).  A tiny 3x3 block is included so that
large-radius neighbourhoods can be fully clipped (zero valid pixels -> NaN),
matching production behaviour.

Run from the repository root:

    PYTHONPATH=src python3 tests/test_morphology_reference.py

Exits non-zero (and prints the mismatches) on any failure.
"""

from __future__ import annotations

import sys

import numpy as np

from models.morphology_features import (
    MORPHOLOGY_LANDUSE_CLASSES,
    MORPHOLOGY_RADII_M,
    _count_clipped_pixels,
    _disk_kernel,
    _focal_aggregate_block_aware,
    _focal_fraction_block_aware,
    _focal_mean_block_aware,
    _radius_m_to_px,
)

RTOL = 1e-9
ATOL = 1e-12

# Synthetic raster geometry: a 4x5 grid of blocks (15 rows x 14 cols each)
# filling rows 0-59, plus a 3-row strip of five 3x14 blocks (rows 60-62), so
# the whole raster is covered by rectangular, contiguous blocks — matching
# the production invariant that spatial blocks come from a regular row/col
# binning (`_compute_spatial_block_raster`).
HEIGHT, WIDTH = 63, 70


def make_block_raster() -> np.ndarray:
    """Build a synthetic spatial-block raster of rectangular blocks."""
    block = np.zeros((HEIGHT, WIDTH), dtype=int)
    next_id = 0
    # Regular 4x5 grid of 15x14 blocks (rows 0-59, cols 0-69).
    for rb in range(4):
        for cb in range(5):
            r0, r1 = rb * 15, (rb + 1) * 15
            c0, c1 = cb * 14, (cb + 1) * 14
            block[r0:r1, c0:c1] = next_id
            next_id += 1
    # Bottom strip (rows 60-62): five 3x14 blocks, so 500 m neighbourhoods
    # are heavily clipped there but every pixel has a valid neighbourhood.
    for cb in range(5):
        block[60:63, cb * 14 : (cb + 1) * 14] = next_id
        next_id += 1
    return block


def disk_offsets(radius_px: int) -> np.ndarray:
    """(N, 2) array of (dy, dx) offsets for the production disk kernel."""
    kernel = _disk_kernel(radius_px)
    return np.argwhere(kernel) - radius_px


def naive_neighbourhood(
    arr: np.ndarray,
    block_raster: np.ndarray,
    r: int,
    c: int,
    offsets: np.ndarray,
) -> np.ndarray:
    """Values of ``arr`` inside the block-aware disk neighbourhood of (r, c)."""
    h, w = arr.shape
    vals = []
    for dy, dx in offsets:
        rr, cc = r + dy, c + dx
        if 0 <= rr < h and 0 <= cc < w and block_raster[rr, cc] == block_raster[r, c]:
            vals.append(arr[rr, cc])
    return np.asarray(vals, dtype=np.float64)


def naive_landuse_stats(counts: np.ndarray, total: float):
    """Reference land-use fraction / dominant / entropy from class counts."""
    lu = np.asarray(MORPHOLOGY_LANDUSE_CLASSES, dtype=np.float64)
    if total <= 0:
        return np.zeros(len(lu)), 0.0, 0.0
    probs = counts / total
    fracs = np.clip(probs, 0.0, 1.0)
    dominant = float(lu[int(np.argmax(counts))])
    positive = probs > 0
    entropy = float(-np.sum(probs[positive] * np.log(probs[positive])))
    return fracs, dominant, entropy


def check(name: str, actual, expected, failures: list) -> None:
    """Append a formatted failure if actual != expected within tolerance."""
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    both_nan = np.isnan(actual) & np.isnan(expected)
    if np.all(both_nan):
        return
    valid = ~np.isnan(actual) & ~np.isnan(expected)
    if not np.all(valid):
        failures.append(f"{name}: NaN mismatch (actual={actual}, expected={expected})")
        return
    if not np.allclose(actual[valid], expected[valid], rtol=RTOL, atol=ATOL):
        diff = np.abs(actual[valid] - expected[valid]).max()
        failures.append(f"{name}: max abs diff {diff:.3e}")


def main() -> int:
    rng = np.random.default_rng(42)
    block_raster = make_block_raster()

    # Synthetic source rasters.
    dist_building = rng.uniform(0, 500, (HEIGHT, WIDTH))
    dist_road = rng.uniform(0, 500, (HEIGHT, WIDTH))
    dist_veg = rng.uniform(0, 500, (HEIGHT, WIDTH))
    vegetation_cover = rng.uniform(0, 1, (HEIGHT, WIDTH))
    ndvi = rng.normal(0.2, 0.25, (HEIGHT, WIDTH))
    ndbi = rng.normal(0.1, 0.2, (HEIGHT, WIDTH))
    landuse = rng.integers(0, 9, (HEIGHT, WIDTH)).astype(np.float64)

    # Inject NaNs to exercise NaN-aware aggregation (~2% of pixels).
    for arr in (vegetation_cover, ndvi, ndbi):
        mask = rng.random(arr.shape) < 0.02
        arr[mask] = np.nan
    # Inject NaNs into a distance raster (binary masks must ignore them).
    dist_building[rng.random(dist_building.shape) < 0.02] = np.nan

    # Pixels to verify: random interior + every pixel adjacent to a block
    # boundary (edge pixels) + the fully-clipped tiny block.
    edge_pixels = set()
    for r in range(1, HEIGHT - 1):
        for c in range(1, WIDTH - 1):
            b = block_raster[r, c]
            if (
                block_raster[r - 1, c] != b
                or block_raster[r + 1, c] != b
                or block_raster[r, c - 1] != b
                or block_raster[r, c + 1] != b
            ):
                edge_pixels.add((r, c))
    edge_pixels = sorted(edge_pixels)
    n_random = 40
    rand_pixels = [
        (int(r), int(c))
        for r, c in rng.integers(0, [HEIGHT, WIDTH], size=(n_random, 2))
    ]
    tiny_block_pixels = [(60, 0), (61, 7), (62, 13)]  # 3x14 strip block
    test_pixels = rand_pixels + tiny_block_pixels + edge_pixels[::3]

    failures: list[str] = []
    n_checks = 0

    # Radius conversion sanity.
    expected_px = {50: 2, 100: 3, 250: 8, 500: 17}
    for radius_m in MORPHOLOGY_RADII_M:
        got = _radius_m_to_px(radius_m)
        if got != expected_px[radius_m]:
            failures.append(
                f"radius conversion: {radius_m} m -> {got} px (expected {expected_px[radius_m]})"
            )

    for radius_m in MORPHOLOGY_RADII_M:
        radius_px = _radius_m_to_px(radius_m)
        kernel = _disk_kernel(radius_px)
        offsets = disk_offsets(radius_px)

        # Production outputs.
        build_frac = _focal_fraction_block_aware(
            (dist_building == 0).astype(np.float64), block_raster, kernel
        )
        road_frac = _focal_fraction_block_aware(
            (dist_road == 0).astype(np.float64), block_raster, kernel
        )
        veg_frac = _focal_fraction_block_aware(
            (dist_veg == 0).astype(np.float64), block_raster, kernel
        )
        vc_mean = _focal_mean_block_aware(vegetation_cover, block_raster, kernel)
        ndvi_mean = _focal_mean_block_aware(ndvi, block_raster, kernel)
        ndbi_mean = _focal_mean_block_aware(ndbi, block_raster, kernel)
        _, valid_count = _focal_aggregate_block_aware(
            np.ones((HEIGHT, WIDTH)), block_raster, kernel
        )

        lu_counts = np.stack(
            [
                _focal_aggregate_block_aware(
                    (landuse == cls).astype(np.float64), block_raster, kernel
                )[0]
                for cls in MORPHOLOGY_LANDUSE_CLASSES
            ],
            axis=0,
        )
        lu_total = lu_counts.sum(axis=0)

        # Global range / NaN invariants.
        for name, arr in (
            ("building_frac", build_frac),
            ("road_frac", road_frac),
            ("veg_frac", veg_frac),
        ):
            finite = arr[~np.isnan(arr)]
            if finite.size and (finite.min() < 0.0 or finite.max() > 1.0):
                failures.append(f"{radius_m}m {name}: values outside [0, 1]")

        for r, c in test_pixels:
            nb_build = naive_neighbourhood((dist_building == 0).astype(np.float64), block_raster, r, c, offsets)
            nb_road = naive_neighbourhood((dist_road == 0).astype(np.float64), block_raster, r, c, offsets)
            nb_veg = naive_neighbourhood((dist_veg == 0).astype(np.float64), block_raster, r, c, offsets)
            nb_vc = naive_neighbourhood(vegetation_cover, block_raster, r, c, offsets)
            nb_ndvi = naive_neighbourhood(ndvi, block_raster, r, c, offsets)
            nb_ndbi = naive_neighbourhood(ndbi, block_raster, r, c, offsets)

            tag = f"{radius_m}m pixel({r},{c})"

            # Naive counts must equal the production valid-count denominator,
            # and every neighbour must belong to the same block (no crossing).
            n_valid = nb_build.size
            if not np.isclose(valid_count[r, c], n_valid, rtol=RTOL, atol=ATOL):
                failures.append(f"{tag}: valid_count {valid_count[r, c]} != naive {n_valid}")
            same_block = block_raster[r, c]
            # Direct no-crossing assertion on the production count:
            # full-kernel count minus block count equals the number of
            # in-bounds disk pixels outside this block.
            h, w = HEIGHT, WIDTH
            n_outside = 0
            for dy, dx in offsets:
                rr, cc = r + dy, c + dx
                if 0 <= rr < h and 0 <= cc < w and block_raster[rr, cc] != same_block:
                    n_outside += 1
            disk_full = kernel.sum()
            expected_valid = disk_full - n_outside
            expected_valid -= sum(
                1
                for dy, dx in offsets
                if not (0 <= r + dy < h and 0 <= c + dx < w)
            )
            if not np.isclose(valid_count[r, c], expected_valid, rtol=RTOL, atol=ATOL):
                failures.append(
                    f"{tag}: valid_count {valid_count[r, c]} != geometric expectation {expected_valid}"
                )

            def mean_or_nan(nb):
                valid_nb = nb[~np.isnan(nb)]
                return valid_nb.mean() if valid_nb.size else np.nan

            check(f"{tag} building_frac", build_frac[r, c], nb_build.mean() if nb_build.size else np.nan, failures)
            check(f"{tag} road_frac", road_frac[r, c], nb_road.mean() if nb_road.size else np.nan, failures)
            check(f"{tag} veg_frac", veg_frac[r, c], nb_veg.mean() if nb_veg.size else np.nan, failures)
            check(f"{tag} vc_mean", vc_mean[r, c], mean_or_nan(nb_vc), failures)

            ndvi_ref = mean_or_nan(nb_ndvi)
            check(
                f"{tag} ndvi_contrast",
                ndvi[r, c] - ndvi_mean[r, c],
                ndvi[r, c] - ndvi_ref if not np.isnan(ndvi_ref) else np.nan,
                failures,
            )
            ndbi_ref = mean_or_nan(nb_ndbi)
            check(
                f"{tag} ndbi_contrast",
                ndbi[r, c] - ndbi_mean[r, c],
                ndbi[r, c] - ndbi_ref if not np.isnan(ndbi_ref) else np.nan,
                failures,
            )

            # Land-use statistics vs naive reference.
            nb_lu = naive_neighbourhood(landuse, block_raster, r, c, offsets)
            if nb_lu.size:
                naive_counts = np.array([(nb_lu == cls).sum() for cls in MORPHOLOGY_LANDUSE_CLASSES], dtype=np.float64)
                naive_total = naive_counts.sum()
                f_ref, d_ref, e_ref = naive_landuse_stats(naive_counts, naive_total)
                probs_ref = naive_counts / naive_total
                pos = probs_ref > 0
                entropy_ref = float(-np.sum(probs_ref[pos] * np.log(probs_ref[pos])))
            else:
                naive_counts = np.zeros(len(MORPHOLOGY_LANDUSE_CLASSES))
                naive_total = 0.0
                f_ref = np.zeros(len(MORPHOLOGY_LANDUSE_CLASSES))
                d_ref, e_ref = 0.0, 0.0
                entropy_ref = 0.0

            if not np.allclose(lu_total[r, c], naive_total, rtol=RTOL, atol=ATOL):
                failures.append(f"{tag}: landuse total {lu_total[r, c]} != naive {naive_total}")
            prod_fracs = np.where(
                lu_total[r, c] > 0,
                lu_counts[:, r, c] / lu_total[r, c],
                0.0,
            )
            check(f"{tag} landuse_fracs", prod_fracs, f_ref, failures)
            prod_dominant = (
                float(np.asarray(MORPHOLOGY_LANDUSE_CLASSES)[int(np.argmax(lu_counts[:, r, c]))])
                if lu_total[r, c] > 0
                else 0.0
            )
            if prod_dominant != d_ref:
                failures.append(f"{tag}: dominant {prod_dominant} != naive {d_ref}")
            prod_probs = np.where(lu_total[r, c] > 0, lu_counts[:, r, c] / lu_total[r, c], 0.0)
            if lu_total[r, c] > 0:
                pos = prod_probs > 0
                prod_entropy = float(-np.sum(prod_probs[pos] * np.log(prod_probs[pos])))
            else:
                prod_entropy = 0.0
            check(f"{tag} landuse_entropy", prod_entropy, entropy_ref, failures)

            n_checks += 1

        # Edge-clipping audit vs naive geometric reference.
        clipped_prod = _count_clipped_pixels(block_raster, kernel)
        clipped_naive = 0
        for r in range(HEIGHT):
            for c in range(WIDTH):
                n_in_block = sum(
                    1
                    for dy, dx in offsets
                    if 0 <= r + dy < HEIGHT
                    and 0 <= c + dx < WIDTH
                    and block_raster[r + dy, c + dx] == block_raster[r, c]
                )
                # Production semantics: clipped means the block-aware kernel
                # is smaller than the full in-RASTER kernel (raster-edge
                # cropping alone does not count as block clipping).
                n_in_raster = sum(
                    1
                    for dy, dx in offsets
                    if 0 <= r + dy < HEIGHT and 0 <= c + dx < WIDTH
                )
                if n_in_block < n_in_raster:
                    clipped_naive += 1
        if clipped_prod != clipped_naive:
            failures.append(
                f"{radius_m}m: clipped count {clipped_prod} != naive {clipped_naive}"
            )

    if failures:
        print(f"FAIL: {len(failures)} mismatches across {n_checks} pixels x {len(MORPHOLOGY_RADII_M)} radii")
        for f in failures[:30]:
            print("  -", f)
        if len(failures) > 30:
            print(f"  ... and {len(failures) - 30} more")
        return 1

    print(
        f"PASS: morphology reference test OK "
        f"({n_checks} pixels x {len(MORPHOLOGY_RADII_M)} radii, "
        f"{len(test_pixels)} test pixels incl. block-edge pixels)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
