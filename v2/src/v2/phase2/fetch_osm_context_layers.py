"""
GreenGrid-AI — V2 Phase 2
Task 2b: OSM vector CONTEXT layers for the V2 pipeline (road lines, vegetation, landuse)

Purpose: the V2 pipeline must never read V1 files, so it needs its OWN copies of
three OSM context layers for full Delhi NCT
(bbox south=28.40, west=76.83, north=28.89, east=77.35):

  1. Road lines   -> constraints/osm_roads_delhi.geojson
                     (or osm_roads_major_delhi.geojson on major-roads fallback)
  2. Vegetation   -> constraints/osm_vegetation_delhi.geojson
  3. Landuse      -> constraints/osm_landuse_delhi.geojson

This script reuses (read-only, no modification) the mirror rotation, Overpass
`out geom` JSON->geometry assembler, bbox clip and provenance patterns from
`fetch_osm_constraints.py`. Raw responses are cached under constraints/raw_cache/
with distinct file names — reruns do not refetch. V1 files are read-only and are
NOT touched.

Usage:
    PYTHONUTF8=1 PYTHONPATH=src ./.venv/Scripts/python.exe \
        src/v2/phase2/fetch_osm_context_layers.py
"""

import contextlib
import io
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
from shapely.geometry import LineString

# Reuse the battle-tested helpers from the constraints fetcher (read-only).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_osm_constraints import (  # noqa: E402
    BBOX_STR,
    CONSTRAINTS_DIR,
    RAW_CACHE,
    area_km2,
    elements_to_features,
    features_to_gdf,
    overpass_fetch,
    validate_and_clip,
    write_layer,
)

PROV_MD = CONSTRAINTS_DIR / "CONTEXT_LAYERS_PROVENANCE.md"

# Tag sets kept as properties on each feature
ROAD_TAGS = ["highway", "name", "ref"]
VEGETATION_TAGS = ["leisure", "natural", "landuse", "name"]
LANDUSE_TAGS = ["landuse", "natural", "name"]

ROADS_QUERY = f"""[out:json][timeout:240][bbox:{BBOX_STR}];
way["highway"];
out geom;
"""

# V1-equivalent major-road classes (motorway/trunk/primary/secondary + _link
# variants), used ONLY if the full highway=* query fails on all mirrors.
ROADS_MAJOR_QUERY = f"""[out:json][timeout:120][bbox:{BBOX_STR}];
way["highway"~"^(motorway|trunk|primary|secondary)(_link)?$"];
out geom;
"""

VEGETATION_QUERY = f"""[out:json][timeout:180][bbox:{BBOX_STR}];
(
  way["leisure"~"^(park|garden|nature_reserve)$"];
  relation["leisure"~"^(park|garden|nature_reserve)$"];
  way["natural"~"^(wood|scrub)$"];
  relation["natural"~"^(wood|scrub)$"];
  way["landuse"~"^(forest|grass|meadow|recreation_ground)$"];
  relation["landuse"~"^(forest|grass|meadow|recreation_ground)$"];
);
out geom;
"""

LANDUSE_QUERY = f"""[out:json][timeout:180][bbox:{BBOX_STR}];
(
  way["landuse"];
  relation["landuse"];
  way["natural"];
  relation["natural"];
);
out geom;
"""

MIRROR_FAILURES = []  # (layer, endpoint host, error text) parsed from fetch log


# ─── Fetch wrapper: reuse overpass_fetch, capture per-attempt failures ──────

def fetch(query, cache_path, label, timeout_s, layer, attempts_per_endpoint=3):
    """
    Call fetch_osm_constraints.overpass_fetch (mirror rotation + raw cache +
    backoff) while capturing its progress output so mirror failures can be
    recorded in the provenance file. Returns (parsed_json, endpoint_used).
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        raw, endpoint = overpass_fetch(query, cache_path, label, timeout_s,
                                       attempts_per_endpoint=attempts_per_endpoint)
    log = buf.getvalue()
    sys.stdout.write(log)
    for m in re.finditer(r"\[WARN\] (.+?) (\S+) attempt \d+: ([\w.]+): (.+)", log):
        MIRROR_FAILURES.append(dict(layer=layer, endpoint=m.group(2),
                                    error=f"{m.group(3)}: {m.group(4)[:200]}"))
    return raw, endpoint


# ─── Overpass `out geom` JSON -> shapely LineStrings (roads) ────────────────

def road_elements_to_lines(elements, keep_tags):
    """
    Roads are ways-only here. Every way becomes a LineString (closed ways keep
    their duplicate endpoint — valid LineString). Dedupes by (type, id).
    Returns (features, n_bad_geom, n_nonway_skipped).
    """
    feats = {}
    n_bad = 0
    n_nonway = 0
    for el in elements:
        if el.get("type") != "way":
            n_nonway += 1
            continue
        eid = el.get("id")
        pts = [(p["lon"], p["lat"]) for p in el.get("geometry", [])]
        if eid is None or len(pts) < 2:
            n_bad += 1
            continue
        props = {"osm_type": "way", "osm_id": int(eid)}
        tags = el.get("tags", {})
        for t in keep_tags:
            if t in tags:
                props[t] = tags[t]
        feats[("way", int(eid))] = (LineString(pts), props)
    return list(feats.values()), n_bad, n_nonway


# ─── Post-write verification ────────────────────────────────────────────────

def verify_on_disk(path, expected_len):
    """Prove the deliverable reloads with geopandas and has 0 null geometries."""
    check = gpd.read_file(path)
    assert len(check) == expected_len, f"{path.name}: reload count mismatch"
    assert not check.geometry.isna().any(), f"{path.name}: null geometries on reload"
    return len(check)


# ─── Layer runners ──────────────────────────────────────────────────────────

def run_roads():
    print("\n[ROAD LINES LAYER]")
    raw, endpoint = fetch(ROADS_QUERY, RAW_CACHE / "roads_lines_raw.json",
                          "road lines (highway=*)", timeout_s=300, layer="roads")
    query, status = ROADS_QUERY, "full"
    if raw is None:
        print("  [FALLBACK] full highway=* query failed on all 3 mirrors — "
              "falling back to V1-equivalent major roads "
              "(motorway/trunk/primary/secondary + _link)")
        raw, endpoint = fetch(ROADS_MAJOR_QUERY,
                              RAW_CACHE / "roads_lines_major_raw.json",
                              "road lines MAJOR (fallback)", timeout_s=180,
                              layer="roads")
        query, status = ROADS_MAJOR_QUERY, "fallback_major"
    if raw is None:
        print("  [ERROR] roads queries failed on all mirrors (full + fallback)")
        return None
    feats, n_bad, n_nonway = road_elements_to_lines(raw["elements"], ROAD_TAGS)
    if n_bad or n_nonway:
        print(f"  [WARN] roads: {n_bad} ways without geometry, "
              f"{n_nonway} non-way elements skipped")
    gdf = features_to_gdf(feats)
    gdf, n_out = validate_and_clip(gdf, "roads")
    out = CONSTRAINTS_DIR / ("osm_roads_major_delhi.geojson" if status == "fallback_major"
                             else "osm_roads_delhi.geojson")
    size = write_layer(gdf, out)
    verify_on_disk(out, len(gdf))
    return dict(layer="roads", query=query, endpoint=endpoint, gdf=gdf, out=out,
                size=size, n_outside=n_out, status=status,
                n_bad_geom=n_bad, n_nonway=n_nonway)


def run_vegetation():
    print("\n[VEGETATION LAYER]")
    raw, endpoint = fetch(VEGETATION_QUERY, RAW_CACHE / "vegetation_raw.json",
                          "vegetation union", timeout_s=240, layer="vegetation")
    if raw is None:
        print("  [ERROR] vegetation query failed on all mirrors")
        return None
    feats = elements_to_features(raw["elements"], VEGETATION_TAGS)  # dedupes by (type,id)
    gdf = features_to_gdf(feats)
    gdf, n_out = validate_and_clip(gdf, "vegetation")
    out = CONSTRAINTS_DIR / "osm_vegetation_delhi.geojson"
    size = write_layer(gdf, out)
    verify_on_disk(out, len(gdf))
    return dict(layer="vegetation", query=VEGETATION_QUERY, endpoint=endpoint,
                gdf=gdf, out=out, size=size, n_outside=n_out, status="full")


def run_landuse():
    print("\n[LANDUSE LAYER]")
    raw, endpoint = fetch(LANDUSE_QUERY, RAW_CACHE / "landuse_raw.json",
                          "landuse+natural union", timeout_s=240, layer="landuse")
    if raw is None:
        print("  [ERROR] landuse query failed on all mirrors")
        return None
    feats = elements_to_features(raw["elements"], LANDUSE_TAGS)  # dedupes by (type,id)
    gdf = features_to_gdf(feats)
    gdf, n_out = validate_and_clip(gdf, "landuse")
    out = CONSTRAINTS_DIR / "osm_landuse_delhi.geojson"
    size = write_layer(gdf, out)
    verify_on_disk(out, len(gdf))
    return dict(layer="landuse", query=LANDUSE_QUERY, endpoint=endpoint,
                gdf=gdf, out=out, size=size, n_outside=n_out, status="full")


# ─── Provenance ─────────────────────────────────────────────────────────────

def write_provenance(results, fetch_ts):
    lines = [
        "# OSM Context Layers (V2 Phase 2) — Provenance",
        "",
        "**Script:** `src/v2/phase2/fetch_osm_context_layers.py` (reuses read-only the",
        "mirror rotation, `out geom` assembler, bbox clip and provenance patterns of",
        "`src/v2/phase2/fetch_osm_constraints.py`; neither V1 files nor that script are modified).",
        "",
        f"**Fetch timestamp (UTC):** {fetch_ts}",
        f"**Study bbox (south,west,north,east):** `{BBOX_STR}` — full Delhi NCT (same bbox as V1 `download_gis_data.py`).",
        "",
        "These are the V2 pipeline's OWN context layers (road lines, vegetation, landuse).",
        "The V2 pipeline reads these files and never reads the V1 copies under `data/raw/gis/`.",
        "",
        "**License (all layers):** OpenStreetMap contributors, **ODbL 1.0**",
        "(https://www.openstreetmap.org/copyright). Attribution required:",
        "“© OpenStreetMap contributors”. Produced from raw OSM data via the Overpass API.",
        "",
        "**Known limitations (all layers):** OSM completeness varies by neighbourhood and",
        "tagger practice; these are context masks, not authoritative inventories.",
        "Areas are computed in EPSG:32643 (UTM zone 43N) over polygonal geometries only.",
        "",
        "**Validation applied to every layer:** written GeoJSON reloads with geopandas,",
        "0 null geometries (checked before and after write), clipped to bbox + 0.01° buffer;",
        "features fully outside that buffer were dropped and counted below (edge-crossing",
        "features were clipped, not deleted). Invalid geometries are KEPT AS-FETCHED",
        "(OSM ring quirks are disclosed per layer below, not silently fixed).",
        "",
    ]
    if MIRROR_FAILURES:
        lines += ["## Mirror failures during this fetch", ""]
        for f in MIRROR_FAILURES:
            lines.append(f"- `{f['layer']}` via `{f['endpoint']}`: {f['error']}")
        lines.append("")
    for res in results:
        gdf = res["gdf"]
        gtypes = gdf.geometry.geom_type.value_counts().to_dict()
        n_invalid = int((~gdf.geometry.is_valid).sum())
        is_line_layer = gdf.geometry.geom_type.isin(["LineString", "MultiLineString"]).all()
        lines += [
            f"## {res['layer']} — `{res['out'].name}`",
            "",
            "**Query executed (verbatim):**",
            "```",
            res["query"].rstrip(),
            "```",
            f"- **Endpoint(s) used:** {res['endpoint']}",
            f"- **Status:** {res['status'].upper()}",
        ]
        if res["layer"] == "roads" and res["status"] == "fallback_major":
            lines += [
                "",
                "> ⚠️ **FALLBACK DISCLOSURE:** the full `highway=*` (all classes) query",
                "> failed on ALL 3 Overpass mirrors after all retries. This file contains ONLY",
                "> V1-equivalent major roads (motorway, trunk, primary, secondary and their",
                "> `_link` variants). Minor/tertiary/residential/unclassified roads are MISSING.",
                "> Re-run the script to retry the full query (the fallback response is cached",
                "> separately as `raw_cache/roads_lines_major_raw.json`).",
            ]
        extra = ""
        if res["layer"] == "roads":
            extra = (f" ({res['n_bad_geom']} ways w/o geometry, "
                     f"{res['n_nonway']} non-way elements skipped)")
        lines += [
            f"- **Feature count:** {len(gdf):,}{extra}",
            f"- **Geometry types:** {gtypes}",
            f"- **Invalid geometries (kept as-fetched):** {n_invalid:,}",
            "- **Total area:** "
            + ("n/a — line layer (no polygons)" if is_line_layer
               else f"{area_km2(gdf):,.2f} km² (EPSG:32643, polygonal geometries only)"),
            f"- **Size on disk:** {res['size'] / 1e6:.1f} MB",
            f"- **Features fully outside bbox+0.01° (dropped by clip):** {res['n_outside']}",
            "",
            "**License:** ODbL 1.0 — © OpenStreetMap contributors (https://www.openstreetmap.org/copyright).",
            "",
        ]
        if res["layer"] == "landuse":
            lines += [
                "**Overlap note:** this layer intentionally includes `natural=water` /",
                "`landuse=reservoir` / `water`-tagged features, which overlap with",
                "`osm_water_delhi.geojson` from the constraints fetch. Keeping them makes the",
                "union query simple (`landuse=*` OR `natural=*` with no exclusions); consumers",
                "that need a water-exclusive mask should intersect/difference against",
                "`osm_water_delhi.geojson`.",
                "",
            ]
        if res["layer"] == "vegetation":
            lines += [
                "**Note:** union of `leisure=park|garden|nature_reserve`, `natural=wood|scrub`,",
                "`landuse=forest|grass|meadow|recreation_ground`, deduped by (osm_type, osm_id).",
                "",
            ]
        if res["layer"] == "roads":
            lines += [
                "**Note:** road *lines* (centreline geometry). Road *surfaces* (`area=yes`) are a",
                "separate layer in `osm_road_surfaces_delhi.geojson`.",
                "",
            ]
    PROV_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[WRITE] {PROV_MD}")


def main():
    t0 = time.time()
    fetch_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    CONSTRAINTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_CACHE.mkdir(parents=True, exist_ok=True)

    results = []
    for runner in (run_roads, run_vegetation, run_landuse):
        res = runner()
        if res is not None:
            results.append(res)

    write_provenance(results, fetch_ts)

    print("\n" + "=" * 60)
    print("SUMMARY")
    for res in results:
        print(f"  {res['layer']:11s} {res['status']:15s} {len(res['gdf']):>9,} feats  "
              f"{res['size'] / 1e6:8.1f} MB  -> {res['out'].name}")
    if any(r["status"] == "fallback_major" for r in results):
        print("  !! roads used MAJOR-ROADS FALLBACK — see CONTEXT_LAYERS_PROVENANCE.md")
    if MIRROR_FAILURES:
        print(f"  !! {len(MIRROR_FAILURES)} mirror attempt(s) failed — see provenance")
    print(f"[DONE] in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
