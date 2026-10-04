"""
GreenGrid-AI — V2 Phase 2
Task 2: OSM vector constraint layers (water, road surfaces, building footprints)

Purpose: Phase 7 V2 needs explicit planting-constraint layers for full Delhi NCT
(bbox south=28.40, west=76.83, north=28.89, east=77.35).

Layers:
  1. Water polygons        -> constraints/osm_water_delhi.geojson
  2. Road surfaces (area)  -> constraints/osm_road_surfaces_delhi.geojson
  3. Building footprints   -> constraints/osm_buildings_delhi.geojson
     (or osm_buildings_delhi_partial.geojson if some sub-bboxes fail)

Overpass endpoint rotation mirrors V1's `src/data_collection/download_gis_data.py`
(read-only reuse of its pattern; that file is NOT modified). Queries use
`out geom` so every element carries its own coordinates and no node assembly
is needed. Raw responses are cached under constraints/raw_cache/ — reruns do
not refetch.

Usage:
    PYTHONUTF8=1 PYTHONPATH=src ./.venv/Scripts/python.exe \
        src/v2/phase2/fetch_osm_constraints.py
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import requests
from shapely.geometry import MultiLineString, MultiPolygon, LineString, Polygon, box
from shapely.ops import unary_union

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

CONSTRAINTS_DIR = PROJECT_ROOT / "data" / "v2" / "phase2" / "constraints"
RAW_CACHE = CONSTRAINTS_DIR / "raw_cache"
PROV_MD = CONSTRAINTS_DIR / "CONSTRAINTS_PROVENANCE.md"

# V1 download_gis_data.py values (read-only reuse — do not modify V1)
BBOX = (28.40, 76.83, 28.89, 77.35)  # south, west, north, east
SOUTH, WEST, NORTH, EAST = BBOX
BBOX_STR = f"{SOUTH},{WEST},{NORTH},{EAST}"
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]
HEADERS = {
    "User-Agent": "GreenGridAI/1.0 (student research project; contact: greengrid-ai)",
    "Content-Type": "application/x-www-form-urlencoded",
}

POLITE_SLEEP_S = 3.0
GRID_N = 3  # 3x3 sub-bbox fallback grid for buildings

WATER_TAGS = ["natural", "water", "landuse", "waterway", "name"]
ROAD_TAGS = ["highway", "area", "name"]
BUILDING_TAGS = ["building", "name"]

WATER_QUERY = f"""[out:json][timeout:180][bbox:{BBOX_STR}];
(
  way["natural"="water"];
  relation["natural"="water"];
  way["water"];
  relation["water"];
  way["landuse"="reservoir"];
  relation["landuse"="reservoir"];
  way["waterway"~"^(riverbank|canal)$"];
  relation["waterway"~"^(riverbank|canal)$"];
);
out geom;
"""

ROAD_SURFACES_QUERY = f"""[out:json][timeout:120][bbox:{BBOX_STR}];
way["highway"]["area"="yes"];
out geom;
"""

BUILDINGS_QUERY_TEMPLATE = """[out:json][timeout:{timeout}][bbox:{bbox}];
(
  way["building"];
  relation["building"];
);
out geom;
"""


# ─── Overpass fetch with mirror rotation + raw cache ────────────────────────

def overpass_fetch(query, cache_path, label, timeout_s, attempts_per_endpoint=3):
    """
    POST the query to each mirror in V1's order; retry with backoff.
    Caches the raw response bytes on success. Returns (parsed_json, endpoint_used)
    or (None, None).
    """
    if cache_path.exists() and cache_path.stat().st_size > 500:
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f), "cache"
        except Exception:
            pass  # corrupted cache -> refetch

    for endpoint in OVERPASS_ENDPOINTS:
        for attempt in range(1, attempts_per_endpoint + 1):
            print(f"  [QUERYING] {label} via {endpoint.split('/')[2]} "
                  f"(attempt {attempt}/{attempts_per_endpoint})", flush=True)
            try:
                r = requests.post(endpoint, data={"data": query},
                                  headers=HEADERS, timeout=timeout_s)
                r.raise_for_status()
                parsed = r.json()
                if "elements" not in parsed:
                    raise ValueError("response missing 'elements' key")
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(r.text)
                print(f"  [OK] {label}: {len(parsed['elements'])} elements, "
                      f"{len(r.content) / 1e6:.1f} MB -> {cache_path.name}")
                time.sleep(POLITE_SLEEP_S)
                return parsed, endpoint
            except Exception as e:
                print(f"  [WARN] {label} {endpoint.split('/')[2]} attempt {attempt}: "
                      f"{type(e).__name__}: {str(e)[:200]}")
                if attempt < attempts_per_endpoint:
                    time.sleep(5 * attempt)
    return None, None


# ─── Overpass `out geom` JSON -> shapely geometries ─────────────────────────

def _chain_rings(segments):
    """Join way segments end-to-end into maximal chains; closed chains = rings."""
    segs = [list(s) for s in segments if len(s) >= 2]
    chains = []
    while segs:
        chain = segs.pop(0)
        extended = True
        while extended:
            extended = False
            for i, s in enumerate(segs):
                if chain[-1] == s[0]:
                    chain += s[1:]
                elif chain[-1] == s[-1]:
                    chain += s[-2::-1]
                elif chain[0] == s[-1]:
                    chain = s[:-1] + chain
                elif chain[0] == s[0]:
                    chain = s[1:][::-1] + chain
                else:
                    continue
                segs.pop(i)
                extended = True
                break
        chains.append(chain)
    return chains


def _make_polygon(shell, holes):
    poly = Polygon(shell, holes)
    if not poly.is_valid:
        fixed = poly.buffer(0)
        if fixed.is_empty:
            return None
        if fixed.geom_type in ("Polygon", "MultiPolygon"):
            return fixed
        return None
    return poly


def _relation_polygon(members):
    """Assemble multipolygon relation members (roles outer/inner) into a geometry."""
    outers_raw, inners_raw = [], []
    for m in members:
        if m.get("type") != "way" or "geometry" not in m:
            continue
        pts = [(p["lon"], p["lat"]) for p in m["geometry"]]
        role = m.get("role", "")
        (outers_raw if role != "inner" else inners_raw).append(pts)

    outer_chains = _chain_rings(outers_raw)
    inner_chains = _chain_rings(inners_raw)

    outer_polys = []
    for ch in outer_chains:
        if len(ch) >= 4 and ch[0] == ch[-1]:
            p = _make_polygon(ch, [])
            if p is not None:
                outer_polys.append(p)
    inner_rings = [ch for ch in inner_chains
                   if len(ch) >= 4 and ch[0] == ch[-1]]

    if not outer_polys:
        leftovers = [ch for ch in outer_chains if len(ch) >= 2]
        if not leftovers:
            return None
        ls = [LineString(c) for c in leftovers]
        return ls[0] if len(ls) == 1 else MultiLineString(ls)

    # Assign inner rings to outers by containment (ring representative point).
    holes_per_outer = [[] for _ in outer_polys]
    for ring in inner_rings:
        pt = Polygon(ring).representative_point()
        for i, op in enumerate(outer_polys):
            if op.covers(pt):
                holes_per_outer[i].append(ring)
                break

    polys = []
    for op, holes in zip(outer_polys, holes_per_outer):
        if holes and op.geom_type == "Polygon":
            p = _make_polygon(list(op.exterior.coords), holes)
        else:
            p = op  # holes on rebuilt multipolygon outers are dropped (rare)
        if p is not None:
            polys.extend([p] if p.geom_type == "Polygon" else list(p.geoms))
    if not polys:
        return None
    return polys[0] if len(polys) == 1 else MultiPolygon(polys)


def elements_to_features(elements, keep_tags):
    """Convert Overpass elements to [(geom, props), ...]; dedupe by (osm_type, osm_id)."""
    feats = {}
    n_bad = 0
    for el in elements:
        etype = el.get("type")
        eid = el.get("id")
        if etype not in ("way", "relation") or eid is None:
            continue
        geom = None
        if etype == "way":
            pts = [(p["lon"], p["lat"]) for p in el.get("geometry", [])]
            if len(pts) >= 4 and pts[0] == pts[-1]:
                geom = _make_polygon(pts, [])
            elif len(pts) >= 2:
                geom = LineString(pts)
        else:  # relation
            geom = _relation_polygon(el.get("members", []))
        if geom is None or geom.is_empty:
            n_bad += 1
            continue
        props = {"osm_type": etype, "osm_id": int(eid)}
        tags = el.get("tags", {})
        for t in keep_tags:
            if t in tags:
                props[t] = tags[t]
        feats[(etype, int(eid))] = (geom, props)
    if n_bad:
        print(f"  [WARN] {n_bad} elements had no/invalid geometry and were dropped")
    return list(feats.values())


def features_to_gdf(features):
    geoms = [g for g, _ in features]
    props = [p for _, p in features]
    return gpd.GeoDataFrame(props, geometry=geoms, crs="EPSG:4326")


def validate_and_clip(gdf, layer_name):
    """Check bbox proximity, clip to bbox+0.01deg, assert no null geometries."""
    assert not gdf.geometry.isna().any(), f"{layer_name}: null geometries present"
    expanded = box(WEST - 0.01, SOUTH - 0.01, EAST + 0.01, NORTH + 0.01)
    mask = gpd.GeoSeries([expanded], crs="EPSG:4326")
    outside = ~gdf.geometry.intersects(expanded)
    n_out = int(outside.sum())
    if n_out:
        print(f"  [WARN] {layer_name}: {n_out} features fully outside "
              f"bbox+0.01 buffer — will be dropped by clip")
    clipped = gpd.clip(gdf, mask)
    n_null = int(clipped.geometry.isna().sum())
    assert n_null == 0, f"{layer_name}: {n_null} null geometries after clip"
    return clipped, n_out


def area_km2(gdf):
    polys = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    if len(polys) == 0:
        return 0.0
    return float(polys.to_crs("EPSG:32643").area.sum() / 1e6)


def write_layer(gdf, path):
    gdf.to_file(path, driver="GeoJSON")
    return path.stat().st_size


# ─── Sub-bbox grid for buildings fallback ───────────────────────────────────

def sub_bboxes():
    cells = []
    for r in range(GRID_N):
        for c in range(GRID_N):
            s = SOUTH + r * (NORTH - SOUTH) / GRID_N
            n = SOUTH + (r + 1) * (NORTH - SOUTH) / GRID_N
            w = WEST + c * (EAST - WEST) / GRID_N
            e = WEST + (c + 1) * (EAST - WEST) / GRID_N
            cells.append((r, c, f"{s:.5f},{w:.5f},{n:.5f},{e:.5f}"))
    return cells


# ─── Layer runners ──────────────────────────────────────────────────────────

def run_water():
    print("\n[WATER LAYER]")
    raw, endpoint = overpass_fetch(WATER_QUERY, RAW_CACHE / "water_raw.json",
                                   "water union", timeout_s=240)
    if raw is None:
        print("  [ERROR] water query failed on all mirrors")
        return None
    feats = elements_to_features(raw["elements"], WATER_TAGS)
    gdf = features_to_gdf(feats)
    gdf, n_out = validate_and_clip(gdf, "water")
    out = CONSTRAINTS_DIR / "osm_water_delhi.geojson"
    size = write_layer(gdf, out)
    return dict(layer="water", query=WATER_QUERY, endpoint=endpoint, gdf=gdf,
                out=out, size=size, n_outside=n_out, status="full")


def run_road_surfaces():
    print("\n[ROAD SURFACES LAYER]")
    raw, endpoint = overpass_fetch(ROAD_SURFACES_QUERY,
                                   RAW_CACHE / "road_surfaces_raw.json",
                                   "road surfaces (highway+area=yes)", timeout_s=180)
    if raw is None:
        print("  [ERROR] road surfaces query failed on all mirrors")
        return None
    feats = elements_to_features(raw["elements"], ROAD_TAGS)
    gdf = features_to_gdf(feats)
    gdf, n_out = validate_and_clip(gdf, "road_surfaces")
    out = CONSTRAINTS_DIR / "osm_road_surfaces_delhi.geojson"
    size = write_layer(gdf, out)
    return dict(layer="road_surfaces", query=ROAD_SURFACES_QUERY, endpoint=endpoint,
                gdf=gdf, out=out, size=size, n_outside=n_out, status="full")


def run_buildings():
    print("\n[BUILDINGS LAYER]")
    full_query = BUILDINGS_QUERY_TEMPLATE.format(timeout=600, bbox=BBOX_STR)
    raw, endpoint = overpass_fetch(full_query, RAW_CACHE / "buildings_full_raw.json",
                                   "buildings FULL bbox", timeout_s=660,
                                   attempts_per_endpoint=1)
    cell_failures = []
    if raw is not None:
        feats = elements_to_features(raw["elements"], BUILDING_TAGS)
        source_desc = "single full-bbox query"
        fetch_desc = f"full bbox, endpoint={endpoint}"
    else:
        print("  [FALLBACK] full bbox failed on all mirrors — trying 3x3 sub-bbox grid")
        feats = []
        merged = {}
        for r, c, bb in sub_bboxes():
            q = BUILDINGS_QUERY_TEMPLATE.format(timeout=600, bbox=bb)
            cache = RAW_CACHE / f"buildings_cell_r{r}c{c}_raw.json"
            cell_raw, cell_endpoint = overpass_fetch(
                q, cache, f"buildings cell r{r}c{c} [{bb}]",
                timeout_s=660, attempts_per_endpoint=3)
            if cell_raw is None:
                cell_failures.append(dict(r=r, c=c, bbox=bb))
                print(f"  [FAIL] cell r{r}c{c} [{bb}] failed on all mirrors/retries")
                continue
            cell_feats = elements_to_features(cell_raw["elements"], BUILDING_TAGS)
            for g, p in cell_feats:
                merged[(p["osm_type"], p["osm_id"])] = (g, p)
            print(f"  [OK] cell r{r}c{c}: running unique total = {len(merged):,}")
        feats = list(merged.values())
        source_desc = "3x3 sub-bbox grid (full bbox failed on all mirrors)"
        fetch_desc = f"9 sub-bboxes, failures={[f['r'] for f in cell_failures]}"

    gdf = features_to_gdf(feats)
    status = "full" if not cell_failures else "partial"
    if cell_failures:
        out = CONSTRAINTS_DIR / "osm_buildings_delhi_partial.geojson"
    else:
        out = CONSTRAINTS_DIR / "osm_buildings_delhi.geojson"
    gdf, n_out = validate_and_clip(gdf, "buildings")
    size = write_layer(gdf, out)
    return dict(layer="buildings", query=full_query if not cell_failures else
                BUILDINGS_QUERY_TEMPLATE.format(timeout=600, bbox="<per-cell>"),
                endpoint=endpoint if raw is not None else "multiple (sub-bbox grid)",
                gdf=gdf, out=out, size=size, n_outside=n_out, status=status,
                cell_failures=cell_failures, source_desc=source_desc,
                fetch_desc=fetch_desc)


# ─── Provenance ─────────────────────────────────────────────────────────────

def write_provenance(results, fetch_ts):
    lines = [
        "# OSM Constraint Layers (V2 Phase 2) — Provenance",
        "",
        f"**Script:** `src/v2/phase2/fetch_osm_constraints.py`",
        f"**Fetch timestamp (UTC):** {fetch_ts}",
        f"**Study bbox (south,west,north,east):** `{BBOX_STR}` — full Delhi NCT (same bbox as V1 `download_gis_data.py`).",
        "**License:** OpenStreetMap contributors, **ODbL 1.0** (https://www.openstreetmap.org/copyright).",
        "Attribution required: “© OpenStreetMap contributors”. Produced from raw OSM data via the Overpass API.",
        "",
        "Known limitations (all layers): OSM completeness varies by neighbourhood and tagger practice;",
        "constraint layers are presence/absence masks, not authoritative inventories.",
        "Road LINES already exist in V1 at `data/raw/gis/roads/` (read-only; not modified).",
        "V1 building SAMPLE at `data/raw/gis/buildings/delhi_buildings_sample_osm.json` is a",
        "Central-Delhi sample only — the layer below aims for the full bbox.",
        "",
        "Each geometry was validated: loads with geopandas, 0 null geometries, clipped to",
        "bbox + 0.01° buffer; features fully outside that buffer were dropped and counted below",
        "(edge-crossing features were clipped, not deleted).",
        "",
    ]
    for res in results:
        gdf = res["gdf"]
        gtypes = gdf.geometry.geom_type.value_counts().to_dict()
        n_invalid = int((~gdf.geometry.is_valid).sum())
        lines += [
            f"## {res['layer']} — `{res['out'].name}`",
            "",
            "**Query executed (verbatim):**",
            "```",
            res["query"].rstrip(),
            "```",
            f"- **Endpoint(s) used:** {res['endpoint']}",
            f"- **Fetch mode:** {res.get('source_desc', 'single bbox query')} "
            f"({res.get('fetch_desc', 'full study bbox')})",
            f"- **Status:** {res['status'].upper()}" +
            (f" — failed sub-bboxes: {[(f['r'], f['c'], f['bbox']) for f in res['cell_failures']]}"
             if res.get('cell_failures') else ""),
            f"- **Feature count:** {len(gdf):,}",
            f"- **Geometry types:** {gtypes}",
            f"- **Invalid geometries (kept as-fetched; OSM ring quirks, see limitations):** "
            f"{n_invalid:,}",
            f"- **Total area:** {area_km2(gdf):,.2f} km² (EPSG:32643, polygonal geometries only)",
            f"- **Size on disk:** {res['size'] / 1e6:.1f} MB",
            f"- **Features fully outside bbox+0.01° (dropped by clip):** {res['n_outside']}",
            "",
            "**License:** ODbL 1.0 — © OpenStreetMap contributors (https://www.openstreetmap.org/copyright).",
            "",
        ]
        if res["layer"] == "road_surfaces":
            lines += [
                "**Note:** road *surfaces* are the rare polygon-represented roads (`highway=*` + `area=yes`).",
                "Most Delhi roads are mapped as lines and live in V1 `data/raw/gis/roads/` —",
                "a small or empty count here is the honest result, not a failure.",
                "",
            ]
        if res["layer"] == "buildings" and res["status"] == "partial":
            lines += [
                "**Partial-success disclosure:** some 3×3 sub-bboxes failed after all mirror retries;",
                "this file contains only the successful cells. Re-run the script to retry",
                "(successful cells are served from `raw_cache/` and are not refetched).",
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
    for runner in (run_water, run_road_surfaces, run_buildings):
        res = runner()
        if res is not None:
            results.append(res)

    write_provenance(results, fetch_ts)

    print("\n" + "=" * 60)
    print("SUMMARY")
    for res in results:
        print(f"  {res['layer']:14s} {res['status']:8s} {len(res['gdf']):>9,} feats  "
              f"{res['size'] / 1e6:8.1f} MB  -> {res['out'].name}")
    if any(r.get("cell_failures") for r in results):
        print("  !! partial building fetch — see CONSTRAINTS_PROVENANCE.md")
    print(f"[DONE] in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
