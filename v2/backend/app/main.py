import os
import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from rio_tiler.io import Reader
from rio_tiler.profiles import img_profiles
from rio_tiler.colormap import cmap

app = FastAPI(title="GreenGrid AI V2 API")

# Allow CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))

def get_raster_path(phase: str, layer: str, year: int) -> str:
    if phase == "phase3":
        return os.path.join(DATA_DIR, "phase3", str(year), f"{layer}_30m.tif")
    elif phase == "phase6":
        return os.path.join(DATA_DIR, "phase6", "rasters", f"{layer}_{year}.tif")
    elif phase == "phase7":
        # layer carries the scenario, e.g. "priority_class_v2_constrained"
        return os.path.join(DATA_DIR, "phase7", layer.rsplit("_", 2)[-2], "rasters", f"{layer}_{year}.tif")
    raise HTTPException(status_code=400, detail="Invalid phase")

@app.get("/api/metadata")
def get_metadata():
    return {
        "years": [2022, 2023, 2024, 2025, 2026],
        "scenarios": ["v1_parity", "v2_constrained"]
    }

@app.get("/api/phase7/zones/{year}/{scenario}")
def get_priority_zones(year: int, scenario: str):
    """Per-year planting-land patches (Phase 8 geojson; Phase 7 v3 scores the
    full area, zones are planting-land patches, not heat zones)."""
    path = os.path.join(DATA_DIR, "phase8", scenario, "vectors", f"recommended_plantations_{scenario}_{year}.geojson")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Data not found")
    with open(path, "r") as f:
        return json.load(f)

@app.get("/api/phase8/capacity/{year}/{scenario}")
def get_capacity(year: int, scenario: str):
    """Theoretical eligible capacity (transparency only - not a recommendation)."""
    path = os.path.join(DATA_DIR, "phase8", scenario, "tables", f"theoretical_capacity_{scenario}_{year}.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Data not found")
    df = pd.read_csv(path)
    return df.to_dict(orient="records")

@app.get("/api/phase7/class_shares/{year}/{scenario}")
def get_class_shares(year: int, scenario: str):
    """Full-area priority class shares (fixed pooled thresholds)."""
    path = os.path.join(DATA_DIR, "phase7", scenario, "tables", f"class_shares_{scenario}_{year}.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Data not found")
    df = pd.read_csv(path)
    return df.to_dict(orient="records")

@app.get("/api/phase7/exclusions/{year}/{scenario}")
def get_exclusions(year: int, scenario: str):
    """Per-reason exclusion accounting (ha), partitioning the valid domain."""
    path = os.path.join(DATA_DIR, "phase7", scenario, "tables", f"exclusion_accounting_{scenario}_{year}.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Data not found")
    df = pd.read_csv(path)
    return df.to_dict(orient="records")

@app.get("/api/phase8/summary/{year}/{scenario}")
def get_tree_requirement_summary(year: int, scenario: str):
    path = os.path.join(DATA_DIR, "phase8", scenario, "tables", f"tree_requirement_summary_{scenario}_{year}.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Data not found")
    df = pd.read_csv(path)
    return df.to_dict(orient="records")

@app.get("/api/phase8/zones/{year}/{scenario}")
def get_tree_requirement_zones(year: int, scenario: str):
    path = os.path.join(DATA_DIR, "phase8", scenario, "tables", f"tree_requirement_by_zone_{scenario}_{year}.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Data not found")
    df = pd.read_csv(path)
    return df.to_dict(orient="records")

@app.get("/api/boundary")
def get_study_area_boundary():
    """Primary dashboard outline: the data-extent boundary (vectorized envelope
    of all pixels valid in >=1 of 5 W4 years). This matches the rasters; the
    GEE exports were clipped to FAO/GAUL/2015 Delhi, not to the OSM polygon."""
    path = os.path.join(DATA_DIR, "gis", "study_area", "study_area_data_extent.geojson")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Boundary not found")
    with open(path, "r") as f:
        return json.load(f)


@app.get("/api/boundary/osm_reference")
def get_osm_reference_boundary():
    """Reference boundary: Delhi NCT per OSM relation 1942586. Differs from the
    data extent along the NCT fringe (~7,850 ha outside / ~5,230 ha inside).
    Served as an optional, unchecked reference layer only."""
    path = os.path.join(DATA_DIR, "gis", "study_area", "study_area.geojson")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Boundary not found")
    with open(path, "r") as f:
        return json.load(f)

# Hardcoded scientific rendering parameters.
# Reference ranges (documented, fixed across years):
#  - LST: pooled p0.1-p99.9 of valid 2022-2026 W4 Landsat-9 LST = 31.4-58.0 C.
#  - severity: categorical class raster {0=Low,1=Medium,2=High} (nodata 255);
#    rendered with an EXACT discrete colormap, no rescale.
LAYER_CONFIGS = {
    "lst": {"rescale": ((31.4, 58.0),), "colormap_name": "inferno"},
    "ndvi": {"rescale": ((-0.4, 0.9),), "colormap_name": "rdylgn"},  # actual 2022-2026 W4 range -0.37..0.90
    "ndbi": {"rescale": ((-0.5, 0.5),), "colormap_name": "rdbu_r"},
    "vegetation_cover": {"rescale": ((0, 1),), "colormap_name": "greens"},
    "severity": {"colormap_dict": {0: (255, 255, 204, 255), 1: (253, 141, 60, 255), 2: (128, 0, 38, 255)}},
    "severity_score": {"rescale": ((0, 2),), "colormap_name": "ylorrd"},
    "confidence": {"rescale": ((0, 1),), "colormap_name": "blues"},
    "probability": {"rescale": ((0, 1),), "colormap_name": "purples"}
}

@app.get("/api/phase8/vectors/{layer}/{year}/{scenario}")
def get_tree_vectors(layer: str, year: int, scenario: str):
    # layer should be "recommended_locations" or "recommended_plantations"
    path = os.path.join(DATA_DIR, "phase8", scenario, "vectors", f"{layer}_{scenario}_{year}.geojson")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Data not found")
    with open(path, "r") as f:
        return json.load(f)

# Tile endpoint using rio-tiler
@app.get("/api/tiles/{phase}/{layer}/{year}/{z}/{x}/{y}.png")
def get_tile(phase: str, layer: str, year: int, z: int, x: int, y: int):
    raster_path = get_raster_path(phase, layer, year)
    if not os.path.exists(raster_path):
        raise HTTPException(status_code=404, detail="Raster not found")
    
    try:
        with Reader(raster_path) as src:
            img = src.tile(x, y, z)
            
            # Setup rendering options based on scientific config
            render_kwargs = {"img_format": "PNG"}
            config = LAYER_CONFIGS.get(layer)
            
            if config:
                if "rescale" in config:
                    img.rescale(in_range=config["rescale"])
                if "colormap_dict" in config:
                    # Exact discrete class colors; raster values must pass through
                    # unrescaled so class ids map directly to legend colors.
                    render_kwargs["colormap"] = config["colormap_dict"]
                elif "colormap_name" in config:
                    cm = cmap.get(config["colormap_name"])
                    render_kwargs["colormap"] = cm
                    
            image_buffer = img.render(**render_kwargs)
            return Response(content=image_buffer, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
