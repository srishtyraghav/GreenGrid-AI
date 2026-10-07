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
    raise HTTPException(status_code=400, detail="Invalid phase")

@app.get("/api/metadata")
def get_metadata():
    return {
        "years": [2022, 2023, 2024, 2025, 2026],
        "scenarios": ["v1_parity", "v2_constrained"]
    }

@app.get("/api/phase7/zones/{year}/{scenario}")
def get_priority_zones(year: int, scenario: str):
    path = os.path.join(DATA_DIR, "phase7", scenario, "zones", f"priority_zones_{scenario}_{year}.geojson")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Data not found")
    with open(path, "r") as f:
        return json.load(f)

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
    path = os.path.join(DATA_DIR, "gis", "study_area", "study_area.geojson")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Boundary not found")
    with open(path, "r") as f:
        return json.load(f)

# Hardcoded scientific rendering parameters
LAYER_CONFIGS = {
    "lst": {"rescale": ((30, 55),), "colormap_name": "inferno"},
    "ndvi": {"rescale": ((-1, 1),), "colormap_name": "rdylgn"},
    "ndbi": {"rescale": ((-0.5, 0.5),), "colormap_name": "rdbu_r"},
    "vegetation_cover": {"rescale": ((0, 1),), "colormap_name": "greens"},
    "severity_score": {"rescale": ((1, 3),), "colormap_name": "ylorrd"},
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
                if "colormap_name" in config:
                    cm = cmap.get(config["colormap_name"])
                    render_kwargs["colormap"] = cm
                    
            image_buffer = img.render(**render_kwargs)
            return Response(content=image_buffer, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
