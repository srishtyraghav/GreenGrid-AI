// ============================================================
// GreenGrid AI — Tier 1 LULC: July-2024 gap fallback export
// Google Earth Engine Script: Dynamic World LULC 2024
// ============================================================
//
// WHY THIS SCRIPT EXISTS:
//   The diagnostic showed Dynamic World has ZERO scenes over Delhi in
//   July 2024 (a DW production gap), while the rest of 2024 has 172
//   scenes. The other four years (2022, 2023, 2025, 2026) exported
//   fine with the July-only window.
//
//   This script exports ONLY 2024, using the nearest same-year window
//   (June 1 – August 31) majority label. Land cover varies slowly, so
//   a summer-majority 2024 layer is a sound stand-in for July 2024.
//
//   PROVENANCE DEVIATION (recorded): 2024 LULC uses Jun–Aug majority;
//   all other years use July majority.
//
// HOW TO USE:
//   1. Open https://code.earthengine.google.com → NEW script
//   2. Paste this file, click Run, check the printed scene count
//   3. Tasks tab → RUN the single export task
//   4. Download from Drive (GreenGridAI_Phase2) to:
//        C:\GreenGrid-AI\data\raw\lulc\2024_07\lulc_2024_10m.tif
//      (same filename as the original recipe — the pipeline needs no
//       changes; only the metadata note differs)
// ============================================================

var gaul = ee.FeatureCollection("FAO/GAUL/2015/level1");
var delhiGeom = gaul.filter(ee.Filter.and(
  ee.Filter.eq("ADM0_NAME", "India"),
  ee.Filter.eq("ADM1_NAME", "Delhi")
)).geometry();

var START = "2024-06-01";
var END = "2024-08-31";

var dw = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
  .filterBounds(delhiGeom)
  .filterDate(START, END);

print("DW scenes over Delhi, Jun–Aug 2024:", dw.size());

// Guard: never queue an export from an empty collection.
dw.size().evaluate(function (n) {
  if (n === 0) {
    print("FAIL: no Dynamic World scenes for Jun–Aug 2024 — do NOT run the task; report back.");
    return;
  }

  var lulc = dw.select("label").mode().clip(delhiGeom).rename("lulc");

  var counts = lulc.reduceRegion({
    reducer: ee.Reducer.frequencyHistogram(),
    geometry: delhiGeom,
    scale: 10,
    maxPixels: 1e12,
    tileScale: 4
  });
  print("LULC 2024 (Jun–Aug majority) class histogram:", counts);

  Export.image.toDrive({
    image: lulc.toFloat(),
    description: "LULC_2024_07_Delhi_10m",
    folder: "GreenGridAI_Phase2",
    fileNamePrefix: "lulc_2024_10m",
    region: delhiGeom,
    scale: 10,
    crs: "EPSG:4326",
    maxPixels: 1e12,
    fileFormat: "GeoTIFF"
  });

  print("✓ Export task queued: LULC_2024_07_Delhi_10m (June–August majority)");
});
