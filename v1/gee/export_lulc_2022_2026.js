// ============================================================
// GreenGrid AI — Tier 1 feature upgrade: land cover export
// Google Earth Engine Script: Dynamic World LULC, July 2022–2026
// ============================================================
//
// PURPOSE:
//   Export a real 10 m land-cover layer per year to replace the
//   OSM-derived landuse raster, which is ~84% "unclassified" and
//   therefore nearly useless as a model feature.
//
//   Product: GOOGLE/DYNAMICWORLD/V1 (Sentinel-2-based, 9-class,
//   near-real-time, 2015–present). We take the per-pixel MAJORITY
//   label over each July window so the layer matches the temporal
//   convention of the other inputs (July composite).
//
//   Classes (label band): 0 water, 1 trees, 2 grass,
//   3 flooded_vegetation, 4 crops, 5 shrub_and_scrub, 6 built,
//   7 bare, 8 snow_and_ice.
//
// HOW TO USE:
//   1. Open https://code.earthengine.google.com (same account as before)
//   2. Paste this entire script into a NEW script (do not re-run the
//      export script — it would queue duplicate satellite exports)
//   3. Click Run; review the printed class histograms
//   4. Tasks tab → RUN the 5 export tasks
//   5. Download from Drive folder GreenGridAI_Phase2 into:
//        data/raw/lulc/2022_07/lulc_2022_10m.tif
//        data/raw/lulc/2023_07/lulc_2023_10m.tif
//        ... (2024, 2025, 2026)
//   6. Tell me when the files are in Downloads — the pipeline picks
//      them up automatically (existence-guarded).
//
// READ-ONLY with respect to the existing project: this script does
// not touch satellite exports; it queues 5 small LULC exports only.
// ============================================================

// ─── DELHI NCT BOUNDARY (identical to all other scripts) ────────────────
var gaul = ee.FeatureCollection("FAO/GAUL/2015/level1");
var delhiNCT = gaul.filter(ee.Filter.and(
  ee.Filter.eq("ADM0_NAME", "India"),
  ee.Filter.eq("ADM1_NAME", "Delhi")
));
var delhiGeom = delhiNCT.geometry();

var DRIVE_FOLDER = "GreenGridAI_Phase2";
var CRS = "EPSG:4326";
var MAX_PIXELS = 1e12;

var CLASS_NAMES = [
  "water", "trees", "grass", "flooded_vegetation", "crops",
  "shrub_and_scrub", "built", "bare", "snow_and_ice"
];

["2022", "2023", "2024", "2025", "2026"].forEach(function (year) {
  var dw = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
    .filterBounds(delhiGeom)
    .filterDate(year + "-07-01", year + "-07-31");

  // Majority (mode) label per pixel across the July window
  var lulc = dw.select("label").mode().clip(delhiGeom).rename("lulc");

  // Diagnostics: pixel count per class (unmasked DW coverage)
  var counts = dw.select("label").mode().reduceRegion({
    reducer: ee.Reducer.frequencyHistogram(),
    geometry: delhiGeom,
    scale: 10,
    maxPixels: MAX_PIXELS,
    tileScale: 4
  });
  print("LULC " + year + " class histogram (label: count):", counts);

  Export.image.toDrive({
    image: lulc.toFloat(),   // Float so UInt8/Int types unify across years
    description: "LULC_" + year + "_07_Delhi_10m",
    folder: DRIVE_FOLDER,
    fileNamePrefix: "lulc_" + year + "_10m",
    region: delhiGeom,
    scale: 10,
    crs: CRS,
    maxPixels: MAX_PIXELS,
    fileFormat: "GeoTIFF"
  });

  print("✓ Export task queued: LULC_" + year + "_07_Delhi_10m");
});

print("=== 5 LULC EXPORT TASKS QUEUED ===");
print("Download into data/raw/lulc/<year>_07/ and tell your assistant.");
