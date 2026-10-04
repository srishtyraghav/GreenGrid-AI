// ============================================================
// GreenGrid AI — V2 Phase 2: Dataset Rebuild
// Google Earth Engine Script 03 — LULC (Dynamic World), W4
// ============================================================
//
// CLONED FROM: gee/export_lulc_2022_2026.js (verified V1 recipe)
//
// V2 CHANGES vs V1 (documented, deliberate):
//   1. Window: W4 = May 1 – Jun 30 (approved common V2 window), so the
//      LULC layer matches the temporal convention of the V2 satellite
//      inputs (V1 used July).
//   2. COVERAGE GUARD: V1's July-2025 export silently produced a
//      ~1.1%-coverage raster (ingestion gap at export time). Here the
//      valid-pixel fraction is computed BEFORE queueing; years below
//      50% coverage are NOT exported and raise an explicit ERROR.
//   3. Class histogram + DW scene count printed per year (V1 did this
//      only for the histogram).
//
// IDENTICAL TO V1:
//   - Product GOOGLE/DYNAMICWORLD/V1, per-pixel MAJORITY (mode) label
//   - Classes: 0 water, 1 trees, 2 grass, 3 flooded_vegetation,
//     4 crops, 5 shrub_and_scrub, 6 built, 7 bare, 8 snow_and_ice
//   - toFloat() export (uniform type across years), 10 m, EPSG:4326
//
// OUTPUTS (5 files):
//   lulc_YYYY_05_06_10m.tif
// → Google Drive folder GreenGridAI_V2_Phase2
// ============================================================

// ─── DELHI NCT BOUNDARY (identical to V1) ───────────────────

var gaul = ee.FeatureCollection("FAO/GAUL/2015/level1");

var delhiNCT = gaul.filter(ee.Filter.and(
  ee.Filter.eq("ADM0_NAME", "India"),
  ee.Filter.eq("ADM1_NAME", "Delhi")
));

var delhiGeom = delhiNCT.geometry();

var DRIVE_FOLDER = "GreenGridAI_V2_Phase2";

var CRS = "EPSG:4326";

var MAX_PIXELS = 1e12;

var YEARS = ["2022", "2023", "2024", "2025", "2026"];

var CLASS_NAMES = [
  "water", "trees", "grass", "flooded_vegetation", "crops",
  "shrub_and_scrub", "built", "bare", "snow_and_ice"
];

// ~Expected valid 10m pixels over Delhi NCT (guard denominator)
var EXPECTED_PX = delhiGeom.area(1).divide(100);

print("Expected 10m pixels over Delhi (guard denominator):", EXPECTED_PX);

YEARS.forEach(function (year) {

  print("============================================");
  print("V2 W4 LULC YEAR:", year);
  print("Window: " + year + "-05-01 to " + year + "-06-30");
  print("============================================");

  var dw = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
    .filterBounds(delhiGeom)
    .filterDate(year + "-05-01", year + "-06-30");

  var n = dw.size().getInfo();

  print("Dynamic World scene count " + year + ":", n);

  if (n === 0) {
    print("ERROR " + year + ": Dynamic World collection EMPTY for W4 — " +
          "NO export queued.");
    return;
  }

  var lulc = dw.select("label").mode().clip(delhiGeom).rename("lulc");

  // ---- Coverage guard (V2 change #2) ----
  var validPx = lulc.reduceRegion({
    reducer: ee.Reducer.count(),
    geometry: delhiGeom,
    scale: 10,
    maxPixels: MAX_PIXELS,
    tileScale: 4
  }).get("lulc");

  var cov = ee.Number(validPx).divide(EXPECTED_PX).getInfo();

  print("LULC " + year + " valid coverage fraction:", cov);

  var hist = lulc.reduceRegion({
    reducer: ee.Reducer.frequencyHistogram(),
    geometry: delhiGeom,
    scale: 10,
    maxPixels: MAX_PIXELS,
    tileScale: 4
  });

  print("LULC " + year + " class histogram (label: count):", hist);
  print("(class names: 0=water 1=trees 2=grass 3=flooded_veg 4=crops " +
        "5=shrub 6=built 7=bare 8=snow)");

  if (cov < 0.5) {
    print("ERROR " + year + ": LULC coverage " + (cov * 100).toFixed(1) +
          "% < 50% threshold — export NOT queued. This is the failure " +
          "mode that silently produced the ~1%-coverage V1 July-2025 " +
          "raster. Re-run later or investigate the ingestion gap.");
    return;
  }

  Export.image.toDrive({
    image: lulc.toFloat(),
    description: "LULC_" + year + "_W4_Delhi_10m",
    folder: DRIVE_FOLDER,
    fileNamePrefix: "lulc_" + year + "_05_06_10m",
    region: delhiGeom,
    scale: 10,
    crs: CRS,
    maxPixels: MAX_PIXELS,
    fileFormat: "GeoTIFF"
  });

  print("✓ Export task queued: LULC_" + year + "_W4_Delhi_10m " +
        "(coverage " + (cov * 100).toFixed(1) + "%)");
});

print("============================================");
print("V2 W4 LULC EXPORT TASKS HANDLED (errors above = not queued)");
print("Drive folder: " + DRIVE_FOLDER);
print("Expected files (if no ERROR above):");
YEARS.forEach(function (y) { print("lulc_" + y + "_05_06_10m"); });
print("============================================");
