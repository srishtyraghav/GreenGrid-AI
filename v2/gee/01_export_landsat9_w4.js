// ============================================================
// GreenGrid AI — V2 Phase 2: Dataset Rebuild
// Google Earth Engine Script 01 — Landsat 9 ONLY, W4 window
// ============================================================
//
// CLONED FROM (verified V1 recipe, unchanged sections marked):
//   gee/export_data_2023_2025.js
//
// SENSOR DECISION (2026-10-04, user): Landsat 9 ONLY — exact sensor
// parity with V1 (LC09/C02/T1_L2, TIRS-2). Landsat 8 was evaluated
// (the 702-scene window investigation used L8+L9) and excluded to
// keep the V2 provenance single-sensor and identical to V1's.
// L9-only window evidence: data/v2/phase2/window_investigation/
//   window_metrics_l9only.csv (recomputed after this decision).
//
// V2 CHANGES vs V1 (documented, deliberate):
//   1. Window: W4 = May 1 – Jun 30 (approved common V2 window
//      2022–2026; see data/v2/phase2/window_investigation/decision_matrix.md)
//   2. EMPTY-COLLECTION GUARD: if a year yields 0 scenes, NO export
//      tasks are queued and the script prints an explicit ERROR
//      (V1 queued exports from empty collections in 2024/2025, which
//      produced invalid no-bands rasters while still "succeeding").
//   3. Extra diagnostic band QA_CLEAR_COUNT per pixel (number of clear
//      observations entering the median) — coverage weight for Phase 3.
//   4. Per-year valid-ST pixel estimate printed BEFORE queueing, so a
//      low-coverage year is caught at run time, not after download.
//
// IDENTICAL TO V1 (do not change):
//   - Collection LANDSAT/LC09/C02/T1_L2 (Landsat 9 only, Tier 1)
//   - Delhi boundary (FAO/GAUL/2015 level1, ADM1_NAME = Delhi)
//   - QA_PIXEL mask: bits 3 (cloud) + 4 (cloud shadow) only
//   - SR scale ×0.0000275 − 0.2; ST_B10 exported RAW (unscaled DN),
//     exactly as V1 (Phase 3 applies ×0.00341802 + 149.0)
//   - Bands SR_B2..SR_B7, ST_B10, QA_PIXEL; per-pixel median composite
//   - EPSG:4326, scale 30, GeoTIFF, clip(delhiGeom)
//   - NO scene-level CLOUD_COVER filter (same as V1)
//
// OUTPUTS (5 years × 1 file):
//   landsat9_YYYY_05_06_composite_30m.tif  (9 bands: the 8 V1 bands + QA_CLEAR_COUNT)
// → Google Drive folder GreenGridAI_V2_Phase2
// ============================================================

// ─── DELHI NCT BOUNDARY (identical to V1) ───────────────────

var gaul = ee.FeatureCollection("FAO/GAUL/2015/level1");

var delhiNCT = gaul.filter(ee.Filter.and(
  ee.Filter.eq("ADM0_NAME", "India"),
  ee.Filter.eq("ADM1_NAME", "Delhi")
));

var delhiGeom = delhiNCT.geometry();

// ─── EXPORT SETTINGS ────────────────────────────────────────

var DRIVE_FOLDER = "GreenGridAI_V2_Phase2";

var CRS = "EPSG:4326";

var MAX_PIXELS = 1e12;

var YEARS = [2022, 2023, 2024, 2025, 2026];

// ─── LANDSAT 9 CLOUD MASK (identical to V1: bits 3 + 4) ────

function maskL9Clouds(image) {

  var qa = image.select("QA_PIXEL");

  return image.updateMask(
    qa.bitwiseAnd(1 << 4).eq(0)
      .and(qa.bitwiseAnd(1 << 3).eq(0))
  );
}

// ─── SCALE FACTORS (identical to V1; ST_B10 stays raw) ──────

function applyL9ScaleFactors(image) {

  var srBands = image
    .select("SR_B.*")
    .multiply(0.0000275)
    .add(-0.2);

  var thermalBand = image.select("ST_B10");

  var qaBand = image.select("QA_PIXEL");

  return ee.Image(
    srBands
      .addBands(thermalBand)
      .addBands(qaBand)
      .copyProperties(image, image.propertyNames())
  );
}

// ============================================================
// EXPORT ONE YEAR
// ============================================================

function exportYear(year) {

  print("============================================");
  print("V2 W4 PROCESSING YEAR:", year);
  print("Window: " + year + "-05-01 to " + year + "-06-30 (Landsat 9 only)");
  print("============================================");

  var startDate = year + "-05-01";
  var endDate = year + "-06-30";

  // LANDSAT 9 ONLY (sensor-parity decision, see header)
  var l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
    .filterBounds(delhiGeom)
    .filterDate(startDate, endDate)
    .sort("CLOUD_COVER");

  // ---- EMPTY-COLLECTION GUARD (V2 change #2) ----
  var n = l9.size().getInfo();

  print("Landsat 9 scene count " + year + ":", n);

  if (n === 0) {
    print("ERROR " + year + ": Landsat 9 collection is EMPTY for W4 — " +
          "NO export queued. Investigate before re-running.");
    return;
  }

  // ---- Provenance log (same fields as V1) ----
  var ids = l9.aggregate_array("system:index").getInfo();
  var dates = l9.aggregate_array("system:time_start")
    .map(function (t) { return ee.Date(t).format("YYYY-MM-dd"); }).getInfo();
  var clouds = l9.aggregate_array("CLOUD_COVER").getInfo();

  print("L9 " + year + " scene IDs:", ids);
  print("L9 " + year + " acquisition dates:", dates);
  print("L9 " + year + " scene cloud cover %:", clouds);

  var masked = l9
    .map(maskL9Clouds)
    .map(applyL9ScaleFactors)
    .select(["SR_B2","SR_B3","SR_B4","SR_B5","SR_B6","SR_B7","ST_B10","QA_PIXEL"]);

  // ---- Per-pixel clear-observation count (V2 change #3) ----
  // NOTE: must be Float64. The composite bands are Float64 (V1 exports
  // were float64) and GEE refuses mixed Float32+Float64 in one export:
  // "Exported bands must have compatible data types".
  var clearCount = masked
    .select("ST_B10")
    .count()
    .toDouble()
    .rename("QA_CLEAR_COUNT")
    .clip(delhiGeom);

  var composite = masked
    .median()
    .clip(delhiGeom);

  // ---- Pre-queue valid-ST coverage estimate (V2 change #4) ----
  var validST = composite.select("ST_B10")
    .reduceRegion({
      reducer: ee.Reducer.count(),
      geometry: delhiGeom,
      scale: 30,
      maxPixels: MAX_PIXELS,
      tileScale: 4
    });

  print("Approx valid ST_B10 pixels " + year + ":", validST.get("ST_B10"));
  print("(V1 reference: full Delhi grid = 1,881,088 study cells @30m)");

  var out = composite.addBands(clearCount);

  Export.image.toDrive({

    image: out,

    description: "L9_" + year + "_W4_Delhi_Composite_30m",

    folder: DRIVE_FOLDER,

    fileNamePrefix: "landsat9_" + year + "_05_06_composite_30m",

    region: delhiGeom,

    scale: 30,

    crs: CRS,

    maxPixels: MAX_PIXELS,

    fileFormat: "GeoTIFF"

  });

  print("✓ Landsat 9 export queued for " + year +
        " (with QA_CLEAR_COUNT band)");
}

// ============================================================
// RUN ALL FIVE YEARS
// ============================================================

YEARS.forEach(exportYear);

print("============================================");
print("ALL V2 W4 LANDSAT 9 EXPORT TASKS HANDLED (L9 only)");
print("Drive folder: " + DRIVE_FOLDER);
print("Expected files (if no ERROR above):");
YEARS.forEach(function (y) {
  print("landsat9_" + y + "_05_06_composite_30m");
});
print("============================================");
