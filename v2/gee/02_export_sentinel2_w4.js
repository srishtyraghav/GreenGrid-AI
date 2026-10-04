// ============================================================
// GreenGrid AI — V2 Phase 2: Dataset Rebuild
// Google Earth Engine Script 02 — Sentinel-2 SR Harmonized, W4
// ============================================================
//
// CLONED FROM (verified V1 recipe, unchanged sections marked):
//   gee/export_data_2023_2025.js
//
// V2 CHANGES vs V1 (documented, deliberate):
//   1. Window: W4 = May 1 – Jun 30 (approved common V2 window
//      2022–2026; see data/v2/phase2/window_investigation/decision_matrix.md)
//   2. EMPTY-COLLECTION GUARD: if a year yields 0 scenes under the
//      cloud filter, NO export tasks are queued and the script prints
//      an explicit ERROR. (V1 queued exports from empty collections in
//      Jul 2024/2025: "Image.divide: ... no bands" failures that still
//      produced queued — and invalid — tasks.)
//   3. Extra diagnostic band VALID_COUNT per pixel (number of SCL-clear
//      observations entering the median) — coverage weight for Phase 3.
//   4. Pre-queue valid-coverage estimate printed per year.
//
// IDENTICAL TO V1 (do not change):
//   - Collection COPERNICUS/S2_SR_HARMONIZED
//   - Scene filter CLOUDY_PIXEL_PERCENTAGE < 60
//   - SCL mask: classes 3 (cloud shadow), 8 (medium cloud),
//     9 (high cloud), 10 (cirrus) — exactly as V1
//   - 10 m bands B2,B3,B4,B8 median ÷10000; 20 m bands
//     B5,B6,B7,B8A,B11,B12 median ÷10000 + SCL median as float
//   - EPSG:4326, scales 10/20, GeoTIFF, clip(delhiGeom)
//
// OUTPUTS (5 years × 2 files):
//   sentinel2_YYYY_05_06_10m_composite.tif  (5 bands: B2,B3,B4,B8,VALID_COUNT)
//   sentinel2_YYYY_05_06_20m_composite.tif  (8 bands: B5..B12,SCL,VALID_COUNT)
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

// ─── SENTINEL-2 CLOUD / SHADOW MASK (identical to V1) ──────

function maskS2Clouds(image) {

  var scl = image.select("SCL");

  return image.updateMask(
    scl.neq(3)     // cloud shadow
      .and(scl.neq(8))   // medium probability cloud
      .and(scl.neq(9))   // high probability cloud
      .and(scl.neq(10))  // cirrus
  );
}

// ============================================================
// EXPORT ONE YEAR
// ============================================================

function exportYear(year) {

  print("============================================");
  print("V2 W4 PROCESSING YEAR:", year);
  print("Window: " + year + "-05-01 to " + year + "-06-30");
  print("============================================");

  var startDate = year + "-05-01";
  var endDate = year + "-06-30";

  var s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(delhiGeom)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
    .sort("CLOUDY_PIXEL_PERCENTAGE");

  // ---- EMPTY-COLLECTION GUARD (V2 change #2) ----
  var n = s2.size().getInfo();

  print("Sentinel-2 scene count " + year + ":", n);

  if (n === 0) {
    print("ERROR " + year + ": Sentinel-2 collection is EMPTY under " +
          "CLOUDY_PIXEL_PERCENTAGE<60 for W4 — NO exports queued. " +
          "Investigate (relax filter? source outage?) before re-running.");
    return;
  }

  // ---- Provenance log (same fields as V1) ----
  print("Sentinel-2 " + year + " cloud percentages:",
    s2.aggregate_array("CLOUDY_PIXEL_PERCENTAGE").getInfo());

  print("Sentinel-2 " + year + " dates:",
    s2.aggregate_array("system:time_start")
      .map(function (t) { return ee.Date(t).format("YYYY-MM-dd"); })
      .getInfo());

  print("Sentinel-2 " + year + " tiles:",
    s2.aggregate_array("MGRS_TILE").getInfo());

  print("Sentinel-2 " + year + " product IDs:",
    s2.aggregate_array("PRODUCT_ID").getInfo());

  var masked = s2.map(maskS2Clouds);

  // ---- Per-pixel valid-observation count (V2 change #3) ----
  // 10m stack is Float64 (median + divide, same as V1) -> count must be
  // Float64; GEE rejects mixed Float32+Float64 exports.
  // 20m stack is explicitly toFloat() (Float32, same as V1) -> count stays
  // Float32.
  var validCount10 = masked.select("B2").count()
    .toDouble().rename("VALID_COUNT").clip(delhiGeom);

  var validCount20 = masked.select("B5").count()
    .toFloat().rename("VALID_COUNT").clip(delhiGeom);

  // ==========================================================
  // 10 m composite (identical band set + scaling to V1)
  // ==========================================================

  var s2_10m = masked
    .select(["B2","B3","B4","B8"])
    .median()
    .divide(10000)
    .addBands(validCount10)
    .clip(delhiGeom);

  Export.image.toDrive({

    image: s2_10m,

    description: "S2_" + year + "_W4_Delhi_10m_Composite",

    folder: DRIVE_FOLDER,

    fileNamePrefix: "sentinel2_" + year + "_05_06_10m_composite",

    region: delhiGeom,

    scale: 10,

    crs: CRS,

    maxPixels: MAX_PIXELS,

    fileFormat: "GeoTIFF"

  });

  // ==========================================================
  // 20 m composite (identical band set + scaling to V1)
  // ==========================================================

  var s2_20m = masked
    .select(["B5","B6","B7","B8A","B11","B12","SCL"])
    .median()
    .clip(delhiGeom);

  var s2_20m_scaled = s2_20m
    .select(["B5","B6","B7","B8A","B11","B12"])
    .divide(10000)
    .toFloat()
    .addBands(s2_20m.select("SCL").toFloat())
    .addBands(validCount20)
    .clip(delhiGeom);

  Export.image.toDrive({

    image: s2_20m_scaled,

    description: "S2_" + year + "_W4_Delhi_20m_Composite",

    folder: DRIVE_FOLDER,

    fileNamePrefix: "sentinel2_" + year + "_05_06_20m_composite",

    region: delhiGeom,

    scale: 20,

    crs: CRS,

    maxPixels: MAX_PIXELS,

    fileFormat: "GeoTIFF"

  });

  // ---- Pre-queue valid-coverage estimate (V2 change #4) ----
  var valid10 = s2_10m.select("B2")
    .reduceRegion({
      reducer: ee.Reducer.count(),
      geometry: delhiGeom,
      scale: 10,
      maxPixels: MAX_PIXELS,
      tileScale: 4
    });

  print("Approx valid 10m pixels " + year + ":", valid10.get("B2"));

  print("✓ Sentinel-2 10m + 20m exports queued for " +
        year + " (both with VALID_COUNT band)");
}

// ============================================================
// RUN ALL FIVE YEARS
// ============================================================

YEARS.forEach(exportYear);

print("============================================");
print("ALL V2 W4 SENTINEL-2 EXPORT TASKS HANDLED");
print("Maximum scene cloud cover: <60% (V1 value)");
print("SCL pixel-level cloud/shadow masking: ENABLED (V1 classes 3/8/9/10)");
print("Drive folder: " + DRIVE_FOLDER);
print("Expected files (if no ERROR above):");
YEARS.forEach(function (y) {
  print("sentinel2_" + y + "_05_06_10m_composite");
  print("sentinel2_" + y + "_05_06_20m_composite");
});
print("============================================");
