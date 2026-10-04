// ============================================================
// GreenGrid AI — Phase 2: Dataset Collection
// Google Earth Engine Script
// Export July 2023–2025 Data
// ============================================================
//
// NEW FILE:
//   gee/export_data_2023_2025.js
//
// EXISTING FILES — DO NOT MODIFY:
//   gee/export_data.js
//   gee/landsat9_collection.js
//   gee/sentinel2_collection.js
//
// Sentinel-2:
//   Maximum scene-level cloud cover = 60%
//   Pixel-level SCL cloud/shadow masking is also applied.
//
// ============================================================


// ─── DELHI NCT BOUNDARY ─────────────────────────────────────

var gaul = ee.FeatureCollection("FAO/GAUL/2015/level1");

var delhiNCT = gaul.filter(ee.Filter.and(
  ee.Filter.eq("ADM0_NAME", "India"),
  ee.Filter.eq("ADM1_NAME", "Delhi")
));

var delhiGeom = delhiNCT.geometry();


// ─── EXPORT SETTINGS ────────────────────────────────────────

var DRIVE_FOLDER = "GreenGridAI_Phase2";

var CRS = "EPSG:4326";

var MAX_PIXELS = 1e12;


// ─── LANDSAT 9 CLOUD MASK ──────────────────────────────────

function maskL9Clouds(image) {

  var qa = image.select("QA_PIXEL");

  return image.updateMask(
    qa.bitwiseAnd(1 << 4).eq(0)
      .and(qa.bitwiseAnd(1 << 3).eq(0))
  );
}


// ─── SENTINEL-2 CLOUD / SHADOW MASK ─────────────────────────

function maskS2Clouds(image) {

  var scl = image.select("SCL");

  return image.updateMask(
    scl.neq(3)     // cloud shadow
      .and(scl.neq(8))   // medium probability cloud
      .and(scl.neq(9))   // high probability cloud
      .and(scl.neq(10))  // cirrus
  );
}


// ─── LANDSAT 9 SCALE FACTORS ────────────────────────────────

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
// FUNCTION: EXPORT ONE YEAR
// ============================================================

function exportYear(year) {

  print("============================================");
  print("PROCESSING YEAR:", year);
  print("============================================");


  var startDate = year + "-07-01";
  var endDate = year + "-07-31";


  // ==========================================================
  // 1. LANDSAT 9
  // ==========================================================

  var l9_raw = ee.ImageCollection(
    "LANDSAT/LC09/C02/T1_L2"
  )
    .filterBounds(delhiGeom)
    .filterDate(startDate, endDate)
    .sort("CLOUD_COVER");


  print(
    "Landsat 9 image count " + year + ":",
    l9_raw.size()
  );


  var l9_best = ee.Image(l9_raw.first());


  print(
    "L9 " + year + " best scene:",
    l9_best.date().format("YYYY-MM-dd")
  );

  print(
    "L9 " + year + " cloud cover:",
    l9_best.get("CLOUD_COVER")
  );


  // ----------------------------------------------------------
  // Landsat 9 July composite
  // ----------------------------------------------------------

  var l9_composite = l9_raw
    .map(maskL9Clouds)
    .map(applyL9ScaleFactors)
    .select([
      "SR_B2",
      "SR_B3",
      "SR_B4",
      "SR_B5",
      "SR_B6",
      "SR_B7",
      "ST_B10",
      "QA_PIXEL"
    ])
    .median()
    .clip(delhiGeom);


  Export.image.toDrive({

    image: l9_composite,

    description:
      "L9_" + year + "_07_Delhi_Composite_30m",

    folder: DRIVE_FOLDER,

    fileNamePrefix:
      "landsat9_" + year + "_07_composite_30m",

    region: delhiGeom,

    scale: 30,

    crs: CRS,

    maxPixels: MAX_PIXELS,

    fileFormat: "GeoTIFF"

  });


  print(
    "✓ Landsat 9 export created for " + year
  );


  // ==========================================================
  // 2. SENTINEL-2
  // ==========================================================
  //
  // Scene-level cloud threshold:
  // MAXIMUM 60%
  //
  // Individual cloudy/shadow pixels are removed using SCL.
  //


  var s2_raw = ee.ImageCollection(
    "COPERNICUS/S2_SR_HARMONIZED"
  )
    .filterBounds(delhiGeom)
    .filterDate(startDate, endDate)
    .filter(
      ee.Filter.lt(
        "CLOUDY_PIXEL_PERCENTAGE",
        60
      )
    )
    .sort("CLOUDY_PIXEL_PERCENTAGE");


  print(
    "Sentinel-2 image count " + year + ":",
    s2_raw.size()
  );


  print(
    "Sentinel-2 " + year + " cloud percentages:",
    s2_raw.aggregate_array(
      "CLOUDY_PIXEL_PERCENTAGE"
    )
  );


  print(
    "Sentinel-2 " + year + " dates:",
    s2_raw.aggregate_array(
      "system:time_start"
    ).map(function(t) {
      return ee.Date(t).format("YYYY-MM-dd");
    })
  );


  print(
    "Sentinel-2 " + year + " tiles:",
    s2_raw.aggregate_array("MGRS_TILE")
  );


  print(
    "Sentinel-2 " + year + " product IDs:",
    s2_raw.aggregate_array("PRODUCT_ID")
  );


  // ==========================================================
  // Sentinel-2 10m composite
  // ==========================================================

  var s2_10m = s2_raw
    .map(maskS2Clouds)
    .select([
      "B2",
      "B3",
      "B4",
      "B8"
    ])
    .median()
    .divide(10000)
    .clip(delhiGeom);


  Export.image.toDrive({

    image: s2_10m,

    description:
      "S2_" + year + "_07_Delhi_10m_Composite",

    folder: DRIVE_FOLDER,

    fileNamePrefix:
      "sentinel2_" + year + "_07_10m_composite",

    region: delhiGeom,

    scale: 10,

    crs: CRS,

    maxPixels: MAX_PIXELS,

    fileFormat: "GeoTIFF"

  });


  // ==========================================================
  // Sentinel-2 20m composite
  // ==========================================================

  var s2_20m = s2_raw
    .map(maskS2Clouds)
    .select([
      "B5",
      "B6",
      "B7",
      "B8A",
      "B11",
      "B12",
      "SCL"
    ])
    .median()
    .clip(delhiGeom);


  var s2_20m_scaled =
  s2_20m
    .select([
      "B5",
      "B6",
      "B7",
      "B8A",
      "B11",
      "B12"
    ])
    .divide(10000)
    .toFloat()
    .addBands(
      s2_20m
        .select("SCL")
        .toFloat()
    )
    .clip(delhiGeom);


  Export.image.toDrive({

    image: s2_20m_scaled,

    description:
      "S2_" + year + "_07_Delhi_20m_Composite",

    folder: DRIVE_FOLDER,

    fileNamePrefix:
      "sentinel2_" + year + "_07_20m_composite",

    region: delhiGeom,

    scale: 20,

    crs: CRS,

    maxPixels: MAX_PIXELS,

    fileFormat: "GeoTIFF"

  });


  print(
    "✓ Sentinel-2 10m export created for " + year
  );

  print(
    "✓ Sentinel-2 20m export created for " + year
  );
}


// ============================================================
// RUN FOR 2023, 2024, 2025
// ============================================================

exportYear(2023);

exportYear(2024);

exportYear(2025);


// ============================================================
// SUMMARY
// ============================================================

print("============================================");
print("ALL 2023–2025 EXPORT TASKS CREATED");
print("============================================");

print(
  "Sentinel-2 maximum scene cloud cover: <60%"
);

print(
  "SCL pixel-level cloud/shadow masking: ENABLED"
);

print(
  "Google Drive folder:",
  DRIVE_FOLDER
);

print("Expected exports:");

print("L9_2023_07_Delhi_Composite_30m");
print("L9_2024_07_Delhi_Composite_30m");
print("L9_2025_07_Delhi_Composite_30m");

print("S2_2023_07_Delhi_10m_Composite");
print("S2_2024_07_Delhi_10m_Composite");
print("S2_2025_07_Delhi_10m_Composite");

print("S2_2023_07_Delhi_20m_Composite");
print("S2_2024_07_Delhi_20m_Composite");
print("S2_2025_07_Delhi_20m_Composite");

print("============================================");