# Buildings — Full Dataset Download Instructions

The file `delhi_buildings_sample_osm.json` contains a sample of building
footprints from Central Delhi only (bbox: 28.55°N–28.75°N, 77.10°E–77.30°E).

## For Full Delhi NCT Buildings:

### Option A: GeoFabrik (Recommended — complete data)
1. Visit: https://download.geofabrik.de/asia/india/northern-zone.html
2. Download: `northern-zone-latest.osm.pbf`
3. Extract Delhi buildings using osmium or osmfilter:
   ```bash
   osmium tags-filter northern-zone-latest.osm.pbf w/building -o delhi_buildings.osm.pbf
   osmium extract --bbox 76.83,28.40,77.35,28.89 delhi_buildings.osm.pbf -o delhi_nct_buildings.osm.pbf
   ```
4. Convert to GeoJSON: `ogr2ogr -f GeoJSON delhi_buildings.geojson delhi_nct_buildings.osm.pbf multipolygons`

### Option B: Overpass Turbo (web interface)
1. Visit: https://overpass-turbo.eu
2. Use query:
   ```
   [out:json][timeout:300][bbox:28.40,76.83,28.89,77.35];
   (way["building"]; relation["building"];);
   out body; >; out skel qt;
   ```
3. Export as GeoJSON

### Option C: India Open Government Data
- data.gov.in may have official building/property maps

Downloaded: 2026-08-27
