import { useEffect, useState } from 'react';
import { MapContainer, TileLayer, LayersControl, useMap, GeoJSON } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import icon from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';
import { fetchBoundary } from '../services/api';

let DefaultIcon = L.icon({
    iconUrl: icon,
    shadowUrl: iconShadow,
    iconSize: [25, 41],
    iconAnchor: [12, 41]
});
L.Marker.prototype.options.icon = DefaultIcon;

interface MapViewerProps {
  center?: [number, number];
  zoom?: number;
  children?: React.ReactNode;
  geoData?: any; // To auto-fit bounds
}

const BoundsComponent = ({ geoData }: { geoData: any }) => {
  const map = useMap();
  useEffect(() => {
    if (geoData && geoData.features && geoData.features.length > 0) {
      const layer = L.geoJSON(geoData);
      const bounds = layer.getBounds();
      if (bounds.isValid()) {
        map.fitBounds(bounds, { padding: [20, 20], maxZoom: 14 });
      }
    }
  }, [geoData, map]);
  return null;
};

export const MapViewer: React.FC<MapViewerProps> = ({ center = [28.6139, 77.2090], zoom = 11, children, geoData }) => {
  const [boundary, setBoundary] = useState<any>(null);

  useEffect(() => {
    fetchBoundary().then(setBoundary).catch(console.error);
  }, []);

  return (
    <MapContainer center={center} zoom={zoom} scrollWheelZoom={true} className="w-full h-full rounded-xl shadow-inner z-0">
      <LayersControl position="topright">
        <LayersControl.BaseLayer checked name="Satellite">
          <TileLayer
            attribution='&copy; Esri'
            url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
          />
        </LayersControl.BaseLayer>
        <LayersControl.BaseLayer name="Street Map">
          <TileLayer
            attribution='&copy; OSM'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
        </LayersControl.BaseLayer>
        <LayersControl.BaseLayer name="Light">
          <TileLayer
            attribution='&copy; <a href="https://carto.com/">CartoDB</a>'
            url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png"
          />
        </LayersControl.BaseLayer>
        
        {boundary && (
          <LayersControl.Overlay checked name="Study Area Boundary">
            <GeoJSON 
              data={boundary} 
              style={{
                color: '#f8fafc',
                weight: 2,
                opacity: 0.8,
                fillColor: '#000000',
                fillOpacity: 0.1,
                dashArray: '5, 5'
              }} 
              interactive={true}
              onEachFeature={(_feature, layer) => {
                layer.bindTooltip("NCR Study Area", { permanent: false, direction: "center", className: "bg-black/70 text-white border-0 shadow-none font-semibold text-xs px-2 py-1" });
              }}
            />
          </LayersControl.Overlay>
        )}
        
        {children}
      </LayersControl>
      {geoData && <BoundsComponent geoData={geoData} />}
    </MapContainer>
  );
};
