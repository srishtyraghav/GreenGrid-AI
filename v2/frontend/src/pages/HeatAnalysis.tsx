import { useState } from 'react';
import { useAppContext } from '../context/AppContext';
import { MapViewer } from '../components/MapViewer';
import { LayersControl, TileLayer } from 'react-leaflet';
import { getTileUrl } from '../services/api';
import { SlidersHorizontal, Info } from 'lucide-react';

type LayerKey = 'severity' | 'lst';

const LAYERS: { key: LayerKey; label: string }[] = [
  { key: 'severity', label: 'Heat Severity (classes)' },
  { key: 'lst', label: 'Land Surface Temperature (°C)' },
];

const SEVERITY_CLASSES = [
  { label: 'High', color: '#800026' },
  { label: 'Medium', color: '#fd8d3c' },
  { label: 'Low', color: '#ffffcc' },
];

const HeatAnalysis = () => {
  const { year } = useAppContext();
  const [activeLayer, setActiveLayer] = useState<LayerKey>('severity');
  const [opacity, setOpacity] = useState(0.75);

  return (
    <div className="h-full flex flex-col p-8 max-w-[1600px] mx-auto space-y-6">
      <div className="flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-bold text-slate-900 tracking-tight">Heat Analysis</h1>
          <p className="text-slate-500 mt-2 text-lg">Categorical relative heat severity and continuous Land Surface Temperature over the Delhi NCT study area.</p>
        </div>

        {/* Opacity Control */}
        <div className="bg-white px-4 py-3 rounded-xl border border-slate-200/60 shadow-sm flex items-center gap-4">
          <SlidersHorizontal size={18} className="text-slate-400" />
          <div className="flex flex-col">
            <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Layer Opacity</label>
            <input
              type="range"
              min="0" max="1" step="0.05"
              value={opacity}
              onChange={(e) => setOpacity(parseFloat(e.target.value))}
              className="w-32 accent-slate-600"
            />
          </div>
          <span className="text-xs font-mono text-slate-500 w-8 text-right">{Math.round(opacity * 100)}%</span>
        </div>
      </div>

      {/* Layer selector: one data layer at a time, each with its own truthful legend */}
      <div className="flex gap-2">
        {LAYERS.map((l) => (
          <button
            key={l.key}
            onClick={() => setActiveLayer(l.key)}
            className={`px-4 py-2 rounded-lg text-sm font-semibold border transition-colors ${
              activeLayer === l.key
                ? 'bg-slate-800 text-white border-slate-800'
                : 'bg-white text-slate-600 border-slate-200 hover:border-slate-400'
            }`}
          >
            {l.label}
          </button>
        ))}
      </div>

      <div className="flex-1 bg-white rounded-2xl shadow-sm border border-slate-200/60 overflow-hidden relative">
        <MapViewer>
          {activeLayer === 'severity' ? (
            <LayersControl.Overlay checked name={`Relative Heat Severity (${year})`}>
              <TileLayer url={getTileUrl('phase6', 'severity', year)} opacity={opacity} />
            </LayersControl.Overlay>
          ) : (
            <LayersControl.Overlay checked name={`Land Surface Temperature (${year})`}>
              <TileLayer url={getTileUrl('phase3', 'lst', year)} opacity={opacity} />
            </LayersControl.Overlay>
          )}
        </MapViewer>

        <div className="absolute bottom-6 right-6 flex flex-col gap-4 z-[1000] w-80">

          {activeLayer === 'severity' && (
            <div className="bg-white/95 backdrop-blur-sm p-5 rounded-xl shadow-lg border border-slate-200/50">
              <h4 className="font-bold text-sm text-slate-800 mb-3 uppercase tracking-wider">Relative Heat Severity</h4>
              <div className="space-y-2 mb-3">
                {SEVERITY_CLASSES.map((c) => (
                  <div key={c.label} className="flex items-center gap-3 text-sm font-medium text-slate-700">
                    <div className="w-10 h-3 rounded-sm" style={{ backgroundColor: c.color }}></div> {c.label}
                  </div>
                ))}
              </div>
              <p className="text-[10px] text-slate-500 italic border-t border-slate-100 pt-2">
                Discrete class raster (Low / Medium / High), {year}.<br/>
                Relative classification — not absolute UHI temperature.
              </p>
            </div>
          )}

          {activeLayer === 'lst' && (
            <div className="bg-white/95 backdrop-blur-sm p-5 rounded-xl shadow-lg border border-slate-200/50">
              <h4 className="font-bold text-sm text-slate-800 mb-2 uppercase tracking-wider">Land Surface Temperature (°C)</h4>
              <div className="h-3 w-full rounded-sm bg-gradient-to-r from-[#000004] via-[#bc3754] to-[#fcffa4] mb-1"></div>
              <div className="flex justify-between text-xs text-slate-500 font-mono">
                <span>31.4</span>
                <span>58.0</span>
              </div>
              <p className="text-[10px] text-slate-500 italic mt-2">
                Fixed pooled range (p0.1–p99.9 of valid 2022–2026 W4 LST); consistent across all years.
              </p>
            </div>
          )}

          <div className="bg-blue-50/95 backdrop-blur-sm p-4 rounded-xl shadow-lg border border-blue-100 flex gap-3 text-blue-900 text-xs">
            <Info size={20} className="shrink-0 mt-0.5" />
            <div>
              <strong>Metadata:</strong><br/>
              LST: Landsat 9 (30m), W4 {year}<br/>
              Severity: V2 Phase 6 categorical model output (per-year relative classes)<br/>
              Boundary: Delhi NCT (OSM relation 1942586)
            </div>
          </div>

        </div>
      </div>
    </div>
  );
};

export default HeatAnalysis;
