import { useState } from 'react';
import { useAppContext } from '../context/AppContext';
import { MapViewer } from '../components/MapViewer';
import { LayersControl, TileLayer } from 'react-leaflet';
import { getTileUrl } from '../services/api';
import { SlidersHorizontal, Info } from 'lucide-react';

type LayerKey = 'ndvi' | 'ndbi' | 'vegetation_cover';

const LAYERS: { key: LayerKey; label: string }[] = [
  { key: 'ndvi', label: 'NDVI' },
  { key: 'ndbi', label: 'NDBI' },
  { key: 'vegetation_cover', label: 'Vegetation Cover' },
];

const LEGENDS: Record<LayerKey, { title: string; desc: string; gradient: string; low: string; high: string }> = {
  ndvi: {
    title: 'NDVI',
    desc: 'Bare/built (low) to dense vegetation (high)',
    gradient: 'from-[#a50026] via-[#ffffbf] to-[#006837]',
    low: '-0.4', high: '+0.9',
  },
  ndbi: {
    title: 'NDBI',
    desc: 'Vegetated/water (low) to dense built-up (high)',
    gradient: 'from-[#053061] via-[#f7f7f7] to-[#67001f]',
    low: '-0.5', high: '+0.5',
  },
  vegetation_cover: {
    title: 'Vegetation Cover',
    desc: 'No cover to full cover',
    gradient: 'from-[#f7fcf5] to-[#00441b]',
    low: '0.0', high: '1.0',
  },
};

const EnvironmentalFactors = () => {
  const { year } = useAppContext();
  const [activeLayer, setActiveLayer] = useState<LayerKey>('ndvi');
  const [opacity, setOpacity] = useState(0.8);
  const legend = LEGENDS[activeLayer];

  return (
    <div className="h-full flex flex-col p-8 max-w-[1600px] mx-auto space-y-6">
      <div className="flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-bold text-slate-900 tracking-tight">Environmental Context</h1>
          <p className="text-slate-500 mt-2 text-lg">Geospatial features associated with heat: NDVI, NDBI, and Vegetation Cover. Toggle one layer at a time.</p>
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

      {/* Layer toggle buttons */}
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
          <LayersControl.Overlay checked name={`${legend.title} (${year})`}>
            <TileLayer key={`${activeLayer}-${year}`} url={getTileUrl('phase3', activeLayer, year)} opacity={opacity} />
          </LayersControl.Overlay>
        </MapViewer>

        <div className="absolute bottom-6 right-6 flex flex-col gap-4 z-[1000] w-[340px]">
          <div className="bg-white/95 backdrop-blur-sm p-5 rounded-xl shadow-lg border border-slate-200/50">
            <h4 className="font-bold text-xs text-slate-800 mb-1 uppercase tracking-wider">{legend.title} ({year})</h4>
            <p className="text-[10px] text-slate-500 mb-2">{legend.desc}</p>
            <div className={`h-3 w-full rounded-sm bg-gradient-to-r ${legend.gradient} mb-1`}></div>
            <div className="flex justify-between text-[11px] text-slate-500 font-mono"><span>{legend.low}</span><span>{legend.high}</span></div>
          </div>

          <div className="bg-blue-50/95 backdrop-blur-sm p-4 rounded-xl shadow-lg border border-blue-100 flex gap-3 text-blue-900 text-xs">
            <Info size={24} className="shrink-0 mt-0.5" />
            <div>
              <strong>Metadata:</strong><br/>
              Landsat 9 + Sentinel-2 fused V2 products (30m), W4 ({year})<br/>
              Boundary: Delhi NCT data extent. These are environmental context layers — inputs to suitability, not suitability scores.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EnvironmentalFactors;
