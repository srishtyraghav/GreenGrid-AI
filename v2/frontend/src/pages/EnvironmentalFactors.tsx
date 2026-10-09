import React, { useState } from 'react';
import { useAppContext } from '../context/AppContext';
import { MapViewer } from '../components/MapViewer';
import { LayersControl, TileLayer } from 'react-leaflet';
import { getTileUrl } from '../services/api';
import { SlidersHorizontal, Info } from 'lucide-react';

const EnvironmentalFactors = () => {
  const { year } = useAppContext();
  const [opacity, setOpacity] = useState(0.8);

  return (
    <div className="h-full flex flex-col p-8 max-w-[1600px] mx-auto space-y-6">
      <div className="flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-bold text-slate-900 tracking-tight">Environmental Context</h1>
          <p className="text-slate-500 mt-2 text-lg">Exploring geospatial features associated with heat: NDVI, NDBI, and Vegetation Cover.</p>
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
      
      <div className="flex-1 bg-white rounded-2xl shadow-sm border border-slate-200/60 overflow-hidden relative">
        <MapViewer>
          <LayersControl.Overlay checked name={`NDVI (${year})`}>
            <TileLayer url={getTileUrl('phase3', 'ndvi', year)} opacity={opacity} />
          </LayersControl.Overlay>
          <LayersControl.Overlay name={`NDBI (${year})`}>
             <TileLayer url={getTileUrl('phase3', 'ndbi', year)} opacity={opacity} />
          </LayersControl.Overlay>
          <LayersControl.Overlay name={`Vegetation Cover (${year})`}>
             <TileLayer url={getTileUrl('phase3', 'vegetation_cover', year)} opacity={opacity} />
          </LayersControl.Overlay>
        </MapViewer>

        <div className="absolute bottom-6 right-6 flex flex-col gap-4 z-[1000] w-[340px]">
          <div className="bg-white/95 backdrop-blur-sm p-5 rounded-xl shadow-lg border border-slate-200/50 space-y-5">
            <div>
              <h4 className="font-bold text-xs text-slate-800 mb-1 uppercase tracking-wider">NDVI</h4>
              <p className="text-[10px] text-slate-500 mb-2">Low vegetation ────────────── High vegetation</p>
              <div className="h-3 w-full rounded-sm bg-gradient-to-r from-[#a50026] via-[#ffffbf] to-[#006837] mb-1"></div>
              <div className="flex justify-between text-[11px] text-slate-500 font-mono"><span>-1.0</span><span>+1.0</span></div>
            </div>
            <div>
              <h4 className="font-bold text-xs text-slate-800 mb-1 uppercase tracking-wider">NDBI</h4>
              <p className="text-[10px] text-slate-500 mb-2">Low built-up ─────────────── High built-up</p>
              <div className="h-3 w-full rounded-sm bg-gradient-to-r from-[#053061] via-[#f7f7f7] to-[#67001f] mb-1"></div>
              <div className="flex justify-between text-[11px] text-slate-500 font-mono"><span>-0.5</span><span>+0.5</span></div>
            </div>
            <div>
              <h4 className="font-bold text-xs text-slate-800 mb-1 uppercase tracking-wider">Vegetation Cover</h4>
              <p className="text-[10px] text-slate-500 mb-2">Low cover ──────────────────── High cover</p>
              <div className="h-3 w-full rounded-sm bg-gradient-to-r from-[#f7fcf5] to-[#00441b] mb-1"></div>
              <div className="flex justify-between text-[11px] text-slate-500 font-mono"><span>0.0</span><span>1.0</span></div>
            </div>
          </div>

          <div className="bg-blue-50/95 backdrop-blur-sm p-4 rounded-xl shadow-lg border border-blue-100 flex gap-3 text-blue-900 text-xs">
            <Info size={24} className="shrink-0 mt-0.5" />
            <div>
              <strong>Metadata & Context:</strong><br/>
              Source: Sentinel-2 (30m), W4 {year}<br/>
              High-severity areas are spatially associated with lower vegetation values, but do not imply strict monotonic causality.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EnvironmentalFactors;
