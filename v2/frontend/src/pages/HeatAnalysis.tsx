import { useState } from 'react';
import { useAppContext } from '../context/AppContext';
import { MapViewer } from '../components/MapViewer';
import { LayersControl, TileLayer } from 'react-leaflet';
import { getTileUrl } from '../services/api';
import { SlidersHorizontal, Info } from 'lucide-react';

const HeatAnalysis = () => {
  const { year } = useAppContext();
  const [opacity, setOpacity] = useState(0.75);

  return (
    <div className="h-full flex flex-col p-8 max-w-[1600px] mx-auto space-y-6">
      <div className="flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-bold text-slate-900 tracking-tight">Heat Analysis</h1>
          <p className="text-slate-500 mt-2 text-lg">Visualizing Land Surface Temperature (LST) and Relative Heat Severity proxy hotspots.</p>
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
          <LayersControl.Overlay checked name={`Relative Heat Severity (${year})`}>
            <TileLayer url={getTileUrl('phase6', 'severity_score', year)} opacity={opacity} />
          </LayersControl.Overlay>
          <LayersControl.Overlay name={`LST (${year})`}>
             <TileLayer url={getTileUrl('phase3', 'lst', year)} opacity={opacity} />
          </LayersControl.Overlay>
        </MapViewer>

        <div className="absolute bottom-6 right-6 flex flex-col gap-4 z-[1000] w-80">
          
          <div className="bg-white/95 backdrop-blur-sm p-5 rounded-xl shadow-lg border border-slate-200/50">
            <h4 className="font-bold text-sm text-slate-800 mb-3 uppercase tracking-wider">Relative Heat Severity</h4>
            <div className="space-y-2 mb-3">
               <div className="flex items-center gap-3 text-sm font-medium text-slate-700">
                 <div className="w-12 h-3 rounded-sm bg-[#800026]"></div> High
               </div>
               <div className="flex items-center gap-3 text-sm font-medium text-slate-700">
                 <div className="w-8 h-3 rounded-sm bg-[#fd8d3c]"></div> Medium
               </div>
               <div className="flex items-center gap-3 text-sm font-medium text-slate-700">
                 <div className="w-4 h-3 rounded-sm bg-[#ffffcc]"></div> Low
               </div>
            </div>
            <p className="text-[10px] text-slate-500 italic border-t border-slate-100 pt-2">
              Relative ranking within {year}.<br/>Not absolute UHI temperature.
            </p>
          </div>

          <div className="bg-white/95 backdrop-blur-sm p-5 rounded-xl shadow-lg border border-slate-200/50">
            <h4 className="font-bold text-sm text-slate-800 mb-2 uppercase tracking-wider">Land Surface Temp (°C)</h4>
            <div className="h-3 w-full rounded-sm bg-gradient-to-r from-[#000004] via-[#bc3754] to-[#fcffa4] mb-1"></div>
            <div className="flex justify-between text-xs text-slate-500 font-mono">
              <span>Low (30)</span>
              <span>High (55)</span>
            </div>
          </div>
          
          <div className="bg-blue-50/95 backdrop-blur-sm p-4 rounded-xl shadow-lg border border-blue-100 flex gap-3 text-blue-900 text-xs">
            <Info size={20} className="shrink-0 mt-0.5" />
            <div>
              <strong>Metadata:</strong><br/>
              LST: Landsat 9 (30m), W4 {year}<br/>
              Severity: Derived from V2 Phase 6 Model. Normalized annually.
            </div>
          </div>

        </div>
      </div>
    </div>
  );
};

export default HeatAnalysis;
