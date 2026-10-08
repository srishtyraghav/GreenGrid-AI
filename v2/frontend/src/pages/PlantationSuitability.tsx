import React, { useEffect, useState } from 'react';
import { useAppContext } from '../context/AppContext';
import { fetchPriorityZones, fetchTreeRequirementSummary, fetchTreeRequirementZones } from '../services/api';
import { MapViewer } from '../components/MapViewer';
import { GeoJSON } from 'react-leaflet';
import { Loader } from '../components/Loader';
import { MapPin, Download, PieChart as PieChartIcon } from 'lucide-react';
import { Doughnut } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';

ChartJS.register(ArcElement, Tooltip, Legend);

const chartOptions = { 
  cutout: '70%', 
  plugins: { legend: { display: false }, tooltip: { enabled: false } },
  animation: { duration: 0 } 
};

const PlantationSuitability = () => {
  const { year, scenario } = useAppContext();
  const [geoData, setGeoData] = useState<any>(null);
  const [summary, setSummary] = useState<any>(null);
  const [selectedZone, setSelectedZone] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetchPriorityZones(year, scenario),
      fetchTreeRequirementSummary(year, scenario),
      fetchTreeRequirementZones(year, scenario)
    ])
      .then(([geo, summData, zonesData]) => {
        if (geo && geo.features) {
          geo.features.forEach((feature: any) => {
            const zoneStats = zonesData.find((z: any) => z.zone_id === feature.properties.zone_id);
            if (zoneStats) {
              feature.properties.rank = zoneStats.rank;
              feature.properties.priority = zoneStats.priority;
              feature.properties.mean_severity_score = zoneStats.mean_severity_score;
              feature.properties.plantable_ha = zoneStats.plantable_ha;
              feature.properties.excluded_ha = zoneStats.excluded_ha;
              feature.properties.trees_400 = zoneStats.recommended_trees_400;
              feature.properties.trees_1000 = zoneStats.recommended_trees_1000;
              feature.properties.trees_2500 = zoneStats.recommended_trees_2500;
            }
          });
        }
        setGeoData(geo);
        setSelectedZone(null);
        const primary = summData.find((row: any) => row.is_primary_density);
        if(primary) setSummary(primary);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [year, scenario]);

  const exportData = () => {
    if (!geoData) return;
    const blob = new Blob([JSON.stringify(geoData, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `priority_zones_${scenario}_${year}.geojson`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const getBaseStyle = (feature: any) => {
    const priority = feature.properties.priority;
    let color = '#22c55e';
    if (priority === 'Medium') color = '#eab308';
    if (priority === 'Low') color = '#ef4444';
    return { fillColor: color, weight: 1.5, opacity: 0.8, color: 'white', fillOpacity: 0.65 };
  };

  const getSelectedStyle = () => {
    return { weight: 4, color: '#2563eb', opacity: 1, fillColor: '#3b82f6', fillOpacity: 0.4 };
  };

  const onEachFeature = (feature: any, layer: any) => {
    layer.on({
      click: () => setSelectedZone(feature.properties)
    });
    // Add tooltip that follows mouse
    layer.bindTooltip(`<strong>Zone ${feature.properties.zone_id}</strong><br/>Priority: ${feature.properties.priority || 'N/A'}<br/>Plantable Area: ${feature.properties.plantable_ha?.toFixed(2)} ha`, {
      sticky: true,
      className: 'bg-slate-900 text-white border-0 shadow-xl rounded px-3 py-2 text-xs'
    });
  };

  const features = geoData?.features || [];
  const selectedFeature = features.find((f: any) => f.properties.zone_id === selectedZone?.zone_id);
  const sortedFeatures = [...features].sort((a, b) => {
    const aRank = a.properties.rank || a.properties.zone_id;
    const bRank = b.properties.rank || b.properties.zone_id;
    return aRank - bRank;
  });

  // Calculate Chart Data (Memoized to prevent canvas redraws on hover)
  const chartData = React.useMemo(() => {
    const priorityCounts = { High: 0, Medium: 0, Low: 0 };
    const priorityArea = { High: 0, Medium: 0, Low: 0 };
    features.forEach((f: any) => {
      const p = f.properties.priority;
      if (p in priorityCounts) {
        priorityCounts[p as keyof typeof priorityCounts]++;
        priorityArea[p as keyof typeof priorityArea] += f.properties.plantable_ha || 0;
      }
    });

    return {
      counts: priorityCounts,
      data: {
        labels: ['High Priority', 'Medium Priority', 'Low Priority'],
        datasets: [{
          data: [priorityArea.High, priorityArea.Medium, priorityArea.Low],
          backgroundColor: ['#22c55e', '#eab308', '#ef4444'],
          borderWidth: 0,
          hoverOffset: 4
        }]
      }
    };
  }, [features]);

  return (
    <div className="h-full flex flex-col p-8 gap-4 max-w-[1600px] mx-auto">
      
      {/* Header & Summary Cards */}
      <div>
        <div className="flex justify-between items-end">
          <div>
            <h1 className="text-3xl font-bold text-slate-900 tracking-tight">Plantation Suitability</h1>
            <p className="text-slate-500 mt-2 text-lg">Where should trees be planted? GIS-based relative suitability ranking.</p>
          </div>
          <button onClick={exportData} className="flex items-center gap-2 bg-white border border-slate-200/60 shadow-sm px-4 py-2.5 rounded-lg text-sm font-semibold text-slate-700 hover:bg-slate-50 hover:text-blue-600 transition-colors">
            <Download size={16} />
            Export GeoJSON
          </button>
        </div>
        
        <div className="grid grid-cols-5 gap-4 mt-6 mb-2">
          <div className="bg-white p-4 rounded-xl border border-slate-200/60 shadow-sm flex items-center justify-between">
             <span className="text-xs font-bold text-slate-400 uppercase">Priority Zones</span>
             <span className="text-2xl font-bold text-slate-800">{summary ? summary.n_zones : '...'}</span>
          </div>
          <div className="bg-white p-4 rounded-xl border border-slate-200/60 shadow-sm flex items-center justify-between">
             <span className="text-xs font-bold text-slate-400 uppercase">High Priority</span>
             <span className="text-2xl font-bold text-green-600">{chartData.counts.High}</span>
          </div>
          <div className="bg-white p-4 rounded-xl border border-slate-200/60 shadow-sm flex items-center justify-between">
             <span className="text-xs font-bold text-slate-400 uppercase">Priority Area</span>
             <span className="text-2xl font-bold text-slate-800">{summary ? summary.plantable_ha.toFixed(2) : '...'} <span className="text-sm font-normal text-slate-400">ha</span></span>
          </div>
          <div className="bg-white p-4 rounded-xl border border-slate-200/60 shadow-sm flex items-center justify-between">
             <span className="text-xs font-bold text-slate-400 uppercase">Planning Density</span>
             <span className="text-2xl font-bold text-slate-800">1000 <span className="text-sm font-normal text-slate-400">/ha</span></span>
          </div>
          <div className="bg-white p-4 rounded-xl border border-slate-200/60 shadow-sm flex items-center justify-between bg-green-50 border-green-100">
             <span className="text-xs font-bold text-green-600 uppercase">Estimated Trees</span>
             <span className="text-2xl font-bold text-green-700">{summary ? summary.recommended_trees.toLocaleString() : '...'}</span>
          </div>
        </div>
      </div>

      <div className="flex-1 flex gap-6 mt-2 h-0">
        <div className="flex-1 bg-white rounded-2xl shadow-sm border border-slate-200/60 overflow-hidden relative group h-full">
          {loading ? (
            <Loader text="Loading spatial data..." />
          ) : (
            <MapViewer geoData={geoData}>
              {geoData && (
                <>
                  {/* Base Layer */}
                  <GeoJSON 
                    key={`${year}-${scenario}-base`} 
                    data={geoData} 
                    style={getBaseStyle} 
                    onEachFeature={onEachFeature} 
                  />
                  {/* Selected Layer */}
                  {selectedFeature && (
                    <GeoJSON 
                      key={`selected-${selectedFeature.properties.zone_id}`} 
                      data={selectedFeature} 
                      style={getSelectedStyle} 
                      interactive={false} 
                    />
                  )}
                </>
              )}
            </MapViewer>
          )}
          
          <div className="absolute bottom-6 left-6 bg-white/95 backdrop-blur-sm p-4 rounded-xl shadow-lg border border-slate-200/50 z-[1000] pointer-events-none flex gap-6 items-center">
             <div>
               <h4 className="font-bold text-xs uppercase tracking-wider text-slate-500 mb-3">Priority Levels</h4>
               <div className="space-y-2">
                 <div className="flex items-center gap-3 text-sm font-medium text-slate-700"><div className="w-4 h-4 rounded bg-[#22c55e] border border-white shadow-sm"></div> High</div>
                 <div className="flex items-center gap-3 text-sm font-medium text-slate-700"><div className="w-4 h-4 rounded bg-[#eab308] border border-white shadow-sm"></div> Medium</div>
                 <div className="flex items-center gap-3 text-sm font-medium text-slate-700"><div className="w-4 h-4 rounded bg-[#ef4444] border border-white shadow-sm"></div> Low</div>
               </div>
             </div>
             <div className="w-px h-16 bg-slate-200 mx-2"></div>
             <div className="w-24 h-24 relative">
                {features.length > 0 && (
                  <Doughnut 
                    data={chartData.data} 
                    options={chartOptions} 
                  />
                )}
                <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                  <PieChartIcon size={20} className="text-slate-300" />
                </div>
             </div>
          </div>
        </div>
        
        {/* Side Panel */}
        <div className="w-[380px] bg-white rounded-2xl shadow-sm border border-slate-200/60 flex flex-col h-full overflow-hidden">
          <div className="p-5 border-b border-slate-100 bg-slate-50/50">
            <h3 className="font-bold text-slate-900 flex items-center gap-2">
              <MapPin size={18} className="text-blue-500" />
              Zone Inspector
            </h3>
            <p className="text-xs text-slate-500 mt-1">Hover or click a polygon to view planning details.</p>
          </div>
          
          <div className="flex-1 overflow-auto flex flex-col scroll-smooth">
            {selectedZone ? (
              <ZoneDetailCard zone={selectedZone} onClear={() => setSelectedZone(null)} />
            ) : (
              <div className="p-8 bg-slate-50/50 border-b border-slate-100 text-sm text-slate-400 text-center flex flex-col items-center justify-center gap-3">
                <MapPin size={32} className="text-slate-300 opacity-50" />
                <span>Select a zone to view detailed metrics.</span>
              </div>
            )}

            {/* List of Zones */}
            <div className="p-3 overflow-auto flex-1">
              <h4 className="text-[11px] font-bold text-slate-400 uppercase tracking-wider px-3 py-3 flex justify-between items-center">
                <span>Available Zones</span>
                <span className="bg-slate-100 text-slate-600 px-2 py-0.5 rounded-full">{sortedFeatures.length}</span>
              </h4>
              
              {loading ? (
                <div className="py-8"><Loader text="Loading zones..." /></div>
              ) : (
                <div className="space-y-1.5">
                  {sortedFeatures.map((f: any) => (
                    <button
                      key={f.properties.zone_id}
                      onClick={() => setSelectedZone(f.properties)}
                      className={`w-full text-left px-4 py-3 rounded-xl text-sm transition-all flex items-center justify-between border ${
                        selectedZone?.zone_id === f.properties.zone_id 
                          ? 'bg-blue-50/80 border-blue-200 text-blue-900 shadow-sm font-semibold' 
                          : 'bg-white border-transparent hover:bg-slate-50 hover:border-slate-200 text-slate-700'
                      }`}
                    >
                      <div className="flex items-center gap-3">
                        <span className="text-slate-400 text-xs w-4">#{f.properties.rank}</span>
                        <span>Zone {f.properties.zone_id}</span>
                      </div>
                      <div className="flex items-center gap-3">
                        <span className="text-xs text-slate-400 font-mono">{f.properties.plantable_ha?.toFixed(1)}ha</span>
                        <div className={`w-2 h-2 rounded-full ${
                          f.properties.priority === 'High' ? 'bg-green-500' : 
                          f.properties.priority === 'Medium' ? 'bg-yellow-500' : 
                          'bg-red-500'
                        }`} title={f.properties.priority} />
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const ZoneDetailCard = ({ zone, onClear }: any) => (
  <div className="p-5 space-y-4 border-b border-slate-100 transition-colors bg-blue-50/30">
    <div className="flex justify-between items-center mb-1">
       <div className="text-[10px] font-bold tracking-widest uppercase px-2 py-0.5 rounded-full text-blue-600 bg-blue-100">
         Selected
       </div>
       <button onClick={onClear} className="text-xs text-slate-400 hover:text-slate-700 font-medium transition-colors">Clear</button>
    </div>
    
    <div className="bg-white p-5 rounded-xl border border-slate-200/60 shadow-sm relative overflow-hidden">
      <div className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-1">Zone {zone.zone_id}</div>
      <div className="text-3xl font-bold text-slate-900 mb-5 tracking-tight">Priority Rank {zone.rank || 'N/A'}</div>
      
      <div className="space-y-3">
        <DetailRow label="Suitability Score" value={<span className="font-bold">{zone.mean_suitability?.toFixed(1)} / 100</span>} />
        <DetailRow label="Heat Severity" value={
          <span className={`font-bold ${
            zone.priority === 'High' ? 'text-red-600' : 
            zone.priority === 'Medium' ? 'text-orange-500' : 
            'text-yellow-600'
          }`}>{zone.priority}</span>
        } />
        <DetailRow label="Plantable Area" value={<span className="font-mono text-green-700 font-bold">{zone.plantable_ha?.toFixed(2)} ha</span>} />
        {zone.excluded_ha > 0 && (
          <DetailRow label="Excluded (Constraints)" value={<span className="font-mono text-red-500 text-xs">{zone.excluded_ha?.toFixed(2)} ha</span>} />
        )}
      </div>

      <div className="mt-5 pt-4 border-t border-slate-100">
        <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-3">Tree Scenarios</h4>
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-slate-500">400 / ha (Low)</span>
            <span className="font-mono font-medium">{zone.trees_400?.toLocaleString() || 'N/A'}</span>
          </div>
          <div className="flex justify-between text-sm bg-green-50 text-green-800 p-2 rounded -mx-2">
            <span className="font-medium">1000 / ha (Primary)</span>
            <span className="font-mono font-bold">{zone.trees_1000?.toLocaleString() || 'N/A'}</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-slate-500">2500 / ha (High)</span>
            <span className="font-mono font-medium">{zone.trees_2500?.toLocaleString() || 'N/A'}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
);

const DetailRow = ({ label, value }: { label: string, value: React.ReactNode }) => (
  <div className="flex justify-between items-center py-2 border-b border-dashed border-slate-100 last:border-0">
    <span className="text-slate-500 text-sm">{label}</span>
    <span className="text-slate-900">{value}</span>
  </div>
);

export default PlantationSuitability;
