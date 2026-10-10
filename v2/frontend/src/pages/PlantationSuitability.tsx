import { useEffect, useState } from 'react';
import { useAppContext } from '../context/AppContext';
import {
  fetchPriorityZones, fetchTreeRequirementZones, fetchTreeRequirementSummary,
} from '../services/api';
import { MapViewer } from '../components/MapViewer';
import { GeoJSON } from 'react-leaflet';
import { Loader } from '../components/Loader';
import { Download } from 'lucide-react';
import { Doughnut } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';

ChartJS.register(ArcElement, Tooltip, Legend);

const CERT_STYLE: Record<string, string> = {
  low: 'bg-amber-100 text-amber-700',
  mixed: 'bg-yellow-100 text-yellow-700',
  identified: 'bg-green-100 text-green-700',
};

const PlantationSuitability = () => {
  const { scenario } = useAppContext();
  const [geoData, setGeoData] = useState<any>(null);
  const [zones, setZones] = useState<any[]>([]);
  const [summary, setSummary] = useState<any>(null);
  const [selected, setSelected] = useState<any>(null);
  const [showAll, setShowAll] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetchPriorityZones(2026, scenario),
      fetchTreeRequirementZones(2026, scenario),
      fetchTreeRequirementSummary(2026, scenario),
    ]).then(([geo, zrows, summ]) => {
      const byId = new Map((zrows as any[]).map((z: any) => [z.zone_id, z]));
      (geo.features || []).forEach((f: any) => {
        const z = byId.get(f.properties.zone_id);
        if (z) {
          f.properties.landuse_certainty = z.landuse_certainty;
          f.properties.untagged_share = z.untagged_share;
          f.properties.rationale = z.rationale;
        }
      });
      setGeoData(geo);
      setZones(zrows as any[]);
      setSummary((summ as any[])[0] || null);
      setSelected(null);
    }).catch(console.error).finally(() => setLoading(false));
  }, [scenario]);

  const features = (geoData?.features || []).filter(
    (f: any) => showAll || f.properties?.shortlist === true);   // shortlist, or all candidates when toggled
  const selectedZone = zones.find((z) => z.zone_id === selected?.zone_id);

  // Priority presentation bands: shortlist rank terciles (fixed, ranking unchanged).
  // High = ranks 1..ceil(n/3), Medium = next third, Low = final third.
  const CLASS_COLORS: Record<string, string> = { High: '#dc2626', Medium: '#f59e0b', Low: '#2563eb' };
  const nSl = zones.filter((z) => z.shortlist).length;
  const classOf = (rank: number, n: number) =>
    rank <= Math.ceil(n / 3) ? 'High' : rank <= Math.ceil((2 * n) / 3) ? 'Medium' : 'Low';
  const zoneColor = (f: any): string => {
    if (f.properties?.oversized) return '#94a3b8';
    if (f.properties?.shortlist !== true) return '#cbd5e1';
    const n = features.length || 1;
    return CLASS_COLORS[classOf(f.properties.rank ?? zones.find((z) => z.zone_id === f.properties.zone_id)?.rank ?? n, n)];
  };

  const exportData = () => {
    if (!geoData) return;
    const blob = new Blob([JSON.stringify(geoData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `stable_zones_${scenario}.geojson`;
    link.click();
    URL.revokeObjectURL(url);
  };

  // Priority class shares over the shortlist (by site count) for the map pie.
  const shortlistZones = zones.filter((z) => z.shortlist);
  const classCounts = (['High', 'Medium', 'Low'] as const).map(
    (c) => shortlistZones.filter((z) => classOf(z.rank_excl, nSl) === c).length);
  const classShare = classCounts.map((n) =>
    shortlistZones.length ? Math.round((n / shortlistZones.length) * 100) : 0);
  const pieData = {
    labels: ['High', 'Medium', 'Low'],
    datasets: [{
      data: classCounts,
      backgroundColor: [CLASS_COLORS.High, CLASS_COLORS.Medium, CLASS_COLORS.Low],
      borderWidth: 1, borderColor: '#ffffff',
    }],
  };
  const pieOptions = {
    plugins: {
      legend: { display: false },
      tooltip: { enabled: false },
    },
    cutout: '0%',
    animation: { duration: 0 },
  };


  return (
    <div className="h-full flex flex-col p-8 gap-4 max-w-[1600px] mx-auto">
      <div className="flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-bold text-slate-900 tracking-tight">Plantation Suitability</h1>
          <p className="text-slate-500 mt-2 text-lg">
            2026 shortlist ranked by the 0-100 planning-priority score. Relative classification, not absolute heat.
          </p>
        </div>
        <button onClick={exportData} className="flex items-center gap-2 bg-white border border-slate-200/60 shadow-sm px-4 py-2.5 rounded-lg text-sm font-semibold text-slate-700 hover:bg-slate-50">
          <Download size={16} /> Export GeoJSON
        </button>
      </div>

      <div className="grid grid-cols-4 gap-4 mt-2">
        <div className="bg-white p-4 rounded-xl border border-slate-200/60 shadow-sm flex items-center justify-between">
          <span className="text-xs font-bold text-slate-400 uppercase">Shortlist zones</span>
          <span className="text-2xl font-bold text-green-700">{summary ? summary.shortlist_zones : '...'}</span>
        </div>
        <div className="bg-white p-4 rounded-xl border border-slate-200/60 shadow-sm flex items-center justify-between">
          <span className="text-xs font-bold text-slate-400 uppercase">Shortlist area</span>
          <span className="text-2xl font-bold text-slate-800">{summary ? Number(summary.shortlist_ha).toLocaleString() : '...'} <span className="text-sm font-normal text-slate-400">ha</span></span>
        </div>
        <div className="bg-white p-4 rounded-xl border border-slate-200/60 shadow-sm flex items-center justify-between">
          <span className="text-xs font-bold text-slate-400 uppercase">All candidate zones</span>
          <span className="text-2xl font-bold text-slate-800">{summary ? summary.n_zones : '...'}</span>
        </div>
        <div className="bg-white p-4 rounded-xl border border-slate-200/60 shadow-sm flex items-center justify-between">
          <span className="text-xs font-bold text-slate-400 uppercase">Plantable 2026</span>
          <span className="text-2xl font-bold text-slate-800">{summary ? Number(summary.annual_plantable_ha).toLocaleString() : '...'} <span className="text-sm font-normal text-slate-400">ha</span></span>
        </div>
      </div>

      <div className="flex-1 flex gap-6 mt-2 h-0">
        <div className="flex-1 bg-white rounded-2xl shadow-sm border border-slate-200/60 overflow-hidden relative h-full">
          {loading ? <Loader text="Loading spatial data..." /> : (
            <MapViewer geoData={geoData}>
              {features.length > 0 && (
                <GeoJSON key={`2026-${scenario}-${showAll ? 'all' : 'short'}`} data={({ type: 'FeatureCollection', features } as any)}
                  style={(f: any) => ({
                    fillColor: zoneColor(f),
                    weight: f?.properties?.zone_id === selected?.zone_id ? 3.5 : f?.properties?.shortlist ? 1.5 : 1,
                    opacity: 0.9,
                    color: f?.properties?.zone_id === selected?.zone_id ? '#7c3aed' : 'white',
                    fillOpacity: f?.properties?.zone_id === selected?.zone_id ? 0.9 : f?.properties?.shortlist ? 0.7 : 0.45,
                  })}
                  onEachFeature={(f: any, layer: any) => {
                    layer.on({ click: () => setSelected(f.properties) });
                    layer.bindTooltip(
                      `<strong>Zone ${f.properties.zone_id}</strong><br/>` +
                      `score ${f.properties.planning_priority_score} ` +
                      `(${f.properties.shortlist ? 'SHORTLIST' : f.properties.oversized ? 'oversized' : 'candidate'})<br/>` +
                      `usable ${Number(f.properties.usable_area_ha).toFixed(1)} ha`,
                      { sticky: true, className: 'bg-slate-900 text-white border-0 shadow-xl rounded px-3 py-2 text-xs' });
                  }} />
              )}
              <div className="absolute bottom-6 left-6 z-[1000] bg-white/95 backdrop-blur-sm p-4 rounded-xl shadow-lg border border-slate-200/50 w-56">
                <h4 className="font-bold text-[11px] text-slate-800 mb-2 uppercase tracking-wider">Planning priority share</h4>
                <div className="flex items-center gap-3">
                  <div className="w-24 h-24 shrink-0"><Doughnut data={pieData} options={pieOptions as any} /></div>
                  <div className="space-y-1 text-xs text-slate-600">
                    {(['High', 'Medium', 'Low'] as const).map((c, i) => (
                      <div key={c} className="flex items-center gap-1.5">
                        <span className="inline-block w-3 h-3 rounded" style={{ background: CLASS_COLORS[c] }} />
                        <span>{c}</span>
                        <span className="font-mono text-slate-500">{classShare[i]}%</span>
                      </div>
                    ))}
                  </div>
                </div>
                <p className="text-[9px] text-slate-400 mt-2">Share of {shortlistZones.length} shortlisted sites (rank terciles)</p>
              </div>
            </MapViewer>
          )}
        </div>

        <div className="w-[500px] bg-white rounded-2xl shadow-sm border border-slate-200/60 flex flex-col h-full overflow-hidden">
          <div className="p-4 border-b border-slate-100">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">
                Recommended shortlist - 2026 ({zones.filter((z) => z.shortlist).length} sites)
              </h3>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setShowAll(!showAll)}
                  className={`px-2 py-0.5 rounded-full text-[10px] font-bold uppercase border transition-colors ${
                    showAll ? 'bg-slate-700 text-white border-slate-700' : 'bg-white text-slate-500 border-slate-300 hover:border-slate-500'
                  }`}
                  title="Toggle all 286 candidate zones on the map"
                >
                  {showAll ? 'Showing all candidates' : 'Show all candidates'}
                </button>
                <span className="bg-slate-100 text-slate-600 px-2 py-0.5 rounded-full text-xs">{features.length}</span>
              </div>
            </div>
          </div>

          {selectedZone ? (
            <div className="p-4 overflow-auto flex-1">
              <div className="flex justify-between items-center mb-2">
                <span className="text-[10px] font-bold tracking-widest uppercase px-2 py-0.5 rounded-full text-blue-600 bg-blue-100">Zone {selectedZone.zone_id}</span>
                <button onClick={() => setSelected(null)} className="text-xs text-slate-400 hover:text-slate-700">Back to list</button>
              </div>
              <div className="text-2xl font-bold text-slate-900">Rank {selectedZone.rank_excl} - score {selectedZone.planning_priority_score}/100</div>
              <p className="text-xs text-slate-500 mt-1">{selectedZone.score_label}</p>
              <p className="text-xs text-slate-600 mt-2 bg-slate-50 rounded-lg p-2">{selectedZone.rationale}</p>
              <div className="mt-3 space-y-1 text-sm">
                <div className="flex justify-between"><span className="text-slate-500">Heat need component</span><span className="font-mono">{selectedZone.heat_need_comp}</span></div>
                <div className="flex justify-between"><span className="text-slate-500">Vegetation deficit component</span><span className="font-mono">{selectedZone.vegetation_deficit_comp}</span></div>
                <div className="flex justify-between"><span className="text-slate-500">Opportunity component</span><span className="font-mono">{selectedZone.opportunity_comp}</span></div>
                <div className="flex justify-between"><span className="text-slate-500">Usable area</span><span className="font-mono">{Number(selectedZone.usable_area_ha).toFixed(2)} ha</span></div>
                <div className="flex justify-between"><span className="text-slate-500">Untagged OSM share</span><span className="font-mono">{(Number(selectedZone.untagged_share) * 100).toFixed(0)}%</span></div>
                <div className="flex justify-between"><span className="text-slate-500">Confidence</span>
                  <span className={`px-2 py-0.5 rounded text-[10px] uppercase font-bold ${CERT_STYLE[selectedZone.landuse_certainty]}`}>{selectedZone.landuse_certainty}</span></div>
                <div className="flex justify-between"><span className="text-slate-500">Water / buildings / roads</span>
                  <span className="font-mono">{`${selectedZone.has_water}/${selectedZone.has_buildings}/${selectedZone.has_road_surfaces}`}</span></div>
              </div>
              <p className="text-[10px] text-amber-700 mt-3">Untagged OSM land is not confirmed available planting space.</p>
            </div>
          ) : (
            <div className="flex-1 overflow-auto">
              {loading ? <Loader /> : (
                <table className="w-full text-left text-sm whitespace-nowrap">
                  <thead className="bg-white text-slate-400 text-xs uppercase tracking-wider sticky top-0 z-10 shadow-sm">
                    <tr>
                      <th className="px-3 py-3 font-semibold">Rank</th>
                      <th className="px-3 py-3 font-semibold">Zone</th>
                      <th className="px-3 py-3 font-semibold">Priority</th>
                      <th className="px-3 py-3 font-semibold text-right">Usable ha</th>
                      <th className="px-3 py-3 font-semibold">Confidence</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {zones.filter((z) => z.shortlist).map((z) => (
                      <tr key={z.zone_id} onClick={() => setSelected(z)}
                        className={`cursor-pointer hover:bg-blue-50/60 ${z.shortlist ? '' : 'text-slate-400'}`}>
                        <td className="px-3 py-2.5 text-slate-400">#{z.rank_excl}{z.oversized ? ' (oversized)' : ''}</td>
                        <td className="px-3 py-2.5 font-bold text-slate-700">Zone {z.zone_id}</td>
                        <td className="px-3 py-2.5">
                          <span className="inline-block px-2 py-0.5 rounded text-[10px] uppercase font-bold text-white" style={{ background: CLASS_COLORS[classOf(z.rank_excl, nSl)] }}>
                            {classOf(z.rank_excl, nSl)}
                          </span>
                          <div className="text-[11px] font-mono text-slate-500 mt-0.5">{z.planning_priority_score}</div>
                        </td>
                        <td className="px-3 py-2.5 text-right font-mono">{Number(z.usable_area_ha).toFixed(1)}</td>
                        <td className="px-3 py-2.5">
                          <span className={`px-2 py-0.5 rounded text-[10px] uppercase font-bold ${CERT_STYLE[z.landuse_certainty]}`}>{z.landuse_certainty}</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
export default PlantationSuitability;
