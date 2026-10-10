import { useEffect, useState } from 'react';
import { useAppContext } from '../context/AppContext';
import { fetchTreeRequirementZones } from '../services/api';
import { Loader } from '../components/Loader';
import { Download } from 'lucide-react';

const QUICK_PICKS = [400, 1000, 2500];

const CERT_STYLE: Record<string, string> = {
  low: 'bg-amber-100 text-amber-700',
  mixed: 'bg-yellow-100 text-yellow-700',
  identified: 'bg-green-100 text-green-700',
};

const TreeRequirements = () => {
  const { scenario, density, setDensity,
          selectedSites, setSelectedSites, treeCapacityLimit, setTreeCapacityLimit } = useAppContext();
  const [allZones, setAllZones] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    fetchTreeRequirementZones(2026, scenario)
      .then((data) => setAllZones((data as any[]).sort((a, b) => a.rank_excl - b.rank_excl)))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [scenario]);

  const zones = allZones.filter((z) => z.shortlist);   // 2026 shortlist ONLY
  const treesAt = (z: any, d: number) => Math.floor(z.usable_area_ha * d);
  const selectedZones = allZones.filter((z) => selectedSites.has(z.zone_id));
  const planTrees = selectedZones.reduce((a, z) => a + treesAt(z, density), 0);
  const planArea = selectedZones.reduce((a, z) => a + z.usable_area_ha, 0);

  // Priority presentation bands: shortlist rank terciles (fixed, ranking unchanged).
  const CLASS_COLORS: Record<string, string> = { High: '#dc2626', Medium: '#f59e0b', Low: '#2563eb' };
  const classOf = (rank: number) => {
    const n = zones.length || 1;
    return rank <= Math.ceil(n / 3) ? 'High' : rank <= Math.ceil((2 * n) / 3) ? 'Medium' : 'Low';
  };

  const toggle = (id: number) => {
    const next = new Set(selectedSites);
    if (next.has(id)) next.delete(id); else next.add(id);
    setSelectedSites(next);
  };

  const autoSelect = () => {
    const next = new Set<number>();
    let acc = 0;
    for (const z of allZones.filter((r) => r.shortlist)) {
      const t = treesAt(z, density);
      if (treeCapacityLimit != null && acc + t > treeCapacityLimit) continue;
      next.add(z.zone_id);
      acc += t;
    }
    setSelectedSites(next);
  };

  const exportCSV = () => {
    if (!allZones.length) return;
    const header = Object.keys(allZones[0]).join(',');
    const rows = allZones.map((z) => Object.values(z).join(','));
    const blob = new Blob([[header, ...rows].join('\n')], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `stable_zones_${scenario}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="p-8 max-w-[1600px] mx-auto space-y-6 h-full flex flex-col">
      <header className="flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-bold text-slate-900 tracking-tight">Tree Requirements</h1>
        </div>
        <button onClick={exportCSV} className="flex items-center gap-2 bg-white border border-slate-200/60 shadow-sm px-4 py-2.5 rounded-lg text-sm font-semibold text-slate-700 hover:bg-slate-50">
          <Download size={16} /> Export CSV
        </button>
      </header>

      <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200/60 flex flex-wrap items-center gap-8">
        <div>
          <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-2">
            Density (trees/ha) - planning assumption
          </h3>
          <div className="flex items-center gap-3">
            <input type="range" min={100} max={2500} step={50} value={density}
                   onChange={(e) => setDensity(Number(e.target.value))} className="w-56 accent-green-600" />
            <span className="font-mono font-bold text-slate-800 w-14">{density}</span>
            {QUICK_PICKS.map((q) => (
              <button key={q} onClick={() => setDensity(q)}
                className={`px-3 py-1.5 text-xs font-semibold rounded-lg border ${density === q ? 'bg-green-50 border-green-300 text-green-700' : 'border-slate-200 text-slate-500'}`}>
                {q}
              </button>
            ))}
          </div>
        </div>
        <div>
          <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-2">Tree capacity limit (optional)</h3>
          <div className="flex items-center gap-2">
            <input type="number" min={0} placeholder="no limit" value={treeCapacityLimit ?? ''}
                   onChange={(e) => setTreeCapacityLimit(e.target.value === '' ? null : Number(e.target.value))}
                   className="w-36 border border-slate-200 rounded-lg px-3 py-1.5 text-sm" />
            <button onClick={autoSelect}
              className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-blue-50 border border-blue-200 text-blue-700 hover:bg-blue-100">
              Auto-select shortlist in priority order
            </button>
            <button onClick={() => setSelectedSites(new Set())}
              className="px-3 py-1.5 text-xs font-semibold rounded-lg border border-slate-200 text-slate-500">
              Clear selection
            </button>
          </div>
        </div>
        <div className="ml-auto text-right">
          <div className="text-sm text-slate-500 font-medium">
            SELECTED PLAN ({selectedZones.length} of {allZones.filter((z) => z.shortlist).length} shortlist zones)
          </div>
          <div className="text-3xl font-bold text-green-700">
            {planTrees.toLocaleString()} <span className="text-base font-normal text-slate-400">trees @{density}/ha</span>
          </div>
          <div className="text-sm text-slate-400 font-mono">{planArea.toFixed(1)} usable ha</div>
          <div className="text-[10px] text-slate-400">Selection is preserved by stable zone id when the year changes.</div>
        </div>
      </div>

      <div className="bg-white rounded-2xl shadow-sm border border-slate-200/60 overflow-hidden flex flex-col flex-1">
        <div className="p-4 border-b border-slate-100 flex items-center justify-between bg-slate-50/50">
          <h3 className="font-bold text-slate-800">Recommended shortlist - 2026 ({zones.length} sites)</h3>
          <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">2026 only</span>
        </div>

        <div className="flex-1 overflow-auto">
          {loading ? <Loader /> : (
            <table className="w-full text-left text-sm whitespace-nowrap">
              <thead className="bg-white text-slate-400 text-xs uppercase tracking-wider sticky top-0 z-10 shadow-sm">
                <tr>
                  <th className="px-4 py-3"></th>
                  <th className="px-4 py-3 font-semibold">Rank</th>
                  <th className="px-4 py-3 font-semibold">Zone</th>
                  <th className="px-4 py-3 font-semibold text-right">Score</th>
                  <th className="px-4 py-3 font-semibold text-right">Usable ha</th>
                  <th className="px-4 py-3 font-semibold text-right">Trees @{density}</th>
                  <th className="px-4 py-3 font-semibold">Confidence</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {zones.map((z) => (
                  <tr key={z.zone_id} className={selectedSites.has(z.zone_id) ? 'bg-green-50/60' : 'hover:bg-slate-50'}>
                    <td className="px-4 py-2.5">
                      <input type="checkbox" checked={selectedSites.has(z.zone_id)} onChange={() => toggle(z.zone_id)} />
                    </td>
                    <td className="px-4 py-2.5 text-slate-400 font-medium">#{z.rank_excl}{z.oversized ? ' (oversized)' : ''}</td>
                    <td className="px-4 py-2.5 font-bold text-slate-700">Zone {z.zone_id}</td>
                    <td className="px-4 py-2.5 text-right">
                      <div className="font-mono text-slate-600">{z.planning_priority_score}</div>
                      <span className="inline-block mt-0.5 px-1.5 py-0.5 rounded text-[10px] font-bold text-white" style={{ background: CLASS_COLORS[classOf(z.rank_excl)] }}>
                        {classOf(z.rank_excl)}
                      </span>
                    </td>
                    <td className="px-4 py-2.5 text-right font-mono text-slate-500">{Number(z.usable_area_ha).toFixed(2)}</td>
                    <td className="px-4 py-2.5 text-right font-bold text-slate-800">{treesAt(z, density).toLocaleString()}</td>
                    <td className="px-4 py-2.5">
                      <span className={`px-2 py-0.5 rounded text-[10px] uppercase font-bold ${CERT_STYLE[z.landuse_certainty]}`}>{z.landuse_certainty}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
        <div className="p-3 border-t border-slate-100 text-[10px] text-slate-400">
          2026 recommendation module. Reference backend columns at 400/1000/2500 exist in the API;
          the plan total above uses the current slider density only.
        </div>
      </div>
    </div>
  );
};

export default TreeRequirements;
