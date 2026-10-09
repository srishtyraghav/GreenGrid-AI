import React, { useEffect, useState } from 'react';
import { useAppContext } from '../context/AppContext';
import { fetchTreeRequirementZones } from '../services/api';
import { Bar } from 'react-chartjs-2';
import { Loader } from '../components/Loader';
import { TreePine, Download } from 'lucide-react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const TreeRequirements = () => {
  const { year, scenario, density, setDensity } = useAppContext();
  const [zones, setZones] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    fetchTreeRequirementZones(year, scenario)
      .then(data => {
        const sorted = data.sort((a: any, b: any) => a.rank - b.rank);
        setZones(sorted);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [year, scenario]);

  const getTreeCountForDensity = (zone: any) => {
    if (density === 400) return zone.recommended_trees_400;
    if (density === 1000) return zone.recommended_trees_1000;
    if (density === 2500) return zone.recommended_trees_2500;
    return zone.recommended_trees; // fallback
  };

  const totalTrees = zones.reduce((acc, zone) => acc + getTreeCountForDensity(zone), 0);
  const totalPlantable = zones.reduce((acc, zone) => acc + zone.plantable_ha, 0);

  const chartData = {
    labels: zones.map(z => `Zone ${z.zone_id}`),
    datasets: [
      {
        label: 'Estimated Trees',
        data: zones.map(z => getTreeCountForDensity(z)),
        backgroundColor: '#22c55e',
        borderRadius: 4,
        hoverBackgroundColor: '#16a34a',
      }
    ]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: '#1e293b',
        padding: 12,
        titleFont: { size: 14, family: 'system-ui' },
        bodyFont: { size: 13, family: 'system-ui' },
        displayColors: false,
        callbacks: {
          label: function(context: any) {
            return context.parsed.y.toLocaleString() + ' trees';
          }
        }
      }
    },
    scales: {
      y: { 
        beginAtZero: true, 
        grid: { color: '#f1f5f9' },
        border: { display: false }
      },
      x: { 
        grid: { display: false },
        border: { display: false }
      }
    }
  };

  const exportCSV = () => {
    if (!zones.length) return;
    const header = Object.keys(zones[0]).join(",");
    const rows = zones.map(z => Object.values(z).join(","));
    const csvContent = [header, ...rows].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `tree_requirements_${scenario}_${year}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="p-8 max-w-[1600px] mx-auto space-y-6 h-full flex flex-col">
      <header className="flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-bold text-slate-900 tracking-tight">Tree Requirements</h1>
          <p className="text-slate-500 mt-2 text-lg">Planning estimates based on plantable area and density scenarios.</p>
        </div>
        <button onClick={exportCSV} className="flex items-center gap-2 bg-white border border-slate-200/60 shadow-sm px-4 py-2.5 rounded-lg text-sm font-semibold text-slate-700 hover:bg-slate-50 hover:text-blue-600 transition-colors">
          <Download size={16} />
          Export CSV
        </button>
      </header>

      <div className="bg-white p-8 rounded-2xl shadow-sm border border-slate-200/60 flex flex-wrap items-center justify-between gap-6 relative overflow-hidden">
        <div className="absolute top-0 right-0 w-32 h-32 bg-green-50 rounded-bl-full -z-10 opacity-50"></div>
        
        <div>
           <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-3">Planning Density Scenarios</h3>
           <div className="flex bg-slate-100 rounded-xl p-1.5 shadow-inner">
             {[
               { val: 400, label: 'Low (400/ha)' },
               { val: 1000, label: 'Primary (1000/ha)' },
               { val: 2500, label: 'High (2500/ha)' }
             ].map(d => (
               <button 
                 key={d.val}
                 onClick={() => setDensity(d.val)}
                 className={`px-5 py-2.5 text-sm font-semibold rounded-lg transition-all duration-200 ${
                   density === d.val 
                    ? 'bg-white shadow-sm text-green-700 border border-slate-200/50' 
                    : 'text-slate-500 hover:text-slate-700 hover:bg-slate-200/50 border border-transparent'
                 }`}
               >
                 {d.label}
               </button>
             ))}
           </div>
           <p className="text-[10px] text-slate-400 mt-3 font-medium">
             * These are planning scenarios, not scientifically validated optimal planting densities.
           </p>
        </div>
        
        <div className="text-right flex items-center gap-6">
          <div className="p-4 bg-green-50 rounded-2xl text-green-600">
            <TreePine size={40} strokeWidth={1.5} />
          </div>
          <div>
            <div className="text-sm text-slate-500 font-medium tracking-wide">Total Estimated Trees</div>
            <div className="text-4xl font-bold text-slate-900 mt-1">
              {loading ? <span className="text-slate-300">...</span> : totalTrees.toLocaleString()}
            </div>
            <div className="text-sm text-slate-400 mt-1 font-mono">
              across <span className="font-semibold text-slate-500">{totalPlantable.toFixed(2)}</span> plantable ha
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 flex-1 min-h-[400px]">
        {/* Chart Card */}
        <div className="bg-white rounded-2xl shadow-sm border border-slate-200/60 p-6 flex flex-col">
          <h3 className="font-bold text-slate-800 mb-6">Distribution by Zone</h3>
          <div className="flex-1 relative w-full h-full min-h-[300px]">
            {loading ? <Loader /> : <Bar data={chartData} options={chartOptions} />}
          </div>
        </div>

        {/* Table Card */}
        <div className="bg-white rounded-2xl shadow-sm border border-slate-200/60 overflow-hidden flex flex-col">
          <div className="p-6 border-b border-slate-100 flex justify-between items-center bg-slate-50/50">
            <h3 className="font-bold text-slate-800">Zone Details Table</h3>
            <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">{zones.length} Zones</span>
          </div>
          
          <div className="flex-1 overflow-auto">
            {loading ? (
              <Loader />
            ) : (
              <table className="w-full text-left text-sm whitespace-nowrap">
                <thead className="bg-white text-slate-400 text-xs uppercase tracking-wider sticky top-0 z-10 shadow-sm">
                  <tr>
                    <th className="px-6 py-4 font-semibold">Rank</th>
                    <th className="px-6 py-4 font-semibold">Zone</th>
                    <th className="px-6 py-4 font-semibold">Priority</th>
                    <th className="px-6 py-4 font-semibold text-right">Plantable Area</th>
                    <th className="px-6 py-4 font-semibold text-right">Trees</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {zones.map((zone) => (
                    <tr key={zone.zone_id} className="hover:bg-blue-50/50 transition-colors group">
                      <td className="px-6 py-4 text-slate-400 font-medium">#{zone.rank}</td>
                      <td className="px-6 py-4 font-bold text-slate-700">Zone {zone.zone_id}</td>
                      <td className="px-6 py-4">
                        <span className={`px-2.5 py-1 rounded-md text-[10px] uppercase font-bold tracking-wider ${
                          zone.priority === 'High' ? 'bg-green-100 text-green-700' : 
                          zone.priority === 'Medium' ? 'bg-yellow-100 text-yellow-700' : 
                          'bg-red-100 text-red-700'
                        }`}>
                          {zone.priority}
                        </span>
                      </td>
                      <td className="px-6 py-4 text-right font-mono text-slate-500">
                        {zone.plantable_ha.toFixed(2)} ha
                      </td>
                      <td className="px-6 py-4 text-right font-bold text-slate-800 text-base">
                        {getTreeCountForDensity(zone).toLocaleString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default TreeRequirements;
