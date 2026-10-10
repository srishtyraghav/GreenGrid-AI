import { useEffect, useState } from 'react';
import { useAppContext } from '../context/AppContext';
import { fetchTreeRequirementSummary } from '../services/api';
import { Leaf, Target, TrendingDown, Layers, Calendar, Cpu, Satellite, Database } from 'lucide-react';

const Overview = ({ setActiveTab }: { setActiveTab: (tab: string) => void }) => {
  const { scenario } = useAppContext();
  const [summary, setSummary] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    fetchTreeRequirementSummary(2026, scenario)
      .then(data => {
        setSummary(Array.isArray(data) && data.length ? data[0] : null);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [scenario]);

  return (
    <div className="p-8 max-w-[1600px] mx-auto space-y-6">
      {/* High-end Technical Header */}
      <header className="relative bg-slate-900 rounded-2xl overflow-hidden shadow-xl border border-slate-800">
        <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] opacity-10"></div>
        <div className="absolute top-0 right-0 p-12 opacity-5 text-slate-100 pointer-events-none">
          <Layers size={400} />
        </div>
        <div className="relative p-10 md:p-12 z-10">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/20 text-blue-300 text-xs font-mono font-bold uppercase tracking-widest mb-6 border border-blue-500/30">
            <span className="w-2 h-2 rounded-full bg-blue-400 animate-pulse"></span>
            System Active
          </div>
          <h1 className="text-4xl md:text-5xl font-black text-white tracking-tight leading-tight">
            GreenGrid AI <span className="text-slate-400 font-light">V2</span>
          </h1>
          <p className="text-xl text-slate-300 mt-4 max-w-2xl leading-relaxed">
            AI-Driven Urban Cooling & Smart Energy Optimization. Decision-support dashboard for geospatial tree-plantation planning.
          </p>
          <div className="flex flex-wrap gap-4 mt-8">
            <Badge icon={Calendar} label="Snapshot: 2026" />
            <Badge icon={Target} label="Delhi NCT Study Area" />
            <Badge icon={Database} label={`Scenario: ${scenario}`} />
          </div>
        </div>
      </header>

      {/* Primary Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
        <StatCard 
          title="Shortlisted Sites" 
          value={loading ? "..." : (summary ? summary.shortlist_zones : "N/A")}
          icon={Target}
          description="Top-ranked 2026 candidate sites"
          color="indigo"
        />
        <StatCard 
          title="Shortlist Area" 
          value={loading ? "..." : (summary ? `${Number(summary.shortlist_ha).toLocaleString()} ha` : "N/A")}
          icon={Layers}
          description="Usable planting area after exclusions"
          color="amber"
        />
        <StatCard 
          title="Shortlist Trees" 
          value={loading ? "..." : (summary ? Math.floor(Number(summary.shortlist_ha) * 1000).toLocaleString() : "N/A")}
          icon={Leaf}
          description="Reference scenario @1,000 trees/ha"
          color="green"
        />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Workflow Pipeline */}
        <div className="xl:col-span-2 bg-white rounded-2xl shadow-sm border border-slate-200/60 p-8">
          <div className="flex justify-between items-end mb-8">
             <div>
               <h2 className="text-xl font-bold text-slate-800">Analytical Pipeline</h2>
               <p className="text-sm text-slate-500 mt-1">From raw satellite telemetry to actionable policy.</p>
             </div>
          </div>
          
          <div className="grid grid-cols-5 gap-3">
            <PipelineNode num="1" title="Heat Analysis" desc="LST UHI Detection" active onClick={() => setActiveTab('heat')} />
            <PipelineNode num="2" title="Environmental" desc="NDVI / NDBI context" active onClick={() => setActiveTab('env')} />
            <PipelineNode num="3" title="Suitability" desc="GIS-based spatial constraints" active onClick={() => setActiveTab('suitability')} />
            <PipelineNode num="4" title="Planning" desc="Scenario generation" active onClick={() => setActiveTab('trees')} />
            <PipelineNode num="5" title="Cooling Predict" desc="ML temperature reduction" active={false} onClick={() => setActiveTab('cooling')} />
          </div>
        </div>

        {/* Technical Specifications */}
        <div className="bg-white rounded-2xl shadow-sm border border-slate-200/60 p-8 flex flex-col">
          <h2 className="text-xl font-bold text-slate-800 mb-6 flex items-center gap-2">
            <Cpu size={20} className="text-slate-400" />
            System Specifications
          </h2>
          <div className="space-y-4 flex-1">
            <SpecRow icon={Satellite} label="LST Telemetry" value="Landsat 9 (30m, W4)" />
            <SpecRow icon={Satellite} label="Multispectral" value="Sentinel-2 (30m, W4)" />
            <SpecRow icon={Layers} label="Spatial Grid" value="EPSG:4326 (30m res)" />
            <SpecRow icon={Database} label="Constraints" value="OSM (Water, Bldgs, Roads)" />
            <SpecRow icon={TrendingDown} label="Phase 9 Status" value="Under Development" highlight />
          </div>
        </div>
      </div>
    </div>
  );
};

const Badge = ({ icon: Icon, label }: any) => (
  <div className="flex items-center gap-2 bg-slate-800/80 backdrop-blur-sm border border-slate-700/50 px-4 py-2 rounded-lg text-slate-200 text-sm font-medium">
    <Icon size={16} className="text-slate-400" />
    {label}
  </div>
);

const StatCard = ({ title, value, icon: Icon, description, color }: any) => {
  const colorMap: Record<string, string> = {
    blue: 'bg-blue-50 text-blue-600',
    green: 'bg-green-50 text-green-600',
    amber: 'bg-amber-50 text-amber-600',
    indigo: 'bg-indigo-50 text-indigo-600',
  };

  return (
    <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200/60 flex flex-col hover:border-slate-300 transition-colors">
      <div className="flex justify-between items-start mb-6">
        <h3 className="text-slate-500 font-bold text-xs uppercase tracking-wider">{title}</h3>
        <div className={`p-2.5 rounded-xl ${colorMap[color] || 'bg-slate-50 text-slate-600'}`}>
          <Icon size={20} strokeWidth={2} />
        </div>
      </div>
      <div className="text-4xl font-black text-slate-800 tracking-tight mb-2">{value}</div>
      <div className="text-sm text-slate-500 mt-auto pt-4 border-t border-slate-100">{description}</div>
    </div>
  );
};

const PipelineNode = ({ num, title, desc, active, onClick }: any) => (
  <button 
    onClick={onClick}
    className={`relative flex flex-col items-center text-center p-4 rounded-xl border transition-all ${
      active 
        ? 'bg-white border-slate-200 shadow-sm hover:border-blue-400 hover:shadow-md cursor-pointer group' 
        : 'bg-slate-50 border-slate-100 opacity-60 cursor-default'
    }`}
  >
    <div className={`w-8 h-8 rounded-full flex items-center justify-center font-black text-xs mb-3 z-10 ${
      active ? 'bg-slate-900 text-white group-hover:bg-blue-500 transition-colors' : 'bg-slate-200 text-slate-500'
    }`}>
      {num}
    </div>
    <h4 className={`text-sm font-bold mb-1 ${active ? 'text-slate-800' : 'text-slate-500'}`}>{title}</h4>
    <p className="text-[10px] text-slate-500 leading-tight">{desc}</p>
  </button>
);

const SpecRow = ({ icon: Icon, label, value, highlight }: any) => (
  <div className="flex items-center justify-between py-3 border-b border-dashed border-slate-100 last:border-0">
    <div className="flex items-center gap-3 text-slate-500">
      <Icon size={16} />
      <span className="text-sm">{label}</span>
    </div>
    <span className={`text-sm font-mono font-medium ${highlight ? 'text-amber-600 bg-amber-50 px-2 py-0.5 rounded' : 'text-slate-700'}`}>
      {value}
    </span>
  </div>
);

export default Overview;
