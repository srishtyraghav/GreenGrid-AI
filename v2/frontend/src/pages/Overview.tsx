import React, { useEffect, useState } from 'react';
import { useAppContext } from '../context/AppContext';
import { fetchTreeRequirementSummary } from '../services/api';
import { Leaf, Map as MapIcon, Target, TrendingDown, ArrowRight, Layers, Calendar } from 'lucide-react';

const Overview = ({ setActiveTab }: { setActiveTab: (tab: string) => void }) => {
  const { year, scenario } = useAppContext();
  const [summary, setSummary] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    fetchTreeRequirementSummary(year, scenario)
      .then(data => {
        const primary = data.find((row: any) => row.is_primary_density);
        if(primary) setSummary(primary);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [year, scenario]);

  return (
    <div className="p-8 max-w-[1600px] mx-auto space-y-8">
      <header className="mb-10">
        <h1 className="text-4xl font-bold text-slate-900 tracking-tight">GreenGrid AI — Urban Cooling Decision Support</h1>
        <p className="text-slate-500 mt-3 text-xl font-light font-mono text-sm">{year} Snapshot | NCR Study Area | {scenario}</p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6">
        <StatCard 
          title="Analysis Year" 
          value={year}
          icon={Calendar}
          description="Annual Snapshot"
          color="blue"
        />
        <StatCard 
          title="Priority Zones" 
          value={loading ? "..." : (summary ? summary.n_zones : "N/A")}
          icon={Target}
          description="High/Medium/Low priority"
          color="indigo"
        />
        <StatCard 
          title="Priority Area" 
          value={loading ? "..." : (summary ? `${summary.plantable_ha.toFixed(2)} ha` : "N/A")}
          icon={Layers}
          description="Plantable priority area after applied constraints"
          color="amber"
        />
        <StatCard 
          title="Trees @ 1000/ha" 
          value={loading ? "..." : (summary ? summary.recommended_trees.toLocaleString() : "N/A")}
          icon={Leaf}
          description="Planning estimate @ 1,000 trees/ha"
          color="green"
        />
      </div>

      <div className="bg-white rounded-2xl shadow-sm border border-slate-200/60 p-8 overflow-hidden relative">
        <div className="absolute -right-20 -top-20 opacity-5 pointer-events-none">
          <TrendingDown size={300} />
        </div>
        
        <h2 className="text-2xl font-bold text-slate-800 mb-2">Project Workflow</h2>
        <p className="text-slate-500 mb-8">Follow the analytical pipeline from raw satellite data to actionable planning.</p>
        
        <div className="flex flex-col md:flex-row gap-3 relative z-10">
          <WorkflowStep 
            number="1"
            label="Heat Analysis" 
            desc="Identify relative heat-severity hotspots"
            onClick={() => setActiveTab('heat')} 
          />
          <ArrowRight className="hidden md:block self-center text-slate-300" />
          <WorkflowStep 
            number="2"
            label="Environmental" 
            desc="Contextualize features"
            onClick={() => setActiveTab('env')} 
          />
          <ArrowRight className="hidden md:block self-center text-slate-300" />
          <WorkflowStep 
            number="3"
            label="Suitability" 
            desc="GIS-based plantation ranking"
            onClick={() => setActiveTab('suitability')} 
          />
          <ArrowRight className="hidden md:block self-center text-slate-300" />
          <WorkflowStep 
            number="4"
            label="Tree Planning" 
            desc="Calculate requirements"
            onClick={() => setActiveTab('trees')} 
          />
          <ArrowRight className="hidden md:block self-center text-slate-300" />
          <WorkflowStep 
            number="5"
            label="Cooling Predict" 
            desc="Under development"
            onClick={() => setActiveTab('cooling')}
            pending 
          />
        </div>
      </div>

      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-100 rounded-2xl p-6 text-blue-900 shadow-sm flex gap-4 items-start">
        <div className="p-3 bg-blue-100 rounded-xl text-blue-600 shrink-0">
          <TrendingDown size={24} />
        </div>
        <div>
          <h3 className="font-bold text-lg mb-1">Phase 9 Status: Cooling Prediction</h3>
          <p className="text-blue-800/80 leading-relaxed">
            Phase 9 temperature-reduction prediction is currently under development. 
            The interface architecture is prepared for immediate integration once the scientific modeling and validation are complete.
          </p>
        </div>
      </div>
    </div>
  );
};

const StatCard = ({ title, value, icon: Icon, description, color }: any) => {
  const colorMap: Record<string, string> = {
    blue: 'bg-blue-50 text-blue-600',
    green: 'bg-green-50 text-green-600',
    amber: 'bg-amber-50 text-amber-600',
    indigo: 'bg-indigo-50 text-indigo-600',
  };

  return (
    <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200/60 flex flex-col hover:shadow-md transition-shadow">
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

const WorkflowStep = ({ number, label, desc, onClick, pending }: any) => (
  <button 
    onClick={onClick}
    className={`flex-1 p-5 rounded-2xl text-left transition-all border group ${
      pending 
        ? 'bg-slate-50 border-slate-200/50 cursor-default opacity-70' 
        : 'bg-white border-slate-200/80 shadow-sm hover:border-green-400 hover:shadow-md cursor-pointer hover:-translate-y-1'
    }`}
  >
    <div className={`text-xs font-black mb-3 w-6 h-6 flex items-center justify-center rounded-full ${pending ? 'bg-slate-200 text-slate-500' : 'bg-slate-900 text-white group-hover:bg-green-500'}`}>
      {number}
    </div>
    <h4 className={`font-bold mb-1 ${pending ? 'text-slate-500' : 'text-slate-800'}`}>{label}</h4>
    <p className="text-xs text-slate-500 leading-relaxed">{desc}</p>
  </button>
);

export default Overview;
