import { useState } from 'react';
import { Leaf, ThermometerSun, Map, TreePine, Droplets, Info, LayoutDashboard } from 'lucide-react';
import { AppProvider, useAppContext } from './context/AppContext';
import Overview from './pages/Overview';
import HeatAnalysis from './pages/HeatAnalysis';
import EnvironmentalFactors from './pages/EnvironmentalFactors';
import PlantationSuitability from './pages/PlantationSuitability';
import TreeRequirements from './pages/TreeRequirements';
import CoolingPrediction from './pages/CoolingPrediction';
import Methodology from './pages/Methodology';

const Navigation = ({ activeTab, setActiveTab }: any) => {
  const tabs = [
    { id: 'overview', name: 'Overview', icon: LayoutDashboard },
    { id: 'heat', name: 'Heat Analysis', icon: ThermometerSun },
    { id: 'env', name: 'Environmental Factors', icon: Droplets },
    { id: 'suitability', name: 'Plantation Suitability', icon: Map },
    { id: 'trees', name: 'Tree Requirements', icon: TreePine },
    { id: 'cooling', name: 'Cooling Prediction', icon: ThermometerSun },
    { id: 'methodology', name: 'Methodology', icon: Info },
  ];

  return (
    <div className="w-72 bg-gradient-to-b from-slate-900 to-slate-950 text-slate-300 flex flex-col h-screen shadow-2xl z-20">
      <div className="p-8 pb-4">
        <h1 className="text-2xl font-bold text-white flex items-center gap-3">
          <div className="p-2 bg-green-500 rounded-lg shadow-lg shadow-green-500/20">
            <Leaf className="text-white" size={24} />
          </div>
          GreenGrid AI
        </h1>
        <p className="text-[11px] mt-3 text-slate-500 font-bold tracking-widest uppercase ml-1">Urban Cooling Module</p>
      </div>
      
      <nav className="flex-1 px-4 mt-6 space-y-1.5 overflow-y-auto">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`w-full flex items-center gap-3 px-4 py-3.5 rounded-xl text-sm font-medium transition-all duration-200 ${
                isActive
                  ? 'bg-gradient-to-r from-green-500/10 to-green-500/5 text-green-400 border border-green-500/20 shadow-sm'
                  : 'hover:bg-slate-800/50 hover:text-white border border-transparent'
              }`}
            >
              <Icon size={18} className={isActive ? "text-green-400" : "text-slate-500"} />
              {tab.name}
            </button>
          );
        })}
      </nav>
      
      <div className="p-6 border-t border-slate-800/50 bg-slate-900/50 backdrop-blur-md">
        <GlobalControls />
      </div>
    </div>
  );
};

const GlobalControls = () => {
  const { year, setYear, scenario, setScenario } = useAppContext();
  
  return (
    <div className="space-y-5">
      <div>
        <label className="flex items-center justify-between text-xs font-semibold text-slate-400 mb-2">
          <span>Analysis Year</span>
          <span className="bg-slate-800 px-2 py-0.5 rounded text-[10px] text-slate-300 border border-slate-700">Snapshot</span>
        </label>
        <select 
          value={year} 
          onChange={(e) => setYear(Number(e.target.value))}
          className="w-full bg-slate-950 border border-slate-700/50 text-sm rounded-lg px-3 py-2.5 text-white outline-none focus:border-green-500 transition-colors shadow-inner appearance-none cursor-pointer"
        >
          {[2022, 2023, 2024, 2025, 2026].map(y => (
            <option key={y} value={y}>{y}</option>
          ))}
        </select>
      </div>
      <div>
        <label className="block text-xs font-semibold text-slate-400 mb-2">Planning Scenario</label>
        <select 
          value={scenario} 
          onChange={(e) => setScenario(e.target.value)}
          className="w-full bg-slate-950 border border-slate-700/50 text-sm rounded-lg px-3 py-2.5 text-white outline-none focus:border-green-500 transition-colors shadow-inner appearance-none cursor-pointer"
        >
          <option value="v2_constrained">v2_constrained (Primary)</option>
          <option value="v1_parity">v1_parity (Baseline)</option>
        </select>
      </div>
    </div>
  );
}

function MainApp() {
  const [activeTab, setActiveTab] = useState('overview');

  return (
    <div className="flex h-screen bg-slate-50 font-sans text-slate-900 overflow-hidden">
      <Navigation activeTab={activeTab} setActiveTab={setActiveTab} />
      <main className="flex-1 overflow-y-auto relative bg-[#f4f7f6]">
        {/* Subtle background pattern */}
        <div className="absolute inset-0 z-0 opacity-[0.03] pointer-events-none" style={{ backgroundImage: 'radial-gradient(#000 1px, transparent 1px)', backgroundSize: '24px 24px' }}></div>
        <div className="relative z-10 h-full">
          {activeTab === 'overview' && <Overview setActiveTab={setActiveTab} />}
          {activeTab === 'heat' && <HeatAnalysis />}
          {activeTab === 'env' && <EnvironmentalFactors />}
          {activeTab === 'suitability' && <PlantationSuitability />}
          {activeTab === 'trees' && <TreeRequirements />}
          {activeTab === 'cooling' && <CoolingPrediction />}
          {activeTab === 'methodology' && <Methodology />}
        </div>
      </main>
    </div>
  );
}

function App() {
  return (
    <AppProvider>
      <MainApp />
    </AppProvider>
  );
}

export default App;
