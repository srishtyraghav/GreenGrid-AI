import { BookOpen, ArrowDown } from 'lucide-react';

const Methodology = () => {
  return (
    <div className="p-8 max-w-4xl mx-auto space-y-8">
      <header className="mb-8">
        <h1 className="text-4xl font-bold text-slate-900 flex items-center gap-4 tracking-tight">
          <BookOpen className="text-slate-500" size={36} />
          Methodology & About
        </h1>
        <p className="text-slate-500 mt-3 text-xl">GreenGrid AI V2 Pipeline Validation and Transparency</p>
      </header>

      <div className="bg-white rounded-2xl shadow-sm border border-slate-200/60 p-10 space-y-10">
        
        <section>
          <h2 className="text-2xl font-bold text-slate-800 mb-6">Scientific Pipeline Architecture</h2>
          <div className="bg-slate-50 p-8 rounded-xl border border-slate-200/60 font-mono text-sm text-slate-700 flex flex-col items-center gap-3">
             
            <PipelineNode text="Satellite + GIS + Weather Data" />
            <ArrowDown size={16} className="text-slate-300" />
            
            <PipelineNode text="30 m Spatial Preprocessing" />
            <ArrowDown size={16} className="text-slate-300" />
            
            <PipelineNode text="Environmental Feature Generation" />
            <ArrowDown size={16} className="text-slate-300" />
            
            <PipelineNode text="Random Forest UHI Proxy" color="blue" />
            <ArrowDown size={16} className="text-blue-300" />
            
            <PipelineNode text="Relative Heat Severity Classification" color="orange" />
            <ArrowDown size={16} className="text-orange-300" />
            
            <PipelineNode text="Plantation Suitability Analysis" color="green" />
            <ArrowDown size={16} className="text-green-300" />
            
            <PipelineNode text="Tree Requirement Estimation" color="green" />
            <ArrowDown size={16} className="text-green-300" />
            
            <PipelineNode text="ML Cooling Prediction (Phase 9)" color="slate" dashed />
            
          </div>
        </section>

        <section>
          <h2 className="text-2xl font-bold text-slate-800 mb-6 pt-6 border-t border-slate-100">Scientific Framing</h2>
          <ul className="space-y-6 text-slate-600 leading-relaxed">
            <li className="flex gap-4 items-start">
              <div className="w-2 h-2 rounded-full bg-orange-500 mt-2 shrink-0"></div>
              <div>
                <strong className="text-slate-800 block mb-1">Relative Heat Severity</strong>
                Phase 6 outputs provide relative, year-specific heat-severity proxies. They are normalized per year and do not represent absolute UHI intensity in °C. Inter-year comparisons do not establish physical temperature trends.
              </div>
            </li>
            <li className="flex gap-4 items-start">
              <div className="w-2 h-2 rounded-full bg-green-500 mt-2 shrink-0"></div>
              <div>
                <strong className="text-slate-800 block mb-1">Plantation Suitability</strong>
                Phase 7 determines relative suitability based on geospatial constraints. It is a decision-support ranking, not a guarantee of legal land availability or ecological survival.
              </div>
            </li>
            <li className="flex gap-4 items-start">
              <div className="w-2 h-2 rounded-full bg-blue-500 mt-2 shrink-0"></div>
              <div>
                <strong className="text-slate-800 block mb-1">Tree Density Scenarios</strong>
                The density options (400, 1000, 2500 trees/ha) are planning scenarios. They are not asserted to be "optimal" without completed Phase 9 modeling.
              </div>
            </li>
          </ul>
        </section>
      </div>
    </div>
  );
};

const PipelineNode = ({ text, color = 'slate', dashed = false }: any) => {
  const colors: Record<string, string> = {
    slate: 'bg-white border-slate-300 text-slate-700',
    blue: 'bg-blue-50 border-blue-300 text-blue-800 font-semibold',
    orange: 'bg-orange-50 border-orange-300 text-orange-800 font-semibold',
    green: 'bg-green-50 border-green-300 text-green-800 font-semibold',
  };
  
  return (
    <div className={`px-6 py-3 rounded-lg shadow-sm border text-center min-w-[300px] ${colors[color]} ${dashed ? 'border-dashed opacity-70' : ''}`}>
      {text}
    </div>
  );
};

export default Methodology;
