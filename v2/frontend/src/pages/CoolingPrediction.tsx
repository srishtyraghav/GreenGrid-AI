import { ThermometerSnowflake, AlertCircle, ArrowDown } from 'lucide-react';

const CoolingPrediction = () => {
  return (
    <div className="p-8 max-w-3xl mx-auto flex flex-col h-full items-center justify-center">
      <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-12 text-center w-full">
        <div className="mx-auto w-16 h-16 bg-blue-50 text-blue-500 rounded-full flex items-center justify-center mb-6">
          <ThermometerSnowflake size={32} />
        </div>
        <h1 className="text-3xl font-bold text-slate-900 mb-4 tracking-tight">COOLING PREDICTION</h1>
        
        <div className="bg-amber-50 border border-amber-200 text-amber-800 rounded-lg p-5 flex items-start gap-4 text-left mb-8 shadow-sm">
          <AlertCircle className="mt-0.5 shrink-0" size={24} />
          <div>
            <strong className="text-sm">Phase 9 model is currently under development.</strong>
            <p className="text-sm mt-2 opacity-90 leading-relaxed">
              Future outputs will map the <strong>Predicted LST reduction under the assumed planting scenario</strong>. 
              <br/><br/>
              <em>Note:</em> This represents modeled potential based on the ML pipeline, and must be distinguished from <em>observed</em> cooling caused by physical planting, as the intervention counterfactual has not yet occurred.
            </p>
          </div>
        </div>

        <div className="text-sm text-slate-600 space-y-4 border-t border-slate-100 pt-8">
          <h3 className="font-bold text-slate-800 uppercase tracking-widest text-xs mb-6">Future Output Pipeline</h3>
          
          <div className="flex flex-col items-center gap-3 font-mono text-xs">
            <div className="bg-slate-50 border border-slate-200 px-6 py-3 rounded-lg shadow-sm">Planting Scenario Selected</div>
            <ArrowDown size={16} className="text-slate-300" />
            <div className="bg-slate-50 border border-slate-200 px-6 py-3 rounded-lg shadow-sm">Assumed vegetation / tree change applied</div>
            <ArrowDown size={16} className="text-slate-300" />
            <div className="bg-blue-50 border border-blue-200 text-blue-800 font-bold px-6 py-3 rounded-lg shadow-sm">ML Cooling Model</div>
            <ArrowDown size={16} className="text-blue-300" />
            <div className="bg-slate-50 border border-slate-200 px-6 py-3 rounded-lg shadow-sm">Predicted ΔLST</div>
            <ArrowDown size={16} className="text-slate-300" />
            <div className="bg-green-50 border border-green-200 text-green-800 font-bold px-6 py-3 rounded-lg shadow-sm">Spatial Cooling Map Generated</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CoolingPrediction;
