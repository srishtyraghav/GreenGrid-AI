import { Loader2 } from 'lucide-react';

export const Loader = ({ text = "Loading..." }) => {
  return (
    <div className="flex flex-col items-center justify-center h-full w-full space-y-3 text-slate-400">
      <Loader2 className="animate-spin text-green-500" size={32} />
      <span className="text-sm font-medium">{text}</span>
    </div>
  );
};
