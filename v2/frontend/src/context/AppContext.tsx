import { createContext, useContext, useState, type ReactNode } from 'react';

interface AppState {
  year: number;
  setYear: (year: number) => void;
  scenario: string;
  setScenario: (scenario: string) => void;
  density: number;                 // trees/ha planning assumption (100-2500, default 400)
  setDensity: (density: number) => void;
  selectedSites: Set<number>;      // stable site ids; PRESERVED across year changes
  setSelectedSites: (s: Set<number>) => void;
  treeCapacityLimit: number | null;
  setTreeCapacityLimit: (n: number | null) => void;
}

const AppContext = createContext<AppState | undefined>(undefined);

export const AppProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [year, setYear] = useState<number>(2026);          // 2026 DEFAULT
  const [scenario, setScenario] = useState<string>('v2_constrained');
  const [density, setDensity] = useState<number>(400);     // default planning assumption
  const [selectedSites, setSelectedSites] = useState<Set<number>>(new Set());
  const [treeCapacityLimit, setTreeCapacityLimit] = useState<number | null>(null);

  return (
    <AppContext.Provider value={{
      year, setYear, scenario, setScenario, density, setDensity,
      selectedSites, setSelectedSites, treeCapacityLimit, setTreeCapacityLimit,
    }}>
      {children}
    </AppContext.Provider>
  );
};

export const useAppContext = () => {
  const context = useContext(AppContext);
  if (context === undefined) {
    throw new Error('useAppContext must be used within an AppProvider');
  }
  return context;
};
