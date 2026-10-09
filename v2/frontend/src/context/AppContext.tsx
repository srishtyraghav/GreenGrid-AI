import React, { createContext, useContext, useState, ReactNode } from 'react';

interface AppState {
  year: number;
  setYear: (year: number) => void;
  scenario: string;
  setScenario: (scenario: string) => void;
  density: number;
  setDensity: (density: number) => void;
}

const AppContext = createContext<AppState | undefined>(undefined);

export const AppProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [year, setYear] = useState<number>(2026);
  const [scenario, setScenario] = useState<string>('v2_constrained');
  const [density, setDensity] = useState<number>(1000);

  return (
    <AppContext.Provider value={{ year, setYear, scenario, setScenario, density, setDensity }}>
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
