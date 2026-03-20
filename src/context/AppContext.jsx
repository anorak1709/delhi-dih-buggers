import { createContext, useContext, useState, useCallback } from 'react';
import { useLocalStorage } from '../hooks/useLocalStorage';

const AppContext = createContext(null);

export function AppProvider({ children }) {
  const [holdings, setHoldings] = useLocalStorage('bloom_holdings', []);
  const [optionsHoldings, setOptionsHoldings] = useLocalStorage('bloom_options', []);
  const [darkMode, setDarkMode] = useLocalStorage('bloom_dark', true);
  const [activeTab, setActiveTab] = useState('dashboard');
  const [toasts, setToasts] = useState([]);

  const addHolding = useCallback((ticker, quantity) => {
    setHoldings(prev => {
      const existing = prev.find(h => h.ticker === ticker.toUpperCase());
      if (existing) {
        return prev.map(h =>
          h.ticker === ticker.toUpperCase()
            ? { ...h, quantity: h.quantity + quantity }
            : h
        );
      }
      return [...prev, { ticker: ticker.toUpperCase(), quantity }];
    });
  }, [setHoldings]);

  const removeHolding = useCallback((ticker) => {
    setHoldings(prev => prev.filter(h => h.ticker !== ticker));
  }, [setHoldings]);

  const updateHolding = useCallback((ticker, quantity) => {
    setHoldings(prev =>
      prev.map(h => (h.ticker === ticker ? { ...h, quantity } : h))
    );
  }, [setHoldings]);

  const notify = useCallback((message, type = 'info') => {
    const id = Date.now();
    setToasts(prev => [...prev, { id, message, type }]);
    setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id));
    }, 4000);
  }, []);

  const dismissToast = useCallback((id) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  }, []);

  const addOptionHolding = useCallback((option) => {
    setOptionsHoldings(prev => [...prev, { id: Date.now(), ...option }]);
  }, [setOptionsHoldings]);

  const removeOptionHolding = useCallback((id) => {
    setOptionsHoldings(prev => prev.filter(o => o.id !== id));
  }, [setOptionsHoldings]);

  const reorderHoldings = useCallback((startIndex, endIndex) => {
    setHoldings(prev => {
      const result = Array.from(prev);
      const [removed] = result.splice(startIndex, 1);
      result.splice(endIndex, 0, removed);
      return result;
    });
  }, [setHoldings]);

  const toggleDark = useCallback(() => {
    setDarkMode(prev => !prev);
  }, [setDarkMode]);

  const tickers = holdings.map(h => h.ticker);

  const holdingsMap = holdings.reduce((acc, h) => {
    acc[h.ticker] = h.quantity;
    return acc;
  }, {});

  return (
    <AppContext.Provider
      value={{
        holdings, tickers, holdingsMap,
        addHolding, removeHolding, updateHolding, reorderHoldings,
        optionsHoldings, addOptionHolding, removeOptionHolding,
        darkMode, toggleDark,
        activeTab, setActiveTab,
        toasts, notify, dismissToast,
      }}
    >
      {children}
    </AppContext.Provider>
  );
}

export function useApp() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error('useApp must be used within AppProvider');
  return ctx;
}
