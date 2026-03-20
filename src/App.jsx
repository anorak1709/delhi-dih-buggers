import { useEffect } from 'react';
import { useApp } from './context/AppContext';
import { useAuth } from './context/AuthContext';
import { motion, AnimatePresence, pageTransition } from './components/ui/Motion';
import Sidebar from './components/layout/Sidebar';
import Header from './components/layout/Header';
import ToastContainer from './components/ui/Toast';
import Loading from './components/ui/Loading';
import LoginPage from './components/auth/LoginPage';
import DashboardPanel from './components/dashboard/DashboardPanel';
import PortfolioPanel from './components/portfolio/PortfolioPanel';
import AnalysisPanel from './components/analysis/AnalysisPanel';
import BacktestPanel from './components/backtest/BacktestPanel';
import OptimizePanel from './components/optimize/OptimizePanel';
import RiskPanel from './components/risk/RiskPanel';
import MarketPanel from './components/market/MarketPanel';
import LivePanel from './components/live/LivePanel';
import RetirementPanel from './components/retirement/RetirementPanel';
import OptionsPanel from './components/options/OptionsPanel';

const panels = {
  dashboard: DashboardPanel,
  portfolio: PortfolioPanel,
  analysis: AnalysisPanel,
  optimize: OptimizePanel,
  risk: RiskPanel,
  backtest: BacktestPanel,
  options: OptionsPanel,
  market: MarketPanel,
  live: LivePanel,
  retirement: RetirementPanel,
};

export default function App() {
  const { darkMode, activeTab } = useApp();
  const { user, loading: authLoading } = useAuth();

  useEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode);
  }, [darkMode]);

  if (authLoading) {
    return (
      <div className="flex h-screen items-center justify-center bg-surface-900 dark:bg-surface-900 bg-surface-50">
        <Loading text="Initializing..." />
      </div>
    );
  }

  if (!user) {
    return <LoginPage />;
  }

  const ActivePanel = panels[activeTab] || DashboardPanel;

  return (
    <div className="flex h-screen overflow-hidden bg-surface-900 dark:bg-surface-900 bg-surface-50">
      <div className="grain-overlay" />
      {/* Background orbs */}
      <div className="orb orb-1" />
      <div className="orb orb-2" />
      <div className="orb orb-3" />
      <Sidebar />
      <div className="flex flex-col flex-1 min-w-0">
        <Header />
        <main className="flex-1 overflow-y-auto p-6 lg:p-8">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              {...pageTransition}
              className="max-w-7xl mx-auto"
            >
              <ActivePanel />
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
      <ToastContainer />
    </div>
  );
}
