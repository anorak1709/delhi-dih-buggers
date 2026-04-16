import { useApp } from '../../context/AppContext';
import { useAuth } from '../../context/AuthContext';
import { motion, AnimatePresence } from 'framer-motion';

const tabLabels = {
  dashboard: 'Dashboard',
  portfolio: 'Portfolio',
  analysis: 'Performance Analysis',
  optimize: 'Portfolio Optimization',
  risk: 'Risk Management',
  backtest: 'Backtesting',
  options: 'Options Pricing',
  market: 'Market Intelligence',
  live: 'Live Prices',
  retirement: 'Retirement Planning',
};

export default function Header() {
  const { activeTab, setActiveTab, darkMode, toggleDark, holdings } = useApp();
  const { user, logOut } = useAuth();

  return (
    <header className="flex items-center justify-between px-6 lg:px-8 py-4 border-b border-surface-700/40 dark:border-surface-700/40 border-surface-200 glass-subtle dark:glass-subtle bg-white/50 backdrop-blur-xl shrink-0">
      <div className="flex items-center gap-4">
        {/* Mobile menu */}
        <div className="md:hidden flex items-center gap-3">
          <h1 className="font-display text-xl font-semibold text-gradient">Bloom</h1>
          <span className="text-surface-700 dark:text-surface-700 text-surface-300">|</span>
        </div>
        <div>
          <AnimatePresence mode="wait">
            <motion.h2
              key={activeTab}
              initial={{ opacity: 0, y: 8, filter: 'blur(4px)' }}
              animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
              exit={{ opacity: 0, y: -8, filter: 'blur(4px)' }}
              transition={{ duration: 0.25, ease: [0.25, 0.46, 0.45, 0.94] }}
              className="text-lg font-semibold text-surface-100 dark:text-surface-100 text-surface-800 tracking-tight"
            >
              {tabLabels[activeTab]}
            </motion.h2>
          </AnimatePresence>
          <p className="text-2xs text-surface-500 mt-0.5">
            {holdings.length} holding{holdings.length !== 1 ? 's' : ''} tracked
          </p>
        </div>
      </div>

      <div className="flex items-center gap-2">
        {/* User info */}
        {user && (
          <div className="hidden sm:flex items-center gap-2 mr-2">
            <div className="flex items-center gap-1.5">
              <span className="text-xs text-surface-400 truncate max-w-[160px]">{user.email}</span>
              {user.emailVerified ? (
                <span
                  title="Email verified"
                  className="inline-flex items-center gap-1 text-2xs px-1.5 py-0.5 rounded-md bg-accent/10 text-accent border border-accent/20"
                >
                  <svg className="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
                  </svg>
                  Verified
                </span>
              ) : (
                <span
                  title="Email not verified"
                  className="text-2xs px-1.5 py-0.5 rounded-md bg-warn/10 text-warn border border-warn/20"
                >
                  Unverified
                </span>
              )}
            </div>
            <motion.button
              onClick={logOut}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="text-xs px-2.5 py-1 rounded-lg text-surface-400 hover:text-down hover:bg-down/10 border border-surface-700/40 dark:border-surface-700/40 border-surface-200 transition-colors duration-200 cursor-pointer"
            >
              Sign Out
            </motion.button>
          </div>
        )}

        {/* Mobile tab selector */}
        <select
          value={activeTab}
          onChange={(e) => setActiveTab(e.target.value)}
          className="md:hidden text-xs bg-surface-800/80 dark:bg-surface-800/80 bg-surface-100 border border-surface-700/60 dark:border-surface-700/60 border-surface-200 rounded-lg px-2 py-1.5 text-surface-200 dark:text-surface-200 text-surface-700"
        >
          {Object.entries(tabLabels).map(([key, label]) => (
            <option key={key} value={key}>{label}</option>
          ))}
        </select>

        {/* Dark mode toggle */}
        <motion.button
          onClick={toggleDark}
          whileHover={{ scale: 1.1, rotate: 15 }}
          whileTap={{ scale: 0.9 }}
          transition={{ type: 'spring', stiffness: 400, damping: 17 }}
          className="p-2 rounded-lg text-surface-400 hover:text-accent hover:bg-surface-700/30 dark:hover:bg-surface-700/30 hover:bg-surface-100 transition-colors duration-200 cursor-pointer"
          title={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
        >
          <AnimatePresence mode="wait">
            {darkMode ? (
              <motion.svg
                key="sun"
                initial={{ opacity: 0, rotate: -90, scale: 0 }}
                animate={{ opacity: 1, rotate: 0, scale: 1 }}
                exit={{ opacity: 0, rotate: 90, scale: 0 }}
                transition={{ duration: 0.3 }}
                className="w-4.5 h-4.5"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth={1.5}
              >
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 3v2.25m6.364.386l-1.591 1.591M21 12h-2.25m-.386 6.364l-1.591-1.591M12 18.75V21m-4.773-4.227l-1.591 1.591M5.25 12H3m4.227-4.773L5.636 5.636M15.75 12a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0z" />
              </motion.svg>
            ) : (
              <motion.svg
                key="moon"
                initial={{ opacity: 0, rotate: 90, scale: 0 }}
                animate={{ opacity: 1, rotate: 0, scale: 1 }}
                exit={{ opacity: 0, rotate: -90, scale: 0 }}
                transition={{ duration: 0.3 }}
                className="w-4.5 h-4.5"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth={1.5}
              >
                <path strokeLinecap="round" strokeLinejoin="round" d="M21.752 15.002A9.718 9.718 0 0118 15.75c-5.385 0-9.75-4.365-9.75-9.75 0-1.33.266-2.597.748-3.752A9.753 9.753 0 003 11.25C3 16.635 7.365 21 12.75 21a9.753 9.753 0 009.002-5.998z" />
              </motion.svg>
            )}
          </AnimatePresence>
        </motion.button>

        {/* Mobile sign out */}
        {user && (
          <motion.button
            onClick={logOut}
            whileTap={{ scale: 0.95 }}
            className="sm:hidden p-2 rounded-lg text-surface-400 hover:text-down hover:bg-down/10 transition-colors duration-200 cursor-pointer"
            title="Sign Out"
          >
            <svg className="w-4.5 h-4.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 9V5.25A2.25 2.25 0 0013.5 3h-6a2.25 2.25 0 00-2.25 2.25v13.5A2.25 2.25 0 007.5 21h6a2.25 2.25 0 002.25-2.25V15m3 0l3-3m0 0l-3-3m3 3H9" />
            </svg>
          </motion.button>
        )}
      </div>
    </header>
  );
}
