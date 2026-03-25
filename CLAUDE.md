# CLAUDE.md

## Project Overview

**Bloom Analytics (Portfolio Optimizer Pro)** — an institutional-grade portfolio analytics web app powered by Monte Carlo simulations, HRP, Black-Litterman, and Black-Scholes options pricing. React frontend with a Flask (Python) backend. Features Firebase authentication, an AI-powered stock research agent (OpenAI), glassmorphism design, and enriched live market tracking.

## Tech Stack

- **Frontend**: React 18, Vite, Tailwind CSS, Framer Motion (used throughout for animations & micro-interactions)
- **Charts**: Chart.js (react-chartjs-2), Plotly.js (react-plotly.js) for 3D visualizations
- **Real-time**: Socket.IO client
- **Auth**: Firebase Authentication (email/password)
- **AI Agent**: OpenAI GPT-4o-mini via `openai` Python SDK (RAG pipeline with yfinance data)
- **Backend**: Flask (Python) with Yahoo Finance data, NumPy/SciPy/Pandas
- **DnD**: react-beautiful-dnd for drag-and-drop portfolio reordering
- **Build**: Vite dev server on port 3000, proxies `/api` and `/socket.io` to Flask on port 5000
- **Fonts**: Cormorant Garamond (display), Plus Jakarta Sans (sans)
- **Theme**: Dark/light mode via Tailwind `class` strategy, gold/bronze accent (#c9985a), glassmorphism (backdrop-blur, glass borders)

## Project Structure

```
src/
  App.jsx              # Main app with auth gate, renders all panels (hidden/visible for state retention)
  main.jsx             # Entry point (AuthProvider → AppProvider → App)
  index.css            # Global styles (Tailwind layers, glassmorphism classes, shimmer, card-glow)
  config/
    firebase.js        # Firebase app initialization & auth export
  constants/
    tooltips.js        # Centralized tooltip descriptions for all financial metrics
  components/
    auth/
      LoginPage.jsx    # Email/password login & signup page
    layout/            # Header (with sign-out), Sidebar (with layoutId nav animation)
    dashboard/         # DashboardPanel — summary cards, holdings list, quick actions
    portfolio/         # PortfolioPanel — holdings CRUD, drag-and-drop reorder, price fetching
    analysis/          # AnalysisPanel — benchmark comparison, correlation matrix
    optimize/          # OptimizePanel — MC optimization, HRP (dendrogram), Black-Litterman (sentiment)
      ConstrainedOptimize.jsx  # Backtracking constraint optimizer + sensitivity tornado chart
    risk/              # RiskPanel — VaR/CVaR, rolling Sharpe, scenario & stress tests
    backtest/          # BacktestPanel — historical simulation with fees, slippage, benchmark comparison
    options/           # OptionsPanel — Black-Scholes calculator, Greeks, options chain, 3D vol surface
    market/
      MarketPanel.jsx  # AI agent, sectors, sentiment, news
      AIAgent.jsx      # OpenAI-powered RAG chat for stock research & advice
    live/              # LivePanel — enriched ticker cards with sparklines, sentiment, headlines
    retirement/        # RetirementPanel — retirement planning with probability cone
    ui/                # Reusable UI: Button, Card, Input, Loading, Toast, Motion, Tooltip
  context/
    AppContext.jsx      # Global app state (holdings, darkMode, activeTab, toasts, reorderHoldings)
    AuthContext.jsx     # Firebase auth state (user, loading, signIn, signUp, logOut)
  hooks/
    useLocalStorage.js  # Local storage persistence hook
  services/
    api.js              # API client (30+ backend endpoints)
new.py                  # Flask backend (30+ API endpoints)
.env                    # Frontend env vars (VITE_API_URL, VITE_FIREBASE_*)
```

## Running the App

1. **Backend**: `python new.py` (runs on port 5000)
   - Requires: `pip install flask flask-cors flask-socketio yfinance numpy pandas scipy matplotlib openai`
   - Set `OPENAI_API_KEY` environment variable for the AI agent feature
2. **Frontend**: `npm run dev` (runs on port 3000, proxies `/api` to backend)
3. **Firebase**: Configure `.env` with your Firebase project credentials (`VITE_FIREBASE_API_KEY`, `VITE_FIREBASE_AUTH_DOMAIN`, `VITE_FIREBASE_PROJECT_ID`, etc.)

## Commands

- `npm run dev` — Start Vite dev server (port 3000)
- `npm run build` — Production build
- `npm run preview` — Preview production build

## Key Backend Endpoints

### Core Analytics
- `POST /api/prices` — Current prices & daily changes (returns `{ prices: { ticker: { current_price, daily_change, daily_change_pct } } }`)
- `POST /api/analyze` — Portfolio performance vs benchmark
- `POST /api/optimize` — Monte Carlo optimization (10k+ simulations, supports standard/antithetic/sobol/full methods)
- `POST /api/correlation` — Asset correlation matrix
- `POST /api/risk-metrics` — VaR, CVaR, Beta, Alpha

### Advanced Optimization
- `POST /api/constrained-optimize` — Backtracking constraint-based optimization with pruning
- `POST /api/sensitivities` — Finite-difference sensitivity analysis (tornado chart data)
- `POST /api/hrp` — Hierarchical Risk Parity with dendrogram data
- `POST /api/black-litterman` — Black-Litterman model with optional AI sentiment views

### Options & Derivatives
- `POST /api/options/price` — Black-Scholes pricing with Greeks
- `POST /api/options/greeks` — Greeks curves (spot, time, volatility)
- `POST /api/options/chain` — Live options chain data
- `POST /api/options/implied-vol` — Implied volatility (Newton-Raphson)
- `POST /api/options/vol-surface` — 3D implied volatility surface data

### Backtesting & Risk
- `POST /api/backtest` — Historical backtesting with fees, slippage, benchmark comparison
- `POST /api/rolling` — Rolling Sharpe ratio
- `POST /api/scenario` — Bull/crash/flat market simulations
- `POST /api/stress` — Historical worst-case stress tests

### Intelligence
- `POST /api/live-analysis` — Enriched data per ticker: price, mini chart, sentiment, headlines
- `POST /api/ai-agent` — RAG pipeline: yfinance data → context → OpenAI GPT-4o-mini
- `POST /api/sentiment` — News sentiment analysis
- `POST /api/retirement/calculate` — Retirement planning with Monte Carlo

## Architecture Decisions

- **Tab state retention**: All panels are always mounted (hidden with `display: none`) to preserve local state when switching tabs. No AnimatePresence unmounting.
- **Tooltip system**: Centralized tooltip text in `src/constants/tooltips.js`, rendered via `src/components/ui/Tooltip.jsx` (Framer Motion animated, glassmorphism styled). Wrap any label: `<Tooltip text={TOOLTIPS.var_95}>VaR (95%)</Tooltip>`
- **API response unwrapping**: `getPrices()` in `api.js` unwraps the nested `data.prices` response so callers get `{ ticker: { current_price, ... } }` directly.

## Guidelines

- Use JSX (not TSX) — project uses plain JavaScript, not TypeScript
- Style with Tailwind CSS utility classes
- Reuse existing UI components from `src/components/ui/` before creating new ones (Button, Card, Input, Loading, Toast, Motion, Tooltip)
- Use shared animation utilities from `src/components/ui/Motion.jsx` (StaggerList, StaggerItem, pageTransition, etc.)
- Keep API calls in `src/services/api.js`
- Use `AppContext` for shared portfolio/UI state, `AuthContext` for authentication state
- Backend API runs on port 5000; frontend proxies `/api` routes to it
- `.env` contains `VITE_API_URL` and Firebase config — do not commit `.env` files
- The AI agent feature requires `OPENAI_API_KEY` set as a backend environment variable
- Add tooltip definitions to `src/constants/tooltips.js` when introducing new financial metrics
- All panels are always mounted (hidden via `display: none`) — never unmount panels on tab switch
- GitHub repo: https://github.com/anorak1709/delhi-dih-buggers
