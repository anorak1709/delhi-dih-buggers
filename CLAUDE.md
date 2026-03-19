# CLAUDE.md

## Project Overview

**Bloom Analytics (Portfolio Optimizer Pro)** — a web app providing institutional-grade portfolio analytics powered by Monte Carlo simulations. React frontend with a Flask (Python) backend.

## Tech Stack

- **Frontend**: React 18, Vite, Tailwind CSS, Framer Motion (used throughout for animations & micro-interactions)
- **Charts**: Chart.js (react-chartjs-2), Plotly.js (react-plotly.js)
- **Real-time**: Socket.IO client
- **Backend**: Flask (Python) with Yahoo Finance data, NumPy/SciPy/Pandas
- **Build**: Vite dev server on port 3000, proxies `/api` and `/socket.io` to Flask on port 5000
- **Fonts**: Cormorant Garamond (display), Plus Jakarta Sans (sans)
- **Theme**: Dark/light mode via Tailwind `class` strategy, gold/bronze accent (#c9985a)

## Project Structure

```
src/
  App.jsx              # Main app with AnimatePresence page transitions
  main.jsx             # Entry point
  index.css            # Global styles (Tailwind layers, shimmer, card-glow)
  components/
    layout/            # Header, Sidebar (with layoutId nav animation)
    portfolio/         # PortfolioPanel — holdings CRUD, price fetching
    analysis/          # AnalysisPanel — benchmark comparison, correlation matrix
    optimize/          # OptimizePanel — Monte Carlo optimization, efficient frontier
    risk/              # RiskPanel — VaR/CVaR, rolling Sharpe, scenario & stress tests
    market/            # MarketPanel — economic indicators, sectors, sentiment, news
    live/              # LivePanel — real-time prices via polling & WebSocket
    retirement/        # RetirementPanel — retirement planning with probability cone
    ui/                # Reusable UI: Button, Card, Input, Loading, Toast, Motion
  context/
    AppContext.jsx      # Global app state (holdings, darkMode, activeTab, toasts)
  hooks/
    useLocalStorage.js  # Local storage persistence hook
  services/
    api.js              # API client (all backend endpoints)
new.py                  # Flask backend (22+ API endpoints)
ft.html, app.js, style.css  # Legacy backup files
.env                    # VITE_API_URL=http://localhost:5000
```

## Running the App

1. **Backend**: `python new.py` (runs on port 5000)
   - Requires: `pip install flask flask-cors flask-socketio yfinance numpy pandas scipy matplotlib`
2. **Frontend**: `npm run dev` (runs on port 3000, proxies `/api` to backend)

## Commands

- `npm run dev` — Start Vite dev server (port 3000)
- `npm run build` — Production build
- `npm run preview` — Preview production build

## Guidelines

- Use JSX (not TSX) — project uses plain JavaScript, not TypeScript
- Style with Tailwind CSS utility classes
- Reuse existing UI components from `src/components/ui/` before creating new ones
- Use shared animation utilities from `src/components/ui/Motion.jsx` (StaggerList, StaggerItem, pageTransition, etc.)
- Keep API calls in `src/services/api.js`
- Use `AppContext` for shared state
- Backend API runs on port 5000; frontend proxies `/api` routes to it
- `.env` contains `VITE_API_URL` for backend URL configuration — do not commit `.env` files
