# CLAUDE.md

## Project Overview

**Bloom Analytics (Portfolio Optimizer Pro)** — a web app providing institutional-grade portfolio analytics powered by Monte Carlo simulations. React frontend with a Flask (Python) backend. Features Firebase authentication, an AI-powered stock research agent (OpenAI), and enriched live market tracking.

## Tech Stack

- **Frontend**: React 18, Vite, Tailwind CSS, Framer Motion (used throughout for animations & micro-interactions)
- **Charts**: Chart.js (react-chartjs-2), Plotly.js (react-plotly.js)
- **Real-time**: Socket.IO client
- **Auth**: Firebase Authentication (email/password)
- **AI Agent**: OpenAI 2.0 Flash via `openai` Python SDK (RAG pipeline with yfinance data)
- **Backend**: Flask (Python) with Yahoo Finance data, NumPy/SciPy/Pandas
- **Build**: Vite dev server on port 3000, proxies `/api` and `/socket.io` to Flask on port 5000
- **Fonts**: Cormorant Garamond (display), Plus Jakarta Sans (sans)
- **Theme**: Dark/light mode via Tailwind `class` strategy, gold/bronze accent (#c9985a)

## Project Structure

```
src/
  App.jsx              # Main app with auth gate + AnimatePresence page transitions
  main.jsx             # Entry point (AuthProvider → AppProvider → App)
  index.css            # Global styles (Tailwind layers, shimmer, card-glow)
  config/
    firebase.js        # Firebase app initialization & auth export
  components/
    auth/
      LoginPage.jsx    # Email/password login & signup page
    layout/            # Header (with sign-out), Sidebar (with layoutId nav animation)
    portfolio/         # PortfolioPanel — holdings CRUD, price fetching
    analysis/          # AnalysisPanel — benchmark comparison, correlation matrix
    optimize/          # OptimizePanel — Monte Carlo optimization, efficient frontier
    risk/              # RiskPanel — VaR/CVaR, rolling Sharpe, scenario & stress tests
    market/
      MarketPanel.jsx  # AI agent, sectors, sentiment, news (no more indices/treasury)
      AIAgent.jsx      # OpenAI-powered RAG chat for stock research & advice
    live/              # LivePanel — enriched ticker cards with sparklines, sentiment, headlines
    retirement/        # RetirementPanel — retirement planning with probability cone
    ui/                # Reusable UI: Button, Card, Input, Loading, Toast, Motion
  context/
    AppContext.jsx      # Global app state (holdings, darkMode, activeTab, toasts)
    AuthContext.jsx     # Firebase auth state (user, loading, signIn, signUp, logOut)
  hooks/
    useLocalStorage.js  # Local storage persistence hook
  services/
    api.js              # API client (all backend endpoints including live-analysis & ai-agent)
new.py                  # Flask backend (24+ API endpoints)
ft.html, app.js, style.css  # Legacy backup files
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

- `POST /api/live-analysis` — Enriched data per ticker: price, mini chart, sentiment, headlines (parallelized with ThreadPoolExecutor)
- `POST /api/ai-agent` — RAG pipeline: gathers yfinance data (financials, recommendations, news, 1Y performance) → builds context → calls OpenAI 2.0 Flash
- `POST /api/prices` — Current prices & daily changes
- `POST /api/analyze` — Portfolio performance vs benchmark
- `POST /api/optimize` — Monte Carlo optimization (10k+ simulations)
- `POST /api/risk-metrics` — VaR, CVaR, Beta, Alpha
- `POST /api/sentiment` — News sentiment analysis (VADER or keyword fallback)
- `POST /api/retirement/calculate` — Retirement planning with Monte Carlo

## Guidelines

- Use JSX (not TSX) — project uses plain JavaScript, not TypeScript
- Style with Tailwind CSS utility classes
- Reuse existing UI components from `src/components/ui/` before creating new ones
- Use shared animation utilities from `src/components/ui/Motion.jsx` (StaggerList, StaggerItem, pageTransition, etc.)
- Keep API calls in `src/services/api.js`
- Use `AppContext` for shared portfolio/UI state, `AuthContext` for authentication state
- Backend API runs on port 5000; frontend proxies `/api` routes to it
- `.env` contains `VITE_API_URL` and Firebase config — do not commit `.env` files
- The AI agent feature requires `OPENAI_API_KEY` set as a backend environment variable
