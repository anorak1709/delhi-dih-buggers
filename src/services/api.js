const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

async function request(endpoint, options = {}) {
  const url = `${BASE_URL}${endpoint}`;
  const config = {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  };

  const res = await fetch(url, config);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || `Request failed: ${res.status}`);
  }
  return res.json();
}

function post(endpoint, body) {
  return request(endpoint, { method: 'POST', body: JSON.stringify(body) });
}

function get(endpoint) {
  return request(endpoint, { method: 'GET' });
}

// ── Core Analytics ──────────────────────────────────────────────
export function analyzePortfolio(holdings, benchmark = '^NSEI', startDate = '2018-01-01', riskFreeRate = 0.065) {
  return post('/api/analyze', { holdings, benchmark, start_date: startDate, risk_free_rate: riskFreeRate });
}

export function optimizePortfolio(tickers, startDate = '2018-01-01', numPortfolios = 10000, riskFreeRate = 0.065, method = 'standard') {
  return post('/api/optimize', { tickers, start_date: startDate, num_portfolios: numPortfolios, risk_free_rate: riskFreeRate, method });
}

export function getCorrelation(tickers, startDate = '2018-01-01') {
  return post('/api/correlation', { tickers, start_date: startDate });
}

export function getRebalance(holdings, targetAllocations) {
  return post('/api/rebalance', { holdings, target_allocations: targetAllocations });
}

// ── Risk Management ─────────────────────────────────────────────
export function getRiskMetrics(tickers, startDate = '2018-01-01', riskFreeRate = 0.065, benchmark = '^NSEI') {
  return post('/api/risk-metrics', { tickers, start_date: startDate, risk_free_rate: riskFreeRate, benchmark });
}

export function getRolling(ticker, startDate = '2018-01-01') {
  return post('/api/rolling', { ticker, start_date: startDate });
}

export function getScenario(ticker, scenarioType = 'crash') {
  return post('/api/scenario', { ticker, type: scenarioType });
}

export function getStress(ticker) {
  return post('/api/stress', { ticker });
}

// ── Market Intelligence ─────────────────────────────────────────
export async function getPrices(tickers) {
  const data = await post('/api/prices', { tickers });
  return data.prices || data;
}

export function getNews(tickers) {
  return post('/api/news', { tickers });
}

export function getSectors(tickers) {
  return post('/api/sectors', { tickers });
}

export function getDividends(tickers) {
  return post('/api/dividends', { tickers });
}

export function getSentiment(tickers) {
  return post('/api/sentiment', { tickers });
}

// ── Advanced Charts ─────────────────────────────────────────────
export function getEfficientFrontier3D(tickers, startDate = '2018-01-01', numPortfolios = 5000) {
  return post('/api/charts/efficient-frontier-3d', { tickers, start_date: startDate, num_portfolios: numPortfolios });
}

export function getTimeSeriesDecomposition(ticker, startDate = '2018-01-01') {
  return post('/api/charts/time-series-decomposition', { ticker, start_date: startDate });
}

export function getCandlestick(ticker, startDate = '2018-01-01') {
  return post('/api/charts/candlestick', { ticker, start_date: startDate });
}

export function getTechnicalIndicators(ticker, startDate = '2018-01-01') {
  return post('/api/charts/technical-indicators', { ticker, start_date: startDate });
}

export function getDrawdown(ticker, startDate = '2018-01-01') {
  return post('/api/charts/drawdown', { ticker, start_date: startDate });
}

export function getRiskContribution(holdings, startDate = '2018-01-01') {
  return post('/api/charts/risk-contribution', { holdings, start_date: startDate });
}

// ── Retirement ──────────────────────────────────────────────────
export function calculateRetirement(holdings, currentAge, retirementAge, withdrawalRate = 0.04, annualReturn = 0.10) {
  return post('/api/retirement/calculate', {
    holdings,
    current_age: currentAge,
    retirement_age: retirementAge,
    withdrawal_rate: withdrawalRate,
    annual_return: annualReturn,
  });
}

// ── Live Analysis ───────────────────────────────────────────────
export function getLiveAnalysis(tickers) {
  return post('/api/live-analysis', { tickers });
}

// ── AI Agent ────────────────────────────────────────────────────
export function askAIAgent(query, tickers = []) {
  return post('/api/ai-agent', { query, tickers });
}

// ── Constrained Optimization ─────────────────────────────────────────
export function constrainedOptimize(tickers, startDate, riskFreeRate, constraints, weightStep) {
  return post('/api/constrained-optimize', { tickers, start_date: startDate, risk_free_rate: riskFreeRate, constraints, weight_step: weightStep });
}

export function getSensitivities(tickers, weights, startDate, riskFreeRate) {
  return post('/api/sensitivities', { tickers, weights, start_date: startDate, risk_free_rate: riskFreeRate });
}

// ── Options ──────────────────────────────────────────────────────────
export function getOptionsPrice(params) {
  return post('/api/options/price', params);
}

export function getOptionsGreeksCurves(params) {
  return post('/api/options/greeks', params);
}

export function getOptionsChain(ticker, expiry) {
  return post('/api/options/chain', { ticker, expiry });
}

export function getImpliedVol(params) {
  return post('/api/options/implied-vol', params);
}

// ── HRP & Black-Litterman ────────────────────────────────────────────
export function getHRP(tickers, startDate = '2020-01-01') {
  return post('/api/hrp', { tickers, start_date: startDate });
}

export function getBlackLitterman(tickers, startDate = '2020-01-01', marketCaps = null, views = null, useSentiment = true) {
  return post('/api/black-litterman', { tickers, start_date: startDate, market_caps: marketCaps, views, use_sentiment: useSentiment });
}

// ── Backtesting ──────────────────────────────────────────────────────
export function runBacktest(params) {
  return post('/api/backtest', params);
}

// ── Volatility Surface ───────────────────────────────────────────────
export function getVolSurface(ticker) {
  return post('/api/options/vol-surface', { ticker });
}

// ── System ──────────────────────────────────────────────────────
export function healthCheck() {
  return get('/api/health');
}
