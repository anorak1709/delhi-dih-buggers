# Patch stdlib for cooperative sockets BEFORE importing anything that opens
# network connections (requests, yfinance, etc.). Required for SocketIO under
# gunicorn with the eventlet worker class in production.
import os
if os.environ.get('USE_EVENTLET', '1') != '0':
    try:
        import eventlet
        eventlet.monkey_patch()
    except ImportError:
        # eventlet isn't required for dev; Werkzeug dev server is fine there.
        pass

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats.qmc import Sobol
from scipy.cluster.hierarchy import linkage, dendrogram as scipy_dendrogram
from scipy.spatial.distance import squareform
from numpy.linalg import inv, pinv
from flask_socketio import SocketIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import time as _time

app = Flask(__name__)

# CORS: restrict to the deployed frontend in production. Comma-separated list
# in ALLOWED_ORIGINS lets us support a preview + prod domain simultaneously.
_default_origins = 'http://localhost:3000,http://127.0.0.1:3000'
_allowed_origins = [
    o.strip()
    for o in os.environ.get('ALLOWED_ORIGINS', _default_origins).split(',')
    if o.strip()
]
CORS(app, resources={r"/api/*": {"origins": _allowed_origins}})

socketio = SocketIO(app, cors_allowed_origins=_allowed_origins)

# Configuration
NEWS_API_KEY = os.environ.get('NEWS_API_KEY', None)  # Get from environment variable
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', None)
TIINGO_API_KEY = None  # Tiingo removed — all data fetched via yfinance

# Configure OpenAI if API key is available
openai_client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except ImportError:
        print("Warning: openai not installed. AI agent will not be available. Run: pip install openai")

@socketio.on('subscribe_prices')
def stream_prices(data):
    tickers = data['tickers']
    while True:
        try:
            prices = yf.download(tickers, period='1d', interval='1m', progress=False)['Close'].iloc[-1]
            socketio.emit('price_update', prices.to_dict())
        except Exception as e:
            socketio.emit('error', {'message': str(e)})
        socketio.sleep(10)

def daily_returns(series):
    """Calculate daily returns from a price series"""
    return series.pct_change().dropna()

def cagr(series, periods_per_year=252):
    """Calculate Compound Annual Growth Rate"""
    if len(series) < 2:
        return 0.0
    total_ret = series.iloc[-1] / series.iloc[0] - 1
    years = len(series) / periods_per_year
    if years <= 0:
        return 0.0
    return (1 + total_ret)**(1/years) - 1

def annual_vol(series, periods_per_year=252):
    """Calculate annualized volatility"""
    r = daily_returns(series)
    if len(r) == 0:
        return 0.0
    return r.std() * np.sqrt(periods_per_year)

def sharpe(series, rf_annual=0.0, periods_per_year=252):
    """Calculate Sharpe ratio"""
    r = daily_returns(series)
    if len(r) == 0 or r.std() == 0:
        return 0.0
    rf_daily = (1 + rf_annual)**(1/periods_per_year) - 1
    excess = r - rf_daily
    return excess.mean() / excess.std() * np.sqrt(periods_per_year)

def max_drawdown(series):
    """Calculate maximum drawdown"""
    if len(series) == 0:
        return 0.0
    running_max = series.cummax()
    drawdown = series / running_max - 1
    return drawdown.min()

def correlation_matrix(prices):
    returns = prices.pct_change().dropna()
    return returns.corr().round(3)

def var_cvar(returns, confidence=0.95):
    if len(returns) == 0:
        return 0.0, 0.0
    mu = returns.mean()
    sigma = returns.std()
    if sigma == 0:
        return 0.0, 0.0
    var = norm.ppf(1 - confidence, mu, sigma)
    cvar_returns = returns[returns <= var]
    cvar = cvar_returns.mean() if len(cvar_returns) > 0 else var
    return float(var), float(cvar)

def beta_alpha(port_returns, bench_returns, rf=0.0):
    if len(port_returns) == 0 or len(bench_returns) == 0:
        return 0.0, 0.0
    # Align the series
    aligned = pd.DataFrame({'port': port_returns, 'bench': bench_returns}).dropna()
    if len(aligned) < 2:
        return 0.0, 0.0
    
    bench_var = np.var(aligned['bench'])
    if bench_var == 0:
        return 0.0, 0.0
    
    cov = np.cov(aligned['port'], aligned['bench'])[0, 1]
    beta = cov / bench_var
    alpha = aligned['port'].mean() - rf - beta * (aligned['bench'].mean() - rf)
    return float(beta), float(alpha)

def rolling_sharpe(returns, window=60, rf=0.0):
    if len(returns) < window:
        return pd.Series(dtype=float)
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    # Avoid division by zero
    sharpe = (rolling_mean - rf) / rolling_std.replace(0, np.nan)
    return sharpe.dropna()

def apply_scenario(returns, multiplier):
    return returns * multiplier

# ── Data Layer (yfinance) ─────────────────────────────────────────────
_cache = {}  # Simple in-memory cache: key -> (data, timestamp)
_CACHE_TTL = 300  # 5 minutes

def _is_tiingo_supported(ticker):
    return False

def _cache_get(key):
    if key in _cache:
        data, ts = _cache[key]
        if _time.time() - ts < _CACHE_TTL:
            return data
        del _cache[key]
    return None

def _cache_set(key, data):
    _cache[key] = (data, _time.time())

def _tiingo_daily(ticker, start, end):
    return None

def _tiingo_meta(ticker):
    return {}

def _tiingo_news(tickers, limit=20):
    return []

def _tiingo_iex(tickers):
    return []

def _fetch_prices(tickers, start, end):
    """Fetch adjusted close prices via yfinance. Returns DataFrame indexed by date."""
    if not isinstance(tickers, list):
        tickers = [tickers]

    all_data = {}
    for t in tickers:
        try:
            yf_data = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)['Close']
            if isinstance(yf_data, pd.DataFrame):
                yf_data = yf_data.iloc[:, 0]
            if len(yf_data) > 0:
                all_data[t] = yf_data
        except Exception as e:
            print(f'Failed to fetch {t}: {e}')

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df.index = pd.to_datetime(df.index)
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df.dropna()

def _fetch_ohlcv(ticker, start, end):
    """Fetch OHLCV data via yfinance."""
    try:
        data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    except Exception:
        return pd.DataFrame()

# ── Variance Reduction Helpers ──────────────────────────────────────
def _mc_standard(n, d):
    """Standard pseudo-random weight generation."""
    w = np.random.random((n, d))
    return w / w.sum(axis=1, keepdims=True)

def _mc_antithetic(n, d):
    """Antithetic variates: pair each sample with its mirror."""
    half = n // 2
    w1 = np.random.random((half, d))
    w2 = 1.0 - w1  # mirror
    w1 = w1 / w1.sum(axis=1, keepdims=True)
    w2 = w2 / w2.sum(axis=1, keepdims=True)
    return np.vstack([w1, w2])

def _mc_sobol(n, d):
    """Quasi-Monte Carlo using Sobol low-discrepancy sequences."""
    # Sobol works best with powers of 2
    m = int(2 ** np.ceil(np.log2(max(n, 2))))
    sampler = Sobol(d=d, scramble=True, seed=42)
    w = sampler.random(m)[:n]
    # Avoid exact 0s or 1s
    w = np.clip(w, 1e-6, 1.0)
    return w / w.sum(axis=1, keepdims=True)

def _mc_full(n, d):
    """Combined: Sobol for first half, antithetic mirrors for second half."""
    half = n // 2
    sampler = Sobol(d=d, scramble=True, seed=42)
    m = int(2 ** np.ceil(np.log2(max(half, 2))))
    w1 = sampler.random(m)[:half]
    w1 = np.clip(w1, 1e-6, 1.0)
    w1 = w1 / w1.sum(axis=1, keepdims=True)
    w2 = 1.0 - w1
    w2 = np.clip(w2, 1e-6, 1.0)
    w2 = w2 / w2.sum(axis=1, keepdims=True)
    return np.vstack([w1, w2])

MC_GENERATORS = {
    'standard': _mc_standard,
    'antithetic': _mc_antithetic,
    'sobol': _mc_sobol,
    'full': _mc_full,
}

# ── Black-Scholes Options Pricing ────────────────────────────────────
def _black_scholes(S, K, T, r, sigma, option_type='call'):
    """Black-Scholes option pricing formula."""
    if T <= 0 or sigma <= 0:
        # At expiry or zero vol: return intrinsic value
        if option_type == 'call':
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    else:
        return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def _bs_greeks(S, K, T, r, sigma, option_type='call'):
    """Compute Black-Scholes Greeks."""
    if T <= 0 or sigma <= 0:
        intrinsic_call = max(S - K, 0.0)
        return {'delta': 1.0 if (option_type == 'call' and S > K) else (-1.0 if (option_type == 'put' and S < K) else 0.0),
                'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0}

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    sqrt_T = np.sqrt(T)

    # Gamma (same for call and put)
    gamma = pdf_d1 / (S * sigma * sqrt_T)

    # Vega (same for call and put) — per 1% move in vol
    vega = S * pdf_d1 * sqrt_T / 100.0

    if option_type == 'call':
        delta = norm.cdf(d1)
        theta = (-S * pdf_d1 * sigma / (2 * sqrt_T) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365.0
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100.0
    else:
        delta = norm.cdf(d1) - 1.0
        theta = (-S * pdf_d1 * sigma / (2 * sqrt_T) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365.0
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100.0

    return {
        'delta': round(float(delta), 6),
        'gamma': round(float(gamma), 6),
        'theta': round(float(theta), 6),
        'vega': round(float(vega), 6),
        'rho': round(float(rho), 6),
    }


def _implied_volatility(market_price, S, K, T, r, option_type='call', tol=1e-6, max_iter=100):
    """Newton-Raphson implied volatility solver with bisection fallback."""
    if T <= 0:
        return None

    sigma = 0.3  # initial guess
    for _ in range(max_iter):
        price = _black_scholes(S, K, T, r, sigma, option_type)
        vega_raw = S * norm.pdf(
            (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        ) * np.sqrt(T)

        if abs(vega_raw) < 1e-10:
            break
        sigma -= (price - market_price) / vega_raw
        sigma = max(0.001, min(sigma, 5.0))  # clamp
        if abs(price - market_price) < tol:
            return round(float(sigma), 6)

    # Bisection fallback
    lo, hi = 0.001, 5.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        price = _black_scholes(S, K, T, r, mid, option_type)
        if abs(price - market_price) < tol:
            return round(float(mid), 6)
        if price < market_price:
            lo = mid
        else:
            hi = mid
    return round(float((lo + hi) / 2.0), 6)


# ── Backtracking Constrained Optimization ────────────────────────────
def _backtrack_portfolios(tickers, mean_returns, cov_matrix, constraints,
                          ticker_info, risk_free_rate, weight_step=0.05):
    """Backtracking search for valid portfolios under constraints."""
    n = len(tickers)
    min_w = constraints.get('min_weight', {})
    max_w = constraints.get('max_weight', {})
    min_ret = constraints.get('min_total_return', None)
    max_vol = constraints.get('max_total_volatility', None)
    sector_limits = constraints.get('sector_limits', {})
    min_div = constraints.get('min_dividend_yield', None)

    valid = []
    explored = [0]
    pruned = [0]
    mean_arr = mean_returns.values
    cov_arr = cov_matrix.values

    # Pre-compute per-ticker bounds
    lo = np.array([min_w.get(t, 0.0) for t in tickers])
    hi = np.array([max_w.get(t, 1.0) for t in tickers])
    step = weight_step

    def backtrack(idx, weights, remaining):
        explored[0] += 1
        if idx == n:
            # Last asset gets all remaining weight
            w = remaining
            if w < lo[idx - 1] - 1e-9 or w > hi[idx - 1] + 1e-9:
                pruned[0] += 1
                return
            weights_full = np.array(weights)

            # Check return constraint
            port_ret = float(weights_full @ mean_arr)
            if min_ret is not None and port_ret < min_ret:
                pruned[0] += 1
                return

            # Check volatility constraint
            port_vol = float(np.sqrt(weights_full @ cov_arr @ weights_full))
            if max_vol is not None and port_vol > max_vol:
                pruned[0] += 1
                return

            # Check sector limits
            if sector_limits:
                sector_alloc = {}
                for i, t in enumerate(tickers):
                    sec = ticker_info.get(t, {}).get('sector', 'Unknown')
                    sector_alloc[sec] = sector_alloc.get(sec, 0) + weights_full[i]
                for sec, limit in sector_limits.items():
                    if sector_alloc.get(sec, 0) > limit + 1e-9:
                        pruned[0] += 1
                        return

            # Check dividend yield
            if min_div is not None:
                weighted_div = sum(
                    weights_full[i] * ticker_info.get(t, {}).get('dividend_yield', 0)
                    for i, t in enumerate(tickers)
                )
                if weighted_div < min_div:
                    pruned[0] += 1
                    return

            sharpe = (port_ret - risk_free_rate) / port_vol if port_vol > 0 else 0
            valid.append({
                'weights': {t: round(float(weights_full[i]), 4) for i, t in enumerate(tickers)},
                'expected_return': round(port_ret, 6),
                'volatility': round(port_vol, 6),
                'sharpe_ratio': round(float(sharpe), 4),
            })
            return

        # Current asset index
        i = idx
        min_remaining_for_rest = sum(lo[j] for j in range(i + 1, n))
        max_remaining_for_rest = sum(hi[j] for j in range(i + 1, n))

        w_lo = max(lo[i], remaining - max_remaining_for_rest)
        w_hi = min(hi[i], remaining - min_remaining_for_rest)

        if w_lo > w_hi + 1e-9:
            pruned[0] += 1
            return

        w = w_lo
        while w <= w_hi + 1e-9:
            backtrack(idx + 1, weights + [w], remaining - w)
            w = round(w + step, 10)

            # Safety: limit total explored to prevent hanging
            if explored[0] > 500000:
                return

    backtrack(0, [], 1.0)

    # Sort by Sharpe and return top 20
    valid.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
    return valid[:20], explored[0], pruned[0]


def _compute_sensitivities(weights_dict, tickers, mean_returns, cov_matrix, risk_free_rate):
    """Finite-difference sensitivity analysis."""
    weights = np.array([weights_dict[t] for t in tickers])
    mean_arr = mean_returns.values
    cov_arr = cov_matrix.values

    # Base metrics
    base_ret = float(weights @ mean_arr)
    base_vol = float(np.sqrt(weights @ cov_arr @ weights))
    base_sharpe = (base_ret - risk_free_rate) / base_vol if base_vol > 0 else 0

    sensitivities = []

    # 1. Volatility bump (+1%): scale diagonal of cov by 1.01^2
    cov_bumped = cov_arr.copy()
    np.fill_diagonal(cov_bumped, np.diag(cov_arr) * 1.01 ** 2)
    bumped_vol = float(np.sqrt(weights @ cov_bumped @ weights))
    bumped_sharpe = (base_ret - risk_free_rate) / bumped_vol if bumped_vol > 0 else 0
    sensitivities.append({
        'parameter': 'Volatility +1%',
        'base_sharpe': round(base_sharpe, 4),
        'bumped_sharpe': round(bumped_sharpe, 4),
        'delta_sharpe': round(bumped_sharpe - base_sharpe, 4),
        'pct_impact': round((bumped_sharpe - base_sharpe) / abs(base_sharpe) * 100, 2) if base_sharpe != 0 else 0,
    })

    # 2. Correlation bump (+5%): scale off-diagonal of cov
    cov_corr = cov_arr.copy()
    diag = np.sqrt(np.diag(cov_arr))
    for i in range(len(tickers)):
        for j in range(len(tickers)):
            if i != j:
                cov_corr[i, j] = cov_arr[i, j] * 1.05
    bumped_vol = float(np.sqrt(weights @ cov_corr @ weights))
    bumped_sharpe = (base_ret - risk_free_rate) / bumped_vol if bumped_vol > 0 else 0
    sensitivities.append({
        'parameter': 'Correlation +5%',
        'base_sharpe': round(base_sharpe, 4),
        'bumped_sharpe': round(bumped_sharpe, 4),
        'delta_sharpe': round(bumped_sharpe - base_sharpe, 4),
        'pct_impact': round((bumped_sharpe - base_sharpe) / abs(base_sharpe) * 100, 2) if base_sharpe != 0 else 0,
    })

    # 3. Expected return bump (+1%)
    mean_bumped = mean_arr + 0.01
    bumped_ret = float(weights @ mean_bumped)
    bumped_sharpe = (bumped_ret - risk_free_rate) / base_vol if base_vol > 0 else 0
    sensitivities.append({
        'parameter': 'Returns +1%',
        'base_sharpe': round(base_sharpe, 4),
        'bumped_sharpe': round(bumped_sharpe, 4),
        'delta_sharpe': round(bumped_sharpe - base_sharpe, 4),
        'pct_impact': round((bumped_sharpe - base_sharpe) / abs(base_sharpe) * 100, 2) if base_sharpe != 0 else 0,
    })

    # 4. Risk-free rate bump (+0.25%)
    rf_bumped = risk_free_rate + 0.0025
    bumped_sharpe = (base_ret - rf_bumped) / base_vol if base_vol > 0 else 0
    sensitivities.append({
        'parameter': 'Risk-Free Rate +0.25%',
        'base_sharpe': round(base_sharpe, 4),
        'bumped_sharpe': round(bumped_sharpe, 4),
        'delta_sharpe': round(bumped_sharpe - base_sharpe, 4),
        'pct_impact': round((bumped_sharpe - base_sharpe) / abs(base_sharpe) * 100, 2) if base_sharpe != 0 else 0,
    })

    # Per-asset weight sensitivities
    per_asset = []
    for idx, t in enumerate(tickers):
        w_bump = weights.copy()
        bump = 0.01
        w_bump[idx] += bump
        # Reduce others proportionally
        others_sum = w_bump.sum() - w_bump[idx]
        if others_sum > 0:
            for j in range(len(tickers)):
                if j != idx:
                    w_bump[j] *= (1 - w_bump[idx]) / others_sum
        w_bump = np.clip(w_bump, 0, 1)
        w_bump /= w_bump.sum()

        b_ret = float(w_bump @ mean_arr)
        b_vol = float(np.sqrt(w_bump @ cov_arr @ w_bump))
        b_sharpe = (b_ret - risk_free_rate) / b_vol if b_vol > 0 else 0
        per_asset.append({
            'ticker': t,
            'weight_bump': bump,
            'return_delta': round(b_ret - base_ret, 6),
            'vol_delta': round(b_vol - base_vol, 6),
            'sharpe_delta': round(b_sharpe - base_sharpe, 4),
        })

    return {
        'base_metrics': {
            'expected_return': round(base_ret, 6),
            'volatility': round(base_vol, 6),
            'sharpe_ratio': round(base_sharpe, 4),
        },
        'sensitivities': sensitivities,
        'per_asset_sensitivities': per_asset,
    }


# ── Hierarchical Risk Parity (HRP) ───────────────────────────────────
def _hrp_tree(cov, corr):
    """Build hierarchical tree from correlation matrix."""
    dist = np.sqrt((1 - corr) / 2)
    np.fill_diagonal(dist.values, 0)
    condensed = squareform(dist.values, checks=False)
    link = linkage(condensed, method='single')
    return link

def _hrp_quasi_diag(link):
    """Quasi-diagonalization: reorder assets by dendrogram leaves."""
    link = link.astype(int, copy=False)
    n = link[-1, -1]
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = n
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]
        df1 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df1])
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()

def _hrp_recursive_bisection(cov, sorted_indices):
    """Allocate weights via inverse-variance recursive bisection."""
    w = pd.Series(1.0, index=sorted_indices)
    cluster_items = [sorted_indices]

    while len(cluster_items) > 0:
        cluster_items_new = []
        for sub in cluster_items:
            if len(sub) <= 1:
                continue
            mid = len(sub) // 2
            left = sub[:mid]
            right = sub[mid:]

            # Cluster variance = inverse of sum of inverse variances
            def cluster_var(items):
                cov_sub = cov.iloc[items, items]
                ivp = 1.0 / np.diag(cov_sub)
                ivp /= ivp.sum()
                return float(np.dot(ivp, np.dot(cov_sub, ivp)))

            lv = cluster_var(left)
            rv = cluster_var(right)
            alpha = 1 - lv / (lv + rv)

            w[left] *= alpha
            w[right] *= (1 - alpha)

            if len(left) > 1:
                cluster_items_new.append(left)
            if len(right) > 1:
                cluster_items_new.append(right)
        cluster_items = cluster_items_new

    return w / w.sum()

def _hrp_optimize(returns_df):
    """Run full HRP optimization. Returns weights, linkage, sorted_order, metrics."""
    cov = returns_df.cov() * 252
    corr = returns_df.corr()
    link = _hrp_tree(cov, corr)
    sorted_ix = _hrp_quasi_diag(link)
    sorted_ix = [int(x) for x in sorted_ix]
    weights_series = _hrp_recursive_bisection(cov, sorted_ix)

    # Map back to ticker names
    tickers = returns_df.columns.tolist()
    weights = {tickers[i]: float(weights_series[i]) for i in range(len(tickers))}

    # Compute metrics
    mean_returns = returns_df.mean() * 252
    w_arr = np.array([weights[t] for t in tickers])
    port_ret = float(np.dot(w_arr, mean_returns.values))
    port_vol = float(np.sqrt(np.dot(w_arr.T, np.dot(cov.values, w_arr))))
    port_sharpe = port_ret / port_vol if port_vol > 0 else 0

    # Dendrogram data for frontend rendering
    ddata = scipy_dendrogram(link, labels=tickers, no_plot=True)

    return {
        'weights': weights,
        'linkage_matrix': link.tolist(),
        'sorted_order': sorted_ix,
        'dendrogram_plot': {
            'icoord': ddata['icoord'],
            'dcoord': ddata['dcoord'],
            'ivl': ddata['ivl'],
            'leaves': ddata['leaves'],
        },
        'metrics': {
            'expected_return': port_ret,
            'volatility': port_vol,
            'sharpe_ratio': port_sharpe,
        },
    }

# ── Black-Litterman Model ────────────────────────────────────────────
def _black_litterman(market_caps, cov_matrix, views_P, views_Q, tau=0.05, omega=None, delta=2.5):
    """
    Black-Litterman model.
    market_caps: array of market cap values
    cov_matrix: NxN covariance matrix (numpy)
    views_P: KxN matrix (K views on N assets)
    views_Q: Kx1 vector of expected returns for views
    """
    n = len(market_caps)
    w_mkt = np.array(market_caps, dtype=float)
    w_mkt = w_mkt / w_mkt.sum()
    cov = np.array(cov_matrix, dtype=float)

    # Implied equilibrium returns
    pi = delta * cov @ w_mkt

    P = np.array(views_P, dtype=float)
    Q = np.array(views_Q, dtype=float).flatten()

    if len(P) == 0 or len(Q) == 0:
        # No views: return market-cap weights and equilibrium returns
        return {
            'weights': {i: float(w) for i, w in enumerate(w_mkt)},
            'expected_returns': pi.tolist(),
            'implied_returns': pi.tolist(),
            'posterior_cov': (tau * cov).tolist(),
        }

    # Omega: uncertainty of views
    if omega is None:
        omega = tau * np.diag(np.diag(P @ (tau * cov) @ P.T))

    # Posterior
    try:
        tau_cov_inv = inv(tau * cov)
    except Exception:
        tau_cov_inv = pinv(tau * cov)

    try:
        omega_inv = inv(omega)
    except Exception:
        omega_inv = pinv(omega)

    try:
        M = inv(tau_cov_inv + P.T @ omega_inv @ P)
    except Exception:
        M = pinv(tau_cov_inv + P.T @ omega_inv @ P)

    mu_bl = M @ (tau_cov_inv @ pi + P.T @ omega_inv @ Q)
    cov_bl = M

    # Optimal weights from posterior
    try:
        w_bl = inv(delta * cov_bl) @ mu_bl
    except Exception:
        w_bl = pinv(delta * cov_bl) @ mu_bl

    # Normalize weights (allow short selling to be clamped)
    w_bl = np.maximum(w_bl, 0)  # No short selling
    if w_bl.sum() > 0:
        w_bl = w_bl / w_bl.sum()

    return {
        'weights': w_bl.tolist(),
        'expected_returns': mu_bl.tolist(),
        'implied_returns': pi.tolist(),
        'posterior_cov': cov_bl.tolist(),
    }

def _sentiment_to_views(sentiment_scores, tickers, magnitude=0.02):
    """Convert sentiment scores to Black-Litterman views P, Q."""
    P_rows = []
    Q_vals = []
    for i, t in enumerate(tickers):
        score = sentiment_scores.get(t, 0)
        if abs(score) > 0.3:
            row = np.zeros(len(tickers))
            row[i] = 1.0
            P_rows.append(row)
            Q_vals.append(magnitude * score)
    if not P_rows:
        return np.array([]).reshape(0, len(tickers)), np.array([])
    return np.array(P_rows), np.array(Q_vals)

# ── Backtesting with Friction ────────────────────────────────────────
def _apply_friction(returns_series, rebalance_mask, fee_pct=0.001, slippage_vol_factor=0.05):
    """Apply trading friction on rebalance dates."""
    adj = returns_series.copy()
    if slippage_vol_factor > 0:
        rolling_vol = returns_series.rolling(20).std().fillna(returns_series.std())
        slippage = rolling_vol * slippage_vol_factor
    else:
        slippage = pd.Series(0, index=returns_series.index)

    total_fees = 0.0
    for date in returns_series.index[rebalance_mask]:
        fee = fee_pct
        slip = float(slippage.get(date, 0))
        adj.loc[date] -= (fee + slip)
        total_fees += fee

    return adj, total_fees

def _backtest_portfolio(tickers, weights, start_date, end_date, rebalance_freq='monthly',
                        fee_pct=0.001, slippage_factor=0.05, benchmark='SPY'):
    """Full backtesting with friction layer."""
    all_tickers = list(tickers) + ([benchmark] if benchmark and benchmark not in tickers else [])
    prices = _fetch_prices(all_tickers, start_date, end_date)

    if prices.empty or len(prices) < 30:
        raise ValueError('Insufficient price data for backtesting')

    returns = prices.pct_change().dropna()

    # Portfolio returns
    w = np.array([weights.get(t, 0) for t in tickers])
    w = w / w.sum()
    port_returns = returns[tickers] @ w

    # Rebalance dates
    freq_map = {'monthly': 'M', 'quarterly': 'Q', 'annual': 'A'}
    freq = freq_map.get(rebalance_freq, 'M')
    rebal_dates = port_returns.resample(freq).last().index
    rebal_mask = port_returns.index.isin(rebal_dates)

    # Apply friction
    port_adj, total_fees = _apply_friction(port_returns, rebal_mask, fee_pct, slippage_factor)

    # Cumulative returns
    port_cum = (1 + port_adj).cumprod()
    port_cum_nofee = (1 + port_returns).cumprod()

    # Benchmark
    bench_metrics = {}
    bench_series = []
    if benchmark and benchmark in prices.columns:
        bench_ret = returns[benchmark]
        bench_cum = (1 + bench_ret).cumprod()
        bench_series = [{'date': d.strftime('%Y-%m-%d'), 'value': float(v)} for d, v in bench_cum.items()]
        bench_metrics = {
            'cagr': float(cagr(bench_cum)),
            'volatility': float(annual_vol(prices[benchmark])),
            'sharpe': float(sharpe(prices[benchmark])),
            'max_drawdown': float(max_drawdown(bench_cum)),
        }
        bv, bc = var_cvar(bench_ret)
        bench_metrics['var_95'] = float(bv)
        bench_metrics['cvar_95'] = float(bc)

    # Portfolio metrics
    port_metrics = {
        'cagr': float(cagr(port_cum)),
        'volatility': float(annual_vol(port_cum)),
        'sharpe': float(sharpe(port_cum)),
        'max_drawdown': float(max_drawdown(port_cum)),
    }
    pv, pc = var_cvar(port_adj)
    port_metrics['var_95'] = float(pv)
    port_metrics['cvar_95'] = float(pc)

    port_series = [{'date': d.strftime('%Y-%m-%d'), 'value': float(v)} for d, v in port_cum.items()]
    rebal_list = [d.strftime('%Y-%m-%d') for d in rebal_dates if d in port_returns.index]

    friction_impact = float(port_cum_nofee.iloc[-1] - port_cum.iloc[-1]) if len(port_cum) > 0 else 0

    return {
        'portfolio_series': port_series,
        'benchmark_series': bench_series,
        'portfolio_metrics': port_metrics,
        'benchmark_metrics': bench_metrics,
        'rebalance_dates': rebal_list,
        'total_fees_paid': float(total_fees),
        'friction_impact': float(friction_impact),
        'num_rebalances': len(rebal_list),
    }


def validate_date_range(start_date, end_date):
    """Validate date range"""
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        if start >= end:
            return False, "Start date must be before end date"
        
        if start < datetime(2000, 1, 1):
            return False, "Start date too far in the past"
        
        if end > datetime.now():
            return False, "End date cannot be in the future"
        
        return True, None
    except ValueError:
        return False, "Invalid date format"

def validate_tickers(tickers):
    """Validate ticker symbols"""
    import re
    ticker_pattern = re.compile(r'^[A-Z0-9\.\-\^]+$')
    
    for ticker in tickers:
        if not ticker_pattern.match(ticker):
            return False, f"Invalid ticker format: {ticker}"
    
    return True, None

def normalize_columns(df):
    """Normalize DataFrame columns by removing MultiIndex levels"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def normalize_series(data):
    """Normalize yfinance data to handle single/multi ticker downloads"""
    if isinstance(data, pd.DataFrame):
        data = normalize_columns(data)
        if len(data.columns) == 1:
            return data.iloc[:, 0]
    elif isinstance(data, pd.Series):
        return data
    return data

@app.route('/api/analyze', methods=['POST'])
def analyze_portfolio():
    try:
        data = request.json
        holdings = data.get('holdings', {})
        benchmark = data.get('benchmark', '^NSEI')
        start_date = data.get('start_date', '2018-01-01')
        risk_free_rate = data.get('risk_free_rate', 0.065)
        end_date = datetime.today().strftime("%Y-%m-%d")
        
        if not holdings:
            return jsonify({'error': 'No holdings provided'}), 400
        
        # Validate date range
        valid, error = validate_date_range(start_date, end_date)
        if not valid:
            return jsonify({'error': error}), 400
        
        # Validate tickers
        all_tickers = list(holdings.keys()) + [benchmark]
        valid, error = validate_tickers(all_tickers)
        if not valid:
            return jsonify({'error': error}), 400
        
        # Download price data
        tickers = list(holdings.keys())
        print(f"Downloading data for: {tickers + [benchmark]}")
        
        # Download portfolio stocks
        px_portfolio = _fetch_prices(tickers, start_date, end_date)

        # Download benchmark separately to avoid issues
        px_benchmark_df = _fetch_prices([benchmark], start_date, end_date)
        px_benchmark = px_benchmark_df.iloc[:, 0] if not px_benchmark_df.empty else pd.Series(dtype=float)

        # Clean data
        px_portfolio = px_portfolio.dropna(how="all").ffill()
        px_benchmark = px_benchmark.dropna().ffill()
        
        # Check if all tickers have data
        missing_tickers = []
        for ticker in tickers:
            if ticker not in px_portfolio.columns or px_portfolio[ticker].isna().all():
                missing_tickers.append(ticker)
        
        if missing_tickers:
            return jsonify({
                'error': f'Unable to fetch data for: {", ".join(missing_tickers)}. Please check ticker symbols.'
            }), 400
        
        # Build portfolio value series
        portfolio_value = pd.Series(0.0, index=px_portfolio.index)
        
        for ticker, qty in holdings.items():
            if ticker in px_portfolio.columns:
                portfolio_value = portfolio_value.add(px_portfolio[ticker] * qty, fill_value=0.0)
        
        # Align dates between portfolio and benchmark
        df_compare = pd.DataFrame({
            "Portfolio": portfolio_value,
            "Benchmark": px_benchmark
        }).dropna()
        
        if len(df_compare) < 30:
            return jsonify({
                'error': 'Insufficient data points. Try a different start date or check if tickers are valid.'
            }), 400
        
        # Normalize for chart
        norm = df_compare / df_compare.iloc[0]
        
        # Calculate metrics
        portfolio_metrics = {
            'cagr': float(cagr(df_compare['Portfolio'])),
            'annual_vol': float(annual_vol(df_compare['Portfolio'])),
            'sharpe': float(sharpe(df_compare['Portfolio'], rf_annual=risk_free_rate)),
            'max_drawdown': float(max_drawdown(df_compare['Portfolio']))
        }
        
        benchmark_metrics = {
            'cagr': float(cagr(df_compare['Benchmark'])),
            'annual_vol': float(annual_vol(df_compare['Benchmark'])),
            'sharpe': float(sharpe(df_compare['Benchmark'], rf_annual=risk_free_rate)),
            'max_drawdown': float(max_drawdown(df_compare['Benchmark']))
        }
        
        # Prepare chart data (sample every 5 days to reduce data size)
        chart_data = []
        sample_interval = max(1, len(norm) // 200)  # Limit to ~200 points
        sampled = norm.iloc[::sample_interval]
        
        for idx, row in sampled.iterrows():
            chart_data.append({
                'date': idx.strftime('%Y-%m-%d'),
                'portfolio': round(row['Portfolio'], 4),
                'benchmark': round(row['Benchmark'], 4)
            })
        
        return jsonify({
            'portfolio': portfolio_metrics,
            'benchmark': benchmark_metrics,
            'chart_data': chart_data,
            'date_range': {
                'start': df_compare.index[0].strftime('%Y-%m-%d'),
                'end': df_compare.index[-1].strftime('%Y-%m-%d')
            }
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/optimize', methods=['POST'])
def optimize_portfolio():
    try:
        data = request.json
        tickers = data.get('tickers', [])
        start_date = data.get('start_date', '2018-01-01')
        num_portfolios = data.get('num_portfolios', 10000)
        risk_free_rate = data.get('risk_free_rate', 0.0)
        end_date = datetime.today().strftime("%Y-%m-%d")

        if not tickers or len(tickers) < 2:
            return jsonify({'error': 'Please provide at least 2 tickers'}), 400

        # Validate inputs
        valid, error = validate_date_range(start_date, end_date)
        if not valid:
            return jsonify({'error': error}), 400
        
        valid, error = validate_tickers(tickers)
        if not valid:
            return jsonify({'error': error}), 400

        print(f"Optimizing portfolio for: {tickers}")

        # Download price data
        prices = _fetch_prices(tickers, start_date, end_date)

        if len(prices) < 30:
            return jsonify({'error': 'Insufficient data. Try a different date range.'}), 400

        # Calculate returns
        returns = prices.pct_change().dropna()
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        # Monte Carlo Simulation with variance reduction
        method = data.get('method', 'standard')
        np.random.seed(42)  # For reproducibility

        generate_weights = MC_GENERATORS.get(method, _mc_standard)
        all_weights = generate_weights(num_portfolios, len(tickers))

        # Vectorized portfolio metric computation
        mean_ret_arr = mean_returns.values
        cov_arr = cov_matrix.values
        portfolio_returns_arr = all_weights @ mean_ret_arr
        portfolio_vols_arr = np.sqrt(np.einsum('ij,jk,ik->i', all_weights, cov_arr, all_weights))
        sharpe_arr = np.where(portfolio_vols_arr > 0, (portfolio_returns_arr - risk_free_rate) / portfolio_vols_arr, 0)

        results = np.array([portfolio_vols_arr, portfolio_returns_arr, sharpe_arr])
        weights_record = all_weights.tolist()

        # Find maximum Sharpe ratio portfolio
        max_sharpe_idx = np.argmax(results[2])
        max_sharpe_weights = weights_record[max_sharpe_idx]
        max_sharpe_return = results[1, max_sharpe_idx]
        max_sharpe_vol = results[0, max_sharpe_idx]
        max_sharpe_ratio = results[2, max_sharpe_idx]

        # Find minimum volatility portfolio
        min_vol_idx = np.argmin(results[0])
        min_vol_weights = weights_record[min_vol_idx]
        min_vol_return = results[1, min_vol_idx]
        min_vol_vol = results[0, min_vol_idx]
        min_vol_sharpe = results[2, min_vol_idx]

        # Generate plot
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(
            results[0, :],
            results[1, :],
            c=results[2, :],
            cmap='viridis',
            s=10,
            alpha=0.6
        )
        plt.colorbar(scatter, label='Sharpe Ratio')
        plt.scatter(max_sharpe_vol, max_sharpe_return, color='red', marker='*',
                   s=500, label='Max Sharpe Ratio', edgecolors='black', linewidths=1.5)
        plt.scatter(min_vol_vol, min_vol_return, color='blue', marker='*',
                   s=500, label='Min Volatility', edgecolors='black', linewidths=1.5)
        plt.xlabel('Annualized Volatility', fontsize=12)
        plt.ylabel('Expected Annual Return', fontsize=12)
        plt.title('Monte Carlo Simulation - Efficient Frontier', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Convert plot to base64 image
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
        plt.close()

        # Prepare response
        optimal_portfolio = {
            'weights': {ticker: float(weight) for ticker, weight in zip(tickers, max_sharpe_weights)},
            'expected_return': float(max_sharpe_return),
            'volatility': float(max_sharpe_vol),
            'sharpe_ratio': float(max_sharpe_ratio)
        }

        min_risk_portfolio = {
            'weights': {ticker: float(weight) for ticker, weight in zip(tickers, min_vol_weights)},
            'expected_return': float(min_vol_return),
            'volatility': float(min_vol_vol),
            'sharpe_ratio': float(min_vol_sharpe)
        }

        # Sample efficient frontier points for chart data
        frontier_points = []
        sample_indices = np.linspace(0, num_portfolios-1, min(500, num_portfolios), dtype=int)
        for idx in sample_indices:
            frontier_points.append({
                'volatility': float(results[0, idx]),
                'return': float(results[1, idx]),
                'sharpe': float(results[2, idx])
            })

        return jsonify({
            'optimal_portfolio': optimal_portfolio,
            'min_risk_portfolio': min_risk_portfolio,
            'frontier_points': frontier_points,
            'plot_image': img_base64,
            'statistics': {
                'num_simulations': num_portfolios,
                'method': method,
                'date_range': {
                    'start': prices.index[0].strftime('%Y-%m-%d'),
                    'end': prices.index[-1].strftime('%Y-%m-%d')
                }
            }
        })

    except Exception as e:
        print(f"Error in optimization: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Optimization failed: {str(e)}'}), 500

@app.route('/api/correlation', methods=['POST'])
def correlation():
    try:
        data = request.json
        tickers = data['tickers']
        start_date = data.get('start_date', '2018-01-01')

        valid, error = validate_tickers(tickers)
        if not valid:
            return jsonify({'error': error}), 400

        end_date = datetime.today().strftime("%Y-%m-%d")
        prices = _fetch_prices(tickers, start_date, end_date)
        
        if len(prices) < 30:
            return jsonify({'error': 'Insufficient data for correlation analysis'}), 400
        
        corr = correlation_matrix(prices)

        return jsonify({
            "tickers": tickers,
            "matrix": corr.values.tolist()
        })
    except Exception as e:
        print(f"Error in correlation: {str(e)}")
        return jsonify({'error': f'Correlation analysis failed: {str(e)}'}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Portfolio Optimizer API is running'})

@app.route('/api/risk-metrics', methods=['POST'])
def risk_metrics():
    try:
        data = request.json
        tickers = data['tickers']
        benchmark = data.get('benchmark', '^GSPC')

        valid, error = validate_tickers(tickers + [benchmark])
        if not valid:
            return jsonify({'error': error}), 400

        if len(tickers) == 0:
            return jsonify({'error': 'No tickers provided'}), 400

        all_tickers = tickers + [benchmark]
        start_3y = (datetime.today() - timedelta(days=3*365)).strftime("%Y-%m-%d")
        end_date = datetime.today().strftime("%Y-%m-%d")
        prices = _fetch_prices(all_tickers, start_3y, end_date)
        
        if len(prices) < 30:
            return jsonify({'error': 'Insufficient data for risk analysis'}), 400
        
        returns = prices.pct_change().dropna()

        # Calculate portfolio returns as equal-weighted average
        portfolio_returns = returns[tickers].mean(axis=1) if len(tickers) > 1 else returns[tickers[0]]
        benchmark_returns = returns[benchmark]

        var, cvar = var_cvar(portfolio_returns)
        beta, alpha = beta_alpha(portfolio_returns, benchmark_returns)

        return jsonify({
            "var": var,
            "cvar": cvar,
            "beta": beta,
            "alpha": alpha
        })
    except Exception as e:
        print(f"Error in risk metrics: {str(e)}")
        return jsonify({'error': f'Risk analysis failed: {str(e)}'}), 500

@app.route('/api/rolling', methods=['POST'])
def rolling():
    try:
        ticker = request.json['ticker']

        valid, error = validate_tickers([ticker])
        if not valid:
            return jsonify({'error': error}), 400

        start_2y = (datetime.today() - timedelta(days=2*365)).strftime("%Y-%m-%d")
        end_date = datetime.today().strftime("%Y-%m-%d")
        prices_df = _fetch_prices([ticker], start_2y, end_date)
        if prices_df.empty:
            return jsonify({'error': 'No price data available'}), 400
        prices = prices_df.iloc[:, 0]

        if len(prices) < 60:
            return jsonify({'error': 'Insufficient data for rolling analysis'}), 400

        returns = prices.pct_change().dropna()
        sharpe_rolling = rolling_sharpe(returns)

        return jsonify({
            "dates": sharpe_rolling.index.strftime('%Y-%m-%d').tolist(),
            "values": sharpe_rolling.values.tolist()
        })
    except Exception as e:
        print(f"Error in rolling analysis: {str(e)}")
        return jsonify({'error': f'Rolling analysis failed: {str(e)}'}), 500

@app.route('/api/scenario', methods=['POST'])
def scenario():
    try:
        data = request.json
        ticker = data['ticker']
        scenario = data['type']

        valid, error = validate_tickers([ticker])
        if not valid:
            return jsonify({'error': error}), 400

        scenarios = {
            "bull": 1.5,
            "crash": 0.5,
            "flat": 1.0
        }

        if scenario not in scenarios:
            return jsonify({'error': 'Invalid scenario type'}), 400

        start_2y = (datetime.today() - timedelta(days=2*365)).strftime("%Y-%m-%d")
        end_date = datetime.today().strftime("%Y-%m-%d")
        prices_df = _fetch_prices([ticker], start_2y, end_date)
        if prices_df.empty:
            return jsonify({'error': 'No price data available'}), 400
        prices = prices_df.iloc[:, 0]

        if len(prices) < 30:
            return jsonify({'error': 'Insufficient data for scenario analysis'}), 400

        returns = prices.pct_change().dropna()
        simulated = apply_scenario(returns, scenarios[scenario])

        mean_val = simulated.mean()
        vol_val = simulated.std()

        return jsonify({
            "mean_return": float(mean_val),
            "volatility": float(vol_val),
            "scenario": scenario
        })
    except Exception as e:
        print(f"Error in scenario analysis: {str(e)}")
        return jsonify({'error': f'Scenario analysis failed: {str(e)}'}), 500


@app.route('/api/stress', methods=['POST'])
def stress():
    try:
        ticker = request.json['ticker']

        valid, error = validate_tickers([ticker])
        if not valid:
            return jsonify({'error': error}), 400

        start_5y = (datetime.today() - timedelta(days=5*365)).strftime("%Y-%m-%d")
        end_date = datetime.today().strftime("%Y-%m-%d")
        prices_df = _fetch_prices([ticker], start_5y, end_date)
        if prices_df.empty:
            return jsonify({'error': 'No price data available'}), 400
        prices = prices_df.iloc[:, 0]

        if len(prices) < 30:
            return jsonify({'error': 'Insufficient data for stress testing'}), 400

        returns = prices.pct_change().dropna()

        worst_day = returns.min()
        percentile_1 = returns.quantile(0.01)
        percentile_5 = returns.quantile(0.05)

        return jsonify({
            "worst_day": float(worst_day),
            "percentile_1": float(percentile_1),
            "percentile_5": float(percentile_5)
        })
    except Exception as e:
        print(f"Error in stress test: {str(e)}")
        return jsonify({'error': f'Stress test failed: {str(e)}'}), 500

@app.route('/api/prices', methods=['POST'])
def prices():
    try:
        data = request.json
        tickers = data.get('tickers', [])

        if not tickers:
            return jsonify({'error': 'No tickers provided'}), 400

        valid, error = validate_tickers(tickers)
        if not valid:
            return jsonify({'error': error}), 400

        prices_data = {}

        # Try Tiingo IEX first for all tickers
        iex_data = {}
        if TIINGO_API_KEY:
            iex_results = _tiingo_iex(tickers)
            for item in iex_results:
                t = item.get('ticker', '').upper()
                if t:
                    iex_data[t] = item

        for ticker in tickers:
            try:
                if ticker in iex_data:
                    item = iex_data[ticker]
                    current_price = item.get('last', item.get('tngoLast', 0))
                    previous_close = item.get('prevClose', 0)
                    if current_price and previous_close:
                        daily_change = current_price - previous_close
                        daily_change_pct = (daily_change / previous_close) * 100 if previous_close != 0 else 0
                        prices_data[ticker] = {
                            'current_price': float(current_price),
                            'previous_close': float(previous_close),
                            'daily_change': float(daily_change),
                            'daily_change_pct': float(daily_change_pct),
                            'currency': 'USD'
                        }
                        continue

                # Fallback to yfinance
                stock = yf.Ticker(ticker)
                info = stock.info

                current_price = info.get('currentPrice') or info.get('regularMarketPrice')
                previous_close = info.get('previousClose') or info.get('regularMarketPreviousClose')

                if current_price and previous_close:
                    daily_change = current_price - previous_close
                    daily_change_pct = (daily_change / previous_close) * 100 if previous_close != 0 else 0

                    prices_data[ticker] = {
                        'current_price': float(current_price),
                        'previous_close': float(previous_close),
                        'daily_change': float(daily_change),
                        'daily_change_pct': float(daily_change_pct),
                        'currency': info.get('currency', 'USD')
                    }
                else:
                    prices_data[ticker] = {
                        'error': 'Price data not available',
                        'current_price': None
                    }
            except Exception as e:
                prices_data[ticker] = {
                    'error': str(e),
                    'current_price': None
                }

        return jsonify({
            'prices': prices_data,
            'timestamp': datetime.today().strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        print(f"Error fetching prices: {str(e)}")
        return jsonify({'error': f'Failed to fetch prices: {str(e)}'}), 500

@app.route('/api/rebalance', methods=['POST'])
def rebalance():
    try:
        weights = request.json['weights']
        
        if not weights or len(weights) == 0:
            return jsonify({'error': 'No weights provided'}), 400
        
        avg = sum(weights.values()) / len(weights)
        threshold = 0.05  # 5% threshold

        suggestions = {}
        for k, v in weights.items():
            diff = abs(v - avg)
            if diff > threshold:
                suggestions[k] = "Reduce" if v > avg else "Increase"

        return jsonify(suggestions)
    except Exception as e:
        print(f"Error in rebalance: {str(e)}")
        return jsonify({'error': f'Rebalance analysis failed: {str(e)}'}), 500

@app.route('/api/news', methods=['POST'])
def news():
    try:
        data = request.json
        # Accept both single ticker (legacy) and multiple tickers
        ticker = data.get('ticker', '')
        tickers = data.get('tickers', [])
        
        # If single ticker provided, convert to list
        if ticker and not tickers:
            tickers = [ticker]
        
        if not tickers:
            return jsonify({'error': 'No tickers provided', 'articles': []}), 200
        
        valid, error = validate_tickers(tickers)
        if not valid:
            return jsonify({'error': error, 'articles': []}), 400

        all_articles = {}

        for t in tickers:
            try:
                articles = []
                # Try Tiingo news first
                if TIINGO_API_KEY:
                    tiingo_items = _tiingo_news([t], limit=5)
                    if tiingo_items:
                        for item in tiingo_items:
                            article = {
                                'title': item.get('title', ''),
                                'description': item.get('description', ''),
                                'url': item.get('url', '#'),
                                'publishedAt': item.get('publishedDate', ''),
                                'source': {'name': item.get('source', 'Unknown')}
                            }
                            articles.append(article)
                        all_articles[t] = articles
                        continue

                # Fallback to yfinance
                stock = yf.Ticker(t)
                news_items = stock.news

                for item in (news_items or [])[:5]:
                    content = item.get('content', item)  # nested under 'content' in newer yfinance
                    provider = content.get('provider', {})
                    canonical = content.get('canonicalUrl', {})
                    article = {
                        'title': content.get('title', ''),
                        'description': content.get('summary', content.get('description', '')),
                        'url': canonical.get('url', content.get('link', '#')),
                        'publishedAt': content.get('pubDate', content.get('providerPublishTime', '')),
                        'source': {'name': provider.get('displayName', provider.get('name', 'Unknown'))}
                    }
                    articles.append(article)

                all_articles[t] = articles
            except Exception as e:
                print(f"Error fetching news for {t}: {str(e)}")
                all_articles[t] = []

        # If single ticker, return articles directly for backward compatibility
        if len(tickers) == 1:
            return jsonify({'articles': all_articles[tickers[0]]})
        
        # For multiple tickers, return per-ticker articles
        return jsonify({'articles_by_ticker': all_articles})

    except Exception as e:
        print(f"Error fetching news: {str(e)}")
        return jsonify({'error': str(e), 'articles': []}), 200


@app.route('/api/sectors', methods=['POST'])
def sectors():
    try:
        tickers = request.json['tickers']
        
        valid, error = validate_tickers(tickers)
        if not valid:
            return jsonify({'error': error}), 400
        
        sectors = {}

        for t in tickers:
            try:
                info = yf.Ticker(t).info
                sector = info.get('sector', 'Unknown')
                sectors[sector] = sectors.get(sector, 0) + 1
            except Exception as e:
                print(f"Error fetching sector for {t}: {str(e)}")
                sectors['Unknown'] = sectors.get('Unknown', 0) + 1

        return jsonify(sectors)
    except Exception as e:
        print(f"Error in sectors: {str(e)}")
        return jsonify({'error': f'Sector analysis failed: {str(e)}'}), 500


@app.route('/api/dividends', methods=['POST'])
def dividends():
    try:
        ticker = request.json['ticker']
        
        valid, error = validate_tickers([ticker])
        if not valid:
            return jsonify({'error': error}), 400
        
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get dividend yield
        div_yield = info.get('dividendYield')
        
        # Get dividend history and convert to serializable format
        dividends = stock.dividends.tail(10)
        div_history = {date.strftime('%Y-%m-%d'): float(value) for date, value in dividends.items()}

        return jsonify({
            "yield": float(div_yield) if div_yield else None,
            "history": div_history
        })
    except Exception as e:
        print(f"Error fetching dividends: {str(e)}")
        return jsonify({'error': f'Dividend analysis failed: {str(e)}'}), 500

@app.route('/api/sentiment', methods=['POST'])
def sentiment():
    try:
        data = request.json
        # Accept both single ticker (legacy) and multiple tickers
        ticker = data.get('ticker', '')
        tickers = data.get('tickers', [])
        texts = data.get('texts', [])
        
        # If single ticker provided, convert to list
        if ticker and not tickers:
            tickers = [ticker]
        
        if not tickers and not texts:
            return jsonify({'error': 'No tickers or texts provided', 'sentiment': 0, 'breakdown': {}}), 200
        
        # Validate tickers if provided
        if tickers:
            valid, error = validate_tickers(tickers)
            if not valid:
                return jsonify({'error': error, 'sentiment': 0, 'breakdown': {}}), 400

        # If tickers provided but no texts, fetch news headlines for sentiment
        all_sentiments = {}
        
        if tickers:
            for t in tickers:
                try:
                    if not texts:
                        ticker_texts = []
                        # Try Tiingo news first
                        if TIINGO_API_KEY:
                            tiingo_items = _tiingo_news([t], limit=10)
                            if tiingo_items:
                                ticker_texts = [item.get('title', '') for item in tiingo_items if item.get('title')]
                        # Fallback to yfinance
                        if not ticker_texts:
                            stock = yf.Ticker(t)
                            news_items = stock.news or []
                            ticker_texts = [item.get('content', item).get('title', '') for item in news_items[:10]
                                           if item.get('content', item).get('title')]
                    else:
                        ticker_texts = texts
                    
                    if not ticker_texts:
                        all_sentiments[t] = {
                            'sentiment': 0.0,
                            'breakdown': {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0},
                            'headlines': []
                        }
                        continue
                    
                    # Calculate sentiment
                    try:
                        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                        analyzer = SentimentIntensityAnalyzer()
                        scores = [analyzer.polarity_scores(text) for text in ticker_texts]
                        compounds = [s['compound'] for s in scores]
                        avg_sentiment = float(np.mean(compounds))
                        positive = len([c for c in compounds if c > 0.05])
                        negative = len([c for c in compounds if c < -0.05])
                        neutral = len(compounds) - positive - negative
                    except ImportError:
                        # Simple keyword-based fallback
                        positive_words = {'up', 'gain', 'rise', 'bull', 'high', 'growth', 'profit', 'surge', 'rally', 'beat', 'strong', 'buy', 'upgrade'}
                        negative_words = {'down', 'loss', 'fall', 'bear', 'low', 'decline', 'crash', 'drop', 'sell', 'cut', 'weak', 'risk', 'fear'}
                        compounds = []
                        for text in ticker_texts:
                            words = set(text.lower().split())
                            pos = len(words & positive_words)
                            neg = len(words & negative_words)
                            if pos + neg == 0:
                                compounds.append(0.0)
                            else:
                                compounds.append((pos - neg) / (pos + neg))
                        avg_sentiment = float(np.mean(compounds))
                        positive = len([c for c in compounds if c > 0])
                        negative = len([c for c in compounds if c < 0])
                        neutral = len(compounds) - positive - negative
                    
                    # Build headline details
                    headlines = []
                    for i, text in enumerate(ticker_texts[:10]):
                        score = compounds[i] if i < len(compounds) else 0
                        label = 'Positive' if score > 0.05 else 'Negative' if score < -0.05 else 'Neutral'
                        headlines.append({'text': text, 'score': round(score, 3), 'label': label})
                    
                    all_sentiments[t] = {
                        'sentiment': round(avg_sentiment, 3),
                        'breakdown': {
                            'positive': positive,
                            'negative': negative,
                            'neutral': neutral,
                            'total': len(ticker_texts)
                        },
                        'headlines': headlines
                    }
                except Exception as e:
                    print(f"Error analyzing sentiment for {t}: {str(e)}")
                    all_sentiments[t] = {
                        'sentiment': 0.0,
                        'breakdown': {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0},
                        'headlines': []
                    }
            
            # If single ticker, return sentiment directly for backward compatibility
            if len(tickers) == 1:
                result = all_sentiments[tickers[0]]
                return jsonify(result)
            
            # For multiple tickers, return per-ticker sentiments
            return jsonify({'sentiment_by_ticker': all_sentiments})
        
        else:
            # If no tickers but texts provided
            if not texts:
                return jsonify({'error': 'No texts available for analysis', 'sentiment': 0, 'breakdown': {}}), 200
            
            # Calculate sentiment from texts
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                analyzer = SentimentIntensityAnalyzer()
                scores = [analyzer.polarity_scores(t) for t in texts]
                compounds = [s['compound'] for s in scores]
                avg_sentiment = float(np.mean(compounds))
                positive = len([c for c in compounds if c > 0.05])
                negative = len([c for c in compounds if c < -0.05])
                neutral = len(compounds) - positive - negative
            except ImportError:
                # Simple keyword-based fallback
                positive_words = {'up', 'gain', 'rise', 'bull', 'high', 'growth', 'profit', 'surge', 'rally', 'beat', 'strong', 'buy', 'upgrade'}
                negative_words = {'down', 'loss', 'fall', 'bear', 'low', 'decline', 'crash', 'drop', 'sell', 'cut', 'weak', 'risk', 'fear'}
                compounds = []
                for text in texts:
                    words = set(text.lower().split())
                    pos = len(words & positive_words)
                    neg = len(words & negative_words)
                    if pos + neg == 0:
                        compounds.append(0.0)
                    else:
                        compounds.append((pos - neg) / (pos + neg))
                avg_sentiment = float(np.mean(compounds))
                positive = len([c for c in compounds if c > 0])
                negative = len([c for c in compounds if c < 0])
                neutral = len(compounds) - positive - negative
            
            # Build headline details
            headlines = []
            for i, text in enumerate(texts[:10]):
                score = compounds[i] if i < len(compounds) else 0
                label = 'Positive' if score > 0.05 else 'Negative' if score < -0.05 else 'Neutral'
                headlines.append({'text': text, 'score': round(score, 3), 'label': label})
            
            return jsonify({
                "sentiment": round(avg_sentiment, 3),
                "breakdown": {
                    "positive": positive,
                    "negative": negative,
                    "neutral": neutral,
                    "total": len(texts)
                },
                "headlines": headlines
            })
    except Exception as e:
        print(f"Error in sentiment analysis: {str(e)}")
        return jsonify({'error': f'Sentiment analysis failed: {str(e)}'}), 500


## ── Retirement Planner ──────────────────────────────────────

PORTFOLIO_PARAMS = {
    "conservative": {"return": 0.05, "volatility": 0.08, "stocks": 30},
    "moderate":     {"return": 0.07, "volatility": 0.12, "stocks": 60},
    "aggressive":   {"return": 0.09, "volatility": 0.18, "stocks": 85},
}


def _run_retirement_mc(current_savings, monthly_contribution, years,
                       expected_return, volatility, num_sims=5000, method='standard'):
    """Run Monte Carlo accumulation simulation with optional variance reduction."""
    months = years * 12
    monthly_ret = expected_return / 12
    monthly_vol = volatility / np.sqrt(12)

    if method == 'antithetic':
        half = num_sims // 2
        rand_base = np.random.normal(monthly_ret, monthly_vol, (half, months))
        # Mirror: reflect around the mean
        rand_mirror = 2 * monthly_ret - rand_base
        rand_returns = np.vstack([rand_base, rand_mirror])
    elif method in ('sobol', 'full'):
        # Sobol uniform samples transformed to normal via inverse CDF
        m_pow2 = int(2 ** np.ceil(np.log2(max(num_sims, 2))))
        sampler = Sobol(d=months, scramble=True, seed=42)
        uniform = sampler.random(m_pow2)[:num_sims]
        uniform = np.clip(uniform, 1e-6, 1 - 1e-6)
        rand_returns = norm.ppf(uniform, loc=monthly_ret, scale=monthly_vol)
    else:
        rand_returns = np.random.normal(monthly_ret, monthly_vol, (num_sims, months))

    values = np.empty((num_sims, months + 1))
    values[:, 0] = current_savings

    for m in range(months):
        values[:, m + 1] = values[:, m] * (1 + rand_returns[:, m]) + monthly_contribution

    return values  # (num_sims, months+1)


def _sustainability_test(portfolio_value, annual_spending, inflation_rate,
                         years_in_retirement, expected_return, volatility,
                         num_sims=5000):
    """Return fraction of simulations where money lasts."""
    successes = 0
    for _ in range(num_sims):
        remaining = portfolio_value
        for yr in range(years_in_retirement):
            withdrawal = annual_spending * (1 + inflation_rate) ** yr
            remaining -= withdrawal
            if remaining <= 0:
                break
            annual_ret = np.random.normal(expected_return, volatility)
            remaining *= (1 + annual_ret)
        if remaining > 0:
            successes += 1
    return successes / num_sims


def _safe_withdrawal_rate(portfolio_value, years_in_retirement,
                          inflation_rate, expected_return, volatility):
    """Find highest withdrawal rate with >= 95 % success (quick search)."""
    for rate in np.linspace(0.02, 0.08, 60):
        annual_w = portfolio_value * rate
        ok = 0
        for _ in range(800):
            rem = portfolio_value
            for yr in range(years_in_retirement):
                rem -= annual_w * (1 + inflation_rate) ** yr
                if rem <= 0:
                    break
                rem *= (1 + np.random.normal(expected_return, volatility))
            if rem > 0:
                ok += 1
        if ok / 800 < 0.95:
            return max(0.02, rate - 0.001)
    return rate


@app.route('/api/retirement/calculate', methods=['POST'])
def retirement_calculate():
    try:
        d = request.json
        current_savings      = float(d.get('current_savings', 0))
        monthly_contribution = float(d.get('monthly_contribution', 0))
        years_to_retirement  = int(d.get('years_to_retirement', 25))
        years_in_retirement  = int(d.get('years_in_retirement', 30))
        annual_spending      = float(d.get('annual_spending', 60000))
        target_amount        = float(d.get('target_amount', 0))
        inflation_rate       = float(d.get('inflation_rate', 0.03))
        risk_tolerance       = d.get('risk_tolerance', 'moderate')

        if risk_tolerance not in PORTFOLIO_PARAMS:
            return jsonify({'error': 'risk_tolerance must be conservative, moderate, or aggressive'}), 400

        params = PORTFOLIO_PARAMS[risk_tolerance]
        exp_ret = params['return']
        vol     = params['volatility']

        # If no explicit target, estimate from spending
        if target_amount <= 0:
            target_amount = annual_spending * 25  # 4 % rule heuristic

        np.random.seed(None)
        values = _run_retirement_mc(current_savings, monthly_contribution,
                                    years_to_retirement, exp_ret, vol, num_sims=5000,
                                    method=d.get('method', 'standard'))
        final_values = values[:, -1]

        success_rate = float(np.mean(final_values >= target_amount))
        median       = float(np.median(final_values))
        p10          = float(np.percentile(final_values, 10))
        p25          = float(np.percentile(final_values, 25))
        p75          = float(np.percentile(final_values, 75))
        p90          = float(np.percentile(final_values, 90))

        # Sustainability test using the median outcome
        sustainability = _sustainability_test(
            median, annual_spending, inflation_rate,
            years_in_retirement, exp_ret, vol, num_sims=3000)

        swr = _safe_withdrawal_rate(
            median, years_in_retirement, inflation_rate, exp_ret, vol)

        # Build a probability cone (yearly medians + bands)
        yearly_indices = list(range(0, values.shape[1], 12))
        if yearly_indices[-1] != values.shape[1] - 1:
            yearly_indices.append(values.shape[1] - 1)
        cone = []
        for idx in yearly_indices:
            col = values[:, idx]
            cone.append({
                'year': idx // 12,
                'p10':  round(float(np.percentile(col, 10)), 0),
                'p25':  round(float(np.percentile(col, 25)), 0),
                'p50':  round(float(np.median(col)), 0),
                'p75':  round(float(np.percentile(col, 75)), 0),
                'p90':  round(float(np.percentile(col, 90)), 0),
            })

        # Recommendations
        recommendations = []
        if success_rate < 0.75:
            recommendations.append(
                f"Your success probability ({success_rate:.0%}) is below 75 %. "
                "Consider increasing contributions, extending your timeline, or adjusting your target."
            )
            if risk_tolerance != 'aggressive':
                recommendations.append(
                    "Switching to a more aggressive allocation could raise expected returns, "
                    "but also increases volatility."
                )
        else:
            recommendations.append(
                f"Good news — your plan has a {success_rate:.0%} chance of meeting the target."
            )

        if swr < 0.035:
            recommendations.append(
                f"Your safe withdrawal rate is {swr:.1%}, which is below the 4 % guideline. "
                "Building a larger nest egg would help."
            )
        else:
            recommendations.append(
                f"Estimated safe withdrawal rate: {swr:.1%} — "
                f"that implies ~${median * swr:,.0f}/year of sustainable income."
            )

        return jsonify({
            'target_amount': target_amount,
            'success_rate': round(success_rate, 3),
            'median': round(median, 0),
            'percentile_10': round(p10, 0),
            'percentile_25': round(p25, 0),
            'percentile_75': round(p75, 0),
            'percentile_90': round(p90, 0),
            'sustainability_rate': round(sustainability, 3),
            'safe_withdrawal_rate': round(swr, 4),
            'safe_annual_income': round(median * swr, 0),
            'cone': cone,
            'portfolio': {
                'stocks': params['stocks'],
                'bonds': max(0, 100 - params['stocks'] - 5),
                'cash': 5,
                'expected_return': exp_ret,
                'volatility': vol,
            },
            'recommendations': recommendations,
        })

    except Exception as e:
        print(f"Error in retirement calculation: {str(e)}")
        import traceback; traceback.print_exc()
        return jsonify({'error': f'Retirement calculation failed: {str(e)}'}), 500


##Interactive Charts API Endpoints 

@app.route('/api/charts/efficient-frontier-3d', methods=['POST'])
def efficient_frontier_3d():
    """3D Efficient Frontier surface with risk, return, and Sharpe ratio."""
    try:
        data = request.json
        tickers = data.get('tickers', [])
        start_date = data.get('start_date', '2018-01-01')
        num_portfolios = data.get('num_portfolios', 5000)
        risk_free_rate = data.get('risk_free_rate', 0.0)
        end_date = datetime.today().strftime("%Y-%m-%d")

        if not tickers or len(tickers) < 2:
            return jsonify({'error': 'Need at least 2 tickers'}), 400

        valid, error = validate_tickers(tickers)
        if not valid:
            return jsonify({'error': error}), 400

        prices = _fetch_prices(tickers, start_date, end_date)

        if len(prices) < 30:
            return jsonify({'error': 'Insufficient data'}), 400

        returns = prices.pct_change().dropna()
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        points = []
        np.random.seed(42)
        for i in range(num_portfolios):
            w = np.random.random(len(tickers))
            w /= w.sum()
            ret = float(np.dot(w, mean_returns))
            vol = float(np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))))
            sr = float((ret - risk_free_rate) / vol) if vol > 0 else 0.0
            weights_dict = {t: round(float(wt), 4) for t, wt in zip(tickers, w)}
            points.append({'volatility': round(vol, 6), 'return': round(ret, 6),
                           'sharpe': round(sr, 4), 'weights': weights_dict})

        return jsonify({'points': points, 'tickers': tickers})
    except Exception as e:
        print(f"Error in 3D frontier: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/charts/time-series-decomposition', methods=['POST'])
def time_series_decomposition():
    """Decompose a stock's price into trend, seasonality, and residual."""
    try:
        ticker = request.json['ticker']
        period = request.json.get('period', '2y')

        valid, error = validate_tickers([ticker])
        if not valid:
            return jsonify({'error': error}), 400

        period_map = {'6mo': 180, '1y': 365, '2y': 730, '5y': 1825}
        days = period_map.get(period, 730)
        start_ts = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
        end_ts = datetime.today().strftime("%Y-%m-%d")
        prices_df = _fetch_prices([ticker], start_ts, end_ts)
        if prices_df.empty:
            return jsonify({'error': 'No price data available'}), 400
        prices = prices_df.iloc[:, 0].dropna()

        if len(prices) < 60:
            return jsonify({'error': 'Insufficient data for decomposition'}), 400

        # Simple moving average as trend
        window = min(63, len(prices) // 4)
        trend = prices.rolling(window=window, center=True).mean()

        # Detrended = original - trend
        detrended = prices - trend

        # Seasonality: average detrended values by day-of-week (simple approach)
        detrended_clean = detrended.dropna()
        seasonal_pattern = detrended_clean.groupby(detrended_clean.index.dayofweek).mean()
        seasonality = detrended.copy()
        for idx in seasonality.index:
            dow = idx.dayofweek
            if dow in seasonal_pattern.index:
                seasonality.loc[idx] = seasonal_pattern[dow]
            else:
                seasonality.loc[idx] = 0

        # Residual = original - trend - seasonality
        residual = prices - trend - seasonality

        dates = prices.index.strftime('%Y-%m-%d').tolist()

        return jsonify({
            'dates': dates,
            'original': [round(float(v), 2) if not pd.isna(v) else None for v in prices.values],
            'trend': [round(float(v), 2) if not pd.isna(v) else None for v in trend.values],
            'seasonality': [round(float(v), 4) if not pd.isna(v) else None for v in seasonality.values],
            'residual': [round(float(v), 4) if not pd.isna(v) else None for v in residual.values],
            'ticker': ticker
        })
    except Exception as e:
        print(f"Error in decomposition: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/charts/candlestick', methods=['POST'])
def candlestick_data():
    """Get OHLCV candlestick data for a stock."""
    try:
        ticker = request.json['ticker']
        period = request.json.get('period', '6mo')
        interval = request.json.get('interval', '1d')

        valid, error = validate_tickers([ticker])
        if not valid:
            return jsonify({'error': error}), 400

        # For daily interval, use _fetch_ohlcv; for intraday, keep yfinance
        if interval == '1d':
            period_map = {'1mo': 30, '3mo': 90, '6mo': 180, '1y': 365, '2y': 730, '5y': 1825}
            days = period_map.get(period, 180)
            start_ts = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
            end_ts = datetime.today().strftime("%Y-%m-%d")
            stock_data = _fetch_ohlcv(ticker, start_ts, end_ts)
        else:
            stock_data = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
            stock_data = normalize_columns(stock_data)

        if len(stock_data) < 5:
            return jsonify({'error': 'Insufficient data'}), 400

        candles = []
        for idx, row in stock_data.iterrows():
            ts = idx.strftime('%Y-%m-%d') if interval == '1d' else idx.strftime('%Y-%m-%d %H:%M')
            candles.append({
                'date': ts,
                'open': round(float(row['Open']), 2),
                'high': round(float(row['High']), 2),
                'low': round(float(row['Low']), 2),
                'close': round(float(row['Close']), 2),
                'volume': int(row['Volume']) if not pd.isna(row['Volume']) else 0
            })

        return jsonify({'candles': candles, 'ticker': ticker})
    except Exception as e:
        print(f"Error in candlestick: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/charts/technical-indicators', methods=['POST'])
def technical_indicators():
    """Calculate RSI, MACD, and Bollinger Bands for a stock."""
    try:
        ticker = request.json['ticker']
        period = request.json.get('period', '1y')

        valid, error = validate_tickers([ticker])
        if not valid:
            return jsonify({'error': error}), 400

        period_map = {'6mo': 180, '1y': 365, '2y': 730, '5y': 1825}
        days = period_map.get(period, 365)
        start_ts = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
        end_ts = datetime.today().strftime("%Y-%m-%d")
        prices_df = _fetch_prices([ticker], start_ts, end_ts)
        if prices_df.empty:
            return jsonify({'error': 'No price data available'}), 400
        prices = prices_df.iloc[:, 0].dropna()

        if len(prices) < 30:
            return jsonify({'error': 'Insufficient data'}), 400

        dates = prices.index.strftime('%Y-%m-%d').tolist()
        close = prices.values

        # RSI (14-day)
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        # MACD
        ema12 = pd.Series(close).ewm(span=12).mean()
        ema26 = pd.Series(close).ewm(span=26).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9).mean()
        histogram = macd_line - signal_line

        # Bollinger Bands (20-day, 2 std)
        sma20 = pd.Series(close).rolling(20).mean()
        std20 = pd.Series(close).rolling(20).std()
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20

        return jsonify({
            'dates': dates,
            'close': [round(float(v), 2) for v in close],
            'rsi': [round(float(v), 2) if not pd.isna(v) else None for v in rsi.values],
            'macd': {
                'macd_line': [round(float(v), 4) if not pd.isna(v) else None for v in macd_line.values],
                'signal_line': [round(float(v), 4) if not pd.isna(v) else None for v in signal_line.values],
                'histogram': [round(float(v), 4) if not pd.isna(v) else None for v in histogram.values]
            },
            'bollinger': {
                'upper': [round(float(v), 2) if not pd.isna(v) else None for v in bb_upper.values],
                'middle': [round(float(v), 2) if not pd.isna(v) else None for v in sma20.values],
                'lower': [round(float(v), 2) if not pd.isna(v) else None for v in bb_lower.values]
            },
            'ticker': ticker
        })
    except Exception as e:
        print(f"Error in technical indicators: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/charts/drawdown', methods=['POST'])
def drawdown_chart():
    """Calculate drawdown (underwater) chart for portfolio or single stock."""
    try:
        data = request.json
        tickers = data.get('tickers', [])
        start_date = data.get('start_date', '2018-01-01')
        benchmark = data.get('benchmark', '^GSPC')

        if not tickers:
            return jsonify({'error': 'No tickers provided'}), 400

        valid, error = validate_tickers(tickers + [benchmark])
        if not valid:
            return jsonify({'error': error}), 400

        end_date = datetime.today().strftime("%Y-%m-%d")
        all_tickers = tickers + [benchmark]
        prices = _fetch_prices(all_tickers, start_date, end_date)

        if len(prices) < 30:
            return jsonify({'error': 'Insufficient data'}), 400

        # Equal-weight portfolio
        if len(tickers) > 1:
            port_prices = prices[tickers].mean(axis=1)
        else:
            port_prices = prices[tickers[0]]

        bench_prices = prices[benchmark]

        # Calculate drawdowns
        def calc_drawdown(series):
            cummax = series.cummax()
            dd = (series / cummax) - 1
            return dd

        port_dd = calc_drawdown(port_prices)
        bench_dd = calc_drawdown(bench_prices)

        dates = port_dd.index.strftime('%Y-%m-%d').tolist()

        # Max drawdown periods
        port_max_dd = float(port_dd.min())
        port_max_dd_date = port_dd.idxmin().strftime('%Y-%m-%d')

        return jsonify({
            'dates': dates,
            'portfolio_drawdown': [round(float(v) * 100, 2) for v in port_dd.values],
            'benchmark_drawdown': [round(float(v) * 100, 2) for v in bench_dd.values],
            'max_drawdown': round(port_max_dd * 100, 2),
            'max_drawdown_date': port_max_dd_date
        })
    except Exception as e:
        print(f"Error in drawdown: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/charts/risk-contribution', methods=['POST'])
def risk_contribution():
    """Calculate each asset's contribution to total portfolio risk."""
    try:
        data = request.json
        tickers = data.get('tickers', [])
        start_date = data.get('start_date', '2018-01-01')

        if not tickers or len(tickers) < 2:
            return jsonify({'error': 'Need at least 2 tickers'}), 400

        valid, error = validate_tickers(tickers)
        if not valid:
            return jsonify({'error': error}), 400

        end_date = datetime.today().strftime("%Y-%m-%d")
        prices = _fetch_prices(tickers, start_date, end_date)

        if len(prices) < 30:
            return jsonify({'error': 'Insufficient data'}), 400

        returns = prices.pct_change().dropna()
        cov_matrix = returns.cov() * 252

        # Equal weights
        n = len(tickers)
        weights = np.array([1.0 / n] * n)

        # Portfolio volatility
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        # Marginal contribution to risk
        mcr = np.dot(cov_matrix, weights) / port_vol

        # Component risk contribution
        crc = weights * mcr

        # Percentage contribution
        pct_contribution = (crc / port_vol) * 100

        contributions = []
        for i, ticker in enumerate(tickers):
            contributions.append({
                'ticker': ticker,
                'weight': round(float(weights[i]) * 100, 2),
                'risk_contribution': round(float(pct_contribution[i]), 2),
                'marginal_risk': round(float(mcr[i]) * 100, 4),
                'annual_vol': round(float(np.sqrt(cov_matrix.iloc[i, i])) * 100, 2)
            })

        return jsonify({
            'contributions': contributions,
            'portfolio_volatility': round(float(port_vol) * 100, 2),
            'tickers': tickers
        })
    except Exception as e:
        print(f"Error in risk contribution: {e}")
        return jsonify({'error': str(e)}), 500


# ── Live Analysis Endpoint ──────────────────────────────────────────
def _fetch_ticker_live_data(t):
    """Fetch enriched live data for a single ticker."""
    try:
        # Try Tiingo IEX for price data first
        price = 0
        prev_close = 0
        change = 0
        currency = 'INR'

        if TIINGO_API_KEY:
            iex_results = _tiingo_iex([t])
            if iex_results:
                item = iex_results[0]
                price = item.get('last', item.get('tngoLast', 0)) or 0
                prev_close = item.get('prevClose', 0) or 0
                change = ((price - prev_close) / prev_close * 100) if prev_close else 0
                currency = 'USD'

        # Fallback to yfinance for price
        if not price:
            stock = yf.Ticker(t)
            info = stock.info or {}
            price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose', 0)
            prev_close = info.get('previousClose') or info.get('regularMarketPreviousClose', 0)
            change = ((price - prev_close) / prev_close * 100) if prev_close else 0
            currency = info.get('currency', 'INR')
        else:
            stock = yf.Ticker(t)

        # Mini chart data (5 day hourly) — keep yfinance for intraday
        hist = stock.history(period='5d', interval='1h')
        mini_chart = []
        if not hist.empty:
            for idx, row in hist.iterrows():
                mini_chart.append({
                    'time': idx.strftime('%Y-%m-%d %H:%M'),
                    'price': round(float(row['Close']), 2)
                })

        # News headlines — try Tiingo first
        headlines = []
        if TIINGO_API_KEY:
            tiingo_items = _tiingo_news([t], limit=3)
            if tiingo_items:
                for item in tiingo_items:
                    headlines.append({
                        'title': item.get('title', 'No title'),
                        'url': item.get('url', ''),
                        'source': item.get('source', ''),
                    })

        # Fallback to yfinance for news
        if not headlines:
            news_items = stock.news or []
            for item in news_items[:3]:
                content = item.get('content', item)
                headlines.append({
                    'title': content.get('title', 'No title'),
                    'url': content.get('canonicalUrl', {}).get('url', '') if isinstance(content.get('canonicalUrl'), dict) else content.get('link', ''),
                    'source': content.get('provider', {}).get('displayName', '') if isinstance(content.get('provider'), dict) else content.get('source', ''),
                })

        # Simple sentiment from headlines
        sentiment_score = 0.0
        if headlines:
            positive_words = {'up', 'gain', 'rise', 'bull', 'high', 'growth', 'profit', 'surge', 'rally', 'beat', 'strong', 'buy', 'upgrade', 'record', 'soar'}
            negative_words = {'down', 'loss', 'fall', 'bear', 'low', 'decline', 'crash', 'drop', 'sell', 'cut', 'weak', 'risk', 'fear', 'miss', 'slump'}
            scores = []
            for h in headlines:
                words = set(h['title'].lower().split())
                pos = len(words & positive_words)
                neg = len(words & negative_words)
                if pos + neg > 0:
                    scores.append((pos - neg) / (pos + neg))
                else:
                    scores.append(0.0)
            sentiment_score = float(np.mean(scores)) if scores else 0.0

        return t, {
            'price': round(float(price), 2),
            'previousClose': round(float(prev_close), 2),
            'change': round(float(change), 2),
            'currency': currency,
            'miniChart': mini_chart,
            'sentiment': round(sentiment_score, 3),
            'headlines': headlines,
        }
    except Exception as e:
        return t, {
            'price': 0, 'previousClose': 0, 'change': 0,
            'currency': 'INR', 'miniChart': [], 'sentiment': 0, 'headlines': [],
            'error': str(e)
        }


@app.route('/api/live-analysis', methods=['POST'])
def live_analysis():
    try:
        data = request.json
        tickers = data.get('tickers', [])
        if not tickers:
            return jsonify({'error': 'No tickers provided'}), 400

        valid, error = validate_tickers(tickers)
        if not valid:
            return jsonify({'error': error}), 400

        result = {}
        with ThreadPoolExecutor(max_workers=min(len(tickers), 8)) as executor:
            futures = {executor.submit(_fetch_ticker_live_data, t): t for t in tickers}
            for future in as_completed(futures):
                ticker, ticker_data = future.result()
                result[ticker] = ticker_data

        return jsonify({'data': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── AI Agent Endpoint ───────────────────────────────────────────────
def _gather_stock_context(ticker):
    """Gather comprehensive stock data for RAG context."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        # Supplement with Tiingo metadata
        tiingo_meta = {}
        if TIINGO_API_KEY:
            tiingo_meta = _tiingo_meta(ticker)

        long_name = info.get('longName', tiingo_meta.get('name', ticker))
        context_parts = [f"\n=== {ticker} ({long_name}) ==="]

        # Tiingo description if available
        if tiingo_meta.get('description'):
            context_parts.append(f"Description: {tiingo_meta['description'][:300]}")

        # Basic info
        context_parts.append(f"Sector: {info.get('sector', 'N/A')}, Industry: {info.get('industry', 'N/A')}")
        context_parts.append(f"Market Cap: {info.get('marketCap', 'N/A')}")
        context_parts.append(f"Current Price: {info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))}")
        context_parts.append(f"52-Week High: {info.get('fiftyTwoWeekHigh', 'N/A')}, Low: {info.get('fiftyTwoWeekLow', 'N/A')}")
        context_parts.append(f"PE Ratio (Trailing): {info.get('trailingPE', 'N/A')}, Forward PE: {info.get('forwardPE', 'N/A')}")
        context_parts.append(f"Dividend Yield: {info.get('dividendYield', 'N/A')}")
        context_parts.append(f"Beta: {info.get('beta', 'N/A')}")
        context_parts.append(f"Analyst Target Price: {info.get('targetMeanPrice', 'N/A')}")
        context_parts.append(f"Recommendation: {info.get('recommendationKey', 'N/A')}")
        context_parts.append(f"Short Ratio: {info.get('shortRatio', 'N/A')}")
        context_parts.append(f"Profit Margins: {info.get('profitMargins', 'N/A')}")
        context_parts.append(f"Revenue Growth: {info.get('revenueGrowth', 'N/A')}")
        context_parts.append(f"Earnings Growth: {info.get('earningsGrowth', 'N/A')}")

        # Financials summary
        try:
            financials = stock.financials
            if financials is not None and not financials.empty:
                latest = financials.iloc[:, 0]
                revenue = latest.get('Total Revenue', 'N/A')
                net_income = latest.get('Net Income', 'N/A')
                context_parts.append(f"Latest Revenue: {revenue}")
                context_parts.append(f"Latest Net Income: {net_income}")
        except Exception:
            pass

        # Analyst recommendations
        try:
            recs = stock.recommendations
            if recs is not None and not recs.empty:
                recent = recs.tail(5).to_string()
                context_parts.append(f"Recent Analyst Recommendations:\n{recent}")
        except Exception:
            pass

        # Recent news — try Tiingo first
        try:
            news_added = False
            if TIINGO_API_KEY:
                tiingo_items = _tiingo_news([ticker], limit=5)
                if tiingo_items:
                    context_parts.append("Recent News:")
                    for item in tiingo_items:
                        title = item.get('title', '')
                        if title:
                            context_parts.append(f"  - {title}")
                    news_added = True
            if not news_added:
                news_items = stock.news or []
                if news_items:
                    context_parts.append("Recent News:")
                    for item in news_items[:5]:
                        content = item.get('content', item)
                        title = content.get('title', '')
                        if title:
                            context_parts.append(f"  - {title}")
        except Exception:
            pass

        # 1Y performance
        try:
            hist = stock.history(period='1y')
            if not hist.empty:
                start_price = hist['Close'].iloc[0]
                end_price = hist['Close'].iloc[-1]
                one_year_return = ((end_price - start_price) / start_price) * 100
                context_parts.append(f"1-Year Return: {one_year_return:.2f}%")
        except Exception:
            pass

        return '\n'.join(context_parts)
    except Exception as e:
        return f"\n=== {ticker} === Error fetching data: {str(e)}"


@app.route('/api/ai-agent', methods=['POST'])
def ai_agent():
    if not openai_client:
        return jsonify({
            'error': 'AI agent not configured. Install openai and set OPENAI_API_KEY environment variable.'
        }), 503

    try:
        data = request.json
        query = data.get('query', '').strip()
        tickers = data.get('tickers', [])

        if not query:
            return jsonify({'error': 'No query provided'}), 400

        # Gather RAG context from stock data
        context_blocks = []
        analyzed_tickers = []
        if tickers:
            valid_tickers = [t for t in tickers if t and isinstance(t, str)]
            with ThreadPoolExecutor(max_workers=min(len(valid_tickers), 6)) as executor:
                futures = {executor.submit(_gather_stock_context, t): t for t in valid_tickers}
                for future in as_completed(futures):
                    context_blocks.append(future.result())
                    analyzed_tickers.append(futures[future])

        context_str = '\n'.join(context_blocks) if context_blocks else 'No specific stock data requested.'

        # Build the prompt
        system_prompt = """You are Bloom AI, an expert financial research analyst integrated into the Bloom Analytics portfolio platform. You have access to real-time market data and financial information provided below.

Your responsibilities:
1. Provide well-researched, balanced financial analysis
2. Consider both bullish and bearish perspectives
3. Reference specific data points from the provided context
4. Give clear, actionable insights while being transparent about uncertainties
5. Format your response with clear sections using **bold** headers and bullet points

IMPORTANT: Always include a brief disclaimer that your analysis is for informational purposes only and does not constitute financial advice."""

        user_prompt = f"""MARKET DATA CONTEXT:
{context_str}

USER QUESTION: {query}

Please provide a thorough, well-researched response based on the data above."""

        # Call OpenAI
        response = openai_client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            temperature=0.7,
            max_tokens=2048,
        )

        return jsonify({
            'response': response.choices[0].message.content,
            'tickers_analyzed': analyzed_tickers,
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Options Pricing Endpoints ────────────────────────────────────────
@app.route('/api/options/price', methods=['POST'])
def options_price():
    try:
        d = request.json
        S = float(d['spot_price'])
        K = float(d['strike_price'])
        T = float(d['time_to_expiry'])  # in years
        r = float(d.get('risk_free_rate', 0.065))
        sigma = float(d['volatility'])
        option_type = d.get('option_type', 'call')

        price = _black_scholes(S, K, T, r, sigma, option_type)
        greeks = _bs_greeks(S, K, T, r, sigma, option_type)

        return jsonify({
            'price': round(price, 4),
            'greeks': greeks,
            'inputs': {'S': S, 'K': K, 'T': T, 'r': r, 'sigma': sigma, 'type': option_type}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/options/greeks', methods=['POST'])
def options_greeks_curves():
    try:
        d = request.json
        S = float(d['spot_price'])
        K = float(d['strike_price'])
        T = float(d['time_to_expiry'])
        r = float(d.get('risk_free_rate', 0.065))
        sigma = float(d['volatility'])
        option_type = d.get('option_type', 'call')

        # Price & Delta & Gamma vs Spot
        spots = np.linspace(0.5 * K, 1.5 * K, 50)
        price_vs_spot = [round(_black_scholes(s, K, T, r, sigma, option_type), 4) for s in spots]
        delta_vs_spot = [_bs_greeks(s, K, T, r, sigma, option_type)['delta'] for s in spots]
        gamma_vs_spot = [_bs_greeks(s, K, T, r, sigma, option_type)['gamma'] for s in spots]

        # Price & Theta vs Time
        times = np.linspace(max(T, 0.02), 0.01, 30)
        price_vs_time = [round(_black_scholes(S, K, t, r, sigma, option_type), 4) for t in times]
        theta_vs_time = [_bs_greeks(S, K, t, r, sigma, option_type)['theta'] for t in times]

        # Price & Vega vs Volatility
        vols = np.linspace(0.05, 1.0, 30)
        price_vs_vol = [round(_black_scholes(S, K, T, r, v, option_type), 4) for v in vols]
        vega_vs_vol = [_bs_greeks(S, K, T, r, v, option_type)['vega'] for v in vols]

        return jsonify({
            'spot_curve': {
                'spots': [round(float(s), 2) for s in spots],
                'prices': price_vs_spot,
                'deltas': delta_vs_spot,
                'gammas': gamma_vs_spot,
            },
            'time_curve': {
                'times': [round(float(t), 4) for t in times],
                'prices': price_vs_time,
                'thetas': theta_vs_time,
            },
            'vol_curve': {
                'vols': [round(float(v), 4) for v in vols],
                'prices': price_vs_vol,
                'vegas': vega_vs_vol,
            },
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/options/chain', methods=['POST'])
def options_chain():
    try:
        d = request.json
        ticker = d['ticker']
        stock = yf.Ticker(ticker)
        expirations = list(stock.options) if stock.options else []

        if not expirations:
            return jsonify({'error': f'No options data available for {ticker}'}), 404

        expiry = d.get('expiry', expirations[0])

        chain = stock.option_chain(expiry)
        cols = ['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']

        calls_df = chain.calls[cols].fillna(0)
        puts_df = chain.puts[cols].fillna(0)

        calls = calls_df.to_dict('records')
        puts = puts_df.to_dict('records')

        spot = stock.info.get('currentPrice', stock.info.get('regularMarketPrice', 0))

        return jsonify({
            'ticker': ticker,
            'spot_price': float(spot) if spot else 0,
            'expirations': expirations,
            'selected_expiry': expiry,
            'calls': calls,
            'puts': puts,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/options/implied-vol', methods=['POST'])
def implied_vol():
    try:
        d = request.json
        iv = _implied_volatility(
            float(d['market_price']),
            float(d['spot_price']),
            float(d['strike_price']),
            float(d['time_to_expiry']),
            float(d.get('risk_free_rate', 0.065)),
            d.get('option_type', 'call'),
        )
        if iv is None:
            return jsonify({'error': 'IV solver did not converge'}), 400
        return jsonify({'implied_volatility': iv})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Constrained Optimization Endpoint ────────────────────────────────
@app.route('/api/constrained-optimize', methods=['POST'])
def constrained_optimize():
    try:
        data = request.json
        tickers = data.get('tickers', [])
        start_date = data.get('start_date', '2020-01-01')
        risk_free_rate = data.get('risk_free_rate', 0.065)
        constraints = data.get('constraints', {})
        weight_step = data.get('weight_step', 0.05)
        end_date = datetime.today().strftime("%Y-%m-%d")

        if not tickers or len(tickers) < 2:
            return jsonify({'error': 'Need at least 2 tickers'}), 400
        if len(tickers) > 10:
            return jsonify({'error': 'Maximum 10 tickers for constrained optimization'}), 400

        valid, error = validate_tickers(tickers)
        if not valid:
            return jsonify({'error': error}), 400

        # Download prices
        prices = _fetch_prices(tickers, start_date, end_date)
        if len(prices) < 30:
            return jsonify({'error': 'Insufficient data'}), 400

        returns = prices.pct_change().dropna()
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        # Fetch ticker info (sector, dividend yield) in parallel
        def fetch_info(t):
            try:
                info = yf.Ticker(t).info
                return t, {
                    'sector': info.get('sector', 'Unknown'),
                    'dividend_yield': float(info.get('dividendYield', 0) or 0),
                    'recommendation': info.get('recommendationKey', 'N/A'),
                }
            except Exception:
                return t, {'sector': 'Unknown', 'dividend_yield': 0, 'recommendation': 'N/A'}

        ticker_info = {}
        with ThreadPoolExecutor(max_workers=min(len(tickers), 6)) as executor:
            futures = [executor.submit(fetch_info, t) for t in tickers]
            for f in as_completed(futures):
                t, info = f.result()
                ticker_info[t] = info

        portfolios, explored, pruned_count = _backtrack_portfolios(
            tickers, mean_returns, cov_matrix, constraints,
            ticker_info, risk_free_rate, weight_step
        )

        # Add dividend yield and sector info to each portfolio
        for p in portfolios:
            p['dividend_yield'] = round(sum(
                p['weights'][t] * ticker_info.get(t, {}).get('dividend_yield', 0)
                for t in tickers
            ), 4)
            sector_alloc = {}
            for t in tickers:
                sec = ticker_info.get(t, {}).get('sector', 'Unknown')
                sector_alloc[sec] = sector_alloc.get(sec, 0) + p['weights'][t]
            p['sector_allocation'] = {k: round(v, 4) for k, v in sector_alloc.items()}

        return jsonify({
            'valid_portfolios': portfolios,
            'total_explored': explored,
            'total_pruned': pruned_count,
            'total_valid': len(portfolios),
            'ticker_info': ticker_info,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/sensitivities', methods=['POST'])
def sensitivities():
    try:
        data = request.json
        tickers = data.get('tickers', [])
        weights = data.get('weights', {})
        start_date = data.get('start_date', '2020-01-01')
        risk_free_rate = data.get('risk_free_rate', 0.065)
        end_date = datetime.today().strftime("%Y-%m-%d")

        if not tickers or not weights:
            return jsonify({'error': 'tickers and weights required'}), 400

        valid, error = validate_tickers(tickers)
        if not valid:
            return jsonify({'error': error}), 400

        prices = _fetch_prices(tickers, start_date, end_date)

        returns = prices.pct_change().dropna()
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        result = _compute_sensitivities(weights, tickers, mean_returns, cov_matrix, risk_free_rate)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── HRP Endpoint ─────────────────────────────────────────────────────
@app.route('/api/hrp', methods=['POST'])
def hrp_endpoint():
    try:
        data = request.json
        tickers = data.get('tickers', [])
        start_date = data.get('start_date', '2020-01-01')
        end_date = datetime.today().strftime('%Y-%m-%d')

        if not tickers or len(tickers) < 2:
            return jsonify({'error': 'Need at least 2 tickers for HRP'}), 400

        prices = _fetch_prices(tickers, start_date, end_date)
        if prices.empty or len(prices) < 30:
            return jsonify({'error': 'Insufficient price data'}), 400

        # Ensure all tickers are present
        missing = [t for t in tickers if t not in prices.columns]
        if missing:
            return jsonify({'error': f'No data for: {", ".join(missing)}'}), 400

        returns = prices[tickers].pct_change().dropna()
        result = _hrp_optimize(returns)
        return jsonify(result)

    except Exception as e:
        print(f'HRP error: {e}')
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'HRP failed: {str(e)}'}), 500

# ── Black-Litterman Endpoint ─────────────────────────────────────────
@app.route('/api/black-litterman', methods=['POST'])
def black_litterman_endpoint():
    try:
        data = request.json
        tickers = data.get('tickers', [])
        start_date = data.get('start_date', '2020-01-01')
        market_caps = data.get('market_caps', None)
        views = data.get('views', None)
        use_sentiment = data.get('use_sentiment', True)
        end_date = datetime.today().strftime('%Y-%m-%d')

        if not tickers or len(tickers) < 2:
            return jsonify({'error': 'Need at least 2 tickers'}), 400

        prices = _fetch_prices(tickers, start_date, end_date)
        if prices.empty or len(prices) < 30:
            return jsonify({'error': 'Insufficient price data'}), 400

        returns = prices[tickers].pct_change().dropna()
        cov_matrix = (returns.cov() * 252).values

        # Fetch market caps if not provided
        if market_caps is None:
            market_caps = []
            def fetch_mcap(t):
                try:
                    info = yf.Ticker(t).info
                    return info.get('marketCap', 1e9)
                except Exception:
                    return 1e9
            with ThreadPoolExecutor(max_workers=6) as ex:
                futures = {ex.submit(fetch_mcap, t): t for t in tickers}
                mcap_dict = {}
                for f in as_completed(futures):
                    mcap_dict[futures[f]] = f.result()
            market_caps = [mcap_dict[t] for t in tickers]

        # Generate views from sentiment or use provided views
        sentiment_scores = {}
        P, Q = np.array([]).reshape(0, len(tickers)), np.array([])

        if views:
            P = np.array(views.get('P', []))
            Q = np.array(views.get('Q', []))
        elif use_sentiment:
            # Run sentiment analysis on Tiingo news
            for t in tickers:
                try:
                    articles = _tiingo_news([t], limit=10)
                    if articles:
                        texts = ' '.join([a.get('title', '') + ' ' + a.get('description', '') for a in articles])
                        # Simple keyword-based sentiment
                        positive = sum(1 for w in ['upgrade', 'beat', 'strong', 'growth', 'profit', 'record', 'surge',
                                                    'outperform', 'positive', 'bullish', 'buy', 'rally'] if w in texts.lower())
                        negative = sum(1 for w in ['downgrade', 'miss', 'weak', 'decline', 'loss', 'risk', 'sell',
                                                    'bearish', 'warning', 'crash', 'default', 'cut'] if w in texts.lower())
                        total = positive + negative
                        sentiment_scores[t] = (positive - negative) / max(total, 1)
                    else:
                        sentiment_scores[t] = 0
                except Exception:
                    sentiment_scores[t] = 0

            P, Q = _sentiment_to_views(sentiment_scores, tickers)

        # Run Black-Litterman
        bl_result = _black_litterman(market_caps, cov_matrix, P, Q)

        # Map weights back to tickers
        weights = {tickers[i]: float(bl_result['weights'][i]) for i in range(len(tickers))}
        expected_rets = {tickers[i]: float(bl_result['expected_returns'][i]) for i in range(len(tickers))}
        implied_rets = {tickers[i]: float(bl_result['implied_returns'][i]) for i in range(len(tickers))}

        views_applied = []
        if len(P) > 0:
            for k in range(len(Q)):
                asset_idx = np.argmax(P[k])
                views_applied.append({
                    'ticker': tickers[asset_idx],
                    'view_return': float(Q[k]),
                    'sentiment': float(sentiment_scores.get(tickers[asset_idx], 0)),
                })

        return jsonify({
            'weights': weights,
            'expected_returns': expected_rets,
            'implied_returns': implied_rets,
            'views_applied': views_applied,
            'sentiment_scores': sentiment_scores,
        })

    except Exception as e:
        print(f'Black-Litterman error: {e}')
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Black-Litterman failed: {str(e)}'}), 500

# ── Backtesting Endpoint ─────────────────────────────────────────────
@app.route('/api/backtest', methods=['POST'])
def backtest_endpoint():
    try:
        data = request.json
        tickers = data.get('tickers', [])
        weights = data.get('weights', {})
        start_date = data.get('start_date', '2018-01-01')
        end_date = data.get('end_date', datetime.today().strftime('%Y-%m-%d'))
        rebalance_freq = data.get('rebalance_freq', 'monthly')
        fee_pct = float(data.get('fee_pct', 0.001))
        slippage_factor = float(data.get('slippage_factor', 0.05))
        benchmark = data.get('benchmark', 'SPY')

        if not tickers or len(tickers) < 1:
            return jsonify({'error': 'Need at least 1 ticker'}), 400

        if not weights:
            # Equal weights if not provided
            weights = {t: 1.0 / len(tickers) for t in tickers}

        result = _backtest_portfolio(tickers, weights, start_date, end_date,
                                      rebalance_freq, fee_pct, slippage_factor, benchmark)
        return jsonify(result)

    except Exception as e:
        print(f'Backtest error: {e}')
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Backtest failed: {str(e)}'}), 500

# ── Volatility Surface Endpoint ──────────────────────────────────────
@app.route('/api/options/vol-surface', methods=['POST'])
def vol_surface():
    try:
        data = request.json
        ticker = data['ticker']
        stock = yf.Ticker(ticker)
        expirations = stock.options

        if not expirations:
            return jsonify({'error': f'No options data for {ticker}'}), 404

        spot = stock.info.get('currentPrice', stock.info.get('regularMarketPrice', 100))
        today = datetime.today()

        strikes_all = set()
        surface_data = []

        for exp in expirations[:8]:  # Limit to 8 expiries for performance
            try:
                chain = stock.option_chain(exp)
                calls = chain.calls
                exp_date = datetime.strptime(exp, '%Y-%m-%d')
                dte = max((exp_date - today).days, 1)

                for _, row in calls.iterrows():
                    iv = row.get('impliedVolatility', 0)
                    strike = row.get('strike', 0)
                    if iv > 0.01 and strike > 0:
                        strikes_all.add(float(strike))
                        surface_data.append({
                            'strike': float(strike),
                            'dte': dte,
                            'iv': float(iv),
                        })
            except Exception:
                continue

        if not surface_data:
            return jsonify({'error': 'No valid IV data found'}), 404

        # Build matrix
        strikes_sorted = sorted(strikes_all)
        expiries_sorted = sorted(set(d['dte'] for d in surface_data))

        iv_matrix = []
        for dte in expiries_sorted:
            row = []
            dte_data = {d['strike']: d['iv'] for d in surface_data if d['dte'] == dte}
            for s in strikes_sorted:
                row.append(dte_data.get(s, None))
            # Interpolate None values
            for i in range(len(row)):
                if row[i] is None:
                    # Find nearest non-None values
                    left = right = None
                    for j in range(i - 1, -1, -1):
                        if row[j] is not None:
                            left = row[j]
                            break
                    for j in range(i + 1, len(row)):
                        if row[j] is not None:
                            right = row[j]
                            break
                    if left and right:
                        row[i] = (left + right) / 2
                    elif left:
                        row[i] = left
                    elif right:
                        row[i] = right
                    else:
                        row[i] = 0.3  # default
            iv_matrix.append(row)

        return jsonify({
            'strikes': strikes_sorted,
            'expiries': expiries_sorted,
            'iv_matrix': iv_matrix,
            'spot_price': float(spot),
            'ticker': ticker,
        })

    except Exception as e:
        print(f'Vol surface error: {e}')
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Vol surface failed: {str(e)}'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    is_prod = os.environ.get('FLASK_ENV', 'development').lower() == 'production'
    debug = not is_prod

    print("Starting Portfolio Optimizer Backend...")
    print(f"Server running on http://localhost:{port}  (debug={debug})")
    print(f"Allowed origins: {_allowed_origins}")
    print(f"News API configured: {NEWS_API_KEY is not None}")
    print(f"OpenAI configured: {openai_client is not None}")
    print("Data source: yfinance (Tiingo removed)")

    # In production we run under gunicorn with the eventlet worker, so this
    # block is only exercised for local dev.
    socketio.run(app, debug=debug, host='0.0.0.0', port=port)
