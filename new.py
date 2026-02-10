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
from flask_socketio import SocketIO
import os

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
        
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration
NEWS_API_KEY = os.environ.get('NEWS_API_KEY', None)  # Get from environment variable

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
        px_portfolio = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)["Close"]
        
        # Download benchmark separately to avoid issues
        px_benchmark = yf.download(benchmark, start=start_date, end=end_date, auto_adjust=True, progress=False)["Close"]
        
        # Handle single ticker case for portfolio
        if len(tickers) == 1:
            px_portfolio = pd.DataFrame({tickers[0]: px_portfolio})
        
        # Ensure benchmark is a Series
        if isinstance(px_benchmark, pd.DataFrame):
            px_benchmark = px_benchmark.iloc[:, 0]
        
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
        prices = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)['Close']

        # Ensure prices is a DataFrame
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])

        # Clean data
        prices = prices.dropna()

        if len(prices) < 30:
            return jsonify({'error': 'Insufficient data. Try a different date range.'}), 400

        # Calculate returns
        returns = prices.pct_change().dropna()
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        # Monte Carlo Simulation
        results = np.zeros((3, num_portfolios))
        weights_record = []

        np.random.seed(42)  # For reproducibility

        for i in range(num_portfolios):
            weights = np.random.random(len(tickers))
            weights /= np.sum(weights)
            weights_record.append(weights)

            portfolio_return = np.dot(weights, mean_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            # Handle edge cases
            if portfolio_vol == 0:
                sharpe_ratio = 0
            else:
                sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol

            results[0, i] = portfolio_vol
            results[1, i] = portfolio_return
            results[2, i] = sharpe_ratio

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

        prices = yf.download(tickers, start=start_date, period='3y', auto_adjust=True, progress=False)['Close']

        # Ensure prices is a DataFrame (yf.download may return Series for single ticker in older versions)
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])

        prices = prices.dropna()
        
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
        prices = yf.download(all_tickers, period='3y', auto_adjust=True, progress=False)['Close']

        # Ensure prices is a DataFrame (yf.download may return Series for single download)
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=all_tickers[0])

        prices = prices.dropna()
        
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

        prices = yf.download(ticker, period='2y', auto_adjust=True, progress=False)['Close']
        
        if isinstance(prices, pd.DataFrame):
            prices = prices.iloc[:, 0]
        
        prices = prices.dropna()
        
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

        prices = yf.download(ticker, period='2y', auto_adjust=True, progress=False)['Close']
        
        if isinstance(prices, pd.DataFrame):
            prices = prices.iloc[:, 0]
        
        prices = prices.dropna()
        
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

        prices = yf.download(ticker, period='5y', auto_adjust=True, progress=False)['Close']
        
        if isinstance(prices, pd.DataFrame):
            prices = prices.iloc[:, 0]
        
        prices = prices.dropna()
        
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

        for ticker in tickers:
            try:
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
                stock = yf.Ticker(t)
                news_items = stock.news
                
                articles = []
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

@app.route('/api/economic-indicators', methods=['GET'])
def economic_indicators():
    try:
        # Use market proxies for economic indicators
        indicators = {}

        # Treasury yields as interest rate proxies
        treasury_tickers = {
            '^TNX': '10Y Treasury Yield',
            '^FVX': '5Y Treasury Yield',
            '^IRX': '13-Week T-Bill',
        }

        # Market indices
        market_tickers = {
            '^GSPC': 'S&P 500',
            '^DJI': 'Dow Jones',
            '^VIX': 'VIX (Fear Index)',
        }

        all_tickers = list(treasury_tickers.keys()) + list(market_tickers.keys())

        data = yf.download(all_tickers, period='5d', auto_adjust=True, progress=False)['Close']
        if isinstance(data, pd.Series):
            data = data.to_frame()

        treasury_data = []
        for ticker, name in treasury_tickers.items():
            if ticker in data.columns:
                vals = data[ticker].dropna()
                if len(vals) >= 1:
                    current = float(vals.iloc[-1])
                    prev = float(vals.iloc[-2]) if len(vals) >= 2 else current
                    treasury_data.append({
                        'name': name,
                        'value': round(current, 2),
                        'change': round(current - prev, 2),
                        'unit': '%'
                    })

        market_data = []
        for ticker, name in market_tickers.items():
            if ticker in data.columns:
                vals = data[ticker].dropna()
                if len(vals) >= 1:
                    current = float(vals.iloc[-1])
                    prev = float(vals.iloc[-2]) if len(vals) >= 2 else current
                    change_pct = ((current - prev) / prev * 100) if prev != 0 else 0
                    market_data.append({
                        'name': name,
                        'value': round(current, 2),
                        'change': round(change_pct, 2),
                        'unit': 'pts' if 'VIX' not in name else ''
                    })

        return jsonify({
            'treasury': treasury_data,
            'market': market_data
        })

    except Exception as e:
        print(f"Error fetching economic indicators: {str(e)}")
        return jsonify({'error': f'Economic indicators failed: {str(e)}'}), 500


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
                       expected_return, volatility, num_sims=5000):
    """Run Monte Carlo accumulation simulation and return final values array."""
    months = years * 12
    monthly_ret = expected_return / 12
    monthly_vol = volatility / np.sqrt(12)

    # Vectorised: shape (num_sims, months)
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
                                    years_to_retirement, exp_ret, vol, num_sims=5000)
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

        prices = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)['Close']
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])
        prices = prices.dropna()

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

        prices = yf.download(ticker, period=period, auto_adjust=True, progress=False)['Close']
        prices = normalize_series(prices)
        prices = prices.dropna()

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

        prices = yf.download(ticker, period=period, auto_adjust=True, progress=False)['Close']
        prices = normalize_series(prices)
        prices = prices.dropna()

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
        prices = yf.download(all_tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)['Close']
        prices = normalize_columns(prices)

        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=all_tickers[0])
        prices = prices.dropna()

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
        prices = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)['Close']
        prices = normalize_columns(prices)

        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])
        prices = prices.dropna()

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


if __name__ == '__main__':
    print("Starting Portfolio Optimizer Backend...")
    print("Server running on http://localhost:5000")
    print(f"News API configured: {NEWS_API_KEY is not None}")
    socketio.run(app, debug=True, port=5000, allow_unsafe_werkzeug=True)
