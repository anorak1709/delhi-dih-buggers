import { useState } from 'react';
import { useApp } from '../../context/AppContext';
import Card, { CardHeader } from '../ui/Card';
import Button from '../ui/Button';
import Input from '../ui/Input';
import { Select } from '../ui/Input';
import Loading from '../ui/Loading';
import { optimizePortfolio, getHRP, getBlackLitterman } from '../../services/api';
import InfoTip from '../ui/Tooltip';
import { TOOLTIPS } from '../../constants/tooltips';
import ConstrainedOptimize from './ConstrainedOptimize';
import Plot from 'react-plotly.js';

function WeightBar({ ticker, weight, isMax }) {
  return (
    <div className="flex items-center gap-3">
      <span className="w-28 text-sm font-semibold text-surface-200 dark:text-surface-200 text-surface-700 truncate">
        {ticker.replace('.NS', '')}
      </span>
      <div className="flex-1 h-2.5 bg-surface-700/40 dark:bg-surface-700/40 bg-surface-200 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ${isMax ? 'bg-accent' : 'bg-accent/60'}`}
          style={{ width: `${Math.max(weight * 100, 1)}%` }}
        />
      </div>
      <span className="w-16 text-right text-sm tabular-nums text-surface-300 dark:text-surface-300 text-surface-600 font-medium">
        {(weight * 100).toFixed(1)}%
      </span>
    </div>
  );
}

function HRPSection({ tickers, startDate, notify, fmt }) {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const run = async () => {
    setLoading(true);
    try {
      const data = await getHRP(tickers, startDate);
      setResult(data);
      notify('HRP optimization complete', 'success');
    } catch (err) {
      notify(err.message, 'error');
    } finally {
      setLoading(false);
    }
  };

  // Build dendrogram traces from icoord/dcoord
  const dendroTraces = result?.dendrogram_plot ? result.dendrogram_plot.icoord.map((ic, idx) => ({
    x: ic,
    y: result.dendrogram_plot.dcoord[idx],
    type: 'scatter',
    mode: 'lines',
    line: { color: '#c9985a', width: 1.5 },
    hoverinfo: 'none',
    showlegend: false,
  })) : [];

  return (
    <Card>
      <CardHeader title={<InfoTip text={TOOLTIPS.hrp}>Hierarchical Risk Parity (HRP)</InfoTip>} subtitle="Dendrogram-based allocation — stable and robust" />
      <Button onClick={run} loading={loading}>Run HRP</Button>

      {loading && <div className="mt-4"><Loading text="Computing hierarchical clustering..." /></div>}

      {result && !loading && (
        <div className="mt-5 space-y-4">
          {/* Metrics */}
          <div className="grid grid-cols-3 gap-3">
            <div>
              <p className="text-2xs uppercase tracking-wider text-surface-500 mb-0.5">Return</p>
              <p className="text-lg font-display font-semibold text-up tabular-nums">{fmt(result.metrics.expected_return)}</p>
            </div>
            <div>
              <p className="text-2xs uppercase tracking-wider text-surface-500 mb-0.5">Volatility</p>
              <p className="text-lg font-display font-semibold text-surface-200 dark:text-surface-200 text-surface-700 tabular-nums">{fmt(result.metrics.volatility)}</p>
            </div>
            <div>
              <p className="text-2xs uppercase tracking-wider text-surface-500 mb-0.5">Sharpe</p>
              <p className="text-lg font-display font-semibold text-accent tabular-nums">{result.metrics.sharpe_ratio.toFixed(2)}</p>
            </div>
          </div>

          {/* Weights */}
          <div className="space-y-2.5">
            {Object.entries(result.weights).sort(([, a], [, b]) => b - a).map(([ticker, weight]) => (
              <WeightBar key={ticker} ticker={ticker} weight={weight} isMax={false} />
            ))}
          </div>

          {/* Dendrogram */}
          {dendroTraces.length > 0 && (
            <div className="mt-4">
              <p className="text-xs font-semibold text-surface-300 dark:text-surface-300 text-surface-600 mb-2">Asset Hierarchy (Dendrogram)</p>
              <Plot
                data={dendroTraces}
                layout={{
                  height: 280,
                  margin: { l: 40, r: 20, t: 10, b: 60 },
                  paper_bgcolor: 'transparent',
                  plot_bgcolor: 'transparent',
                  xaxis: {
                    tickvals: result.dendrogram_plot.ivl.map((_, i) => (i + 0.5) * 10),
                    ticktext: result.dendrogram_plot.ivl.map(l => l.replace('.NS', '')),
                    tickfont: { color: '#8b949e', size: 10 },
                    gridcolor: 'rgba(33,38,45,0.3)',
                  },
                  yaxis: {
                    title: { text: 'Distance', font: { color: '#656d76', size: 10 } },
                    tickfont: { color: '#656d76', size: 9 },
                    gridcolor: 'rgba(33,38,45,0.3)',
                  },
                }}
                config={{ displayModeBar: false, responsive: true }}
                className="w-full"
              />
            </div>
          )}
        </div>
      )}
    </Card>
  );
}

function BlackLittermanSection({ tickers, startDate, notify, fmt }) {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [useSentiment, setUseSentiment] = useState(true);

  const run = async () => {
    setLoading(true);
    try {
      const data = await getBlackLitterman(tickers, startDate, null, null, useSentiment);
      setResult(data);
      notify('Black-Litterman model complete', 'success');
    } catch (err) {
      notify(err.message, 'error');
    } finally {
      setLoading(false);
    }
  };

  const sentimentColor = (score) => {
    if (score > 0.5) return 'text-up bg-up/10';
    if (score < -0.5) return 'text-down bg-down/10';
    return 'text-amber-400 bg-amber-400/10';
  };

  return (
    <Card>
      <CardHeader title={<InfoTip text={TOOLTIPS.black_litterman}>Black-Litterman Model</InfoTip>} subtitle="Bayesian approach combining market equilibrium with AI sentiment views" />
      <div className="flex items-center gap-4 mb-4">
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={useSentiment}
            onChange={(e) => setUseSentiment(e.target.checked)}
            className="w-4 h-4 rounded border-surface-600 text-accent focus:ring-accent/30 bg-surface-800 cursor-pointer"
          />
          <span className="text-xs text-surface-300 dark:text-surface-300 text-surface-600">Use AI Sentiment Views</span>
        </label>
        <Button onClick={run} loading={loading}>Run Black-Litterman</Button>
      </div>

      {loading && <Loading text="Computing posterior estimates..." />}

      {result && !loading && (
        <div className="mt-4 space-y-4">
          {/* Sentiment Scores */}
          {result.sentiment_scores && Object.keys(result.sentiment_scores).length > 0 && (
            <div>
              <p className="text-2xs font-medium text-surface-500 uppercase tracking-wider mb-2">AI Sentiment Analysis</p>
              <div className="flex flex-wrap gap-2">
                {Object.entries(result.sentiment_scores).map(([t, s]) => (
                  <span
                    key={t}
                    className={`text-xs px-2.5 py-1 rounded-full font-medium ${sentimentColor(s)}`}
                  >
                    {t.replace('.NS', '')}: {s >= 0 ? '+' : ''}{(s * 100).toFixed(0)}%
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Views Applied */}
          {result.views_applied?.length > 0 && (
            <div>
              <p className="text-2xs font-medium text-surface-500 uppercase tracking-wider mb-2">Views Applied</p>
              <div className="space-y-1">
                {result.views_applied.map((v, i) => (
                  <div key={i} className="flex items-center justify-between text-xs">
                    <span className="text-surface-300 dark:text-surface-300 text-surface-600">{v.ticker.replace('.NS', '')}</span>
                    <span className={`tabular-nums font-medium ${v.view_return >= 0 ? 'text-up' : 'text-down'}`}>
                      {v.view_return >= 0 ? '+' : ''}{(v.view_return * 100).toFixed(2)}% expected
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Implied vs BL Returns */}
          <div>
            <p className="text-2xs font-medium text-surface-500 uppercase tracking-wider mb-2">Expected Returns: Equilibrium vs BL-Adjusted</p>
            <div className="space-y-2">
              {Object.keys(result.weights).sort((a, b) => result.weights[b] - result.weights[a]).map(t => (
                <div key={t} className="flex items-center gap-3">
                  <span className="w-20 text-xs text-surface-300 dark:text-surface-300 text-surface-600 truncate">{t.replace('.NS', '')}</span>
                  <div className="flex-1 flex items-center gap-2">
                    <div className="flex-1 h-1.5 bg-surface-700/30 dark:bg-surface-700/30 bg-surface-200 rounded-full overflow-hidden relative">
                      <div
                        className="absolute h-full rounded-full bg-blue-500/40"
                        style={{ width: `${Math.min(Math.abs(result.implied_returns[t] || 0) * 300, 100)}%` }}
                      />
                      <div
                        className="absolute h-full rounded-full bg-accent"
                        style={{ width: `${Math.min(Math.abs(result.expected_returns[t] || 0) * 300, 100)}%` }}
                      />
                    </div>
                    <span className="w-14 text-right text-2xs tabular-nums text-accent">{fmt(result.expected_returns[t] || 0)}</span>
                  </div>
                </div>
              ))}
            </div>
            <div className="flex gap-4 mt-2 text-2xs text-surface-500">
              <span className="flex items-center gap-1"><span className="w-3 h-1.5 rounded bg-blue-500/40" /> Equilibrium</span>
              <span className="flex items-center gap-1"><span className="w-3 h-1.5 rounded bg-accent" /> BL-Adjusted</span>
            </div>
          </div>

          {/* BL Weights */}
          <div>
            <p className="text-2xs font-medium text-surface-500 uppercase tracking-wider mb-2">Optimal BL Weights</p>
            <div className="space-y-2.5">
              {Object.entries(result.weights).sort(([, a], [, b]) => b - a).map(([ticker, weight]) => (
                <WeightBar key={ticker} ticker={ticker} weight={weight} isMax={false} />
              ))}
            </div>
          </div>
        </div>
      )}
    </Card>
  );
}

export default function OptimizePanel() {
  const { tickers, notify } = useApp();
  const [startDate, setStartDate] = useState('2020-01-01');
  const [numSims, setNumSims] = useState('10000');
  const [method, setMethod] = useState('standard');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const runOptimize = async () => {
    if (tickers.length < 2) return notify('Need at least 2 tickers to optimize', 'warning');
    setLoading(true);
    try {
      const data = await optimizePortfolio(tickers, startDate, parseInt(numSims, 10), 0.065, method);
      setResult(data);
      notify('Optimization complete', 'success');
    } catch (err) {
      notify(err.message, 'error');
    } finally {
      setLoading(false);
    }
  };

  const fmt = (v) => (v * 100).toFixed(2) + '%';

  const maxWeight = (weights) => {
    const entries = Object.entries(weights);
    return entries.reduce((max, [, w]) => Math.max(max, w), 0);
  };

  return (
    <div className="space-y-6">
      {/* Controls */}
      <Card>
        <CardHeader
          title="Monte Carlo Optimization"
          subtitle="Simulate thousands of portfolios to find optimal allocations"
        />
        <div className="flex flex-col sm:flex-row gap-3 items-end">
          <Input
            label="Start Date"
            type="date"
            value={startDate}
            onChange={(e) => setStartDate(e.target.value)}
            className="flex-1"
          />
          <Input
            label={<InfoTip text={TOOLTIPS.simulations}>Simulations</InfoTip>}
            type="number"
            value={numSims}
            min="1000"
            max="50000"
            step="1000"
            onChange={(e) => setNumSims(e.target.value)}
            className="w-full sm:w-36"
          />
          <Select
            label={<InfoTip text={TOOLTIPS.mc_method}>MC Method</InfoTip>}
            value={method}
            onChange={(e) => setMethod(e.target.value)}
            className="w-full sm:w-44"
          >
            <option value="standard">Standard</option>
            <option value="antithetic">Antithetic Variates</option>
            <option value="sobol">Sobol (Quasi-MC)</option>
            <option value="full">Full (Combined)</option>
          </Select>
          <Button onClick={runOptimize} loading={loading}>
            Optimize
          </Button>
        </div>
      </Card>

      {loading && <Loading text={`Running ${parseInt(numSims, 10).toLocaleString()} simulations...`} />}

      {result && !loading && (
        <>
          {/* Two-column optimal portfolios */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Max Sharpe */}
            <Card>
              <CardHeader title="Max Sharpe Ratio" subtitle="Highest risk-adjusted return" />
              <div className="grid grid-cols-3 gap-3 mb-5">
                <div>
                  <p className="text-2xs uppercase tracking-wider text-surface-500 mb-0.5">Return</p>
                  <p className="text-lg font-display font-semibold text-up tabular-nums">
                    {fmt(result.optimal_portfolio.expected_return)}
                  </p>
                </div>
                <div>
                  <p className="text-2xs uppercase tracking-wider text-surface-500 mb-0.5">Volatility</p>
                  <p className="text-lg font-display font-semibold text-surface-200 dark:text-surface-200 text-surface-700 tabular-nums">
                    {fmt(result.optimal_portfolio.volatility)}
                  </p>
                </div>
                <div>
                  <p className="text-2xs uppercase tracking-wider text-surface-500 mb-0.5">Sharpe</p>
                  <p className="text-lg font-display font-semibold text-accent tabular-nums">
                    {result.optimal_portfolio.sharpe_ratio.toFixed(2)}
                  </p>
                </div>
              </div>
              <div className="space-y-2.5">
                {Object.entries(result.optimal_portfolio.weights)
                  .sort(([, a], [, b]) => b - a)
                  .map(([ticker, weight]) => (
                    <WeightBar
                      key={ticker}
                      ticker={ticker}
                      weight={weight}
                      isMax={weight === maxWeight(result.optimal_portfolio.weights)}
                    />
                  ))}
              </div>
            </Card>

            {/* Min Volatility */}
            <Card>
              <CardHeader title="Min Volatility" subtitle="Lowest risk portfolio" />
              <div className="grid grid-cols-3 gap-3 mb-5">
                <div>
                  <p className="text-2xs uppercase tracking-wider text-surface-500 mb-0.5">Return</p>
                  <p className="text-lg font-display font-semibold text-up tabular-nums">
                    {fmt(result.min_risk_portfolio.expected_return)}
                  </p>
                </div>
                <div>
                  <p className="text-2xs uppercase tracking-wider text-surface-500 mb-0.5">Volatility</p>
                  <p className="text-lg font-display font-semibold text-surface-200 dark:text-surface-200 text-surface-700 tabular-nums">
                    {fmt(result.min_risk_portfolio.volatility)}
                  </p>
                </div>
                <div>
                  <p className="text-2xs uppercase tracking-wider text-surface-500 mb-0.5">Sharpe</p>
                  <p className="text-lg font-display font-semibold text-accent tabular-nums">
                    {result.min_risk_portfolio.sharpe_ratio.toFixed(2)}
                  </p>
                </div>
              </div>
              <div className="space-y-2.5">
                {Object.entries(result.min_risk_portfolio.weights)
                  .sort(([, a], [, b]) => b - a)
                  .map(([ticker, weight]) => (
                    <WeightBar
                      key={ticker}
                      ticker={ticker}
                      weight={weight}
                      isMax={weight === maxWeight(result.min_risk_portfolio.weights)}
                    />
                  ))}
              </div>
            </Card>
          </div>

          {/* Efficient Frontier Image */}
          {result.plot_image && (
            <Card>
              <CardHeader
                title="Efficient Frontier"
                subtitle={`${result.statistics.num_simulations.toLocaleString()} simulated portfolios`}
              />
              <div className="flex justify-center">
                <img
                  src={`data:image/png;base64,${result.plot_image}`}
                  alt="Efficient Frontier"
                  className="rounded-lg max-w-full max-h-[480px] object-contain"
                />
              </div>
            </Card>
          )}

          {/* Stats */}
          <Card hover={false} className="!p-4">
            <div className="flex flex-wrap gap-6 text-xs text-surface-500">
              <span>Simulations: <strong className="text-surface-300 dark:text-surface-300 text-surface-600">{result.statistics.num_simulations.toLocaleString()}</strong></span>
              <span>Data: <strong className="text-surface-300 dark:text-surface-300 text-surface-600">{result.statistics.date_range.start} — {result.statistics.date_range.end}</strong></span>
              <span>Assets: <strong className="text-surface-300 dark:text-surface-300 text-surface-600">{tickers.length}</strong></span>
              <span>Method: <strong className="text-surface-300 dark:text-surface-300 text-surface-600">{result.statistics.method || 'Standard'}</strong></span>
            </div>
          </Card>
        </>
      )}

      {/* Constrained Optimization */}
      {tickers.length >= 2 && <ConstrainedOptimize />}

      {/* HRP Section */}
      {tickers.length >= 2 && <HRPSection tickers={tickers} startDate={startDate} notify={notify} fmt={fmt} />}

      {/* Black-Litterman Section */}
      {tickers.length >= 2 && <BlackLittermanSection tickers={tickers} startDate={startDate} notify={notify} fmt={fmt} />}

      {tickers.length < 2 && (
        <Card hover={false}>
          <div className="text-center py-16">
            <p className="text-surface-500 text-sm">Add at least 2 holdings in the Portfolio tab to run optimization.</p>
          </div>
        </Card>
      )}
    </div>
  );
}
