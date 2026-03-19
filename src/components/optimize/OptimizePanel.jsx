import { useState } from 'react';
import { useApp } from '../../context/AppContext';
import Card, { CardHeader } from '../ui/Card';
import Button from '../ui/Button';
import Input from '../ui/Input';
import Loading from '../ui/Loading';
import { optimizePortfolio } from '../../services/api';

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

export default function OptimizePanel() {
  const { tickers, notify } = useApp();
  const [startDate, setStartDate] = useState('2020-01-01');
  const [numSims, setNumSims] = useState('10000');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const runOptimize = async () => {
    if (tickers.length < 2) return notify('Need at least 2 tickers to optimize', 'warning');
    setLoading(true);
    try {
      const data = await optimizePortfolio(tickers, startDate, parseInt(numSims, 10));
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
            label="Simulations"
            type="number"
            value={numSims}
            min="1000"
            max="50000"
            step="1000"
            onChange={(e) => setNumSims(e.target.value)}
            className="w-full sm:w-36"
          />
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
            </div>
          </Card>
        </>
      )}

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
