import { useState } from 'react';
import { useApp } from '../../context/AppContext';
import Card, { CardHeader } from '../ui/Card';
import Button from '../ui/Button';
import Input from '../ui/Input';
import { Select } from '../ui/Input';
import Loading from '../ui/Loading';
import { constrainedOptimize, getSensitivities } from '../../services/api';
import { motion, AnimatePresence } from 'framer-motion';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  BarElement,
  CategoryScale,
  LinearScale,
  Tooltip as ChartTooltip,
} from 'chart.js';

ChartJS.register(BarElement, CategoryScale, LinearScale, ChartTooltip);

function SmallWeightBar({ ticker, weight }) {
  return (
    <div className="flex items-center gap-2">
      <span className="w-20 text-2xs font-medium text-surface-300 dark:text-surface-300 text-surface-600 truncate">
        {ticker.replace('.NS', '')}
      </span>
      <div className="flex-1 h-1.5 bg-surface-700/30 dark:bg-surface-700/30 bg-surface-200 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full bg-accent/70"
          style={{ width: `${Math.max(weight * 100, 1)}%` }}
        />
      </div>
      <span className="w-12 text-right text-2xs tabular-nums text-surface-400">
        {(weight * 100).toFixed(1)}%
      </span>
    </div>
  );
}

export default function ConstrainedOptimize() {
  const { tickers, notify } = useApp();
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [sensLoading, setSensLoading] = useState(null); // index of portfolio being analyzed
  const [sensResult, setSensResult] = useState(null);
  const [sensIdx, setSensIdx] = useState(null);

  // Per-ticker constraints
  const [minWeights, setMinWeights] = useState({});
  const [maxWeights, setMaxWeights] = useState({});

  // Global constraints
  const [minReturn, setMinReturn] = useState('');
  const [maxVol, setMaxVol] = useState('');
  const [minDiv, setMinDiv] = useState('');
  const [weightStep, setWeightStep] = useState('0.05');

  const run = async () => {
    if (tickers.length < 2) return notify('Need at least 2 tickers', 'warning');
    setLoading(true);
    setResult(null);
    setSensResult(null);
    try {
      const constraints = {};
      const mw = {};
      const xw = {};
      tickers.forEach((t) => {
        if (minWeights[t]) mw[t] = parseFloat(minWeights[t]) / 100;
        if (maxWeights[t]) xw[t] = parseFloat(maxWeights[t]) / 100;
      });
      if (Object.keys(mw).length) constraints.min_weight = mw;
      if (Object.keys(xw).length) constraints.max_weight = xw;
      if (minReturn) constraints.min_total_return = parseFloat(minReturn) / 100;
      if (maxVol) constraints.max_total_volatility = parseFloat(maxVol) / 100;
      if (minDiv) constraints.min_dividend_yield = parseFloat(minDiv) / 100;

      const data = await constrainedOptimize(
        tickers, '2020-01-01', 0.065, constraints, parseFloat(weightStep)
      );
      setResult(data);
      if (data.valid_portfolios?.length > 0) {
        notify(`Found ${data.total_valid} valid portfolios`, 'success');
      } else {
        notify('No portfolios matched all constraints. Try relaxing them.', 'warning');
      }
    } catch (err) {
      notify(err.message, 'error');
    } finally {
      setLoading(false);
    }
  };

  const runSensitivity = async (portfolio, idx) => {
    setSensLoading(idx);
    setSensResult(null);
    setSensIdx(idx);
    try {
      const data = await getSensitivities(tickers, portfolio.weights, '2020-01-01', 0.065);
      setSensResult(data);
    } catch (err) {
      notify(err.message, 'error');
    } finally {
      setSensLoading(null);
    }
  };

  const tornadoData = sensResult
    ? {
        labels: sensResult.sensitivities.map((s) => s.parameter),
        datasets: [
          {
            data: sensResult.sensitivities.map((s) => s.pct_impact),
            backgroundColor: sensResult.sensitivities.map((s) =>
              s.pct_impact >= 0 ? 'rgba(63, 185, 80, 0.7)' : 'rgba(248, 81, 73, 0.7)'
            ),
            borderRadius: 4,
          },
        ],
      }
    : null;

  const tornadoOptions = {
    indexAxis: 'y',
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: '#161b22',
        borderColor: '#21262d',
        borderWidth: 1,
        titleColor: '#f6f8fa',
        bodyColor: '#afb8c1',
        callbacks: { label: (ctx) => `${ctx.parsed.x > 0 ? '+' : ''}${ctx.parsed.x.toFixed(2)}% Sharpe impact` },
      },
    },
    scales: {
      x: {
        grid: { color: 'rgba(33,38,45,0.4)' },
        ticks: { color: '#656d76', font: { size: 10 }, callback: (v) => `${v > 0 ? '+' : ''}${v}%` },
      },
      y: {
        grid: { display: false },
        ticks: { color: '#8b949e', font: { size: 10 } },
      },
    },
  };

  return (
    <Card>
      <CardHeader
        title="Constrained Portfolio Search"
        subtitle="Backtracking algorithm with custom constraints"
      />

      {/* Per-ticker weight constraints */}
      <div className="mb-4">
        <p className="text-xs font-medium text-surface-400 uppercase tracking-wider mb-2">
          Weight Constraints (%)
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
          {tickers.map((t) => (
            <div key={t} className="flex items-center gap-2">
              <span className="text-xs text-surface-300 dark:text-surface-300 text-surface-600 w-20 truncate">
                {t.replace('.NS', '')}
              </span>
              <input
                type="number"
                placeholder="Min"
                value={minWeights[t] || ''}
                onChange={(e) => setMinWeights((p) => ({ ...p, [t]: e.target.value }))}
                className="w-16 rounded px-2 py-1 text-2xs bg-surface-850/80 dark:bg-surface-850/80 bg-surface-50 border border-surface-700/60 dark:border-surface-700/60 border-surface-200 text-surface-200 dark:text-surface-200 text-surface-700 focus:outline-none focus:border-accent/50"
              />
              <input
                type="number"
                placeholder="Max"
                value={maxWeights[t] || ''}
                onChange={(e) => setMaxWeights((p) => ({ ...p, [t]: e.target.value }))}
                className="w-16 rounded px-2 py-1 text-2xs bg-surface-850/80 dark:bg-surface-850/80 bg-surface-50 border border-surface-700/60 dark:border-surface-700/60 border-surface-200 text-surface-200 dark:text-surface-200 text-surface-700 focus:outline-none focus:border-accent/50"
              />
            </div>
          ))}
        </div>
      </div>

      {/* Global constraints */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
        <Input label="Min Return (%)" type="number" value={minReturn} onChange={(e) => setMinReturn(e.target.value)} />
        <Input label="Max Volatility (%)" type="number" value={maxVol} onChange={(e) => setMaxVol(e.target.value)} />
        <Input label="Min Dividend (%)" type="number" value={minDiv} onChange={(e) => setMinDiv(e.target.value)} />
        <Select label="Weight Step" value={weightStep} onChange={(e) => setWeightStep(e.target.value)}>
          <option value="0.025">2.5%</option>
          <option value="0.05">5%</option>
          <option value="0.10">10%</option>
        </Select>
      </div>

      <Button onClick={run} loading={loading}>
        Find Portfolios
      </Button>

      {loading && <div className="mt-4"><Loading text="Searching portfolio space..." /></div>}

      {result && !loading && (
        <div className="mt-5 space-y-4">
          {/* Stats */}
          <div className="flex flex-wrap gap-4 text-xs text-surface-500">
            <span>Explored: <strong className="text-surface-300 dark:text-surface-300 text-surface-600">{result.total_explored?.toLocaleString()}</strong></span>
            <span>Pruned: <strong className="text-surface-300 dark:text-surface-300 text-surface-600">{result.total_pruned?.toLocaleString()}</strong></span>
            <span>Valid: <strong className="text-accent">{result.total_valid}</strong></span>
          </div>

          {/* Portfolios */}
          <AnimatePresence>
            {result.valid_portfolios?.map((p, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.05 }}
                className="rounded-lg border border-surface-700/30 dark:border-surface-700/30 border-surface-200 p-4"
              >
                <div className="flex items-center justify-between mb-3">
                  <span className="text-xs font-semibold text-surface-300 dark:text-surface-300 text-surface-600">
                    Portfolio #{i + 1}
                  </span>
                  <div className="flex items-center gap-3 text-xs tabular-nums">
                    <span className="text-up">{(p.expected_return * 100).toFixed(2)}% ret</span>
                    <span className="text-surface-400">{(p.volatility * 100).toFixed(2)}% vol</span>
                    <span className="text-accent font-semibold">{p.sharpe_ratio.toFixed(2)} SR</span>
                  </div>
                </div>
                <div className="space-y-1.5 mb-3">
                  {Object.entries(p.weights)
                    .sort(([, a], [, b]) => b - a)
                    .map(([t, w]) => (
                      <SmallWeightBar key={t} ticker={t} weight={w} />
                    ))}
                </div>
                <div className="flex items-center gap-3">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => runSensitivity(p, i)}
                    loading={sensLoading === i}
                  >
                    Sensitivity Analysis
                  </Button>
                  {p.dividend_yield > 0 && (
                    <span className="text-2xs text-surface-500">
                      Div Yield: <strong className="text-surface-300 dark:text-surface-300 text-surface-600">{(p.dividend_yield * 100).toFixed(2)}%</strong>
                    </span>
                  )}
                </div>

                {/* Sensitivity tornado chart */}
                {sensIdx === i && sensResult && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    className="mt-4 pt-4 border-t border-surface-700/20 dark:border-surface-700/20 border-surface-100"
                  >
                    <p className="text-xs font-semibold text-surface-300 dark:text-surface-300 text-surface-600 mb-2">
                      Sharpe Ratio Sensitivity
                    </p>
                    <div className="h-40">
                      <Bar data={tornadoData} options={tornadoOptions} />
                    </div>
                    {/* Per-asset sensitivity */}
                    <div className="mt-3 space-y-1">
                      <p className="text-2xs font-medium text-surface-500 uppercase tracking-wider">Per-Asset Impact (+1% weight)</p>
                      {sensResult.per_asset_sensitivities.map((a) => (
                        <div key={a.ticker} className="flex items-center justify-between text-2xs">
                          <span className="text-surface-400">{a.ticker.replace('.NS', '')}</span>
                          <span className={`tabular-nums font-medium ${a.sharpe_delta >= 0 ? 'text-up' : 'text-down'}`}>
                            {a.sharpe_delta >= 0 ? '+' : ''}{a.sharpe_delta.toFixed(4)} Sharpe
                          </span>
                        </div>
                      ))}
                    </div>
                  </motion.div>
                )}
              </motion.div>
            ))}
          </AnimatePresence>

          {result.valid_portfolios?.length === 0 && (
            <p className="text-sm text-surface-500 text-center py-6">
              No portfolios matched all constraints. Try relaxing the bounds.
            </p>
          )}
        </div>
      )}
    </Card>
  );
}
