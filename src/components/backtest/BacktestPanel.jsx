import { useState } from 'react';
import { useApp } from '../../context/AppContext';
import Card, { CardHeader } from '../ui/Card';
import Button from '../ui/Button';
import Input from '../ui/Input';
import { Select } from '../ui/Input';
import Loading from '../ui/Loading';
import { runBacktest } from '../../services/api';
import { motion } from 'framer-motion';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';

ChartJS.register(LineElement, PointElement, LinearScale, CategoryScale, Tooltip, Legend, Filler);

function MetricCard({ label, portfolio, benchmark }) {
  return (
    <div className="rounded-lg border border-surface-700/30 dark:border-surface-700/30 border-surface-200 p-3">
      <p className="text-2xs uppercase tracking-wider text-surface-500 mb-2">{label}</p>
      <div className="flex justify-between">
        <div>
          <p className="text-2xs text-accent mb-0.5">Portfolio</p>
          <p className="text-sm font-semibold text-surface-100 dark:text-surface-100 text-surface-800 tabular-nums">{portfolio}</p>
        </div>
        <div className="text-right">
          <p className="text-2xs text-blue-400 mb-0.5">Benchmark</p>
          <p className="text-sm font-semibold text-surface-300 dark:text-surface-300 text-surface-600 tabular-nums">{benchmark}</p>
        </div>
      </div>
    </div>
  );
}

export default function BacktestPanel() {
  const { tickers, holdings, notify } = useApp();
  const [startDate, setStartDate] = useState('2019-01-01');
  const [endDate, setEndDate] = useState(new Date().toISOString().split('T')[0]);
  const [rebalFreq, setRebalFreq] = useState('monthly');
  const [feePct, setFeePct] = useState('0.1');
  const [slippage, setSlippage] = useState('5');
  const [benchmark, setBenchmark] = useState('SPY');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const run = async () => {
    if (tickers.length < 1) return notify('Add at least 1 holding to backtest', 'warning');
    setLoading(true);
    try {
      const weights = {};
      const totalQty = holdings.reduce((s, h) => s + h.quantity, 0);
      holdings.forEach(h => { weights[h.ticker] = h.quantity / totalQty; });

      const data = await runBacktest({
        tickers,
        weights,
        start_date: startDate,
        end_date: endDate,
        rebalance_freq: rebalFreq,
        fee_pct: parseFloat(feePct) / 100,
        slippage_factor: parseFloat(slippage) / 100,
        benchmark,
      });
      setResult(data);
      notify('Backtest complete', 'success');
    } catch (err) {
      notify(err.message, 'error');
    } finally {
      setLoading(false);
    }
  };

  const fmt = (v) => typeof v === 'number' ? (v * 100).toFixed(2) + '%' : '—';

  const chartData = result ? {
    labels: result.portfolio_series.map(p => p.date),
    datasets: [
      {
        label: 'Portfolio',
        data: result.portfolio_series.map(p => p.value),
        borderColor: '#c9985a',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.3,
        fill: true,
        backgroundColor: 'rgba(201, 152, 90, 0.05)',
      },
      ...(result.benchmark_series.length > 0 ? [{
        label: `Benchmark (${benchmark})`,
        data: result.benchmark_series.map(p => p.value),
        borderColor: '#58a6ff',
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.3,
        borderDash: [4, 2],
      }] : []),
    ],
  } : null;

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: '#8b949e', font: { size: 10 }, usePointStyle: true, pointStyle: 'line' } },
      tooltip: {
        backgroundColor: '#161b22',
        borderColor: '#21262d',
        borderWidth: 1,
        titleColor: '#f6f8fa',
        bodyColor: '#afb8c1',
        padding: 10,
        callbacks: {
          label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(4)}x`,
        },
      },
    },
    scales: {
      x: {
        grid: { color: 'rgba(33,38,45,0.4)' },
        ticks: { color: '#656d76', font: { size: 9 }, maxTicksLimit: 8 },
      },
      y: {
        grid: { color: 'rgba(33,38,45,0.4)' },
        ticks: { color: '#656d76', font: { size: 9 }, callback: (v) => v.toFixed(2) + 'x' },
      },
    },
  };

  return (
    <div className="space-y-6">
      {/* Controls */}
      <Card>
        <CardHeader title="Portfolio Backtesting" subtitle="Historical simulation with realistic market friction" />
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          <Input label="Start Date" type="date" value={startDate} onChange={e => setStartDate(e.target.value)} />
          <Input label="End Date" type="date" value={endDate} onChange={e => setEndDate(e.target.value)} />
          <Select label="Rebalance" value={rebalFreq} onChange={e => setRebalFreq(e.target.value)}>
            <option value="monthly">Monthly</option>
            <option value="quarterly">Quarterly</option>
            <option value="annual">Annual</option>
          </Select>
          <Input label="Fees (%)" type="number" value={feePct} onChange={e => setFeePct(e.target.value)} step="0.01" min="0" />
          <Input label="Slippage (%)" type="number" value={slippage} onChange={e => setSlippage(e.target.value)} step="0.1" min="0" />
          <Input label="Benchmark" type="text" value={benchmark} onChange={e => setBenchmark(e.target.value.toUpperCase())} />
        </div>
        <div className="mt-4">
          <Button onClick={run} loading={loading}>Run Backtest</Button>
        </div>
      </Card>

      {loading && <Loading text="Running historical backtest..." />}

      {result && !loading && (
        <>
          {/* Chart */}
          <Card>
            <CardHeader title="Cumulative Returns" subtitle="Portfolio vs benchmark performance" />
            <div className="h-72 sm:h-80">
              <Line data={chartData} options={chartOptions} />
            </div>
          </Card>

          {/* Metrics Comparison */}
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            <MetricCard
              label="CAGR"
              portfolio={fmt(result.portfolio_metrics.cagr)}
              benchmark={fmt(result.benchmark_metrics?.cagr)}
            />
            <MetricCard
              label="Sharpe Ratio"
              portfolio={result.portfolio_metrics.sharpe?.toFixed(2) || '—'}
              benchmark={result.benchmark_metrics?.sharpe?.toFixed(2) || '—'}
            />
            <MetricCard
              label="Max Drawdown"
              portfolio={fmt(result.portfolio_metrics.max_drawdown)}
              benchmark={fmt(result.benchmark_metrics?.max_drawdown)}
            />
            <MetricCard
              label="VaR (95%)"
              portfolio={fmt(result.portfolio_metrics.var_95)}
              benchmark={fmt(result.benchmark_metrics?.var_95)}
            />
            <MetricCard
              label="CVaR (95%)"
              portfolio={fmt(result.portfolio_metrics.cvar_95)}
              benchmark={fmt(result.benchmark_metrics?.cvar_95)}
            />
            <MetricCard
              label="Volatility"
              portfolio={fmt(result.portfolio_metrics.volatility)}
              benchmark={fmt(result.benchmark_metrics?.volatility)}
            />
          </div>

          {/* Friction Impact */}
          <Card hover={false} className="!p-4">
            <div className="flex flex-wrap gap-6 text-xs text-surface-500">
              <span>Rebalances: <strong className="text-surface-300 dark:text-surface-300 text-surface-600">{result.num_rebalances}</strong></span>
              <span>Total Fees: <strong className="text-down">{(result.total_fees_paid * 100).toFixed(2)}%</strong></span>
              <span>Friction Impact: <strong className="text-down">{result.friction_impact?.toFixed(4) || '—'}</strong></span>
              <span>Benchmark: <strong className="text-surface-300 dark:text-surface-300 text-surface-600">{benchmark}</strong></span>
            </div>
          </Card>
        </>
      )}

      {tickers.length < 1 && (
        <Card hover={false}>
          <div className="text-center py-16">
            <p className="text-surface-500 text-sm">Add holdings in the Portfolio tab to run a backtest.</p>
          </div>
        </Card>
      )}
    </div>
  );
}
