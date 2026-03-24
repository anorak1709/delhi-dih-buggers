import { useState, useRef, useEffect } from 'react';
import { useApp } from '../../context/AppContext';
import Card, { CardHeader } from '../ui/Card';
import Button from '../ui/Button';
import Input from '../ui/Input';
import { Select } from '../ui/Input';
import Loading from '../ui/Loading';
import { analyzePortfolio, getCorrelation } from '../../services/api';
import InfoTip from '../ui/Tooltip';
import { TOOLTIPS } from '../../constants/tooltips';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler } from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler);

function MetricCard({ label, value, suffix = '', color = 'text-surface-100 dark:text-surface-100 text-surface-800' }) {
  return (
    <div>
      <p className="text-2xs uppercase tracking-wider text-surface-500 mb-1">{label}</p>
      <p className={`text-xl font-display font-semibold tabular-nums ${color}`}>
        {value}
        {suffix && <span className="text-sm text-surface-400 ml-0.5">{suffix}</span>}
      </p>
    </div>
  );
}

export default function AnalysisPanel() {
  const { holdings, holdingsMap, tickers, notify } = useApp();
  const [benchmark, setBenchmark] = useState('^NSEI');
  const [startDate, setStartDate] = useState('2020-01-01');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [corrLoading, setCorrLoading] = useState(false);
  const [correlation, setCorrelation] = useState(null);

  const runAnalysis = async () => {
    if (!holdings.length) return notify('Add holdings first', 'warning');
    setLoading(true);
    try {
      const data = await analyzePortfolio(holdingsMap, benchmark, startDate);
      setResult(data);
      notify('Analysis complete', 'success');
    } catch (err) {
      notify(err.message, 'error');
    } finally {
      setLoading(false);
    }
  };

  const runCorrelation = async () => {
    if (tickers.length < 2) return notify('Need at least 2 tickers for correlation', 'warning');
    setCorrLoading(true);
    try {
      const data = await getCorrelation(tickers, startDate);
      setCorrelation(data);
      notify('Correlation matrix ready', 'success');
    } catch (err) {
      notify(err.message, 'error');
    } finally {
      setCorrLoading(false);
    }
  };

  const fmt = (v, decimals = 2) => {
    if (v == null) return '—';
    return (v * 100).toFixed(decimals) + '%';
  };

  const chartData = result?.chart_data
    ? {
        labels: result.chart_data.map((d) => d.date),
        datasets: [
          {
            label: 'Portfolio',
            data: result.chart_data.map((d) => d.portfolio),
            borderColor: '#c9985a',
            backgroundColor: 'rgba(201,152,90,0.08)',
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.3,
            fill: true,
          },
          {
            label: 'Benchmark',
            data: result.chart_data.map((d) => d.benchmark),
            borderColor: '#58a6ff',
            backgroundColor: 'transparent',
            borderWidth: 1.5,
            pointRadius: 0,
            tension: 0.3,
            borderDash: [4, 4],
          },
        ],
      }
    : null;

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { labels: { color: '#8b949e', font: { size: 11 }, usePointStyle: true, pointStyle: 'line' } },
      tooltip: {
        backgroundColor: '#161b22',
        borderColor: '#21262d',
        borderWidth: 1,
        titleColor: '#f6f8fa',
        bodyColor: '#afb8c1',
        padding: 10,
        callbacks: { label: (ctx) => `${ctx.dataset.label}: ${(ctx.parsed.y).toFixed(4)}x` },
      },
    },
    scales: {
      x: { grid: { color: 'rgba(33,38,45,0.4)' }, ticks: { color: '#656d76', maxTicksLimit: 8, font: { size: 10 } } },
      y: { grid: { color: 'rgba(33,38,45,0.4)' }, ticks: { color: '#656d76', font: { size: 10 }, callback: (v) => v.toFixed(1) + 'x' } },
    },
  };

  const corrColor = (v) => {
    if (v >= 0.7) return 'bg-up/20 text-up';
    if (v >= 0.3) return 'bg-up/10 text-up/80';
    if (v <= -0.3) return 'bg-down/20 text-down';
    if (v <= -0.1) return 'bg-down/10 text-down/80';
    return 'text-surface-400';
  };

  return (
    <div className="space-y-6">
      {/* Controls */}
      <Card>
        <CardHeader title="Performance Analysis" subtitle="Compare your portfolio against a benchmark index" />
        <div className="flex flex-col sm:flex-row gap-3 items-end">
          <Select label="Benchmark" value={benchmark} onChange={(e) => setBenchmark(e.target.value)} className="flex-1">
            <option value="^NSEI">NIFTY 50</option>
            <option value="^BSESN">SENSEX</option>
            <option value="^GSPC">S&P 500</option>
            <option value="^DJI">Dow Jones</option>
          </Select>
          <Input
            label="Start Date"
            type="date"
            value={startDate}
            onChange={(e) => setStartDate(e.target.value)}
            className="flex-1"
          />
          <Button onClick={runAnalysis} loading={loading}>
            Analyze
          </Button>
        </div>
      </Card>

      {loading && <Loading text="Running performance analysis..." />}

      {result && !loading && (
        <>
          {/* Metrics Comparison */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card hover={false}>
              <CardHeader title="Portfolio" />
              <div className="grid grid-cols-2 gap-4">
                <MetricCard label={<InfoTip text={TOOLTIPS.cagr}>CAGR</InfoTip>} value={fmt(result.portfolio.cagr)} color={result.portfolio.cagr >= 0 ? 'text-up' : 'text-down'} />
                <MetricCard label={<InfoTip text={TOOLTIPS.volatility}>Volatility</InfoTip>} value={fmt(result.portfolio.annual_vol)} />
                <MetricCard label={<InfoTip text={TOOLTIPS.sharpe}>Sharpe Ratio</InfoTip>} value={result.portfolio.sharpe?.toFixed(2)} />
                <MetricCard label={<InfoTip text={TOOLTIPS.max_drawdown}>Max Drawdown</InfoTip>} value={fmt(result.portfolio.max_drawdown)} color="text-down" />
              </div>
            </Card>
            <Card hover={false}>
              <CardHeader title="Benchmark" />
              <div className="grid grid-cols-2 gap-4">
                <MetricCard label="CAGR" value={fmt(result.benchmark.cagr)} color={result.benchmark.cagr >= 0 ? 'text-up' : 'text-down'} />
                <MetricCard label="Volatility" value={fmt(result.benchmark.annual_vol)} />
                <MetricCard label="Sharpe Ratio" value={result.benchmark.sharpe?.toFixed(2)} />
                <MetricCard label="Max Drawdown" value={fmt(result.benchmark.max_drawdown)} color="text-down" />
              </div>
            </Card>
          </div>

          {/* Chart */}
          {chartData && (
            <Card>
              <CardHeader title="Normalized Performance" subtitle={`${result.date_range.start} — ${result.date_range.end}`} />
              <div className="h-72 sm:h-80">
                <Line data={chartData} options={chartOptions} />
              </div>
            </Card>
          )}
        </>
      )}

      {/* Correlation Matrix */}
      {tickers.length >= 2 && (
        <Card>
          <CardHeader
            title={<InfoTip text={TOOLTIPS.correlation}>Correlation Matrix</InfoTip>}
            subtitle="Pearson correlation between asset returns"
            action={
              <Button variant="secondary" size="sm" onClick={runCorrelation} loading={corrLoading}>
                Calculate
              </Button>
            }
          />
          {corrLoading && <Loading text="Calculating correlations..." />}
          {correlation && !corrLoading && (
            <div className="overflow-x-auto -mx-5">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-surface-700/40 dark:border-surface-700/40 border-surface-200">
                    <th className="text-left text-2xs uppercase tracking-wider text-surface-500 font-medium px-4 py-2" />
                    {correlation.tickers.map((t) => (
                      <th key={t} className="text-center text-2xs uppercase tracking-wider text-surface-500 font-medium px-3 py-2">
                        {t.replace('.NS', '')}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {correlation.tickers.map((rowT, i) => (
                    <tr key={rowT} className="border-b border-surface-700/20 dark:border-surface-700/20 border-surface-100">
                      <td className="px-4 py-2.5 font-semibold text-surface-200 dark:text-surface-200 text-surface-700 text-2xs uppercase tracking-wider">
                        {rowT.replace('.NS', '')}
                      </td>
                      {correlation.matrix[i].map((v, j) => (
                        <td key={j} className="px-3 py-2.5 text-center">
                          <span className={`inline-block rounded px-2 py-0.5 text-xs tabular-nums font-medium ${i === j ? 'text-surface-600' : corrColor(v)}`}>
                            {v.toFixed(2)}
                          </span>
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      )}

      {/* Empty state */}
      {!holdings.length && (
        <Card hover={false}>
          <div className="text-center py-16">
            <p className="text-surface-500 text-sm">Add holdings in the Portfolio tab to run performance analysis.</p>
          </div>
        </Card>
      )}
    </div>
  );
}
