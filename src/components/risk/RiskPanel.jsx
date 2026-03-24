import { useState } from 'react';
import { useApp } from '../../context/AppContext';
import Card, { CardHeader } from '../ui/Card';
import Button from '../ui/Button';
import Input from '../ui/Input';
import { Select } from '../ui/Input';
import Loading from '../ui/Loading';
import { getRiskMetrics, getRolling, getScenario, getStress } from '../../services/api';
import InfoTip from '../ui/Tooltip';
import { TOOLTIPS } from '../../constants/tooltips';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend } from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend);

function RiskMetricCard({ label, value, description, color = 'text-surface-100 dark:text-surface-100 text-surface-800' }) {
  return (
    <Card hover={false} className="!p-4">
      <p className="text-2xs uppercase tracking-wider text-surface-500 mb-1">{label}</p>
      <p className={`text-2xl font-display font-semibold tabular-nums ${color}`}>{value}</p>
      {description && <p className="text-2xs text-surface-600 mt-1.5">{description}</p>}
    </Card>
  );
}

export default function RiskPanel() {
  const { tickers, holdings, notify } = useApp();
  const [loading, setLoading] = useState(false);
  const [metrics, setMetrics] = useState(null);
  const [rollingData, setRollingData] = useState(null);
  const [rollingTicker, setRollingTicker] = useState('');
  const [rollingLoading, setRollingLoading] = useState(false);
  const [scenarioTicker, setScenarioTicker] = useState('');
  const [scenarioType, setScenarioType] = useState('crash');
  const [scenarioResult, setScenarioResult] = useState(null);
  const [scenarioLoading, setScenarioLoading] = useState(false);
  const [stressTicker, setStressTicker] = useState('');
  const [stressResult, setStressResult] = useState(null);
  const [stressLoading, setStressLoading] = useState(false);

  const fetchMetrics = async () => {
    if (!tickers.length) return notify('Add holdings first', 'warning');
    setLoading(true);
    try {
      const data = await getRiskMetrics(tickers);
      setMetrics(data);
      notify('Risk metrics calculated', 'success');
    } catch (err) {
      notify(err.message, 'error');
    } finally {
      setLoading(false);
    }
  };

  const fetchRolling = async () => {
    const t = rollingTicker || tickers[0];
    if (!t) return notify('Select a ticker', 'warning');
    setRollingLoading(true);
    try {
      const data = await getRolling(t);
      setRollingData(data);
    } catch (err) {
      notify(err.message, 'error');
    } finally {
      setRollingLoading(false);
    }
  };

  const runScenario = async () => {
    const t = scenarioTicker || tickers[0];
    if (!t) return notify('Select a ticker', 'warning');
    setScenarioLoading(true);
    try {
      const data = await getScenario(t, scenarioType);
      setScenarioResult(data);
    } catch (err) {
      notify(err.message, 'error');
    } finally {
      setScenarioLoading(false);
    }
  };

  const runStress = async () => {
    const t = stressTicker || tickers[0];
    if (!t) return notify('Select a ticker', 'warning');
    setStressLoading(true);
    try {
      const data = await getStress(t);
      setStressResult(data);
    } catch (err) {
      notify(err.message, 'error');
    } finally {
      setStressLoading(false);
    }
  };

  const pct = (v) => (v * 100).toFixed(2) + '%';

  const rollingChartData = rollingData
    ? {
        labels: rollingData.dates,
        datasets: [
          {
            label: '60-Day Rolling Sharpe',
            data: rollingData.values,
            borderColor: '#c9985a',
            backgroundColor: 'transparent',
            borderWidth: 1.5,
            pointRadius: 0,
            tension: 0.3,
          },
          {
            label: 'Zero Line',
            data: rollingData.values.map(() => 0),
            borderColor: '#484f58',
            borderWidth: 1,
            borderDash: [4, 4],
            pointRadius: 0,
          },
        ],
      }
    : null;

  const rollingChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: '#8b949e', font: { size: 11 }, usePointStyle: true, pointStyle: 'line' } },
      tooltip: {
        backgroundColor: '#161b22',
        borderColor: '#21262d',
        borderWidth: 1,
        titleColor: '#f6f8fa',
        bodyColor: '#afb8c1',
        padding: 10,
      },
    },
    scales: {
      x: { grid: { color: 'rgba(33,38,45,0.4)' }, ticks: { color: '#656d76', maxTicksLimit: 8, font: { size: 10 } } },
      y: { grid: { color: 'rgba(33,38,45,0.4)' }, ticks: { color: '#656d76', font: { size: 10 } } },
    },
  };

  return (
    <div className="space-y-6">
      {/* Core Risk Metrics */}
      <Card>
        <CardHeader
          title="Risk Metrics"
          subtitle="Value at Risk, Conditional VaR, Beta & Alpha"
          action={
            <Button onClick={fetchMetrics} loading={loading} size="sm">
              Calculate
            </Button>
          }
        />
        {loading && <Loading text="Calculating risk metrics..." />}
        {metrics && !loading && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-2">
            <RiskMetricCard
              label={<InfoTip text={TOOLTIPS.var_95}>VaR (95%)</InfoTip>}
              value={pct(metrics.var)}
              description="Max daily loss at 95% confidence"
              color="text-down"
            />
            <RiskMetricCard
              label={<InfoTip text={TOOLTIPS.cvar_95}>CVaR (95%)</InfoTip>}
              value={pct(metrics.cvar)}
              description="Expected loss beyond VaR"
              color="text-down"
            />
            <RiskMetricCard
              label={<InfoTip text={TOOLTIPS.beta}>Beta</InfoTip>}
              value={metrics.beta?.toFixed(3)}
              description="Market sensitivity"
              color={metrics.beta > 1 ? 'text-warn' : 'text-up'}
            />
            <RiskMetricCard
              label={<InfoTip text={TOOLTIPS.alpha}>Jensen's Alpha</InfoTip>}
              value={pct(metrics.alpha)}
              description="Excess return over CAPM"
              color={metrics.alpha >= 0 ? 'text-up' : 'text-down'}
            />
          </div>
        )}
      </Card>

      {/* Rolling Sharpe */}
      <Card>
        <CardHeader title={<InfoTip text={TOOLTIPS.rolling_sharpe}>Rolling Sharpe Ratio</InfoTip>} subtitle="60-day rolling window" />
        <div className="flex flex-col sm:flex-row gap-3 items-end mb-4">
          <Select
            label="Ticker"
            value={rollingTicker}
            onChange={(e) => setRollingTicker(e.target.value)}
            className="flex-1"
          >
            <option value="">Select ticker</option>
            {tickers.map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </Select>
          <Button variant="secondary" size="sm" onClick={fetchRolling} loading={rollingLoading}>
            Fetch
          </Button>
        </div>
        {rollingLoading && <Loading text="Loading rolling Sharpe..." />}
        {rollingChartData && !rollingLoading && (
          <div className="h-56 sm:h-64">
            <Line data={rollingChartData} options={rollingChartOptions} />
          </div>
        )}
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Scenario Analysis */}
        <Card>
          <CardHeader title="Scenario Analysis" subtitle="Simulate market conditions" />
          <div className="space-y-3">
            <Select label="Ticker" value={scenarioTicker} onChange={(e) => setScenarioTicker(e.target.value)}>
              <option value="">Select ticker</option>
              {tickers.map((t) => (
                <option key={t} value={t}>{t}</option>
              ))}
            </Select>
            <Select label="Scenario" value={scenarioType} onChange={(e) => setScenarioType(e.target.value)}>
              <option value="crash">Market Crash (−50%)</option>
              <option value="bull">Bull Market (+50%)</option>
              <option value="flat">Flat Market</option>
            </Select>
            <Button variant="secondary" size="sm" onClick={runScenario} loading={scenarioLoading} className="w-full">
              Run Scenario
            </Button>
          </div>
          {scenarioResult && !scenarioLoading && (
            <div className="mt-4 pt-4 border-t border-surface-700/40 dark:border-surface-700/40 border-surface-200 grid grid-cols-2 gap-3">
              <div>
                <p className="text-2xs uppercase tracking-wider text-surface-500 mb-0.5">Mean Return</p>
                <p className={`text-lg font-display font-semibold tabular-nums ${scenarioResult.mean_return >= 0 ? 'text-up' : 'text-down'}`}>
                  {pct(scenarioResult.mean_return)}
                </p>
              </div>
              <div>
                <p className="text-2xs uppercase tracking-wider text-surface-500 mb-0.5">Volatility</p>
                <p className="text-lg font-display font-semibold tabular-nums text-surface-200 dark:text-surface-200 text-surface-700">
                  {pct(scenarioResult.volatility)}
                </p>
              </div>
            </div>
          )}
        </Card>

        {/* Stress Test */}
        <Card>
          <CardHeader title="Stress Test" subtitle="Historical worst-case analysis" />
          <div className="space-y-3">
            <Select label="Ticker" value={stressTicker} onChange={(e) => setStressTicker(e.target.value)}>
              <option value="">Select ticker</option>
              {tickers.map((t) => (
                <option key={t} value={t}>{t}</option>
              ))}
            </Select>
            <Button variant="secondary" size="sm" onClick={runStress} loading={stressLoading} className="w-full">
              Run Stress Test
            </Button>
          </div>
          {stressResult && !stressLoading && (
            <div className="mt-4 pt-4 border-t border-surface-700/40 dark:border-surface-700/40 border-surface-200 space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-xs text-surface-500">Worst Single Day</span>
                <span className="text-sm font-semibold text-down tabular-nums">{pct(stressResult.worst_day)}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-surface-500">1st Percentile</span>
                <span className="text-sm font-semibold text-down tabular-nums">{pct(stressResult.percentile_1)}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-surface-500">5th Percentile</span>
                <span className="text-sm font-semibold text-warn tabular-nums">{pct(stressResult.percentile_5)}</span>
              </div>
            </div>
          )}
        </Card>
      </div>

      {!tickers.length && (
        <Card hover={false}>
          <div className="text-center py-16">
            <p className="text-surface-500 text-sm">Add holdings in the Portfolio tab to analyze risk.</p>
          </div>
        </Card>
      )}
    </div>
  );
}
