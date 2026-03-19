import { useState } from 'react';
import { useApp } from '../../context/AppContext';
import Card, { CardHeader } from '../ui/Card';
import Button from '../ui/Button';
import Input from '../ui/Input';
import { Select } from '../ui/Input';
import Loading from '../ui/Loading';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler } from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler);

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

function StatCard({ label, value, color = 'text-surface-100 dark:text-surface-100 text-surface-800', sub }) {
  return (
    <Card hover={false} className="!p-4">
      <p className="text-2xs uppercase tracking-wider text-surface-500 mb-1">{label}</p>
      <p className={`text-xl font-display font-semibold tabular-nums ${color}`}>{value}</p>
      {sub && <p className="text-2xs text-surface-600 mt-1">{sub}</p>}
    </Card>
  );
}

function SuccessGauge({ rate }) {
  const pct = Math.round(rate * 100);
  const color = pct >= 80 ? 'text-up' : pct >= 50 ? 'text-warn' : 'text-down';
  const bgColor = pct >= 80 ? 'bg-up' : pct >= 50 ? 'bg-warn' : 'bg-down';
  return (
    <div className="flex flex-col items-center">
      <div className="relative w-28 h-28">
        <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
          <circle cx="50" cy="50" r="42" fill="none" stroke="currentColor" strokeWidth="6"
            className="text-surface-700/30 dark:text-surface-700/30 text-surface-200" />
          <circle cx="50" cy="50" r="42" fill="none" strokeWidth="6"
            className={bgColor}
            strokeLinecap="round"
            strokeDasharray={`${pct * 2.64} 264`}
            style={{ transition: 'stroke-dasharray 1s ease' }}
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className={`text-2xl font-display font-bold tabular-nums ${color}`}>{pct}%</span>
        </div>
      </div>
      <p className="text-2xs text-surface-500 mt-2 uppercase tracking-wider">Success Rate</p>
    </div>
  );
}

export default function RetirementPanel() {
  const { notify } = useApp();
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const [currentSavings, setCurrentSavings] = useState('500000');
  const [monthlyContribution, setMonthlyContribution] = useState('20000');
  const [yearsToRetirement, setYearsToRetirement] = useState('25');
  const [yearsInRetirement, setYearsInRetirement] = useState('30');
  const [annualSpending, setAnnualSpending] = useState('600000');
  const [inflationRate, setInflationRate] = useState('0.06');
  const [riskTolerance, setRiskTolerance] = useState('moderate');

  const calculate = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${BASE_URL}/api/retirement/calculate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          current_savings: parseFloat(currentSavings),
          monthly_contribution: parseFloat(monthlyContribution),
          years_to_retirement: parseInt(yearsToRetirement, 10),
          years_in_retirement: parseInt(yearsInRetirement, 10),
          annual_spending: parseFloat(annualSpending),
          inflation_rate: parseFloat(inflationRate),
          risk_tolerance: riskTolerance,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Calculation failed');
      setResult(data);
      notify('Retirement plan calculated', 'success');
    } catch (err) {
      notify(err.message, 'error');
    } finally {
      setLoading(false);
    }
  };

  const fmtCurrency = (v) => '₹' + Math.round(v).toLocaleString('en-IN');

  const coneChart = result?.cone
    ? {
        labels: result.cone.map((c) => `Year ${c.year}`),
        datasets: [
          {
            label: '90th Percentile',
            data: result.cone.map((c) => c.p90),
            borderColor: 'rgba(63,185,80,0.3)',
            backgroundColor: 'rgba(63,185,80,0.05)',
            borderWidth: 1,
            pointRadius: 0,
            fill: '+1',
            tension: 0.3,
          },
          {
            label: '75th Percentile',
            data: result.cone.map((c) => c.p75),
            borderColor: 'rgba(63,185,80,0.5)',
            backgroundColor: 'rgba(63,185,80,0.08)',
            borderWidth: 1,
            pointRadius: 0,
            fill: '+1',
            tension: 0.3,
          },
          {
            label: 'Median (50th)',
            data: result.cone.map((c) => c.p50),
            borderColor: '#c9985a',
            backgroundColor: 'rgba(201,152,90,0.1)',
            borderWidth: 2,
            pointRadius: 0,
            fill: '+1',
            tension: 0.3,
          },
          {
            label: '25th Percentile',
            data: result.cone.map((c) => c.p25),
            borderColor: 'rgba(248,81,73,0.5)',
            backgroundColor: 'rgba(248,81,73,0.05)',
            borderWidth: 1,
            pointRadius: 0,
            fill: '+1',
            tension: 0.3,
          },
          {
            label: '10th Percentile',
            data: result.cone.map((c) => c.p10),
            borderColor: 'rgba(248,81,73,0.3)',
            backgroundColor: 'transparent',
            borderWidth: 1,
            pointRadius: 0,
            tension: 0.3,
          },
        ],
      }
    : null;

  const coneOptions = {
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
          label: (ctx) => `${ctx.dataset.label}: ${fmtCurrency(ctx.parsed.y)}`,
        },
      },
    },
    scales: {
      x: { grid: { color: 'rgba(33,38,45,0.4)' }, ticks: { color: '#656d76', font: { size: 10 }, maxTicksLimit: 10 } },
      y: {
        grid: { color: 'rgba(33,38,45,0.4)' },
        ticks: {
          color: '#656d76',
          font: { size: 10 },
          callback: (v) => {
            if (v >= 10000000) return '₹' + (v / 10000000).toFixed(1) + 'Cr';
            if (v >= 100000) return '₹' + (v / 100000).toFixed(1) + 'L';
            return '₹' + v.toLocaleString();
          },
        },
      },
    },
  };

  return (
    <div className="space-y-6">
      {/* Input Form */}
      <Card>
        <CardHeader title="Retirement Planner" subtitle="Monte Carlo simulation-powered retirement planning" />
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          <Input
            label="Current Savings (₹)"
            type="number"
            value={currentSavings}
            onChange={(e) => setCurrentSavings(e.target.value)}
            min="0"
          />
          <Input
            label="Monthly Contribution (₹)"
            type="number"
            value={monthlyContribution}
            onChange={(e) => setMonthlyContribution(e.target.value)}
            min="0"
          />
          <Input
            label="Years to Retirement"
            type="number"
            value={yearsToRetirement}
            onChange={(e) => setYearsToRetirement(e.target.value)}
            min="1"
            max="60"
          />
          <Input
            label="Years in Retirement"
            type="number"
            value={yearsInRetirement}
            onChange={(e) => setYearsInRetirement(e.target.value)}
            min="1"
            max="60"
          />
          <Input
            label="Annual Spending (₹)"
            type="number"
            value={annualSpending}
            onChange={(e) => setAnnualSpending(e.target.value)}
            min="0"
          />
          <Input
            label="Inflation Rate"
            type="number"
            value={inflationRate}
            onChange={(e) => setInflationRate(e.target.value)}
            min="0"
            max="0.2"
            step="0.01"
          />
        </div>
        <div className="flex flex-col sm:flex-row gap-3 items-end mt-4">
          <Select
            label="Risk Tolerance"
            value={riskTolerance}
            onChange={(e) => setRiskTolerance(e.target.value)}
            className="flex-1"
          >
            <option value="conservative">Conservative (5% return, 8% vol)</option>
            <option value="moderate">Moderate (7% return, 12% vol)</option>
            <option value="aggressive">Aggressive (9% return, 18% vol)</option>
          </Select>
          <Button onClick={calculate} loading={loading}>
            Calculate
          </Button>
        </div>
      </Card>

      {loading && <Loading text="Running 5,000 Monte Carlo simulations..." />}

      {result && !loading && (
        <>
          {/* Key Outcomes */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <StatCard label="Target Amount" value={fmtCurrency(result.target_amount)} color="text-accent" />
            <StatCard label="Median Outcome" value={fmtCurrency(result.median)} color="text-surface-100 dark:text-surface-100 text-surface-800" />
            <StatCard
              label="Safe Withdrawal"
              value={(result.safe_withdrawal_rate * 100).toFixed(1) + '%'}
              color="text-info"
              sub={`${fmtCurrency(result.safe_annual_income)}/year`}
            />
            <StatCard
              label="Sustainability"
              value={(result.sustainability_rate * 100).toFixed(0) + '%'}
              color={result.sustainability_rate >= 0.8 ? 'text-up' : 'text-warn'}
              sub="Money lasts through retirement"
            />
          </div>

          {/* Success Rate + Percentile Range */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card hover={false} className="flex items-center justify-center !py-8">
              <SuccessGauge rate={result.success_rate} />
            </Card>
            <Card hover={false} className="md:col-span-2">
              <CardHeader title="Outcome Distribution" subtitle="Range of possible outcomes at retirement" />
              <div className="space-y-3">
                {[
                  { label: '90th Percentile (Optimistic)', value: result.percentile_90, color: 'bg-up' },
                  { label: '75th Percentile', value: result.percentile_75, color: 'bg-up/60' },
                  { label: 'Median (50th)', value: result.median, color: 'bg-accent' },
                  { label: '25th Percentile', value: result.percentile_25, color: 'bg-down/60' },
                  { label: '10th Percentile (Pessimistic)', value: result.percentile_10, color: 'bg-down' },
                ].map((row) => (
                  <div key={row.label} className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className={`w-2 h-2 rounded-full ${row.color}`} />
                      <span className="text-xs text-surface-400">{row.label}</span>
                    </div>
                    <span className="text-sm font-semibold text-surface-200 dark:text-surface-200 text-surface-700 tabular-nums">
                      {fmtCurrency(row.value)}
                    </span>
                  </div>
                ))}
              </div>
            </Card>
          </div>

          {/* Probability Cone */}
          {coneChart && (
            <Card>
              <CardHeader title="Probability Cone" subtitle="Projected portfolio growth with confidence bands" />
              <div className="h-72 sm:h-80">
                <Line data={coneChart} options={coneOptions} />
              </div>
            </Card>
          )}

          {/* Portfolio Allocation */}
          <Card hover={false}>
            <CardHeader title="Recommended Allocation" />
            <div className="flex items-center gap-4">
              <div className="flex-1">
                <div className="flex h-4 rounded-full overflow-hidden mb-3">
                  <div className="bg-accent" style={{ width: `${result.portfolio.stocks}%` }} />
                  <div className="bg-info" style={{ width: `${result.portfolio.bonds}%` }} />
                  <div className="bg-surface-500" style={{ width: `${result.portfolio.cash}%` }} />
                </div>
                <div className="flex gap-6 text-xs">
                  <div className="flex items-center gap-1.5">
                    <div className="w-2.5 h-2.5 rounded-sm bg-accent" />
                    <span className="text-surface-400">Stocks {result.portfolio.stocks}%</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <div className="w-2.5 h-2.5 rounded-sm bg-info" />
                    <span className="text-surface-400">Bonds {result.portfolio.bonds}%</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <div className="w-2.5 h-2.5 rounded-sm bg-surface-500" />
                    <span className="text-surface-400">Cash {result.portfolio.cash}%</span>
                  </div>
                </div>
              </div>
            </div>
          </Card>

          {/* Recommendations */}
          {result.recommendations?.length > 0 && (
            <Card hover={false}>
              <CardHeader title="Recommendations" />
              <div className="space-y-2">
                {result.recommendations.map((rec, i) => (
                  <div key={i} className="flex gap-3 py-2">
                    <span className="shrink-0 mt-0.5 w-5 h-5 flex items-center justify-center rounded-full bg-accent/10 text-accent text-2xs font-semibold">
                      {i + 1}
                    </span>
                    <p className="text-sm text-surface-300 dark:text-surface-300 text-surface-600 leading-relaxed">{rec}</p>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </>
      )}
    </div>
  );
}
