import { useState, useEffect } from 'react';
import { useApp } from '../../context/AppContext';
import Card, { CardHeader } from '../ui/Card';
import Button from '../ui/Button';
import Loading from '../ui/Loading';
import { getPrices, getRiskMetrics } from '../../services/api';
import { motion } from 'framer-motion';

function SummaryCard({ label, value, sub, color = 'text-surface-100 dark:text-surface-100 text-surface-800' }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ type: 'spring', stiffness: 300, damping: 25 }}
    >
      <Card hover={false} className="!p-5">
        <p className="text-2xs uppercase tracking-wider text-surface-500 mb-1.5">{label}</p>
        <p className={`text-2xl font-display font-bold tabular-nums ${color}`}>{value}</p>
        {sub && <p className="text-2xs text-surface-500 mt-1">{sub}</p>}
      </Card>
    </motion.div>
  );
}

function HoldingRow({ ticker, price, change, changePct, value }) {
  const isUp = change >= 0;
  return (
    <motion.div
      initial={{ opacity: 0, x: -8 }}
      animate={{ opacity: 1, x: 0 }}
      className="flex items-center justify-between py-3 border-b border-surface-700/20 dark:border-surface-700/20 border-surface-100 last:border-0"
    >
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-lg bg-accent/10 flex items-center justify-center">
          <span className="text-xs font-bold text-accent">{ticker.replace('.NS', '').slice(0, 3)}</span>
        </div>
        <div>
          <p className="text-sm font-semibold text-surface-100 dark:text-surface-100 text-surface-800">{ticker.replace('.NS', '')}</p>
          <p className="text-2xs text-surface-500 tabular-nums">{price ? `$${price.toFixed(2)}` : '—'}</p>
        </div>
      </div>
      <div className="text-right">
        <p className="text-sm font-semibold text-surface-200 dark:text-surface-200 text-surface-700 tabular-nums">
          {value ? `$${value.toLocaleString(undefined, { maximumFractionDigits: 0 })}` : '—'}
        </p>
        <p className={`text-2xs font-medium tabular-nums ${isUp ? 'text-up' : 'text-down'}`}>
          {isUp ? '+' : ''}{changePct ? changePct.toFixed(2) : '0.00'}%
        </p>
      </div>
    </motion.div>
  );
}

export default function DashboardPanel() {
  const { holdings, tickers, setActiveTab, notify } = useApp();
  const [loading, setLoading] = useState(false);
  const [priceData, setPriceData] = useState(null);
  const [riskData, setRiskData] = useState(null);

  useEffect(() => {
    if (tickers.length === 0) return;
    const fetchData = async () => {
      setLoading(true);
      try {
        const [prices, risk] = await Promise.allSettled([
          getPrices(tickers),
          tickers.length >= 2 ? getRiskMetrics(tickers) : Promise.resolve(null),
        ]);
        if (prices.status === 'fulfilled') setPriceData(prices.value);
        if (risk.status === 'fulfilled') setRiskData(risk.value);
      } catch (err) {
        notify(err.message, 'error');
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [tickers.join(',')]);

  // Calculate totals
  let totalValue = 0;
  let totalChange = 0;
  const holdingDetails = holdings.map(h => {
    const p = priceData?.[h.ticker];
    const price = p?.price || 0;
    const change = p?.daily_change || 0;
    const changePct = p?.daily_change_pct || 0;
    const value = price * h.quantity;
    totalValue += value;
    totalChange += change * h.quantity;
    return { ...h, price, change, changePct, value };
  });
  const totalChangePct = totalValue > 0 ? (totalChange / (totalValue - totalChange)) * 100 : 0;

  const fmtCurrency = (v) => '$' + v.toLocaleString(undefined, { maximumFractionDigits: 0 });

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <SummaryCard
          label="Total Portfolio Value"
          value={fmtCurrency(totalValue)}
          color="text-accent"
        />
        <SummaryCard
          label="Daily Change"
          value={`${totalChange >= 0 ? '+' : ''}${totalChangePct.toFixed(2)}%`}
          sub={`${totalChange >= 0 ? '+' : ''}${fmtCurrency(Math.abs(totalChange))}`}
          color={totalChange >= 0 ? 'text-up' : 'text-down'}
        />
        <SummaryCard
          label="Value at Risk (95%)"
          value={riskData?.var_95 ? `${(riskData.var_95 * 100).toFixed(2)}%` : '—'}
          sub="Daily parametric VaR"
          color="text-down"
        />
        <SummaryCard
          label="Sharpe Ratio"
          value={riskData?.sharpe ? riskData.sharpe.toFixed(2) : '—'}
          sub="Risk-adjusted return"
          color="text-info"
        />
      </div>

      {/* Holdings List */}
      <Card>
        <CardHeader title="Holdings" subtitle={`${holdings.length} positions`} />
        {loading && <Loading text="Fetching live data..." />}
        {!loading && holdings.length > 0 && (
          <div>
            {holdingDetails.map(h => (
              <HoldingRow
                key={h.ticker}
                ticker={h.ticker}
                price={h.price}
                change={h.change}
                changePct={h.changePct}
                value={h.value}
              />
            ))}
          </div>
        )}
        {holdings.length === 0 && !loading && (
          <div className="text-center py-12">
            <p className="text-surface-500 text-sm mb-4">No holdings yet. Add some stocks to get started.</p>
            <Button onClick={() => setActiveTab('portfolio')}>Go to Portfolio</Button>
          </div>
        )}
      </Card>

      {/* Quick Actions */}
      <Card hover={false}>
        <CardHeader title="Quick Actions" />
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {[
            { label: 'Optimize', tab: 'optimize', desc: 'Run Monte Carlo' },
            { label: 'Risk Analysis', tab: 'risk', desc: 'VaR & stress tests' },
            { label: 'Backtest', tab: 'backtest', desc: 'Historical simulation' },
            { label: 'Options', tab: 'options', desc: 'Black-Scholes pricing' },
          ].map(action => (
            <button
              key={action.tab}
              onClick={() => setActiveTab(action.tab)}
              className="p-3 rounded-lg border border-surface-700/30 dark:border-surface-700/30 border-surface-200 hover:border-accent/30 transition-colors text-left cursor-pointer group"
            >
              <p className="text-sm font-semibold text-surface-200 dark:text-surface-200 text-surface-700 group-hover:text-accent transition-colors">
                {action.label}
              </p>
              <p className="text-2xs text-surface-500 mt-0.5">{action.desc}</p>
            </button>
          ))}
        </div>
      </Card>
    </div>
  );
}
