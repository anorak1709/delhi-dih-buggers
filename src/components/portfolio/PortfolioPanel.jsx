import { useState } from 'react';
import { useApp } from '../../context/AppContext';
import Card, { CardHeader } from '../ui/Card';
import Button from '../ui/Button';
import Input from '../ui/Input';
import { getPrices } from '../../services/api';
import { motion, AnimatePresence } from 'framer-motion';
import { StaggerList, StaggerItem } from '../ui/Motion';

export default function PortfolioPanel() {
  const { holdings, addHolding, removeHolding, updateHolding, notify } = useApp();
  const [ticker, setTicker] = useState('');
  const [quantity, setQuantity] = useState('');
  const [prices, setPrices] = useState({});
  const [loadingPrices, setLoadingPrices] = useState(false);

  const handleAdd = () => {
    const t = ticker.trim().toUpperCase();
    const q = parseInt(quantity, 10);
    if (!t) return notify('Enter a ticker symbol', 'warning');
    if (!q || q <= 0) return notify('Enter a valid quantity', 'warning');
    addHolding(t, q);
    notify(`Added ${q} shares of ${t}`, 'success');
    setTicker('');
    setQuantity('');
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') handleAdd();
  };

  const fetchPrices = async () => {
    if (!holdings.length) return notify('Add holdings first', 'warning');
    setLoadingPrices(true);
    try {
      const data = await getPrices(holdings.map(h => h.ticker));
      setPrices(data.prices || {});
      notify('Prices updated', 'success');
    } catch (err) {
      notify(err.message, 'error');
    } finally {
      setLoadingPrices(false);
    }
  };

  const totalValue = holdings.reduce((sum, h) => {
    const p = prices[h.ticker];
    return sum + (p ? p.price * h.quantity : 0);
  }, 0);

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <StaggerList className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <StaggerItem>
          <Card hover={false} className="!p-4">
            <p className="text-2xs uppercase tracking-wider text-surface-500 mb-1">Holdings</p>
            <motion.p
              key={holdings.length}
              initial={{ opacity: 0.5, y: -4 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-2xl font-display font-semibold text-surface-100 dark:text-surface-100 text-surface-800 tabular-nums"
            >
              {holdings.length}
            </motion.p>
          </Card>
        </StaggerItem>
        <StaggerItem>
          <Card hover={false} className="!p-4">
            <p className="text-2xs uppercase tracking-wider text-surface-500 mb-1">Tickers</p>
            <p className="text-sm text-surface-300 dark:text-surface-300 text-surface-600 truncate">
              {holdings.length ? holdings.map(h => h.ticker).join(', ') : '—'}
            </p>
          </Card>
        </StaggerItem>
        <StaggerItem>
          <Card hover={false} className="!p-4">
            <p className="text-2xs uppercase tracking-wider text-surface-500 mb-1">Portfolio Value</p>
            <motion.p
              key={totalValue}
              initial={{ opacity: 0.5, y: -4 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-2xl font-display font-semibold text-accent tabular-nums"
            >
              {totalValue > 0 ? `₹${totalValue.toLocaleString('en-IN', { maximumFractionDigits: 0 })}` : '—'}
            </motion.p>
          </Card>
        </StaggerItem>
      </StaggerList>

      {/* Add Holdings */}
      <Card>
        <CardHeader title="Add Holding" subtitle="Enter NSE/BSE ticker symbols" />
        <div className="flex flex-col sm:flex-row gap-3">
          <Input
            label="Ticker"
            placeholder="e.g. RELIANCE.NS"
            value={ticker}
            onChange={(e) => setTicker(e.target.value)}
            onKeyDown={handleKeyDown}
            className="flex-1"
          />
          <Input
            label="Quantity"
            type="number"
            placeholder="e.g. 50"
            min="1"
            value={quantity}
            onChange={(e) => setQuantity(e.target.value)}
            onKeyDown={handleKeyDown}
            className="w-full sm:w-32"
          />
          <div className="flex items-end">
            <Button onClick={handleAdd} size="md">
              Add
            </Button>
          </div>
        </div>
      </Card>

      {/* Holdings Table */}
      <Card>
        <CardHeader
          title="Current Holdings"
          action={
            <Button variant="secondary" size="sm" onClick={fetchPrices} loading={loadingPrices}>
              Refresh Prices
            </Button>
          }
        />
        {holdings.length === 0 ? (
          <div className="text-center py-12">
            <p className="text-surface-500 text-sm">No holdings yet. Add a ticker above to get started.</p>
          </div>
        ) : (
          <div className="overflow-x-auto -mx-5">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-surface-700/40 dark:border-surface-700/40 border-surface-200">
                  <th className="text-left text-2xs uppercase tracking-wider text-surface-500 font-medium px-5 py-2.5">Ticker</th>
                  <th className="text-right text-2xs uppercase tracking-wider text-surface-500 font-medium px-5 py-2.5">Qty</th>
                  <th className="text-right text-2xs uppercase tracking-wider text-surface-500 font-medium px-5 py-2.5">Price</th>
                  <th className="text-right text-2xs uppercase tracking-wider text-surface-500 font-medium px-5 py-2.5">Change</th>
                  <th className="text-right text-2xs uppercase tracking-wider text-surface-500 font-medium px-5 py-2.5">Value</th>
                  <th className="text-right text-2xs uppercase tracking-wider text-surface-500 font-medium px-5 py-2.5"></th>
                </tr>
              </thead>
              <AnimatePresence mode="popLayout">
                <tbody>
                  {holdings.map((h, i) => {
                    const p = prices[h.ticker];
                    const value = p ? p.price * h.quantity : null;
                    const changeColor = p?.change >= 0 ? 'text-up' : 'text-down';
                    return (
                      <motion.tr
                        key={h.ticker}
                        layout
                        initial={{ opacity: 0, x: -16 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: 24, height: 0 }}
                        transition={{ type: 'spring', stiffness: 300, damping: 24, delay: i * 0.03 }}
                        className="border-b border-surface-700/20 dark:border-surface-700/20 border-surface-100 hover:bg-surface-800/30 dark:hover:bg-surface-800/30 hover:bg-surface-50 transition-colors"
                      >
                        <td className="px-5 py-3 font-semibold text-surface-100 dark:text-surface-100 text-surface-800">
                          {h.ticker}
                        </td>
                        <td className="px-5 py-3 text-right tabular-nums">
                          <input
                            type="number"
                            value={h.quantity}
                            min="1"
                            onChange={(e) => updateHolding(h.ticker, parseInt(e.target.value, 10) || 0)}
                            className="w-20 text-right bg-transparent border-b border-transparent hover:border-surface-600 focus:border-accent/50 focus:outline-none py-0.5 tabular-nums text-surface-200 dark:text-surface-200 text-surface-700"
                          />
                        </td>
                        <td className="px-5 py-3 text-right tabular-nums text-surface-300 dark:text-surface-300 text-surface-600">
                          {p ? `₹${p.price.toLocaleString('en-IN', { minimumFractionDigits: 2 })}` : '—'}
                        </td>
                        <td className={`px-5 py-3 text-right tabular-nums ${p ? changeColor : 'text-surface-500'}`}>
                          {p ? `${p.change >= 0 ? '+' : ''}${p.change.toFixed(2)}%` : '—'}
                        </td>
                        <td className="px-5 py-3 text-right tabular-nums font-medium text-surface-100 dark:text-surface-100 text-surface-800">
                          {value ? `₹${value.toLocaleString('en-IN', { maximumFractionDigits: 0 })}` : '—'}
                        </td>
                        <td className="px-5 py-3 text-right">
                          <motion.button
                            whileHover={{ scale: 1.2 }}
                            whileTap={{ scale: 0.85 }}
                            onClick={() => { removeHolding(h.ticker); notify(`Removed ${h.ticker}`, 'info'); }}
                            className="text-surface-600 hover:text-down transition-colors cursor-pointer"
                            title="Remove"
                          >
                            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}>
                              <path strokeLinecap="round" strokeLinejoin="round" d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0" />
                            </svg>
                          </motion.button>
                        </td>
                      </motion.tr>
                    );
                  })}
                </tbody>
              </AnimatePresence>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}
