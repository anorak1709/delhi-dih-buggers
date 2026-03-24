import { useState, useEffect } from 'react';
import { useApp } from '../../context/AppContext';
import Card, { CardHeader } from '../ui/Card';
import Button from '../ui/Button';
import Input from '../ui/Input';
import { Select } from '../ui/Input';
import Loading from '../ui/Loading';
import { getOptionsPrice, getOptionsGreeksCurves, getOptionsChain, getImpliedVol, getPrices, getVolSurface } from '../../services/api';
import InfoTip from '../ui/Tooltip';
import { TOOLTIPS } from '../../constants/tooltips';
import Plot from 'react-plotly.js';
import { motion, AnimatePresence } from 'framer-motion';
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

const chartColors = {
  accent: '#c9985a',
  blue: '#58a6ff',
  green: 'rgba(63, 185, 80, 0.8)',
  red: 'rgba(248, 81, 73, 0.8)',
};

function GreekCard({ label, value, sub }) {
  return (
    <div className="rounded-lg border border-surface-700/30 dark:border-surface-700/30 border-surface-200 p-3">
      <p className="text-2xs uppercase tracking-wider text-surface-500 mb-0.5">{label}</p>
      <p className="text-lg font-display font-semibold text-surface-100 dark:text-surface-100 text-surface-800 tabular-nums">
        {typeof value === 'number' ? value.toFixed(4) : value}
      </p>
      {sub && <p className="text-2xs text-surface-600 mt-0.5">{sub}</p>}
    </div>
  );
}

function makeLineOptions(title) {
  return {
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
      },
    },
    scales: {
      x: { grid: { color: 'rgba(33,38,45,0.4)' }, ticks: { color: '#656d76', font: { size: 9 }, maxTicksLimit: 8 } },
      y: { grid: { color: 'rgba(33,38,45,0.4)' }, ticks: { color: '#656d76', font: { size: 9 } } },
    },
  };
}

export default function OptionsPanel() {
  const { tickers, notify } = useApp();

  // Calculator state
  const [spot, setSpot] = useState('');
  const [strike, setStrike] = useState('');
  const [expDays, setExpDays] = useState('30');
  const [rfRate, setRfRate] = useState('6.5');
  const [vol, setVol] = useState('25');
  const [optType, setOptType] = useState('call');
  const [calcResult, setCalcResult] = useState(null);
  const [calcLoading, setCalcLoading] = useState(false);
  const [curves, setCurves] = useState(null);
  const [curvesLoading, setCurvesLoading] = useState(false);

  // Options chain state
  const [chainTicker, setChainTicker] = useState('');
  const [chainData, setChainData] = useState(null);
  const [chainLoading, setChainLoading] = useState(false);
  const [chainExpiry, setChainExpiry] = useState('');
  const [chainTab, setChainTab] = useState('calls');

  // Volatility surface state
  const [surfaceTicker, setSurfaceTicker] = useState('');
  const [surfaceData, setSurfaceData] = useState(null);
  const [surfaceLoading, setSurfaceLoading] = useState(false);

  // Autofill ticker
  const [autoTicker, setAutoTicker] = useState('');

  const loadSurface = async () => {
    if (!surfaceTicker) return notify('Enter a ticker', 'warning');
    setSurfaceLoading(true);
    try {
      const data = await getVolSurface(surfaceTicker);
      setSurfaceData(data);
      notify('Volatility surface loaded', 'success');
    } catch (err) {
      notify(err.message, 'error');
    } finally {
      setSurfaceLoading(false);
    }
  };

  const autoFill = async () => {
    if (!autoTicker) return;
    try {
      const data = await getPrices([autoTicker]);
      if (data?.[autoTicker]) {
        setSpot(String(data[autoTicker].current_price || ''));
        setStrike(String(Math.round(data[autoTicker].current_price || 0)));
        notify(`Filled spot price for ${autoTicker}`, 'success');
      }
    } catch (err) {
      notify(err.message, 'error');
    }
  };

  const calculate = async () => {
    if (!spot || !strike) return notify('Enter spot and strike prices', 'warning');
    setCalcLoading(true);
    try {
      const params = {
        spot_price: parseFloat(spot),
        strike_price: parseFloat(strike),
        time_to_expiry: parseInt(expDays, 10) / 365,
        risk_free_rate: parseFloat(rfRate) / 100,
        volatility: parseFloat(vol) / 100,
        option_type: optType,
      };
      const data = await getOptionsPrice(params);
      setCalcResult(data);
      notify('Option priced', 'success');

      // Auto-fetch curves
      setCurvesLoading(true);
      try {
        const c = await getOptionsGreeksCurves(params);
        setCurves(c);
      } catch (_) {}
      setCurvesLoading(false);
    } catch (err) {
      notify(err.message, 'error');
    } finally {
      setCalcLoading(false);
    }
  };

  const loadChain = async (expiry) => {
    if (!chainTicker) return notify('Enter a ticker symbol', 'warning');
    setChainLoading(true);
    try {
      const data = await getOptionsChain(chainTicker, expiry || undefined);
      setChainData(data);
      setChainExpiry(data.selected_expiry);
    } catch (err) {
      notify(err.message, 'error');
    } finally {
      setChainLoading(false);
    }
  };

  const spotCurveChart = curves?.spot_curve
    ? {
        labels: curves.spot_curve.spots.map((s) => s.toFixed(0)),
        datasets: [
          {
            label: 'Option Price',
            data: curves.spot_curve.prices,
            borderColor: chartColors.accent,
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.3,
            yAxisID: 'y',
          },
          {
            label: 'Delta',
            data: curves.spot_curve.deltas,
            borderColor: chartColors.blue,
            borderWidth: 1.5,
            pointRadius: 0,
            tension: 0.3,
            borderDash: [4, 2],
            yAxisID: 'y1',
          },
        ],
      }
    : null;

  const spotCurveOptions = {
    ...makeLineOptions(),
    scales: {
      ...makeLineOptions().scales,
      y: { ...makeLineOptions().scales.y, position: 'left', title: { display: true, text: 'Price', color: '#656d76', font: { size: 9 } } },
      y1: { position: 'right', grid: { drawOnChartArea: false }, ticks: { color: chartColors.blue, font: { size: 9 } }, title: { display: true, text: 'Delta', color: chartColors.blue, font: { size: 9 } } },
    },
  };

  const timeCurveChart = curves?.time_curve
    ? {
        labels: curves.time_curve.times.map((t) => Math.round(t * 365) + 'd'),
        datasets: [
          {
            label: 'Price (Time Decay)',
            data: curves.time_curve.prices,
            borderColor: chartColors.red,
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.3,
            fill: true,
            backgroundColor: 'rgba(248, 81, 73, 0.05)',
          },
        ],
      }
    : null;

  const volCurveChart = curves?.vol_curve
    ? {
        labels: curves.vol_curve.vols.map((v) => (v * 100).toFixed(0) + '%'),
        datasets: [
          {
            label: 'Price vs Volatility',
            data: curves.vol_curve.prices,
            borderColor: chartColors.green,
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.3,
            fill: true,
            backgroundColor: 'rgba(63, 185, 80, 0.05)',
          },
        ],
      }
    : null;

  return (
    <div className="space-y-6">
      {/* Section 1: Options Calculator */}
      <Card>
        <CardHeader
          title="Black-Scholes Calculator"
          subtitle="Analytical options pricing with Greeks"
        />

        {/* Auto-fill from ticker */}
        <div className="flex gap-2 mb-4">
          <input
            type="text"
            placeholder="Auto-fill from ticker (e.g. AAPL)"
            value={autoTicker}
            onChange={(e) => setAutoTicker(e.target.value.toUpperCase())}
            className="flex-1 rounded-lg px-3 py-2 text-sm bg-surface-850/80 dark:bg-surface-850/80 bg-surface-50 border border-surface-700/60 dark:border-surface-700/60 border-surface-200 text-surface-50 dark:text-surface-50 text-surface-900 placeholder:text-surface-600 focus:outline-none focus:border-accent/50 transition-colors"
          />
          <Button variant="secondary" size="sm" onClick={autoFill}>
            Fetch
          </Button>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          <Input label={<InfoTip text={TOOLTIPS.spot_price}>Spot Price</InfoTip>} type="number" value={spot} onChange={(e) => setSpot(e.target.value)} step="0.01" />
          <Input label={<InfoTip text={TOOLTIPS.strike_price}>Strike Price</InfoTip>} type="number" value={strike} onChange={(e) => setStrike(e.target.value)} step="0.01" />
          <Input label={<InfoTip text={TOOLTIPS.dte}>Days to Expiry</InfoTip>} type="number" value={expDays} onChange={(e) => setExpDays(e.target.value)} min="1" />
          <Input label={<InfoTip text={TOOLTIPS.risk_free_rate}>Risk-Free (%)</InfoTip>} type="number" value={rfRate} onChange={(e) => setRfRate(e.target.value)} step="0.1" />
          <Input label={<InfoTip text={TOOLTIPS.volatility}>Volatility (%)</InfoTip>} type="number" value={vol} onChange={(e) => setVol(e.target.value)} step="0.1" />
          <Select label="Option Type" value={optType} onChange={(e) => setOptType(e.target.value)}>
            <option value="call">Call</option>
            <option value="put">Put</option>
          </Select>
        </div>

        <div className="mt-4">
          <Button onClick={calculate} loading={calcLoading}>
            Price Option
          </Button>
        </div>

        {/* Results */}
        {calcResult && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-5 space-y-4"
          >
            <div className="flex items-baseline gap-3">
              <span className="text-2xs uppercase tracking-wider text-surface-500">Option Price</span>
              <span className="text-3xl font-display font-bold text-accent tabular-nums">
                {calcResult.price?.toFixed(2)}
              </span>
              <span className="text-xs text-surface-500">
                ({calcResult.inputs?.type === 'call' ? 'Call' : 'Put'})
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
              <GreekCard label={<InfoTip text={TOOLTIPS.delta}>Delta (Δ)</InfoTip>} value={calcResult.greeks?.delta} sub="Price sensitivity" />
              <GreekCard label={<InfoTip text={TOOLTIPS.gamma}>Gamma (Γ)</InfoTip>} value={calcResult.greeks?.gamma} sub="Delta sensitivity" />
              <GreekCard label={<InfoTip text={TOOLTIPS.theta}>Theta (Θ)</InfoTip>} value={calcResult.greeks?.theta} sub="Time decay / day" />
              <GreekCard label={<InfoTip text={TOOLTIPS.vega}>Vega (ν)</InfoTip>} value={calcResult.greeks?.vega} sub="Per 1% vol change" />
              <GreekCard label={<InfoTip text={TOOLTIPS.rho}>Rho (ρ)</InfoTip>} value={calcResult.greeks?.rho} sub="Per 1% rate change" />
            </div>
          </motion.div>
        )}
      </Card>

      {/* Section 2: Greeks Visualization */}
      {curves && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {spotCurveChart && (
            <Card>
              <CardHeader title="Price & Delta vs Spot" />
              <div className="h-56">
                <Line data={spotCurveChart} options={spotCurveOptions} />
              </div>
            </Card>
          )}
          {timeCurveChart && (
            <Card>
              <CardHeader title="Time Decay (Theta)" />
              <div className="h-56">
                <Line data={timeCurveChart} options={makeLineOptions()} />
              </div>
            </Card>
          )}
          {volCurveChart && (
            <Card>
              <CardHeader title="Price vs Volatility" />
              <div className="h-56">
                <Line data={volCurveChart} options={makeLineOptions()} />
              </div>
            </Card>
          )}
        </div>
      )}

      {/* Section 3: Options Chain */}
      <Card>
        <CardHeader
          title="Options Chain"
          subtitle="Live options data from market"
        />
        <div className="flex gap-2 mb-4">
          <input
            type="text"
            placeholder="Ticker (e.g. AAPL)"
            value={chainTicker}
            onChange={(e) => setChainTicker(e.target.value.toUpperCase())}
            className="flex-1 rounded-lg px-3 py-2 text-sm bg-surface-850/80 dark:bg-surface-850/80 bg-surface-50 border border-surface-700/60 dark:border-surface-700/60 border-surface-200 text-surface-50 dark:text-surface-50 text-surface-900 placeholder:text-surface-600 focus:outline-none focus:border-accent/50 transition-colors"
          />
          <Button variant="secondary" onClick={() => loadChain()} loading={chainLoading}>
            Load Chain
          </Button>
        </div>

        {chainData && (
          <div>
            {/* Expiry selector */}
            <div className="flex items-center gap-3 mb-4">
              <Select
                label="Expiration"
                value={chainExpiry}
                onChange={(e) => {
                  setChainExpiry(e.target.value);
                  loadChain(e.target.value);
                }}
                className="w-48"
              >
                {chainData.expirations?.map((exp) => (
                  <option key={exp} value={exp}>{exp}</option>
                ))}
              </Select>
              <p className="text-xs text-surface-500 mt-5">
                Spot: <strong className="text-surface-200 dark:text-surface-200 text-surface-700">${chainData.spot_price?.toFixed(2)}</strong>
              </p>
            </div>

            {/* Calls / Puts tabs */}
            <div className="flex gap-2 mb-3">
              {['calls', 'puts'].map((tab) => (
                <button
                  key={tab}
                  onClick={() => setChainTab(tab)}
                  className={`text-xs px-3 py-1.5 rounded-lg font-medium transition-colors cursor-pointer ${
                    chainTab === tab
                      ? 'bg-accent/10 text-accent border border-accent/20'
                      : 'text-surface-400 border border-surface-700/30 dark:border-surface-700/30 border-surface-200 hover:text-surface-200'
                  }`}
                >
                  {tab === 'calls' ? 'Calls' : 'Puts'}
                </button>
              ))}
            </div>

            {/* Data table */}
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-surface-700/30 dark:border-surface-700/30 border-surface-200">
                    {[
                      { label: 'Strike', tip: null },
                      { label: 'Last', tip: null },
                      { label: 'Bid', tip: null },
                      { label: 'Ask', tip: null },
                      { label: 'Vol', tip: TOOLTIPS.option_vol },
                      { label: 'OI', tip: TOOLTIPS.open_interest },
                      { label: 'IV', tip: TOOLTIPS.implied_vol },
                    ].map((h) => (
                      <th key={h.label} className="py-2 px-2 text-left font-medium text-surface-500 uppercase tracking-wider">
                        {h.tip ? <InfoTip text={h.tip}>{h.label}</InfoTip> : h.label}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(chainTab === 'calls' ? chainData.calls : chainData.puts)?.map((row, i) => {
                    const itm = chainTab === 'calls'
                      ? row.strike < chainData.spot_price
                      : row.strike > chainData.spot_price;
                    return (
                      <tr
                        key={i}
                        className={`border-b border-surface-700/10 dark:border-surface-700/10 border-surface-100 ${
                          itm ? 'bg-accent/5' : ''
                        }`}
                      >
                        <td className="py-1.5 px-2 tabular-nums font-medium text-surface-200 dark:text-surface-200 text-surface-700">
                          {row.strike?.toFixed(2)}
                        </td>
                        <td className="py-1.5 px-2 tabular-nums text-surface-300 dark:text-surface-300 text-surface-600">
                          {row.lastPrice?.toFixed(2)}
                        </td>
                        <td className="py-1.5 px-2 tabular-nums text-surface-400">{row.bid?.toFixed(2)}</td>
                        <td className="py-1.5 px-2 tabular-nums text-surface-400">{row.ask?.toFixed(2)}</td>
                        <td className="py-1.5 px-2 tabular-nums text-surface-400">{row.volume || 0}</td>
                        <td className="py-1.5 px-2 tabular-nums text-surface-400">{row.openInterest || 0}</td>
                        <td className="py-1.5 px-2 tabular-nums text-surface-400">
                          {((row.impliedVolatility || 0) * 100).toFixed(1)}%
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {!chainData && !chainLoading && (
          <p className="text-sm text-surface-500 text-center py-8">
            Enter a ticker to load options chain data.
          </p>
        )}
        {chainLoading && <Loading text="Loading options chain..." />}
      </Card>

      {/* Section: 3D Volatility Surface */}
      <Card>
        <CardHeader title="3D Volatility Surface" subtitle="Implied volatility across strikes and expirations" />
        <div className="flex gap-2 mb-4">
          <input
            type="text"
            placeholder="Ticker (e.g. AAPL)"
            value={surfaceTicker}
            onChange={(e) => setSurfaceTicker(e.target.value.toUpperCase())}
            className="flex-1 rounded-lg px-3 py-2 text-sm bg-surface-850/80 dark:bg-surface-850/80 bg-surface-50 border border-surface-700/60 dark:border-surface-700/60 border-surface-200 text-surface-50 dark:text-surface-50 text-surface-900 placeholder:text-surface-600 focus:outline-none focus:border-accent/50 transition-colors"
          />
          <Button variant="secondary" onClick={loadSurface} loading={surfaceLoading}>
            Load Surface
          </Button>
        </div>

        {surfaceLoading && <Loading text="Building volatility surface..." />}

        {surfaceData && !surfaceLoading && (
          <div className="mt-2">
            <Plot
              data={[{
                type: 'surface',
                x: surfaceData.strikes,
                y: surfaceData.expiries,
                z: surfaceData.iv_matrix,
                colorscale: [
                  [0, '#1a1a2e'],
                  [0.25, '#6b4c2a'],
                  [0.5, '#c9985a'],
                  [0.75, '#e6c88a'],
                  [1, '#f5d799'],
                ],
                colorbar: {
                  title: { text: 'IV', font: { color: '#8b949e', size: 10 } },
                  tickfont: { color: '#656d76', size: 9 },
                },
                hovertemplate: 'Strike: %{x}<br>DTE: %{y}<br>IV: %{z:.1%}<extra></extra>',
              }]}
              layout={{
                height: 450,
                margin: { l: 0, r: 0, t: 30, b: 0 },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                title: {
                  text: `${surfaceData.ticker} Vol Surface (Spot: $${surfaceData.spot_price.toFixed(2)})`,
                  font: { color: '#8b949e', size: 12 },
                },
                scene: {
                  xaxis: { title: 'Strike', titlefont: { color: '#656d76', size: 10 }, tickfont: { color: '#656d76', size: 9 }, gridcolor: 'rgba(33,38,45,0.5)', backgroundcolor: 'transparent' },
                  yaxis: { title: 'Days to Expiry', titlefont: { color: '#656d76', size: 10 }, tickfont: { color: '#656d76', size: 9 }, gridcolor: 'rgba(33,38,45,0.5)', backgroundcolor: 'transparent' },
                  zaxis: { title: 'Implied Vol', titlefont: { color: '#656d76', size: 10 }, tickfont: { color: '#656d76', size: 9 }, gridcolor: 'rgba(33,38,45,0.5)', backgroundcolor: 'transparent' },
                  bgcolor: 'transparent',
                  camera: { eye: { x: 1.5, y: 1.5, z: 0.8 } },
                },
              }}
              config={{ displayModeBar: false, responsive: true }}
              className="w-full"
            />
          </div>
        )}

        {!surfaceData && !surfaceLoading && (
          <p className="text-sm text-surface-500 text-center py-8">
            Enter a US ticker with options data to view the implied volatility surface.
          </p>
        )}
      </Card>
    </div>
  );
}
