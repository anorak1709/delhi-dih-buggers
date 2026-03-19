import { useState, useEffect, useRef } from 'react';
import { useApp } from '../../context/AppContext';
import Card, { CardHeader } from '../ui/Card';
import Button from '../ui/Button';
import Loading from '../ui/Loading';
import { getLiveAnalysis, getPrices } from '../../services/api';
import { io } from 'socket.io-client';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Filler,
} from 'chart.js';

ChartJS.register(LineElement, PointElement, LinearScale, CategoryScale, Filler);

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

function SentimentBadge({ score }) {
  if (score > 0.05) return <span className="text-2xs px-2 py-0.5 rounded-full bg-up/10 text-up font-medium">Bullish</span>;
  if (score < -0.05) return <span className="text-2xs px-2 py-0.5 rounded-full bg-down/10 text-down font-medium">Bearish</span>;
  return <span className="text-2xs px-2 py-0.5 rounded-full bg-surface-700/30 text-surface-400 font-medium">Neutral</span>;
}

function MiniChart({ data, isUp }) {
  if (!data || data.length < 2) return null;

  const color = isUp ? 'rgba(34, 197, 94, 1)' : 'rgba(239, 68, 68, 1)';
  const bgColor = isUp ? 'rgba(34, 197, 94, 0.08)' : 'rgba(239, 68, 68, 0.08)';

  const chartData = {
    labels: data.map((d) => d.time),
    datasets: [
      {
        data: data.map((d) => d.price),
        borderColor: color,
        backgroundColor: bgColor,
        borderWidth: 1.5,
        pointRadius: 0,
        fill: true,
        tension: 0.3,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false }, tooltip: { enabled: false } },
    scales: {
      x: { display: false },
      y: { display: false },
    },
    elements: { line: { borderJoinStyle: 'round' } },
    animation: { duration: 600 },
  };

  return (
    <div className="h-16 w-full">
      <Line data={chartData} options={options} />
    </div>
  );
}

function LiveTickerCard({ ticker, data, prevPrice }) {
  const flash = prevPrice && data.price !== prevPrice;
  const direction = data.change >= 0 ? 'up' : 'down';
  const isUp = data.change >= 0;

  return (
    <Card hover={false} className="!p-4 space-y-3">
      {/* Header: ticker + change badge */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-surface-100 dark:text-surface-100 text-surface-800">
          {ticker.replace('.NS', '')}
        </h3>
        <span
          className={`text-2xs px-2 py-0.5 rounded-full font-medium ${
            isUp ? 'bg-up/10 text-up' : 'bg-down/10 text-down'
          }`}
        >
          {isUp ? '+' : ''}{data.change?.toFixed(2)}%
        </span>
      </div>

      {/* Price */}
      <div className="flex items-baseline gap-2">
        <span
          className={`text-2xl font-display font-semibold tabular-nums transition-colors duration-300 ${
            flash
              ? direction === 'up' ? 'text-up' : 'text-down'
              : 'text-surface-100 dark:text-surface-100 text-surface-800'
          }`}
        >
          ₹{data.price?.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
        </span>
        {data.previousClose > 0 && (
          <span className="text-2xs text-surface-500">
            prev ₹{data.previousClose?.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
          </span>
        )}
      </div>

      {/* Mini Chart */}
      <MiniChart data={data.miniChart} isUp={isUp} />

      {/* Sentiment */}
      <div className="flex items-center gap-2">
        <span className="text-2xs text-surface-500">Sentiment:</span>
        <SentimentBadge score={data.sentiment} />
        <span className="text-2xs text-surface-500 tabular-nums">{data.sentiment?.toFixed(2)}</span>
      </div>

      {/* Headlines */}
      {data.headlines && data.headlines.length > 0 && (
        <div className="space-y-1 pt-1 border-t border-surface-700/20 dark:border-surface-700/20 border-surface-100">
          {data.headlines.map((h, i) => (
            <a
              key={i}
              href={h.url}
              target="_blank"
              rel="noopener noreferrer"
              className="block text-2xs text-surface-400 hover:text-accent leading-snug truncate transition-colors"
              title={h.title}
            >
              {h.title}
            </a>
          ))}
        </div>
      )}
    </Card>
  );
}

export default function LivePanel() {
  const { tickers, holdings, notify } = useApp();
  const [liveData, setLiveData] = useState({});
  const [prevPrices, setPrevPrices] = useState({});
  const [loading, setLoading] = useState(false);
  const [connected, setConnected] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [lastUpdate, setLastUpdate] = useState(null);
  const socketRef = useRef(null);
  const intervalRef = useRef(null);

  const fetchLiveData = async () => {
    if (!tickers.length) return;
    setLoading(true);
    try {
      const res = await getLiveAnalysis(tickers);
      if (res.data) {
        setPrevPrices((prev) => {
          const pp = {};
          for (const t of tickers) pp[t] = prev[t]?.price || liveData[t]?.price;
          return pp;
        });
        setLiveData(res.data);
        setLastUpdate(new Date());
      }
    } catch (err) {
      notify(err.message, 'error');
    } finally {
      setLoading(false);
    }
  };

  // Auto-fetch when tickers change
  useEffect(() => {
    if (tickers.length > 0) {
      fetchLiveData();
    }
  }, [tickers.join(',')]);

  const toggleAutoRefresh = () => {
    if (autoRefresh) {
      clearInterval(intervalRef.current);
      setAutoRefresh(false);
    } else {
      fetchLiveData();
      intervalRef.current = setInterval(fetchLiveData, 30000);
      setAutoRefresh(true);
    }
  };

  const connectSocket = () => {
    if (socketRef.current) {
      socketRef.current.disconnect();
    }
    try {
      const socket = io(BASE_URL);
      socketRef.current = socket;

      socket.on('connect', () => {
        setConnected(true);
        socket.emit('subscribe_prices', { tickers });
        notify('Connected to live feed', 'success');
      });

      socket.on('price_update', (data) => {
        setPrevPrices((prev) => {
          const pp = {};
          for (const t of tickers) pp[t] = liveData[t]?.price || prev[t];
          return pp;
        });
        setLiveData((prev) => {
          const updated = { ...prev };
          for (const [t, price] of Object.entries(data)) {
            if (updated[t]) {
              updated[t] = { ...updated[t], price };
            }
          }
          return updated;
        });
        setLastUpdate(new Date());
      });

      socket.on('disconnect', () => setConnected(false));
      socket.on('error', (err) => notify(err.message || 'Socket error', 'error'));
    } catch (err) {
      notify('Failed to connect: ' + err.message, 'error');
    }
  };

  useEffect(() => {
    return () => {
      if (socketRef.current) socketRef.current.disconnect();
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  const totalValue = holdings.reduce((sum, h) => {
    const d = liveData[h.ticker];
    return sum + (d?.price ? d.price * h.quantity : 0);
  }, 0);

  const totalChange = holdings.reduce((sum, h) => {
    const d = liveData[h.ticker];
    if (!d?.price || !d?.previousClose) return sum;
    return sum + (d.price - d.previousClose) * h.quantity;
  }, 0);

  return (
    <div className="space-y-6">
      {/* Status Bar */}
      <Card hover={false} className="!p-4">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${connected ? 'bg-up animate-pulse-soft' : autoRefresh ? 'bg-warn animate-pulse-soft' : 'bg-surface-600'}`} />
              <span className="text-xs text-surface-400">
                {connected ? 'Live WebSocket' : autoRefresh ? 'Auto-refresh (30s)' : 'Manual'}
              </span>
            </div>
            {lastUpdate && (
              <span className="text-2xs text-surface-600">
                Updated {lastUpdate.toLocaleTimeString()}
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <Button variant="secondary" size="sm" onClick={fetchLiveData} loading={loading}>
              Refresh
            </Button>
            <Button
              variant={autoRefresh ? 'danger' : 'secondary'}
              size="sm"
              onClick={toggleAutoRefresh}
            >
              {autoRefresh ? 'Stop' : 'Auto'}
            </Button>
            <Button
              variant={connected ? 'danger' : 'ghost'}
              size="sm"
              onClick={() => {
                if (connected) {
                  socketRef.current?.disconnect();
                  setConnected(false);
                } else {
                  connectSocket();
                }
              }}
            >
              {connected ? 'Disconnect' : 'Live'}
            </Button>
          </div>
        </div>
      </Card>

      {/* Portfolio Summary */}
      {totalValue > 0 && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <Card hover={false} className="!p-4">
            <p className="text-2xs uppercase tracking-wider text-surface-500 mb-1">Total Value</p>
            <p className="text-3xl font-display font-semibold text-accent tabular-nums">
              ₹{totalValue.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
            </p>
          </Card>
          <Card hover={false} className="!p-4">
            <p className="text-2xs uppercase tracking-wider text-surface-500 mb-1">Day Change</p>
            <p className={`text-3xl font-display font-semibold tabular-nums ${totalChange >= 0 ? 'text-up' : 'text-down'}`}>
              {totalChange >= 0 ? '+' : ''}₹{totalChange.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
            </p>
          </Card>
        </div>
      )}

      {loading && Object.keys(liveData).length === 0 && <Loading text="Fetching live analysis..." />}

      {/* Enriched Ticker Grid */}
      {Object.keys(liveData).length > 0 && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {tickers.map((t) => {
            const data = liveData[t];
            if (!data || data.price == null) return null;
            return (
              <LiveTickerCard
                key={t}
                ticker={t}
                data={data}
                prevPrice={prevPrices[t]}
              />
            );
          })}
        </div>
      )}

      {!tickers.length && (
        <Card hover={false}>
          <div className="text-center py-16">
            <p className="text-surface-500 text-sm">Add holdings in the Portfolio tab to track live prices.</p>
          </div>
        </Card>
      )}
    </div>
  );
}
