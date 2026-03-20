import { useState } from 'react';
import { useApp } from '../../context/AppContext';
import Card, { CardHeader } from '../ui/Card';
import Button from '../ui/Button';
import Loading from '../ui/Loading';
import { getNews, getSectors, getSentiment } from '../../services/api';
import AIAgent from './AIAgent';

function SentimentBadge({ score }) {
  if (score > 0.05) return <span className="text-2xs px-2 py-0.5 rounded-full bg-up/10 text-up font-medium">Bullish</span>;
  if (score < -0.05) return <span className="text-2xs px-2 py-0.5 rounded-full bg-down/10 text-down font-medium">Bearish</span>;
  return <span className="text-2xs px-2 py-0.5 rounded-full bg-surface-700/30 text-surface-400 font-medium">Neutral</span>;
}

export default function MarketPanel() {
  const { tickers, notify } = useApp();
  const [newsLoading, setNewsLoading] = useState(false);
  const [news, setNews] = useState(null);
  const [sectorsLoading, setSectorsLoading] = useState(false);
  const [sectors, setSectors] = useState(null);
  const [sentimentLoading, setSentimentLoading] = useState(false);
  const [sentiment, setSentiment] = useState(null);

  const fetchNews = async () => {
    if (!tickers.length) return notify('Add holdings first', 'warning');
    setNewsLoading(true);
    try {
      const data = await getNews(tickers);
      setNews(data);
      notify('News loaded', 'success');
    } catch (err) {
      notify(err.message, 'error');
    } finally {
      setNewsLoading(false);
    }
  };

  const fetchSectors = async () => {
    if (!tickers.length) return notify('Add holdings first', 'warning');
    setSectorsLoading(true);
    try {
      const data = await getSectors(tickers);
      setSectors(data);
    } catch (err) {
      notify(err.message, 'error');
    } finally {
      setSectorsLoading(false);
    }
  };

  const fetchSentiment = async () => {
    if (!tickers.length) return notify('Add holdings first', 'warning');
    setSentimentLoading(true);
    try {
      const data = await getSentiment(tickers);
      setSentiment(data);
      notify('Sentiment analysis complete', 'success');
    } catch (err) {
      notify(err.message, 'error');
    } finally {
      setSentimentLoading(false);
    }
  };

  const totalSectors = sectors ? Object.values(sectors).reduce((a, b) => a + b, 0) : 0;
  const sectorColors = ['bg-accent', 'bg-info', 'bg-up', 'bg-warn', 'bg-down', 'bg-surface-400'];

  // Determine news articles (handle single vs multi-ticker response)
  const newsArticles = news?.articles || [];
  const newsByTicker = news?.articles_by_ticker || null;

  // Determine sentiment (single ticker returns directly, multi returns sentiment_by_ticker)
  const sentimentByTicker = sentiment?.sentiment_by_ticker || null;
  const singleSentiment = sentiment && !sentimentByTicker ? sentiment : null;

  return (
    <div className="space-y-6">
      {/* AI Research Agent */}
      <AIAgent />

      {/* Sector Allocation + Sentiment */}
      {tickers.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Sectors */}
          <Card>
            <CardHeader
              title="Sector Breakdown"
              action={
                <Button variant="secondary" size="sm" onClick={fetchSectors} loading={sectorsLoading}>
                  Load
                </Button>
              }
            />
            {sectorsLoading && <Loading text="Loading sectors..." />}
            {sectors && !sectorsLoading && (
              <div className="space-y-3">
                {/* Horizontal stacked bar */}
                <div className="flex h-3 rounded-full overflow-hidden">
                  {Object.entries(sectors).map(([name, count], i) => (
                    <div
                      key={name}
                      className={`${sectorColors[i % sectorColors.length]} transition-all duration-500`}
                      style={{ width: `${(count / totalSectors) * 100}%` }}
                      title={`${name}: ${count}`}
                    />
                  ))}
                </div>
                <div className="space-y-1.5">
                  {Object.entries(sectors).map(([name, count], i) => (
                    <div key={name} className="flex items-center justify-between text-sm">
                      <div className="flex items-center gap-2">
                        <div className={`w-2.5 h-2.5 rounded-sm ${sectorColors[i % sectorColors.length]}`} />
                        <span className="text-surface-300 dark:text-surface-300 text-surface-600">{name}</span>
                      </div>
                      <span className="tabular-nums text-surface-400">{count} stock{count > 1 ? 's' : ''}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </Card>

          {/* Sentiment */}
          <Card>
            <CardHeader
              title="News Sentiment"
              subtitle="AI-powered headline analysis"
              action={
                <Button variant="secondary" size="sm" onClick={fetchSentiment} loading={sentimentLoading}>
                  Analyze
                </Button>
              }
            />
            {sentimentLoading && <Loading text="Analyzing sentiment..." />}
            {singleSentiment && !sentimentLoading && (
              <div>
                <div className="flex items-center gap-3 mb-4">
                  <div className="text-3xl font-display font-semibold tabular-nums text-surface-100 dark:text-surface-100 text-surface-800">
                    {singleSentiment.sentiment?.toFixed(2)}
                  </div>
                  <SentimentBadge score={singleSentiment.sentiment} />
                </div>
                <div className="flex gap-4 text-xs text-surface-500 mb-4">
                  <span className="text-up">+{singleSentiment.breakdown?.positive || 0} positive</span>
                  <span className="text-down">{singleSentiment.breakdown?.negative || 0} negative</span>
                  <span>{singleSentiment.breakdown?.neutral || 0} neutral</span>
                </div>
                {singleSentiment.headlines?.slice(0, 5).map((h, i) => (
                  <div key={i} className="flex items-start gap-2 py-1.5 border-b border-surface-700/20 dark:border-surface-700/20 border-surface-100 last:border-0">
                    <span className={`mt-0.5 w-1.5 h-1.5 rounded-full shrink-0 ${h.label === 'Positive' ? 'bg-up' : h.label === 'Negative' ? 'bg-down' : 'bg-surface-500'}`} />
                    <span className="text-xs text-surface-400 leading-relaxed">{h.text}</span>
                  </div>
                ))}
              </div>
            )}
            {sentimentByTicker && !sentimentLoading && (
              <div className="space-y-4">
                {Object.entries(sentimentByTicker).map(([ticker, s]) => {
                  const score = s.sentiment || 0;
                  const suggestion = score > 0.5
                    ? 'Consider increasing weight'
                    : score < -0.5
                    ? 'Consider reducing weight'
                    : 'Hold current position';
                  const suggestionColor = score > 0.5
                    ? 'text-up'
                    : score < -0.5
                    ? 'text-down'
                    : 'text-amber-400';
                  return (
                    <div key={ticker} className="border-b border-surface-700/20 dark:border-surface-700/20 border-surface-100 pb-3 last:border-0">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-sm font-semibold text-surface-200 dark:text-surface-200 text-surface-700">{ticker}</span>
                        <div className="flex items-center gap-2">
                          <span className="text-sm tabular-nums text-surface-300 dark:text-surface-300 text-surface-600">{score.toFixed(2)}</span>
                          <SentimentBadge score={score} />
                        </div>
                      </div>
                      <p className={`text-2xs ${suggestionColor} mt-0.5`}>{suggestion}</p>
                      {/* Sentiment bar */}
                      <div className="mt-1.5 h-1.5 w-full bg-surface-700/30 dark:bg-surface-700/30 bg-surface-200 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all duration-500 ${score > 0.05 ? 'bg-up' : score < -0.05 ? 'bg-down' : 'bg-amber-400'}`}
                          style={{ width: `${Math.min(Math.abs(score) * 100, 100)}%`, marginLeft: score < 0 ? 'auto' : '0' }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </Card>
        </div>
      )}

      {/* News Feed */}
      <Card>
        <CardHeader
          title="Market News"
          subtitle="Latest headlines for your holdings"
          action={
            <Button variant="secondary" size="sm" onClick={fetchNews} loading={newsLoading}>
              Refresh
            </Button>
          }
        />
        {newsLoading && <Loading text="Fetching news..." />}
        {newsArticles.length > 0 && !newsLoading && (
          <div className="space-y-0.5">
            {newsArticles.map((article, i) => (
              <a
                key={i}
                href={article.url}
                target="_blank"
                rel="noopener noreferrer"
                className="block rounded-lg px-3 py-3 -mx-3 hover:bg-surface-700/20 dark:hover:bg-surface-700/20 hover:bg-surface-50 transition-colors"
              >
                <p className="text-sm text-surface-200 dark:text-surface-200 text-surface-700 font-medium leading-snug">
                  {article.title}
                </p>
                <div className="flex items-center gap-2 mt-1.5">
                  <span className="text-2xs text-surface-500">{article.source?.name}</span>
                  {article.publishedAt && (
                    <>
                      <span className="text-surface-700 dark:text-surface-700 text-surface-300">·</span>
                      <span className="text-2xs text-surface-600">{new Date(article.publishedAt).toLocaleDateString()}</span>
                    </>
                  )}
                </div>
              </a>
            ))}
          </div>
        )}
        {newsByTicker && !newsLoading && (
          <div className="space-y-4">
            {Object.entries(newsByTicker).map(([ticker, articles]) => (
              <div key={ticker}>
                <p className="text-xs font-semibold uppercase tracking-wider text-accent mb-2">{ticker}</p>
                <div className="space-y-0.5">
                  {articles.slice(0, 3).map((article, i) => (
                    <a
                      key={i}
                      href={article.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="block rounded-lg px-3 py-2.5 -mx-3 hover:bg-surface-700/20 dark:hover:bg-surface-700/20 hover:bg-surface-50 transition-colors"
                    >
                      <p className="text-sm text-surface-200 dark:text-surface-200 text-surface-700 leading-snug">
                        {article.title}
                      </p>
                      <span className="text-2xs text-surface-500 mt-1 inline-block">{article.source?.name}</span>
                    </a>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
        {!news && !newsLoading && tickers.length === 0 && (
          <div className="text-center py-10">
            <p className="text-surface-500 text-sm">Add holdings to see market news.</p>
          </div>
        )}
      </Card>
    </div>
  );
}
