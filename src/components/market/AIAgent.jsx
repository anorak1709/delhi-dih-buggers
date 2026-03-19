import { useState, useRef, useEffect } from 'react';
import { useApp } from '../../context/AppContext';
import Card, { CardHeader } from '../ui/Card';
import Button from '../ui/Button';
import { askAIAgent } from '../../services/api';
import { motion, AnimatePresence } from 'framer-motion';

function formatMarkdown(text) {
  // Simple markdown rendering: bold, bullets, paragraphs
  const lines = text.split('\n');
  const elements = [];
  let key = 0;

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) {
      elements.push(<div key={key++} className="h-2" />);
      continue;
    }

    // Bold headers (**text**)
    let formatted = trimmed.replace(/\*\*(.*?)\*\*/g, '<strong class="text-surface-100 dark:text-surface-100 text-surface-800 font-semibold">$1</strong>');
    // Italic (*text*)
    formatted = formatted.replace(/\*(.*?)\*/g, '<em>$1</em>');

    if (trimmed.startsWith('- ') || trimmed.startsWith('• ')) {
      elements.push(
        <div key={key++} className="flex gap-2 pl-2 py-0.5">
          <span className="text-accent mt-0.5 shrink-0">•</span>
          <span dangerouslySetInnerHTML={{ __html: formatted.slice(2) }} />
        </div>
      );
    } else if (/^\d+\.\s/.test(trimmed)) {
      const num = trimmed.match(/^(\d+)\.\s/)[1];
      elements.push(
        <div key={key++} className="flex gap-2 pl-2 py-0.5">
          <span className="text-accent shrink-0">{num}.</span>
          <span dangerouslySetInnerHTML={{ __html: formatted.replace(/^\d+\.\s/, '') }} />
        </div>
      );
    } else {
      elements.push(
        <p key={key++} className="py-0.5" dangerouslySetInnerHTML={{ __html: formatted }} />
      );
    }
  }

  return elements;
}

const suggestions = [
  'Analyze my portfolio holdings',
  'Which of my stocks has the best growth potential?',
  'What are the risks in my current portfolio?',
  'Should I diversify into other sectors?',
];

export default function AIAgent() {
  const { tickers } = useApp();
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const sendMessage = async (text) => {
    const query = text || input.trim();
    if (!query) return;

    const userMsg = { role: 'user', content: query, timestamp: new Date() };
    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const res = await askAIAgent(query, tickers);
      if (res.error) {
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: `Error: ${res.error}`, timestamp: new Date() },
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: res.response,
            tickers_analyzed: res.tickers_analyzed,
            timestamp: new Date(),
          },
        ]);
      }
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `Error: ${err.message}`, timestamp: new Date() },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <Card hover={false} className="!p-0 overflow-hidden">
      {/* Header */}
      <div className="px-5 pt-5 pb-3">
        <CardHeader
          title="Bloom AI Research Agent"
          subtitle="Powered by Gemini — Ask about any stock in your portfolio"
        />
      </div>

      {/* Chat Area */}
      <div className="h-96 overflow-y-auto px-5 space-y-3 scroll-smooth">
        {messages.length === 0 && !loading && (
          <div className="flex flex-col items-center justify-center h-full gap-4">
            <div className="text-center">
              <p className="text-sm text-surface-400 mb-1">Ask me anything about your investments</p>
              <p className="text-2xs text-surface-600">I'll analyze real-time data and provide research-backed insights</p>
            </div>
            <div className="flex flex-wrap justify-center gap-2 max-w-md">
              {suggestions.map((s, i) => (
                <motion.button
                  key={i}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.08 }}
                  onClick={() => sendMessage(s)}
                  className="text-2xs px-3 py-1.5 rounded-full border border-surface-700/40 dark:border-surface-700/40 border-surface-200 text-surface-400 hover:text-accent hover:border-accent/30 transition-colors cursor-pointer"
                >
                  {s}
                </motion.button>
              ))}
            </div>
          </div>
        )}

        <AnimatePresence>
          {messages.map((msg, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.2 }}
              className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[85%] rounded-xl px-4 py-3 text-sm leading-relaxed ${
                  msg.role === 'user'
                    ? 'bg-accent/10 text-surface-100 dark:text-surface-100 text-surface-800 border border-accent/20'
                    : 'bg-surface-700/20 dark:bg-surface-700/20 bg-surface-50 text-surface-300 dark:text-surface-300 text-surface-600 border border-surface-700/20 dark:border-surface-700/20 border-surface-200'
                }`}
              >
                {msg.role === 'user' ? (
                  <p>{msg.content}</p>
                ) : (
                  <div className="space-y-0.5">{formatMarkdown(msg.content)}</div>
                )}
                {msg.tickers_analyzed && msg.tickers_analyzed.length > 0 && (
                  <div className="flex gap-1.5 mt-2 pt-2 border-t border-surface-700/20">
                    <span className="text-2xs text-surface-500">Analyzed:</span>
                    {msg.tickers_analyzed.map((t) => (
                      <span key={t} className="text-2xs px-1.5 py-0.5 rounded bg-accent/10 text-accent font-medium">
                        {t.replace('.NS', '')}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {/* Typing indicator */}
        {loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex justify-start"
          >
            <div className="bg-surface-700/20 dark:bg-surface-700/20 bg-surface-50 rounded-xl px-4 py-3 border border-surface-700/20 dark:border-surface-700/20 border-surface-200">
              <div className="flex gap-1.5">
                {[0, 1, 2].map((i) => (
                  <motion.div
                    key={i}
                    className="w-2 h-2 rounded-full bg-accent/60"
                    animate={{ y: [0, -6, 0] }}
                    transition={{ duration: 0.6, repeat: Infinity, delay: i * 0.15 }}
                  />
                ))}
              </div>
            </div>
          </motion.div>
        )}

        <div ref={chatEndRef} />
      </div>

      {/* Input Area */}
      <div className="px-5 py-4 border-t border-surface-700/20 dark:border-surface-700/20 border-surface-100">
        <div className="flex gap-2">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={tickers.length ? 'Ask about your holdings...' : 'Add holdings first, then ask me anything...'}
            disabled={loading}
            className="flex-1 rounded-lg px-3.5 py-2.5 text-sm bg-surface-850/80 dark:bg-surface-850/80 bg-surface-50 border border-surface-700/60 dark:border-surface-700/60 border-surface-200 text-surface-50 dark:text-surface-50 text-surface-900 placeholder:text-surface-600 dark:placeholder:text-surface-600 placeholder:text-surface-400 focus:outline-none focus:border-accent/50 focus:ring-1 focus:ring-accent/20 transition-colors duration-200 disabled:opacity-50"
          />
          <Button
            onClick={() => sendMessage()}
            loading={loading}
            disabled={!input.trim() && !loading}
            size="md"
          >
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5" />
            </svg>
          </Button>
        </div>
        {tickers.length > 0 && (
          <p className="text-2xs text-surface-600 mt-1.5">
            Tracking: {tickers.map((t) => t.replace('.NS', '')).join(', ')}
          </p>
        )}
      </div>
    </Card>
  );
}
