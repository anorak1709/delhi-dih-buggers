/* ═══════════════════════════════════════════════════════════
   Portfolio Optimizer Pro — Application Logic
   Palette: #00D9FF #B026FF #000000 #ffffff
   ═══════════════════════════════════════════════════════════ */

let holdings = [];
let holdingIdCounter = 0;
let currentTab = 'portfolio';
let performanceChartInstance = null;
let frontierChartInstance = null;
let currentTheme = 'dark';

// ── Initialisation ────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    loadFromLocalStorage();
    initializeTabButtons();
    setupKeyboardShortcuts();
});

// ── Local Storage ─────────────────────────────────────────

function saveToLocalStorage() {
    const data = {
        holdings: holdings,
        benchmark: document.getElementById('benchmark').value,
        startDate: document.getElementById('startDate').value,
        riskFreeRate: document.getElementById('riskFreeRate').value,
        theme: currentTheme
    };
    localStorage.setItem('portfolioOptimizerData', JSON.stringify(data));
}

function loadFromLocalStorage() {
    const saved = localStorage.getItem('portfolioOptimizerData');
    if (saved) {
        try {
            const data = JSON.parse(saved);
            holdings = data.holdings || [];
            holdingIdCounter = Math.max(...holdings.map(h => h.id), 0) + 1;

            if (data.benchmark) document.getElementById('benchmark').value = data.benchmark;
            if (data.startDate) document.getElementById('startDate').value = data.startDate;
            if (data.riskFreeRate) document.getElementById('riskFreeRate').value = data.riskFreeRate;
            if (data.theme) {
                currentTheme = data.theme;
                if (currentTheme === 'light') {
                    document.body.classList.add('light-mode');
                    updateThemeIcon();
                }
            }

            updateHoldingsList();
        } catch (e) {
            console.error('Error loading saved data:', e);
        }
    }
}

// ── Theme Toggle ──────────────────────────────────────────

function toggleTheme() {
    currentTheme = currentTheme === 'dark' ? 'light' : 'dark';
    document.body.classList.toggle('light-mode');
    updateThemeIcon();
    saveToLocalStorage();
}

function updateThemeIcon() {
    const icon = document.getElementById('themeIcon');
    if (currentTheme === 'light') {
        icon.innerHTML = '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"></path>';
    } else {
        icon.innerHTML = '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"></path>';
    }
}

// ── Export to CSV ──────────────────────────────────────────

function exportToCSV() {
    if (holdings.length === 0) {
        showError('No holdings to export');
        return;
    }

    let csv = 'Ticker,Quantity\n';
    holdings.forEach(h => {
        csv += `${h.ticker},${h.quantity}\n`;
    });

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `portfolio_${new Date().toISOString().split('T')[0]}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);

    showSuccess('Portfolio exported successfully!');
}

// ── Keyboard Shortcuts ────────────────────────────────────

function setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'a') { e.preventDefault(); document.getElementById('ticker').focus(); }
        if (e.ctrlKey && e.key === 'Enter') { e.preventDefault(); analyzePortfolio(); }
        if (e.ctrlKey && e.key === 'o') { e.preventDefault(); optimizePortfolio(); }
        if (e.ctrlKey && e.key === 'e') { e.preventDefault(); exportToCSV(); }
        if (e.ctrlKey && e.key === 'd') { e.preventDefault(); toggleTheme(); }
        if (e.ctrlKey && e.key === '/') { e.preventDefault(); showShortcutsModal(); }
    });
}

function showShortcutsModal() {
    document.getElementById('shortcutsModal').classList.add('show');
}

function closeShortcutsModal() {
    document.getElementById('shortcutsModal').classList.remove('show');
}

// ── Loading Overlay ───────────────────────────────────────

function showLoading(text = 'Processing...') {
    document.getElementById('loadingText').textContent = text;
    document.getElementById('loadingOverlay').classList.add('show');
}

function hideLoading() {
    document.getElementById('loadingOverlay').classList.remove('show');
}

// ── Notifications ─────────────────────────────────────────

function showSuccess(message) {
    const alert = document.createElement('div');
    alert.className = 'fixed top-4 right-4 glass-strong rounded-xl p-4 neon-border-blue fade-in z-50';
    alert.innerHTML = `
        <div class="flex items-center gap-3">
            <svg class="w-6 h-6 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
            </svg>
            <p class="text-sm text-white">${message}</p>
        </div>
    `;
    document.body.appendChild(alert);
    setTimeout(() => { alert.remove(); }, 3000);
}

function showError(message) {
    document.getElementById('errorMessage').textContent = message;
    document.getElementById('errorAlert').classList.remove('hidden');
    setTimeout(() => hideError(), 5000);
}

function hideError() {
    document.getElementById('errorAlert').classList.add('hidden');
}

// ── Tab Navigation ────────────────────────────────────────

function switchTab(tabName) {
    currentTab = tabName;
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById(tabName + 'Tab').classList.add('active');
    document.querySelectorAll('.tab-content').forEach(content => content.classList.add('hidden'));
    document.getElementById(tabName + 'Content').classList.remove('hidden');
    document.getElementById(tabName + 'Content').classList.add('fade-in');
}

function initializeTabButtons() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            document.querySelectorAll('.tab-btn').forEach(b => {
                b.classList.remove('bg-gradient-to-r', 'from-primary', 'to-secondary', 'neon-glow-blue', 'text-darkest');
                b.classList.add('text-muted', 'hover:text-primary', 'hover:bg-primary/10');
            });
            this.classList.remove('text-muted', 'hover:text-primary', 'hover:bg-primary/10');
            this.classList.add('bg-gradient-to-r', 'from-primary', 'to-secondary', 'neon-glow-blue', 'text-darkest', 'font-bold');
        });
    });
    document.getElementById('portfolioTab').click();
}

// ── Holdings Management ───────────────────────────────────

function addHolding() {
    const ticker = document.getElementById('ticker').value.trim().toUpperCase();
    const quantity = parseInt(document.getElementById('quantity').value);

    if (!ticker) { showError('Please enter a stock ticker'); return; }
    if (!quantity || quantity <= 0) { showError('Please enter a valid quantity'); return; }
    if (holdings.some(h => h.ticker === ticker)) { showError(`${ticker} is already in your portfolio`); return; }

    holdings.push({ id: holdingIdCounter++, ticker: ticker, quantity: quantity });
    updateHoldingsList();
    document.getElementById('ticker').value = '';
    document.getElementById('quantity').value = '';
    hideError();
    saveToLocalStorage();
    showSuccess(`${ticker} added to portfolio`);
}

function removeHolding(id) {
    holdings = holdings.filter(h => h.id !== id);
    updateHoldingsList();
    saveToLocalStorage();
}

function updateHoldingsList() {
    const section = document.getElementById('holdingsSection');
    const list = document.getElementById('holdingsList');
    const count = document.getElementById('holdingsCount');

    if (holdings.length === 0) { section.classList.add('hidden'); return; }

    section.classList.remove('hidden');
    count.textContent = holdings.length;

    list.innerHTML = holdings.map(h => `
        <div class="glass rounded-lg p-3 flex items-center justify-between hover:bg-primary/10 transition-all border border-primary/20">
            <div class="flex items-center gap-3">
                <div class="w-10 h-10 bg-gradient-to-br from-primary to-secondary rounded-lg flex items-center justify-center text-darkest font-bold text-sm neon-glow-blue">
                    ${h.ticker.substring(0, 2)}
                </div>
                <div>
                    <p class="font-semibold text-white">${h.ticker}</p>
                    <p class="text-xs text-muted">${h.quantity} shares</p>
                </div>
            </div>
            <button onclick="removeHolding(${h.id})" class="w-8 h-8 bg-secondary/20 hover:bg-secondary border border-secondary/40 rounded-lg flex items-center justify-center text-secondary hover:text-white transition-all no-print">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                </svg>
            </button>
        </div>
    `).join('');
}

// ── Portfolio Analysis ────────────────────────────────────

async function analyzePortfolio() {
    if (holdings.length === 0) { showError('Please add at least one stock'); return; }

    const btn = document.getElementById('analyzeBtn');
    const originalHTML = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<div class="spinner"></div>Analyzing...';
    showLoading('Analyzing portfolio performance...');
    hideError();

    const holdingsObj = {};
    holdings.forEach(h => holdingsObj[h.ticker] = h.quantity);

    try {
        const response = await fetch('http://localhost:5000/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                holdings: holdingsObj,
                benchmark: document.getElementById('benchmark').value,
                start_date: document.getElementById('startDate').value,
                risk_free_rate: parseFloat(document.getElementById('riskFreeRate').value) / 100
            })
        });
        const data = await response.json();
        if (data.error) { showError(data.error); }
        else { displayAnalysisResults(data); switchTab('analysis'); showSuccess('Analysis complete!'); }
    } catch (err) {
        showError('Failed to connect to backend. Make sure the server is running on port 5000.');
    } finally {
        btn.disabled = false;
        btn.innerHTML = originalHTML;
        hideLoading();
    }
}

async function optimizePortfolio() {
    if (holdings.length < 2) { showError('Need at least 2 stocks for optimization'); return; }

    const btn = document.getElementById('optimizeBtn');
    const originalHTML = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<div class="spinner"></div>Optimizing...';
    showLoading('Running Monte Carlo simulation...');
    hideError();

    try {
        const response = await fetch('http://localhost:5000/api/optimize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                tickers: holdings.map(h => h.ticker),
                start_date: document.getElementById('startDate').value,
                num_portfolios: 10000,
                risk_free_rate: parseFloat(document.getElementById('riskFreeRate').value) / 100
            })
        });
        const data = await response.json();
        if (data.error) { showError(data.error); }
        else { displayOptimizationResults(data); switchTab('optimization'); showSuccess('Optimization complete!'); }
    } catch (err) {
        showError('Failed to connect to backend. Make sure the server is running on port 5000.');
    } finally {
        btn.disabled = false;
        btn.innerHTML = originalHTML;
        hideLoading();
    }
}

// ── Display Analysis Results ──────────────────────────────

function displayAnalysisResults(data) {
    const formatPercent = (val) => (val * 100).toFixed(2) + '%';
    const formatNumber = (val) => val.toFixed(2);

    document.getElementById('portCagr').textContent = formatPercent(data.portfolio.cagr);
    document.getElementById('portVol').textContent = formatPercent(data.portfolio.annual_vol);
    document.getElementById('portSharpe').textContent = formatNumber(data.portfolio.sharpe);
    document.getElementById('portDrawdown').textContent = formatPercent(data.portfolio.max_drawdown);

    document.getElementById('benchName').textContent = document.getElementById('benchmark').value;
    document.getElementById('benchCagr').textContent = formatPercent(data.benchmark.cagr);
    document.getElementById('benchVol').textContent = formatPercent(data.benchmark.annual_vol);
    document.getElementById('benchSharpe').textContent = formatNumber(data.benchmark.sharpe);
    document.getElementById('benchDrawdown').textContent = formatPercent(data.benchmark.max_drawdown);

    const ctx = document.getElementById('performanceChart').getContext('2d');
    if (performanceChartInstance) performanceChartInstance.destroy();

    performanceChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.chart_data.map(d => d.date),
            datasets: [
                {
                    label: 'Portfolio',
                    data: data.chart_data.map(d => d.portfolio),
                    borderColor: '#00D9FF',
                    backgroundColor: 'rgba(0, 217, 255, 0.1)',
                    borderWidth: 3, tension: 0.4, fill: true, pointRadius: 0, pointHoverRadius: 6
                },
                {
                    label: 'Benchmark',
                    data: data.chart_data.map(d => d.benchmark),
                    borderColor: '#B026FF',
                    backgroundColor: 'rgba(176, 38, 255, 0.1)',
                    borderWidth: 3, tension: 0.4, fill: true, pointRadius: 0, pointHoverRadius: 6
                }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    labels: { color: currentTheme === 'dark' ? '#ffffff' : '#1a1a1a', font: { size: 14, weight: 'bold' }, padding: 20 }
                },
                tooltip: {
                    mode: 'index', intersect: false,
                    backgroundColor: 'rgba(0, 0, 0, 0.95)',
                    titleColor: '#00D9FF', bodyColor: '#ffffff',
                    borderColor: 'rgba(0, 217, 255, 0.4)', borderWidth: 2, padding: 12, displayColors: true
                }
            },
            scales: {
                x: { grid: { color: 'rgba(0, 217, 255, 0.08)' }, ticks: { color: '#999999', maxTicksLimit: 8 } },
                y: { grid: { color: 'rgba(0, 217, 255, 0.08)' }, ticks: { color: '#999999' } }
            },
            interaction: { mode: 'nearest', axis: 'x', intersect: false }
        }
    });
}

// ── Display Optimisation Results ──────────────────────────

function displayOptimizationResults(data) {
    const formatPercent = (val) => (val * 100).toFixed(2) + '%';
    const formatNumber = (val) => val.toFixed(2);

    document.getElementById('simCount').textContent = '10,000';
    document.getElementById('maxSharpeValue').textContent = formatNumber(data.optimal_portfolio.sharpe_ratio);
    document.getElementById('minVolValue').textContent = formatPercent(data.min_risk_portfolio.volatility);

    document.getElementById('optReturn').textContent = formatPercent(data.optimal_portfolio.expected_return);
    document.getElementById('optVol').textContent = formatPercent(data.optimal_portfolio.volatility);
    document.getElementById('optSharpe').textContent = formatNumber(data.optimal_portfolio.sharpe_ratio);

    document.getElementById('minReturn').textContent = formatPercent(data.min_risk_portfolio.expected_return);
    document.getElementById('minVol').textContent = formatPercent(data.min_risk_portfolio.volatility);
    document.getElementById('minSharpe').textContent = formatNumber(data.min_risk_portfolio.sharpe_ratio);

    const optWeightsHTML = Object.entries(data.optimal_portfolio.weights)
        .sort((a, b) => b[1] - a[1])
        .map(([ticker, weight]) => `
            <div class="flex items-center justify-between">
                <span class="text-white font-semibold">${ticker}</span>
                <div class="flex items-center gap-2">
                    <div class="w-32 h-2 bg-darkest rounded-full overflow-hidden border border-primary/30">
                        <div class="h-full bg-gradient-to-r from-primary to-secondary rounded-full" style="width: ${weight * 100}%"></div>
                    </div>
                    <span class="text-primary font-bold text-sm min-w-[50px] text-right">${formatPercent(weight)}</span>
                </div>
            </div>
        `).join('');

    const minWeightsHTML = Object.entries(data.min_risk_portfolio.weights)
        .sort((a, b) => b[1] - a[1])
        .map(([ticker, weight]) => `
            <div class="flex items-center justify-between">
                <span class="text-white font-semibold">${ticker}</span>
                <div class="flex items-center gap-2">
                    <div class="w-32 h-2 bg-darkest rounded-full overflow-hidden border border-secondary/30">
                        <div class="h-full bg-gradient-to-r from-secondary to-cream rounded-full" style="width: ${weight * 100}%"></div>
                    </div>
                    <span class="text-secondary font-bold text-sm min-w-[50px] text-right">${formatPercent(weight)}</span>
                </div>
            </div>
        `).join('');

    document.getElementById('optWeights').innerHTML = optWeightsHTML;
    document.getElementById('minWeights').innerHTML = minWeightsHTML;
    createFrontierScatterPlot(data);
}

// ── Frontier Scatter Plot ─────────────────────────────────

function createFrontierScatterPlot(data) {
    const ctx = document.getElementById('frontierChart').getContext('2d');
    if (frontierChartInstance) frontierChartInstance.destroy();

    const scatterData = data.frontier_points.map(point => ({
        x: point.volatility * 100, y: point.return * 100, sharpe: point.sharpe
    }));

    function getColor(sharpe) {
        const normalized = Math.min(Math.max((sharpe + 1) / 3, 0), 1);
        const r = Math.floor(181 + (253 - 181) * normalized);
        const g = Math.floor(127 + (228 - 127) * normalized);
        const b = Math.floor(105 + (208 - 105) * normalized);
        return `rgba(${r}, ${g}, ${b}, 0.6)`;
    }

    const pointColors = scatterData.map(d => getColor(d.sharpe));

    frontierChartInstance = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Simulated Portfolios',
                    data: scatterData,
                    backgroundColor: pointColors,
                    borderColor: 'rgba(0, 217, 255, 0.2)',
                    borderWidth: 0.5, pointRadius: 3, pointHoverRadius: 6,
                },
                {
                    label: 'Max Sharpe Ratio',
                    data: [{ x: data.optimal_portfolio.volatility * 100, y: data.optimal_portfolio.expected_return * 100 }],
                    backgroundColor: '#00D9FF', borderColor: '#000000', borderWidth: 3,
                    pointRadius: 15, pointHoverRadius: 18, pointStyle: 'star',
                },
                {
                    label: 'Min Volatility',
                    data: [{ x: data.min_risk_portfolio.volatility * 100, y: data.min_risk_portfolio.expected_return * 100 }],
                    backgroundColor: '#B026FF', borderColor: '#000000', borderWidth: 3,
                    pointRadius: 15, pointHoverRadius: 18, pointStyle: 'star',
                }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true, position: 'top',
                    labels: { color: currentTheme === 'dark' ? '#ffffff' : '#1a1a1a', font: { size: 14, weight: 'bold' }, padding: 20, usePointStyle: true }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.97)', titleColor: '#00D9FF', bodyColor: '#ffffff',
                    borderColor: 'rgba(0, 217, 255, 0.4)', borderWidth: 2, padding: 12,
                    callbacks: {
                        label: function(context) {
                            if (context.datasetIndex === 0) {
                                return [`Return: ${context.parsed.y.toFixed(2)}%`, `Volatility: ${context.parsed.x.toFixed(2)}%`, `Sharpe: ${context.raw.sharpe.toFixed(3)}`];
                            } else {
                                return [context.dataset.label, `Return: ${context.parsed.y.toFixed(2)}%`, `Volatility: ${context.parsed.x.toFixed(2)}%`];
                            }
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Annualized Volatility (%)', color: '#00D9FF', font: { size: 14, weight: 'bold' } },
                    grid: { color: 'rgba(0, 217, 255, 0.08)' },
                    ticks: { color: '#999999', callback: function(value) { return value.toFixed(1) + '%'; } }
                },
                y: {
                    title: { display: true, text: 'Expected Annual Return (%)', color: '#00D9FF', font: { size: 14, weight: 'bold' } },
                    grid: { color: 'rgba(0, 217, 255, 0.08)' },
                    ticks: { color: '#999999', callback: function(value) { return value.toFixed(1) + '%'; } }
                }
            }
        }
    });
}

// ── Input Listeners ───────────────────────────────────────

document.getElementById('ticker').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') document.getElementById('quantity').focus();
});

document.getElementById('quantity').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') addHolding();
});

window.onclick = function(event) {
    if (event.target.classList.contains('modal')) {
        event.target.classList.remove('show');
    }
}

// ══════════════════════════════════════════════════════════
// Interactive Charts Functions
// ══════════════════════════════════════════════════════════

const plotlyDarkLayout = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0.4)',
    font: { color: '#B026FF', family: 'Inter, sans-serif' },
    xaxis: { gridcolor: 'rgba(0,217,255,0.1)', zerolinecolor: 'rgba(0,217,255,0.2)' },
    yaxis: { gridcolor: 'rgba(0,217,255,0.1)', zerolinecolor: 'rgba(0,217,255,0.2)' },
    margin: { l: 50, r: 30, t: 40, b: 40 },
    hoverlabel: { bgcolor: 'rgba(0,0,0,0.95)', bordercolor: '#00D9FF', font: { color: '#ffffff' } }
};

const plotlyConfig = {
    responsive: true, displayModeBar: true, displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d']
};

function getChartTicker() {
    const t = document.getElementById('chartTicker').value.trim().toUpperCase();
    if (!t && holdings.length > 0) return holdings[0].ticker;
    return t || 'AAPL';
}

function getChartPeriod() {
    return document.getElementById('chartPeriod').value;
}

// ── 1. Candlestick Chart ──────────────────────────────────

async function loadCandlestickChart() {
    const ticker = getChartTicker();
    const period = getChartPeriod();
    const statusEl = document.getElementById('candlestickStatus');
    statusEl.textContent = `Loading ${ticker} data...`;

    try {
        const resp = await fetch('http://localhost:5000/api/charts/candlestick', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ticker, period })
        });
        const data = await resp.json();
        if (data.error) { statusEl.textContent = data.error; return; }

        const candles = data.candles;
        const dates = candles.map(c => c.date);

        const candleTrace = {
            x: dates, open: candles.map(c => c.open), high: candles.map(c => c.high),
            low: candles.map(c => c.low), close: candles.map(c => c.close),
            type: 'candlestick', name: ticker,
            increasing: { line: { color: '#00D9FF' }, fillcolor: 'rgba(0,217,255,0.3)' },
            decreasing: { line: { color: '#FF4444' }, fillcolor: 'rgba(255,68,68,0.3)' },
            yaxis: 'y2'
        };

        const colors = candles.map(c => c.close >= c.open ? 'rgba(0,217,255,0.4)' : 'rgba(255,68,68,0.4)');
        const volumeTrace = {
            x: dates, y: candles.map(c => c.volume), type: 'bar',
            marker: { color: colors }, name: 'Volume', yaxis: 'y'
        };

        const layout = {
            ...plotlyDarkLayout,
            title: { text: `${ticker} — Candlestick & Volume`, font: { color: '#00D9FF', size: 16 } },
            yaxis: { domain: [0, 0.25], gridcolor: 'rgba(0,217,255,0.1)', title: 'Volume' },
            yaxis2: { domain: [0.3, 1], gridcolor: 'rgba(0,217,255,0.1)', title: 'Price' },
            xaxis: { gridcolor: 'rgba(0,217,255,0.1)', rangeslider: { visible: false } },
            showlegend: false, margin: { l: 60, r: 30, t: 50, b: 40 }
        };

        Plotly.newPlot('candlestickChart', [volumeTrace, candleTrace], layout, plotlyConfig);
        statusEl.textContent = `${ticker} — ${candles.length} data points loaded`;
    } catch (err) {
        statusEl.textContent = 'Failed to load. Is the server running?';
        console.error(err);
    }
}

// ── 2. Technical Indicators ───────────────────────────────

async function loadTechnicalIndicators() {
    const ticker = getChartTicker();
    const period = getChartPeriod();
    const statusEl = document.getElementById('technicalStatus');
    statusEl.textContent = `Calculating indicators for ${ticker}...`;

    try {
        const resp = await fetch('http://localhost:5000/api/charts/technical-indicators', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ticker, period })
        });
        const data = await resp.json();
        if (data.error) { statusEl.textContent = data.error; return; }

        // Bollinger Bands
        const bbLayout = {
            ...plotlyDarkLayout,
            title: { text: `${ticker} — Bollinger Bands (20, 2)`, font: { color: '#00D9FF', size: 14 } },
            showlegend: true, legend: { orientation: 'h', y: -0.15, font: { size: 11 } },
            margin: { l: 55, r: 20, t: 40, b: 50 }
        };
        Plotly.newPlot('bollingerChart', [
            { x: data.dates, y: data.bollinger.upper, type: 'scatter', mode: 'lines', line: { color: 'rgba(176,38,255,0.5)', width: 1 }, name: 'Upper Band' },
            { x: data.dates, y: data.bollinger.lower, type: 'scatter', mode: 'lines', line: { color: 'rgba(176,38,255,0.5)', width: 1 }, name: 'Lower Band', fill: 'tonexty', fillcolor: 'rgba(176,38,255,0.08)' },
            { x: data.dates, y: data.bollinger.middle, type: 'scatter', mode: 'lines', line: { color: '#B026FF', width: 1.5, dash: 'dot' }, name: 'SMA 20' },
            { x: data.dates, y: data.close, type: 'scatter', mode: 'lines', line: { color: '#00D9FF', width: 2 }, name: 'Close' }
        ], bbLayout, plotlyConfig);

        // RSI
        const rsiLayout = {
            ...plotlyDarkLayout,
            title: { text: 'RSI (14)', font: { color: '#00D9FF', size: 13 } },
            yaxis: { ...plotlyDarkLayout.yaxis, range: [0, 100] },
            shapes: [
                { type: 'line', x0: data.dates[0], x1: data.dates[data.dates.length-1], y0: 70, y1: 70, line: { color: 'rgba(255,68,68,0.5)', dash: 'dash', width: 1 } },
                { type: 'line', x0: data.dates[0], x1: data.dates[data.dates.length-1], y0: 30, y1: 30, line: { color: 'rgba(0,217,255,0.5)', dash: 'dash', width: 1 } }
            ],
            showlegend: false, margin: { l: 45, r: 15, t: 35, b: 30 }
        };
        Plotly.newPlot('rsiChart', [{
            x: data.dates, y: data.rsi, type: 'scatter', mode: 'lines',
            line: { color: '#B026FF', width: 2 }, name: 'RSI'
        }], rsiLayout, plotlyConfig);

        // MACD
        const histColors = data.macd.histogram.map(v => v === null ? 'gray' : v >= 0 ? 'rgba(0,217,255,0.6)' : 'rgba(255,68,68,0.6)');
        const macdLayout = {
            ...plotlyDarkLayout,
            title: { text: 'MACD (12, 26, 9)', font: { color: '#00D9FF', size: 13 } },
            showlegend: true, legend: { orientation: 'h', y: -0.25, font: { size: 10 } },
            margin: { l: 45, r: 15, t: 35, b: 45 }
        };
        Plotly.newPlot('macdChart', [
            { x: data.dates, y: data.macd.histogram, type: 'bar', marker: { color: histColors }, name: 'Histogram' },
            { x: data.dates, y: data.macd.macd_line, type: 'scatter', mode: 'lines', line: { color: '#00D9FF', width: 2 }, name: 'MACD' },
            { x: data.dates, y: data.macd.signal_line, type: 'scatter', mode: 'lines', line: { color: '#ffffff', width: 2 }, name: 'Signal' }
        ], macdLayout, plotlyConfig);

        statusEl.textContent = `${ticker} technical analysis complete`;
    } catch (err) {
        statusEl.textContent = 'Failed to load. Is the server running?';
        console.error(err);
    }
}

// ── 3. Time Series Decomposition ──────────────────────────

async function loadTimeSeriesDecomposition() {
    const ticker = getChartTicker();
    const period = getChartPeriod();
    const statusEl = document.getElementById('decompositionStatus');
    statusEl.textContent = `Decomposing ${ticker} time series...`;

    try {
        const resp = await fetch('http://localhost:5000/api/charts/time-series-decomposition', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ticker, period })
        });
        const data = await resp.json();
        if (data.error) { statusEl.textContent = data.error; return; }

        const layout = {
            ...plotlyDarkLayout,
            title: { text: `${ticker} — Time Series Decomposition`, font: { color: '#B026FF', size: 16 } },
            grid: { rows: 4, columns: 1, subplots: [['xy'], ['xy2'], ['xy3'], ['xy4']] },
            xaxis: { ...plotlyDarkLayout.xaxis },
            yaxis: { ...plotlyDarkLayout.yaxis, domain: [0.78, 1], title: { text: 'Original', font: { size: 11 } } },
            xaxis2: { ...plotlyDarkLayout.xaxis, anchor: 'y2' },
            yaxis2: { ...plotlyDarkLayout.yaxis, domain: [0.53, 0.74], title: { text: 'Trend', font: { size: 11 } } },
            xaxis3: { ...plotlyDarkLayout.xaxis, anchor: 'y3' },
            yaxis3: { ...plotlyDarkLayout.yaxis, domain: [0.28, 0.49], title: { text: 'Seasonal', font: { size: 11 } } },
            xaxis4: { ...plotlyDarkLayout.xaxis, anchor: 'y4' },
            yaxis4: { ...plotlyDarkLayout.yaxis, domain: [0, 0.24], title: { text: 'Residual', font: { size: 11 } } },
            showlegend: false, margin: { l: 65, r: 20, t: 50, b: 40 }
        };

        Plotly.newPlot('decompositionChart', [
            { x: data.dates, y: data.original, type: 'scatter', mode: 'lines', line: { color: '#00D9FF', width: 1.5 }, name: 'Original', xaxis: 'x', yaxis: 'y' },
            { x: data.dates, y: data.trend, type: 'scatter', mode: 'lines', line: { color: '#ffffff', width: 2 }, name: 'Trend', xaxis: 'x2', yaxis: 'y2' },
            { x: data.dates, y: data.seasonality, type: 'scatter', mode: 'lines', line: { color: '#B026FF', width: 1.5 }, name: 'Seasonality', xaxis: 'x3', yaxis: 'y3' },
            { x: data.dates, y: data.residual, type: 'scatter', mode: 'lines', line: { color: '#FF4444', width: 1 }, name: 'Residual', xaxis: 'x4', yaxis: 'y4' }
        ], layout, plotlyConfig);

        statusEl.textContent = `${ticker} decomposition complete`;
    } catch (err) {
        statusEl.textContent = 'Failed to load. Is the server running?';
        console.error(err);
    }
}

// ── 4. Drawdown Chart ─────────────────────────────────────

async function loadDrawdownChart() {
    const statusEl = document.getElementById('drawdownStatus');
    const tickers = holdings.map(h => h.ticker);
    if (tickers.length === 0) { const t = getChartTicker(); if (t) tickers.push(t); }
    if (tickers.length === 0) { statusEl.textContent = 'Add holdings or enter a ticker'; return; }

    statusEl.textContent = 'Calculating drawdowns...';

    try {
        const resp = await fetch('http://localhost:5000/api/charts/drawdown', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ tickers, start_date: document.getElementById('startDate').value, benchmark: document.getElementById('benchmark').value })
        });
        const data = await resp.json();
        if (data.error) { statusEl.textContent = data.error; return; }

        const layout = {
            ...plotlyDarkLayout,
            title: { text: 'Underwater (Drawdown) Chart', font: { color: '#FF4444', size: 14 } },
            yaxis: { ...plotlyDarkLayout.yaxis, title: 'Drawdown %', ticksuffix: '%' },
            showlegend: true, legend: { orientation: 'h', y: -0.15, font: { size: 11 } },
            margin: { l: 55, r: 20, t: 40, b: 50 }
        };

        Plotly.newPlot('drawdownChart', [
            { x: data.dates, y: data.portfolio_drawdown, type: 'scatter', mode: 'lines', fill: 'tozeroy', fillcolor: 'rgba(255,68,68,0.15)', line: { color: '#FF4444', width: 2 }, name: 'Portfolio' },
            { x: data.dates, y: data.benchmark_drawdown, type: 'scatter', mode: 'lines', line: { color: 'rgba(0,217,255,0.6)', width: 1.5, dash: 'dot' }, name: 'Benchmark' }
        ], layout, plotlyConfig);

        document.getElementById('drawdownStats').classList.remove('hidden');
        document.getElementById('maxDrawdownVal').textContent = data.max_drawdown.toFixed(2) + '%';
        document.getElementById('maxDrawdownDate').textContent = 'On ' + data.max_drawdown_date;
        statusEl.textContent = 'Drawdown analysis complete';
    } catch (err) {
        statusEl.textContent = 'Failed to load. Is the server running?';
        console.error(err);
    }
}

// ── 5. Risk Contribution Pie Chart ────────────────────────

async function loadRiskContribution() {
    const statusEl = document.getElementById('riskContribStatus');
    const tickers = holdings.map(h => h.ticker);
    if (tickers.length < 2) { statusEl.textContent = 'Need at least 2 holdings'; return; }

    statusEl.textContent = 'Calculating risk contributions...';

    try {
        const resp = await fetch('http://localhost:5000/api/charts/risk-contribution', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ tickers, start_date: document.getElementById('startDate').value })
        });
        const data = await resp.json();
        if (data.error) { statusEl.textContent = data.error; return; }

        const chartColors = ['#00D9FF', '#B026FF', '#FF4444', '#ffffff', '#999999',
                             '#FFD700', '#FF69B4', '#1a1a1a', '#9370DB', '#32CD32'];

        const layout = {
            ...plotlyDarkLayout,
            title: { text: `Risk Contribution (Portfolio Vol: ${data.portfolio_volatility}%)`, font: { color: '#ffffff', size: 14 } },
            showlegend: true, margin: { l: 20, r: 20, t: 45, b: 20 }
        };

        Plotly.newPlot('riskContributionChart', [{
            labels: data.contributions.map(c => c.ticker),
            values: data.contributions.map(c => Math.max(0, c.risk_contribution)),
            type: 'pie', hole: 0.45,
            marker: { colors: chartColors.slice(0, data.contributions.length), line: { color: 'rgba(0,0,0,0.5)', width: 2 } },
            textinfo: 'label+percent', textfont: { color: '#ffffff', size: 12 },
            hovertemplate: '%{label}<br>Risk: %{value:.1f}%<br>Weight: %{customdata:.1f}%<extra></extra>',
            customdata: data.contributions.map(c => c.weight)
        }], layout, plotlyConfig);

        let tableHtml = '<table class="w-full text-sm"><thead><tr>';
        tableHtml += '<th class="p-2 text-left text-primary">Ticker</th>';
        tableHtml += '<th class="p-2 text-center text-primary">Weight</th>';
        tableHtml += '<th class="p-2 text-center text-primary">Risk Contrib.</th>';
        tableHtml += '<th class="p-2 text-center text-primary">Ann. Vol</th>';
        tableHtml += '</tr></thead><tbody>';
        data.contributions.forEach(c => {
            tableHtml += `<tr class="border-t border-muted/30">`;
            tableHtml += `<td class="p-2 font-semibold text-white">${c.ticker}</td>`;
            tableHtml += `<td class="p-2 text-center text-cream">${c.weight.toFixed(1)}%</td>`;
            tableHtml += `<td class="p-2 text-center text-secondary font-semibold">${c.risk_contribution.toFixed(1)}%</td>`;
            tableHtml += `<td class="p-2 text-center text-cream">${c.annual_vol.toFixed(1)}%</td>`;
            tableHtml += '</tr>';
        });
        tableHtml += '</tbody></table>';
        document.getElementById('riskContribTable').innerHTML = tableHtml;

        statusEl.textContent = 'Risk decomposition complete';
    } catch (err) {
        statusEl.textContent = 'Failed to load. Is the server running?';
        console.error(err);
    }
}

// ── 6. 3D Efficient Frontier ──────────────────────────────

async function loadEfficientFrontier3D() {
    const statusEl = document.getElementById('frontier3dStatus');
    const tickers = holdings.map(h => h.ticker);
    if (tickers.length < 2) { statusEl.textContent = 'Need at least 2 holdings'; return; }

    statusEl.textContent = 'Generating 5000 random portfolios...';

    try {
        const resp = await fetch('http://localhost:5000/api/charts/efficient-frontier-3d', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                tickers, start_date: document.getElementById('startDate').value,
                num_portfolios: 5000, risk_free_rate: parseFloat(document.getElementById('riskFreeRate').value) / 100
            })
        });
        const data = await resp.json();
        if (data.error) { statusEl.textContent = data.error; return; }

        const pts = data.points;
        const hoverTexts = pts.map(p => {
            let txt = `Return: ${(p.return * 100).toFixed(2)}%<br>Risk: ${(p.volatility * 100).toFixed(2)}%<br>Sharpe: ${p.sharpe.toFixed(3)}<br>---`;
            for (const [t, w] of Object.entries(p.weights)) { txt += `<br>${t}: ${(w * 100).toFixed(1)}%`; }
            return txt;
        });

        let maxSharpeIdx = 0, minVolIdx = 0;
        pts.forEach((p, i) => {
            if (p.sharpe > pts[maxSharpeIdx].sharpe) maxSharpeIdx = i;
            if (p.volatility < pts[minVolIdx].volatility) minVolIdx = i;
        });

        const trace3d = {
            x: pts.map(p => (p.volatility * 100).toFixed(2)),
            y: pts.map(p => (p.return * 100).toFixed(2)),
            z: pts.map(p => p.sharpe.toFixed(3)),
            mode: 'markers', type: 'scatter3d',
            marker: {
                size: 3, color: pts.map(p => p.sharpe),
                colorscale: [[0, '#FF4444'], [0.25, '#B026FF'], [0.5, '#ffffff'], [0.75, '#00D9FF'], [1, '#1a1a1a']],
                colorbar: { title: 'Sharpe', tickfont: { color: '#B026FF' }, titlefont: { color: '#B026FF' } },
                opacity: 0.7
            },
            text: hoverTexts, hoverinfo: 'text', name: 'Portfolios'
        };

        const starTrace = {
            x: [(pts[maxSharpeIdx].volatility * 100).toFixed(2), (pts[minVolIdx].volatility * 100).toFixed(2)],
            y: [(pts[maxSharpeIdx].return * 100).toFixed(2), (pts[minVolIdx].return * 100).toFixed(2)],
            z: [pts[maxSharpeIdx].sharpe.toFixed(3), pts[minVolIdx].sharpe.toFixed(3)],
            mode: 'markers', type: 'scatter3d',
            marker: { size: 10, color: ['#ffffff', '#00D9FF'], symbol: 'diamond', line: { color: '#fff', width: 2 } },
            text: ['Max Sharpe', 'Min Volatility'], hoverinfo: 'text', name: 'Optimal'
        };

        const layout3d = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#B026FF', family: 'Inter, sans-serif' },
            title: { text: '3D Efficient Frontier', font: { color: '#00D9FF', size: 16 } },
            scene: {
                xaxis: { title: 'Volatility (%)', gridcolor: 'rgba(0,217,255,0.15)', color: '#999999' },
                yaxis: { title: 'Return (%)', gridcolor: 'rgba(176,38,255,0.15)', color: '#999999' },
                zaxis: { title: 'Sharpe Ratio', gridcolor: 'rgba(255,255,255,0.15)', color: '#999999' },
                bgcolor: 'rgba(0,0,0,0.05)',
                camera: { eye: { x: 1.5, y: 1.5, z: 1.2 } }
            },
            showlegend: false, margin: { l: 0, r: 0, t: 50, b: 0 }
        };

        Plotly.newPlot('frontier3dChart', [trace3d, starTrace], layout3d, plotlyConfig);
        statusEl.textContent = `${pts.length} portfolios plotted — Drag to rotate, scroll to zoom`;
    } catch (err) {
        statusEl.textContent = 'Failed to load. Is the server running?';
        console.error(err);
    }
}

// ══════════════════════════════════════════════════════════
// Market Intel Functions
// ══════════════════════════════════════════════════════════

let sectorChartInstance = null;

function displaySectorChart(data) {
    const ctx = document.getElementById('sectorChart').getContext('2d');
    if (sectorChartInstance) sectorChartInstance.destroy();

    const labels = Object.keys(data);
    const values = Object.values(data);
const colors = ['#00D9FF', '#B026FF', '#ffffff', '#999999', '#1a1a1a', '#FF4444', '#FFD700', '#000000'];

    sectorChartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{ data: values, backgroundColor: colors.slice(0, labels.length), borderColor: 'rgba(0, 0, 0, 0.8)', borderWidth: 2, hoverOffset: 8 }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom', labels: { color: currentTheme === 'dark' ? '#ffffff' : '#1a1a1a', font: { size: 12, weight: 'bold' }, padding: 15 } },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.97)', titleColor: '#00D9FF', bodyColor: '#ffffff',
                    borderColor: 'rgba(0, 217, 255, 0.4)', borderWidth: 2, padding: 10,
                    callbacks: { label: function(context) { return context.label + ': ' + context.parsed + ' stock(s)'; } }
                }
            }
        }
    });
}

function displayNews(articlesData) {
    const newsList = document.getElementById('newsList');
    if (!articlesData || Object.keys(articlesData).length === 0) {
        newsList.innerHTML = '<li class="text-muted text-sm">No news available</li>';
        return;
    }

    let allArticles = [];
    
    // Handle both legacy format (array) and new format (per-ticker object)
    if (Array.isArray(articlesData)) {
        allArticles = articlesData;
    } else {
        // articlesData is an object with ticker keys
        Object.entries(articlesData).forEach(([ticker, articles]) => {
            if (Array.isArray(articles)) {
                articles.forEach(article => {
                    article.ticker = ticker;  // Add ticker to article for display
                    allArticles.push(article);
                });
            }
        });
    }

    if (allArticles.length === 0) {
        newsList.innerHTML = '<li class="text-muted text-sm">No news available</li>';
        return;
    }

    newsList.innerHTML = allArticles.slice(0, 10).map(article => `
        <li class="glass rounded-lg p-4 border border-secondary/20 hover:border-secondary/50 transition-all">
            <a href="${article.url}" target="_blank" class="block group">
                <div class="flex justify-between items-start mb-2">
                    <h4 class="font-semibold text-primary group-hover:text-cream transition-colors text-sm mb-2 line-clamp-2 flex-1">
                        ${article.title || 'Untitled'}
                    </h4>
                    ${article.ticker ? `<span class="text-xs bg-secondary/20 text-secondary px-2 py-1 rounded ml-2 whitespace-nowrap">${article.ticker}</span>` : ''}
                </div>
                <p class="text-xs text-muted mb-2 line-clamp-2">
                    ${article.description || 'No description available'}
                </p>
                <div class="flex items-center justify-between">
                    <span class="text-xs text-muted">${article.source?.name || 'Unknown'}</span>
                    <svg class="w-4 h-4 text-secondary opacity-0 group-hover:opacity-100 transition-opacity" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"></path>
                    </svg>
                </div>
            </a>
        </li>
    `).join('');
}

async function fetchSectorData() {
    try {
        const response = await fetch('http://localhost:5000/api/sectors', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ tickers: holdings.map(h => h.ticker) })
        });
        const data = await response.json();
        if (!data.error) displaySectorChart(data);
    } catch (err) { console.log('Sector data unavailable'); }
}

async function fetchNewsData() {
    try {
        if (holdings.length > 0) {
            const response = await fetch('http://localhost:5000/api/news', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ tickers: holdings.map(h => h.ticker) })
            });
            const data = await response.json();
            if (Array.isArray(data)) displayNews(data);
            else if (data.articles) displayNews(data.articles);
            else if (data.articles_by_ticker) displayNews(data.articles_by_ticker);
        }
    } catch (err) { console.log('News data unavailable'); }
}


// ── Live Prices ───────────────────────────────────────────

async function refreshLivePrices() {
    if (holdings.length === 0) { showError('No holdings to display'); return; }

    const btn = document.querySelector('[onclick="refreshLivePrices()"]');
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<div class="spinner" style="display: inline-block; margin-right: 4px;"></div>Refreshing...';
    }

    try {
        const response = await fetch('http://localhost:5000/api/prices', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ tickers: holdings.map(h => h.ticker) })
        });
        const data = await response.json();
        if (data.error) { showError('Failed to fetch prices: ' + data.error); }
        else { displayLivePrices(data.prices); updatePortfolioValue(data.prices); showSuccess('Prices updated successfully'); }
    } catch (err) {
        showError('Failed to connect to price server');
        console.log('Error fetching prices:', err);
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = '<svg class="w-4 h-4 inline mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path></svg>Refresh';
        }
    }
}

function displayLivePrices(pricesData) {
    const livePricesList = document.getElementById('livePrices');
    if (!pricesData || Object.keys(pricesData).length === 0) {
        livePricesList.innerHTML = '<li class="text-muted text-sm p-2">No price data available</li>';
        return;
    }

    livePricesList.innerHTML = holdings.map(h => {
        const priceInfo = pricesData[h.ticker];
        if (!priceInfo || priceInfo.error) {
            return `<li class="flex items-center justify-between py-2 px-3 bg-red-500/10 border border-red-500/20 rounded mb-2 text-red-400 text-xs">
                <span>${h.ticker}</span><span>${priceInfo?.error || 'No data'}</span></li>`;
        }
        const price = priceInfo.current_price;
        const dailyChange = priceInfo.daily_change;
        const dailyChangePct = priceInfo.daily_change_pct;
        const changeColor = dailyChange >= 0 ? 'text-green-400' : 'text-red-500';
        const changeIcon = dailyChange >= 0 ? '▲' : '▼';
        const holdingValue = (price * h.quantity).toFixed(2);

        return `
            <li class="flex flex-col py-2 px-3 bg-primary/5 border border-primary/20 rounded mb-2 text-xs font-mono">
                <div class="flex items-center justify-between mb-1">
                    <span class="font-semibold text-white">${h.ticker}</span>
                    <span class="text-primary">$${price.toFixed(2)}</span>
                </div>
                <div class="flex items-center justify-between">
                    <span class="text-muted">${h.quantity} shares</span>
                    <span class="${changeColor}"> ${changeIcon} ${Math.abs(dailyChangePct).toFixed(2)}% ($${Math.abs(dailyChange).toFixed(2)})</span>
                </div>
                <div class="text-right text-muted mt-1">Value: $${holdingValue}</div>
            </li>`;
    }).join('');
}

function updatePortfolioValue(pricesData) {
    let totalValue = 0;
    let totalPreviousValue = 0;

    holdings.forEach(h => {
        const priceInfo = pricesData[h.ticker];
        if (priceInfo && priceInfo.current_price) {
            totalValue += priceInfo.current_price * h.quantity;
            totalPreviousValue += priceInfo.previous_close * h.quantity;
        }
    });

    const totalChange = totalValue - totalPreviousValue;
    const totalChangePct = totalPreviousValue > 0 ? (totalChange / totalPreviousValue) * 100 : 0;
    const changeColor = totalChange >= 0 ? 'text-green-400' : 'text-red-500';

    document.getElementById('portfolioValue').textContent = '$' + totalValue.toFixed(2);
    document.getElementById('dailyChange').innerHTML = `<span class="${changeColor}">${totalChange >= 0 ? '+' : ''}$${totalChange.toFixed(2)} (${totalChangePct.toFixed(2)}%)</span>`;
}

// ── Dividend Tracking ─────────────────────────────────────

async function fetchDividendData() {
    if (holdings.length === 0) return;
    try {
        const ticker = holdings[0].ticker;
        const response = await fetch('http://localhost:5000/api/dividends', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ticker: ticker })
        });
        const data = await response.json();
        if (!data.error) displayDividendData(data, ticker);
    } catch (err) { console.log('Dividend data unavailable:', err); }
}

function displayDividendData(data, ticker) {
    document.getElementById('divTicker').textContent = ticker;
    document.getElementById('divYield').textContent = data.yield != null ? (data.yield * 100).toFixed(2) + '%' : 'N/A';

    const historyEl = document.getElementById('divHistory');
    if (data.history && Object.keys(data.history).length > 0) {
        historyEl.innerHTML = Object.entries(data.history).reverse().map(([date, amount]) => `
            <div class="glass rounded-lg p-3 border border-green-500/10 flex justify-between items-center">
                <span class="text-sm text-cream">${date}</span>
                <span class="text-sm font-semibold text-green-400">$${amount.toFixed(4)}</span>
            </div>
        `).join('');
    } else {
        historyEl.innerHTML = '<p class="text-muted text-sm">No dividend history available for this stock</p>';
    }
}

// ── Economic Indicators ───────────────────────────────────

async function fetchEconomicIndicators() {
    try {
        const response = await fetch('http://localhost:5000/api/economic-indicators');
        const data = await response.json();
        if (!data.error) displayEconomicIndicators(data);
    } catch (err) { console.log('Economic indicators unavailable:', err); }
}

function displayEconomicIndicators(data) {
    const treasuryEl = document.getElementById('treasuryData');
    const marketEl = document.getElementById('marketData');

    if (data.treasury && data.treasury.length > 0) {
        treasuryEl.innerHTML = data.treasury.map(item => {
            const changeColor = item.change >= 0 ? 'text-green-400' : 'text-red-400';
            const arrow = item.change >= 0 ? '&#9650;' : '&#9660;';
            return `<div class="glass rounded-xl p-4 border border-secondary/20">
                <p class="text-xs text-muted uppercase tracking-wider mb-1">${item.name}</p>
                <p class="text-xl font-bold text-secondary">${item.value}${item.unit}</p>
                <p class="text-xs ${changeColor} mt-1">${arrow} ${Math.abs(item.change).toFixed(2)}${item.unit}</p>
            </div>`;
        }).join('');
    }

    if (data.market && data.market.length > 0) {
        marketEl.innerHTML = data.market.map(item => {
            const changeColor = item.change >= 0 ? 'text-green-400' : 'text-red-400';
            const arrow = item.change >= 0 ? '&#9650;' : '&#9660;';
            return `<div class="glass rounded-xl p-4 border border-secondary/20">
                <p class="text-xs text-muted uppercase tracking-wider mb-1">${item.name}</p>
                <p class="text-xl font-bold text-secondary">${item.value.toLocaleString()}</p>
                <p class="text-xs ${changeColor} mt-1">${arrow} ${Math.abs(item.change).toFixed(2)}%</p>
            </div>`;
        }).join('');
    }
}

// ── Market Sentiment ──────────────────────────────────────

async function fetchSentimentData() {
    if (holdings.length === 0) return;
    try {
        const response = await fetch('http://localhost:5000/api/sentiment', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ tickers: holdings.map(h => h.ticker) })
        });
        const data = await response.json();
        if (!data.error || data.sentiment != null) displaySentimentData(data);
    } catch (err) { console.log('Sentiment data unavailable:', err); }
}

function displaySentimentData(data) {
    const scoreEl = document.getElementById('sentimentScore');
    const labelEl = document.getElementById('sentimentLabel');
    const posEl = document.getElementById('sentimentPos');
    const negEl = document.getElementById('sentimentNeg');
    const headlinesEl = document.getElementById('sentimentHeadlines');

    let aggregatedScore = 0;
    let aggregatedBreakdown = { positive: 0, negative: 0, neutral: 0, total: 0 };
    let aggregatedHeadlines = [];

    // Handle both legacy format (single sentiment) and new format (per-ticker)
    if (data.sentiment_by_ticker) {
        // Multiple tickers - aggregate and show per-ticker details
        const tickers = Object.keys(data.sentiment_by_ticker);
        const sentiments = Object.values(data.sentiment_by_ticker);
        
        // Calculate aggregate sentiment
        aggregatedScore = sentiments.reduce((sum, s) => sum + (s.sentiment || 0), 0) / sentiments.length;
        
        // Aggregate breakdowns
        sentiments.forEach(s => {
            if (s.breakdown) {
                aggregatedBreakdown.positive += s.breakdown.positive || 0;
                aggregatedBreakdown.negative += s.breakdown.negative || 0;
                aggregatedBreakdown.neutral += s.breakdown.neutral || 0;
                aggregatedBreakdown.total += s.breakdown.total || 0;
            }
            if (s.headlines) {
                s.headlines.forEach(h => {
                    const headline = { ...h, ticker: tickers[sentiments.indexOf(s)] };
                    aggregatedHeadlines.push(headline);
                });
            }
        });
    } else {
        // Single ticker (legacy format)
        aggregatedScore = data.sentiment || 0;
        aggregatedBreakdown = data.breakdown || { positive: 0, negative: 0, neutral: 0, total: 0 };
        aggregatedHeadlines = data.headlines || [];
    }

    // Display aggregated sentiment score
    scoreEl.textContent = aggregatedScore.toFixed(3);
    if (aggregatedScore > 0.05) {
        scoreEl.className = 'text-3xl font-bold text-green-400';
        labelEl.textContent = 'Bullish';
        labelEl.className = 'text-sm text-green-400 mt-1';
    } else if (aggregatedScore < -0.05) {
        scoreEl.className = 'text-3xl font-bold text-red-400';
        labelEl.textContent = 'Bearish';
        labelEl.className = 'text-sm text-red-400 mt-1';
    } else {
        scoreEl.className = 'text-3xl font-bold text-secondary';
        labelEl.textContent = 'Neutral';
        labelEl.className = 'text-sm text-secondary mt-1';
    }

    // Display breakdown
    posEl.textContent = aggregatedBreakdown.positive || 0;
    negEl.textContent = aggregatedBreakdown.negative || 0;

    // Display headlines with ticker info
    if (aggregatedHeadlines.length > 0) {
        headlinesEl.innerHTML = aggregatedHeadlines.slice(0, 10).map(h => {
            const color = h.label === 'Positive' ? 'text-green-400' : h.label === 'Negative' ? 'text-red-400' : 'text-secondary';
            const badge = h.label === 'Positive' ? 'bg-green-500/20 border-green-500/30' : h.label === 'Negative' ? 'bg-red-500/20 border-red-500/30' : 'bg-secondary/20 border-secondary/30';
            return `<li class="glass rounded-lg p-3 border border-secondary/10 flex items-start gap-3">
                <div class="flex gap-2 flex-wrap">
                    <span class="text-xs px-2 py-1 rounded border ${badge} ${color} whitespace-nowrap">${h.label}</span>
                    ${h.ticker ? `<span class="text-xs bg-secondary/20 text-secondary px-2 py-1 rounded border border-secondary/30 whitespace-nowrap">${h.ticker}</span>` : ''}
                </div>
                <span class="text-sm text-cream flex-1">${h.text}</span>
                <span class="text-xs ${color} whitespace-nowrap">${h.score}</span>
            </li>`;
        }).join('');
    }
}


// ── Hook Analysis to Fetch Everything ─────────────────────

const originalAnalyzePortfolioLive = analyzePortfolio;
analyzePortfolio = async function() {
    await originalAnalyzePortfolioLive.call(this);
    if (holdings.length > 0) {
        fetchSectorData(); fetchNewsData(); refreshLivePrices(); fetchDividendData(); fetchSentimentData();
    }
    fetchEconomicIndicators();
}

// Fetch economic indicators on page load
fetchEconomicIndicators();

// ══════════════════════════════════════════════════════════
// Retirement Planner
// ══════════════════════════════════════════════════════════

let retireConeChartInstance = null;
let retireAllocChartInstance = null;

async function calculateRetirement() {
    const age = parseInt(document.getElementById('retireAge').value);
    const retireAge = parseInt(document.getElementById('retireTargetAge').value);
    const lifeExpect = parseInt(document.getElementById('retireLifeExpect').value);
    const savings = parseFloat(document.getElementById('retireSavings').value);
    const contribution = parseFloat(document.getElementById('retireContribution').value);
    const spending = parseFloat(document.getElementById('retireSpending').value);
    const target = parseFloat(document.getElementById('retireTarget').value);
    const inflation = parseFloat(document.getElementById('retireInflation').value) / 100;
    const risk = document.getElementById('retireRisk').value;

    if (retireAge <= age) { showError('Retirement age must be greater than current age'); return; }
    if (lifeExpect <= retireAge) { showError('Life expectancy must be greater than retirement age'); return; }

    const btn = document.querySelector('[onclick="calculateRetirement()"]');
    btn.disabled = true;
    btn.innerHTML = '<div class="spinner" style="display:inline-block;margin-right:8px"></div>Running 5,000 simulations...';

    try {
        const response = await fetch('http://localhost:5000/api/retirement/calculate', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                current_savings: savings, monthly_contribution: contribution,
                years_to_retirement: retireAge - age, years_in_retirement: lifeExpect - retireAge,
                annual_spending: spending, target_amount: target, inflation_rate: inflation, risk_tolerance: risk
            })
        });
        const data = await response.json();
        if (data.error) { showError(data.error); return; }
        displayRetirementResults(data);
        document.getElementById('retireResults').classList.remove('hidden');
        showSuccess('Retirement simulation complete!');
    } catch (err) {
        showError('Failed to connect to server: ' + err.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = 'Run Retirement Simulation';
    }
}

function displayRetirementResults(data) {
    const successPct = (data.success_rate * 100).toFixed(1);
    const successEl = document.getElementById('retireSuccess');
    successEl.textContent = successPct + '%';
    if (data.success_rate >= 0.80) successEl.className = 'text-3xl font-bold text-green-400';
    else if (data.success_rate >= 0.60) successEl.className = 'text-3xl font-bold text-secondary';
    else successEl.className = 'text-3xl font-bold text-red-400';

    document.getElementById('retireMedian').textContent = '$' + Number(data.median).toLocaleString(undefined, {maximumFractionDigits:0});

    const sustainEl = document.getElementById('retireSustain');
    sustainEl.textContent = (data.sustainability_rate * 100).toFixed(1) + '%';
    sustainEl.className = data.sustainability_rate >= 0.90 ? 'text-3xl font-bold text-green-400' :
                          data.sustainability_rate >= 0.70 ? 'text-3xl font-bold text-secondary' :
                          'text-3xl font-bold text-red-400';

    document.getElementById('retireIncome').textContent = '$' + Number(data.safe_annual_income).toLocaleString(undefined, {maximumFractionDigits:0});
    document.getElementById('retireSWR').textContent = 'Safe withdrawal rate: ' + (data.safe_withdrawal_rate * 100).toFixed(2) + '%';

    document.getElementById('retireP90').textContent = '$' + Number(data.percentile_90).toLocaleString(undefined, {maximumFractionDigits:0});
    document.getElementById('retireP75').textContent = '$' + Number(data.percentile_75).toLocaleString(undefined, {maximumFractionDigits:0});
    document.getElementById('retireP50').textContent = '$' + Number(data.median).toLocaleString(undefined, {maximumFractionDigits:0});
    document.getElementById('retireP25').textContent = '$' + Number(data.percentile_25).toLocaleString(undefined, {maximumFractionDigits:0});
    document.getElementById('retireP10').textContent = '$' + Number(data.percentile_10).toLocaleString(undefined, {maximumFractionDigits:0});
    document.getElementById('retireTargetDisplay').textContent = '$' + Number(data.target_amount).toLocaleString(undefined, {maximumFractionDigits:0});

    const p90 = data.percentile_90 || 1;
    document.getElementById('retireP90Bar').style.width = '100%';
    document.getElementById('retireP75Bar').style.width = Math.min(100, (data.percentile_75 / p90) * 100) + '%';
    document.getElementById('retireP50Bar').style.width = Math.min(100, (data.median / p90) * 100) + '%';
    document.getElementById('retireP25Bar').style.width = Math.min(100, (data.percentile_25 / p90) * 100) + '%';
    document.getElementById('retireP10Bar').style.width = Math.min(100, (data.percentile_10 / p90) * 100) + '%';

    const recEl = document.getElementById('retireRecommendations');
    recEl.innerHTML = data.recommendations.map(rec => {
        const isWarning = rec.includes('below') || rec.includes('Success probability') && data.success_rate < 0.75;
        const borderColor = isWarning ? 'border-secondary/30' : 'border-green-500/30';
        const textColor = isWarning ? 'text-secondary' : 'text-green-300';
        return `<li class="glass rounded-xl p-4 border ${borderColor}"><p class="text-sm ${textColor}">${rec}</p></li>`;
    }).join('');

    drawConeChart(data.cone, data.target_amount);
    drawAllocChart(data.portfolio);
}

function drawConeChart(cone, target) {
    const ctx = document.getElementById('retireConeChart').getContext('2d');
    if (retireConeChartInstance) retireConeChartInstance.destroy();

    const labels = cone.map(c => 'Year ' + c.year);

    retireConeChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                { label: '90th Percentile', data: cone.map(c => c.p90), borderColor: 'rgba(34, 197, 94, 0.6)', backgroundColor: 'rgba(34, 197, 94, 0.05)', fill: '+1', borderWidth: 1, pointRadius: 0, tension: 0.3 },
                { label: '75th Percentile', data: cone.map(c => c.p75), borderColor: 'rgba(0, 217, 255, 0.6)', backgroundColor: 'rgba(0, 217, 255, 0.08)', fill: '+1', borderWidth: 1, pointRadius: 0, tension: 0.3 },
                { label: 'Median', data: cone.map(c => c.p50), borderColor: '#B026FF', backgroundColor: 'rgba(176, 38, 255, 0.1)', fill: '+1', borderWidth: 3, pointRadius: 0, tension: 0.3 },
                { label: '25th Percentile', data: cone.map(c => c.p25), borderColor: 'rgba(249, 115, 22, 0.6)', backgroundColor: 'rgba(249, 115, 22, 0.08)', fill: '+1', borderWidth: 1, pointRadius: 0, tension: 0.3 },
                { label: '10th Percentile', data: cone.map(c => c.p10), borderColor: 'rgba(239, 68, 68, 0.6)', backgroundColor: 'transparent', fill: false, borderWidth: 1, pointRadius: 0, tension: 0.3 },
                { label: 'Target', data: cone.map(() => target), borderColor: 'rgba(0, 217, 255, 0.8)', borderDash: [8, 4], borderWidth: 2, pointRadius: 0, fill: false }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            interaction: { intersect: false, mode: 'index' },
            plugins: {
                legend: { labels: { color: '#999999', font: { size: 11 } } },
                tooltip: { callbacks: { label: function(ctx) { return ctx.dataset.label + ': $' + Number(ctx.parsed.y).toLocaleString(undefined, {maximumFractionDigits:0}); } } }
            },
            scales: {
                x: { ticks: { color: '#999999', maxTicksLimit: 10 }, grid: { color: 'rgba(0,217,255,0.06)' } },
                y: { ticks: { color: '#999999', callback: v => '$' + (v >= 1e6 ? (v/1e6).toFixed(1)+'M' : (v/1e3).toFixed(0)+'K') }, grid: { color: 'rgba(0,217,255,0.06)' } }
            }
        }
    });
}

function drawAllocChart(portfolio) {
    const ctx = document.getElementById('retireAllocChart').getContext('2d');
    if (retireAllocChartInstance) retireAllocChartInstance.destroy();

    retireAllocChartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Stocks', 'Bonds', 'Cash'],
            datasets: [{ data: [portfolio.stocks, portfolio.bonds, portfolio.cash], backgroundColor: ['#00D9FF', '#B026FF', '#999999'], borderColor: 'rgba(0,0,0,0.8)', borderWidth: 2, hoverOffset: 8 }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { labels: { color: '#999999', font: { size: 12 }, padding: 15 } } }
        }
    });

    document.getElementById('retireAllocDetails').innerHTML = `
        <div class="flex justify-between text-sm">
            <span class="text-muted">Expected Return</span>
            <span class="text-secondary font-semibold">${(portfolio.expected_return * 100).toFixed(1)}%</span>
        </div>
        <div class="flex justify-between text-sm">
            <span class="text-muted">Expected Volatility</span>
            <span class="text-primary font-semibold">${(portfolio.volatility * 100).toFixed(1)}%</span>
        </div>
    `;
}
