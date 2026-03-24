export const TOOLTIPS = {
  // Risk metrics
  var_95: 'Value at Risk at 95% confidence: the maximum expected daily loss, exceeded only 5% of trading days.',
  cvar_95: 'Conditional VaR (Expected Shortfall): the average loss on days when VaR is breached. Captures tail risk better than VaR.',
  beta: 'Portfolio sensitivity to market movements. Beta > 1 means more volatile than the market; < 1 means less.',
  alpha: "Jensen's Alpha: excess return over what CAPM predicts. Positive alpha indicates outperformance.",
  rolling_sharpe: 'Sharpe ratio over a rolling 60-day window, showing how risk-adjusted performance changes over time.',
  sharpe: 'Sharpe Ratio: excess return per unit of risk (volatility). Higher is better; above 1.0 is generally good.',
  volatility: 'Annualized standard deviation of returns, measuring total price variability.',
  cagr: 'Compound Annual Growth Rate: the annualized return assuming profits are reinvested.',
  max_drawdown: 'Maximum Drawdown: the largest peak-to-trough decline in portfolio value during the period.',
  correlation: 'Pearson correlation between asset returns. +1 = move together, −1 = move opposite, 0 = no relationship.',

  // Options Greeks
  delta: 'Delta (Δ): option price change per $1 move in the underlying. Calls: 0 to 1, Puts: −1 to 0.',
  gamma: 'Gamma (Γ): rate of change of Delta per $1 move. High near at-the-money and expiry.',
  theta: 'Theta (Θ): daily time decay — how much option value erodes each day, all else equal.',
  vega: 'Vega (ν): option price sensitivity to a 1% change in implied volatility.',
  rho: 'Rho (ρ): option price sensitivity to a 1% change in the risk-free interest rate.',

  // Options parameters
  spot_price: 'Current market price of the underlying asset.',
  strike_price: 'The price at which the option holder can buy (call) or sell (put) the underlying.',
  dte: 'Days to Expiry: calendar days remaining until the option expires.',
  risk_free_rate: 'Theoretical return on a zero-risk investment, typically based on government bond yields.',
  implied_vol: "Implied Volatility: the market's forecast of future price volatility, derived from option prices.",
  open_interest: 'Open Interest: total number of outstanding option contracts not yet settled.',
  option_vol: 'Volume: number of contracts traded during the current session.',

  // Backtest parameters
  fees: 'Transaction fees applied at each rebalance as a percentage of trade value.',
  slippage: 'Slippage: difference between expected and actual execution price due to market impact.',
  rebalance_freq: 'How often the portfolio is rebalanced to target weights. More frequent = higher costs.',

  // Dashboard
  total_value: 'Total current market value of all holdings in your portfolio.',
  daily_change: 'Percentage change in total portfolio value since the previous market close.',

  // Retirement
  success_rate: 'Percentage of Monte Carlo simulations where savings lasted through the entire retirement period.',
  safe_withdrawal: 'Maximum annual withdrawal rate (% of portfolio) that sustains savings through retirement in most scenarios.',
  sustainability: 'Probability that your portfolio balance stays positive throughout the entire retirement period.',
  target_amount: 'Estimated total savings needed at retirement to fund planned spending, adjusted for inflation.',
  inflation_rate: 'Assumed annual rate of price increases, eroding purchasing power over time.',
  median_outcome: '50th percentile outcome from Monte Carlo simulation — half of scenarios are above, half below.',

  // Optimize
  simulations: 'Number of random portfolio weight combinations generated to map the efficient frontier.',
  mc_method: 'Monte Carlo sampling method: Standard (random), Antithetic (variance reduction), Sobol (quasi-random), or Full (combined).',
  expected_return: 'Annualized expected portfolio return based on historical data.',
  hrp: 'Hierarchical Risk Parity: a clustering-based allocation method that builds diversified portfolios without inverting the covariance matrix.',
  black_litterman: 'Black-Litterman: a Bayesian framework blending market equilibrium returns with investor views to produce optimal allocations.',
};
