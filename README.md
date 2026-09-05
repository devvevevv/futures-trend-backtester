# Index Futures Trend-Following Backtester
Simulates a Moving-Average Crossover strategy on an index futures contract (e.g. NIFTY futures) and benchmarks it against buy-and-hold using standard risk/return metrics used by trading desks: CAGR, Sharpe Ratio, Max Drawdown, Win Rate, and Profit Factor. Pulls real NIFTY/BankNifty futures-proxy data via yfinance (^NSEI, ^NSEBANK).

## What it does
 
- Generates a long/short trading signal from a 10-day vs. 30-day SMA crossover
- Simulates daily P&L with a realistic transaction cost (5 bps per trade) applied only on position changes
- Computes CAGR, annualized volatility, Sharpe ratio, max drawdown, win rate, and profit factor
- Plots the strategy's equity curve against buy-and-hold, plus a drawdown chart
## Why this design
 
Futures (unlike equities) let you go long or short with equal ease and no borrow constraint, which is what makes a crossover strategy that flips direction on signal changes practical to test in the first place. The backtest also shifts signals by one day to avoid lookahead bias, you can only trade on a signal *after* it's confirmed, not on the same bar it appears.
 
## Data
 
By default the script runs on synthetic price data generated via Geometric Brownian Motion, so it runs anywhere without an internet connection. Set `USE_REAL_DATA = True` and install `yfinance` to pull real index data (e.g. `^NSEI` for NIFTY) instead.
 
```bash
pip install yfinance
```
 
## Usage
 
```bash
pip install pandas numpy matplotlib
python backtester.py
```
 
Outputs:
- Console printout comparing strategy vs. buy-and-hold metrics
- `performance_summary.csv` — the metrics table
- `equity_curve.png` — equity curve and drawdown chart

## Equity curve
<img width="1500" height="1200" alt="equity_curve" src="https://github.com/user-attachments/assets/4cc6fb94-e9fb-4a52-bf8e-8d3581675284" />


## Key finding
 
On the test data, the SMA crossover strategy underperformed simple buy-and-hold, mainly due to whipsaws (frequent false signals) in non-trending periods and the transaction costs those whipsaws generate. This is a common and honest result for basic trend-following; it's a reminder that a strategy has to clear the bar of *both* costs and out-of-sample robustness, not just look good on a backtest.
 
## Possible extensions
 
- Walk-forward / out-of-sample validation instead of a single backtest window
- Volatility-based position sizing instead of a fixed +1/-1 position
- Testing across multiple SMA lookback pairs to check sensitivity to parameter choice
