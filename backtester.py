"""
STRATEGY LOGIC:
  - Go LONG when fast SMA (10-day) crosses above slow SMA (30-day)
  - Go FLAT/SHORT when fast SMA crosses below slow SMA
  - This is a classic trend-following signal used to trade futures because
    futures let you go long or short with equal ease (no borrow constraint,
    unlike shorting equity).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

USE_REAL_DATA = True
np.random.seed(42)

def get_real_data(ticker="^NSEI", period="3y"):
    import yfinance as yf
    df = yf.download(ticker, period=period, interval="1d", progress=False)
    df = df[["Close"]].rename(columns={"Close": "close"})
    df.index.name = "date"
    return df

def get_synthetic_data(n_days=750, s0=22000, mu=0.10, sigma=0.16):
    """Synthetic daily closes via GBM, mimicking an index futures contract."""
    dt = 1 / 252
    rets = np.random.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), n_days)
    price = s0 * np.exp(np.cumsum(rets))
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    return pd.DataFrame({"close": price}, index=dates)

df = get_real_data() if USE_REAL_DATA else get_synthetic_data()

# SIGNAL GENERATION
FAST, SLOW = 10, 30
df["sma_fast"] = df["close"].rolling(FAST).mean()
df["sma_slow"] = df["close"].rolling(SLOW).mean()

# Position: +1 long, -1 short. Shift by 1 to avoid lookahead bias
# (you can only act on a signal the day AFTER it's confirmed)
df["signal"] = np.where(df["sma_fast"] > df["sma_slow"], 1, -1)
df["position"] = df["signal"].shift(1)

# P&L / RETURNS
df["daily_ret"] = df["close"].pct_change()
df["strategy_ret"] = df["position"] * df["daily_ret"]

# Transaction cost: charge a small cost only on days position changes
# (proxy for bid-ask spread + brokerage on futures roll/trade)
COST_PER_TRADE = 0.0005  # 5 bps
df["trade"] = df["position"].diff().abs()
df["strategy_ret_net"] = df["strategy_ret"] - df["trade"] * COST_PER_TRADE

df["equity_strategy"] = (1 + df["strategy_ret_net"].fillna(0)).cumprod()
df["equity_buyhold"] = (1 + df["daily_ret"].fillna(0)).cumprod()

# PERFORMANCE METRICS
def performance_summary(returns, equity, label):
    n_years = len(returns) / 252
    cagr = equity.iloc[-1] ** (1 / n_years) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252) / ann_vol if ann_vol > 0 else np.nan
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min()
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    win_rate = len(wins) / (len(wins) + len(losses))
    profit_factor = wins.sum() / abs(losses.sum())
    return {
        "Strategy": label,
        "CAGR": f"{cagr:.2%}",
        "Ann. Volatility": f"{ann_vol:.2%}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Max Drawdown": f"{max_dd:.2%}",
        "Win Rate": f"{win_rate:.2%}",
        "Profit Factor": f"{profit_factor:.2f}",
    }

summary_strategy = performance_summary(
    df["strategy_ret_net"].dropna(), df["equity_strategy"].dropna(), "SMA Crossover"
)
summary_bh = performance_summary(
    df["daily_ret"].dropna(), df["equity_buyhold"].dropna(), "Buy & Hold"
)

results = pd.DataFrame([summary_strategy, summary_bh]).set_index("Strategy")
print("\n=== PERFORMANCE COMPARISON ===")
print(results.to_string())
results.to_csv("performance_summary.csv")

# PLOT
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                          gridspec_kw={"height_ratios": [2, 1]})

axes[0].plot(df.index, df["equity_strategy"], label="SMA Crossover Strategy", color="#1f4e8c", linewidth=1.6)
axes[0].plot(df.index, df["equity_buyhold"], label="Buy & Hold", color="#c94f4f", linewidth=1.2, alpha=0.8)
axes[0].set_title("Equity Curve: Trend-Following vs Buy & Hold")
axes[0].set_ylabel("Growth of ₹1")
axes[0].legend()
axes[0].grid(alpha=0.3)

dd = (df["equity_strategy"] - df["equity_strategy"].cummax()) / df["equity_strategy"].cummax()
axes[1].fill_between(df.index, dd, 0, color="#c94f4f", alpha=0.4)
axes[1].set_title("Strategy Drawdown")
axes[1].set_ylabel("Drawdown")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("equity_curve.png", dpi=150)
print("\nSaved chart to equity_curve.png")