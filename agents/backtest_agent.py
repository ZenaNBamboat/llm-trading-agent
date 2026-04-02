"""
Agent 7: Backtest Agent — Historical Validation

Responsibilities:
- Backtest the strategy on the last 12 months of data
- Enforce NO look-ahead bias (only use data available at each point in time)
- Compare performance vs SPY Buy-and-Hold benchmark
- Compute key metrics: Total Return, Sharpe Ratio, Max Drawdown, Win Rate
- Generate equity curve visualization

Assignment requirement:
  "Before running 'live' on Paper, the code must run a backtest...
   Test the strategy on the last 12 months of data.
   Compare the bot's performance vs. a simple Buy and Hold strategy of S&P 500 (SPY)."

Design rationale:
  The backtest uses only SMA-based technical signals (no LLM calls for
  historical data, since we don't have historical headlines). This is
  acknowledged as a limitation. For the live/semi-live demo, the full
  pipeline including LLM sentiment is demonstrated.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from agents.market_data_handler import MarketDataHandler
from config.settings import settings
from utils.logger import get_logger

logger = get_logger("BacktestAgent")


class BacktestAgent:
    """
    Backtests the SMA-crossover + simulated sentiment strategy
    against SPY buy-and-hold over the last 12 months.
    """

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        self.market_handler = MarketDataHandler(period=settings.backtest.lookback_period)
        os.makedirs(output_dir, exist_ok=True)
        logger.info("BacktestAgent initialized")

    def run_backtest(self, ticker: str) -> Dict:
        """
        Run a full backtest for a single ticker.

        Strategy rules (matching the live agent):
          BUY when: Close > SMA50 AND SMA20 > SMA50 (no position held)
          SELL when: Close < SMA50 OR SMA20 < SMA50 (position held)
          Stop-loss: Exit if price drops 2% from entry
          Take-profit: Exit if price rises 4% from entry

        No look-ahead bias:
          Signals are generated using data available up to day t-1.
          Trades execute at day t's close (simulating next-day execution).
        """
        logger.info("Running backtest for %s", ticker)

        # Fetch data
        df = self.market_handler.fetch_historical_data(ticker)
        benchmark_df = self.market_handler.fetch_historical_data(settings.backtest.benchmark_ticker)

        # ------------------------------------------------------------------
        # Strategy simulation
        # ------------------------------------------------------------------
        capital = settings.trading.initial_capital
        position = 0        # shares held
        entry_price = 0.0
        cash = capital
        trades = []
        portfolio_values = []

        for i in range(1, len(df)):
            # Data available up to yesterday (no look-ahead bias)
            prev = df.iloc[i - 1]
            today = df.iloc[i]
            today_date = df.index[i]

            current_close = float(today["close"])

            # Portfolio value
            portfolio_value = cash + position * current_close
            portfolio_values.append({"date": today_date, "value": portfolio_value})

            # ------------------------------------------------------------------
            # Check stop-loss / take-profit on existing position
            # ------------------------------------------------------------------
            if position > 0:
                pnl_pct = (current_close - entry_price) / entry_price

                if pnl_pct <= -settings.trading.stop_loss_pct:
                    # Stop-loss triggered
                    cash += position * current_close
                    trades.append({
                        "date": str(today_date.date()),
                        "action": "SELL (STOP-LOSS)",
                        "price": current_close,
                        "shares": position,
                        "pnl_pct": round(pnl_pct * 100, 2),
                    })
                    logger.info("STOP-LOSS at %.2f (%.1f%%)", current_close, pnl_pct * 100)
                    position = 0
                    entry_price = 0
                    continue

                if pnl_pct >= settings.trading.take_profit_pct:
                    # Take-profit triggered
                    cash += position * current_close
                    trades.append({
                        "date": str(today_date.date()),
                        "action": "SELL (TAKE-PROFIT)",
                        "price": current_close,
                        "shares": position,
                        "pnl_pct": round(pnl_pct * 100, 2),
                    })
                    logger.info("TAKE-PROFIT at %.2f (%.1f%%)", current_close, pnl_pct * 100)
                    position = 0
                    entry_price = 0
                    continue

            # ------------------------------------------------------------------
            # Signal generation using PREVIOUS day's indicators (no look-ahead)
            # ------------------------------------------------------------------
            prev_close = float(prev["close"])
            prev_sma20 = float(prev["sma20"])
            prev_sma50 = float(prev["sma50"])
            trend_up = prev_close > prev_sma50 and prev_sma20 > prev_sma50

            # BUY signal
            if trend_up and position == 0:
                shares_to_buy = int((cash * settings.trading.max_position_pct) // current_close)
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_close
                    cash -= cost
                    position = shares_to_buy
                    entry_price = current_close
                    trades.append({
                        "date": str(today_date.date()),
                        "action": "BUY",
                        "price": current_close,
                        "shares": shares_to_buy,
                        "pnl_pct": 0.0,
                    })

            # SELL signal (trend reversal)
            elif not trend_up and position > 0:
                pnl_pct = (current_close - entry_price) / entry_price
                cash += position * current_close
                trades.append({
                    "date": str(today_date.date()),
                    "action": "SELL (TREND REVERSAL)",
                    "price": current_close,
                    "shares": position,
                    "pnl_pct": round(pnl_pct * 100, 2),
                })
                position = 0
                entry_price = 0

        # Close any remaining position
        if position > 0:
            final_price = float(df.iloc[-1]["close"])
            pnl_pct = (final_price - entry_price) / entry_price
            cash += position * final_price
            trades.append({
                "date": str(df.index[-1].date()),
                "action": "SELL (END-OF-BACKTEST)",
                "price": final_price,
                "shares": position,
                "pnl_pct": round(pnl_pct * 100, 2),
            })
            position = 0

        # ------------------------------------------------------------------
        # Compute metrics
        # ------------------------------------------------------------------
        pv_df = pd.DataFrame(portfolio_values).set_index("date")
        pv_df["daily_return"] = pv_df["value"].pct_change()

        final_value = cash
        total_return = (final_value - capital) / capital

        # Benchmark (SPY buy-and-hold)
        bench_start = float(benchmark_df.iloc[0]["close"])
        bench_end = float(benchmark_df.iloc[-1]["close"])
        benchmark_return = (bench_end - bench_start) / bench_start

        # Sharpe Ratio (annualized) — drop NaN from first row
        daily_returns = pv_df["daily_return"].dropna().values
        risk_free_daily = settings.backtest.risk_free_rate / 252
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            excess_return = np.mean(daily_returns) - risk_free_daily
            sharpe = (excess_return / np.std(daily_returns)) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Benchmark Sharpe for comparison
        bench_daily = benchmark_df["close"].pct_change().dropna().values
        if len(bench_daily) > 1 and np.std(bench_daily) > 0:
            bench_excess = np.mean(bench_daily) - risk_free_daily
            bench_sharpe = (bench_excess / np.std(bench_daily)) * np.sqrt(252)
        else:
            bench_sharpe = 0.0

        # Max Drawdown
        cummax = pv_df["value"].cummax()
        drawdown = (pv_df["value"] - cummax) / cummax
        max_drawdown = float(drawdown.min())

        # Benchmark max drawdown
        bench_cummax = benchmark_df["close"].cummax()
        bench_drawdown = (benchmark_df["close"] - bench_cummax) / bench_cummax
        bench_max_dd = float(bench_drawdown.min())

        # Win rate
        winning_trades = [t for t in trades if t["action"].startswith("SELL") and t["pnl_pct"] > 0]
        losing_trades = [t for t in trades if t["action"].startswith("SELL") and t["pnl_pct"] <= 0]
        sell_trades = winning_trades + losing_trades
        win_rate = len(winning_trades) / len(sell_trades) if sell_trades else 0.0

        metrics = {
            "ticker": ticker,
            "period": settings.backtest.lookback_period,
            "initial_capital": capital,
            "final_value": round(final_value, 2),
            "total_return_pct": round(total_return * 100, 2),
            "benchmark_return_pct": round(benchmark_return * 100, 2),
            "sharpe_ratio": round(sharpe, 2),
            "benchmark_sharpe": round(bench_sharpe, 2),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "bench_max_drawdown_pct": round(bench_max_dd * 100, 2),
            "total_trades": len(trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate_pct": round(win_rate * 100, 1),
        }

        logger.info("Backtest complete: Return=%.2f%% vs Benchmark=%.2f%% | Sharpe=%.2f | MaxDD=%.2f%%",
                     metrics["total_return_pct"], metrics["benchmark_return_pct"],
                     metrics["sharpe_ratio"], metrics["max_drawdown_pct"])

        # ------------------------------------------------------------------
        # Generate plots
        # ------------------------------------------------------------------
        self._plot_equity_curve(pv_df, benchmark_df, capital, ticker, metrics)
        self._save_trade_log(trades, ticker)

        return {
            "metrics": metrics,
            "trades": trades,
            "portfolio_values": pv_df,
        }

    def _plot_equity_curve(self, pv_df: pd.DataFrame, benchmark_df: pd.DataFrame,
                           capital: float, ticker: str, metrics: Dict) -> None:
        """Generate equity curve vs benchmark visualization."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

        # --- Equity curve ---
        ax1 = axes[0]
        ax1.plot(pv_df.index, pv_df["value"], label=f"Strategy ({ticker})", color="#2563EB", linewidth=2)

        # Benchmark normalized to same capital
        bench_norm = (benchmark_df["close"] / float(benchmark_df["close"].iloc[0])) * capital
        ax1.plot(benchmark_df.index, bench_norm, label="SPY Buy & Hold", color="#9CA3AF", linewidth=1.5, linestyle="--")

        ax1.set_title(f"LLM Trading Agent — Equity Curve vs SPY Benchmark ({ticker})", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        # Add metrics annotation
        metrics_text = (
            f"Strategy Return: {metrics['total_return_pct']:.1f}%\n"
            f"Benchmark Return: {metrics['benchmark_return_pct']:.1f}%\n"
            f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
            f"Max Drawdown: {metrics['max_drawdown_pct']:.1f}%\n"
            f"Win Rate: {metrics['win_rate_pct']:.0f}%"
        )
        ax1.text(0.02, 0.02, metrics_text, transform=ax1.transAxes, fontsize=10,
                 verticalalignment="bottom", bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))

        # --- Drawdown ---
        ax2 = axes[1]
        cummax = pv_df["value"].cummax()
        drawdown = ((pv_df["value"] - cummax) / cummax) * 100
        ax2.fill_between(pv_df.index, drawdown, 0, color="#EF4444", alpha=0.3)
        ax2.plot(pv_df.index, drawdown, color="#EF4444", linewidth=1)
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Date")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = os.path.join(self.output_dir, "equity_curve.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Equity curve saved to %s", output_path)

    def _save_trade_log(self, trades: list, ticker: str) -> None:
        """Save backtest trade log to CSV."""
        if not trades:
            return
        df = pd.DataFrame(trades)
        path = os.path.join(self.output_dir, f"backtest_trades_{ticker}.csv")
        df.to_csv(path, index=False)
        logger.info("Trade log saved to %s (%d trades)", path, len(trades))
