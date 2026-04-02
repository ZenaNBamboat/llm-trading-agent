"""
===============================================================================
  LLM-Driven Automated Trading Agent — Orchestrator
===============================================================================

  This is the MANAGER AGENT that coordinates all specialist agents:

    1. MarketDataHandler  →  Fetches OHLCV data + technical indicators
    2. NewsFetcher         →  Retrieves financial headlines
    3. SentimentAgent      →  LLM-powered sentiment classification (the "Brain")
    4. SignalAgent          →  Fuses technical + sentiment signals
    5. RiskAgent           →  Applies risk management guardrails
    6. ExecutionAgent      →  Places simulated trades
    7. BacktestAgent       →  Validates strategy on historical data

  Architecture style:
    Multi-agent orchestration where each specialist agent has a single
    clear responsibility and the manager coordinates them in sequence.
    This mirrors enterprise agentic AI patterns: structured coordination
    of models, tools, data sources, and oversight mechanisms with control
    layers, escalation, and governance.

  The LLM is used ONLY for unstructured text interpretation (headline
  sentiment). All numerical computation (indicators, risk, backtesting)
  is handled deterministically in code.

  Usage:
    python app.py                    # Run full pipeline for AAPL
    python app.py --ticker TSLA      # Run for a specific ticker
    python app.py --backtest-only    # Run only the backtest
    python app.py --demo             # Run the full demo sequence

===============================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.market_data_handler import MarketDataHandler
from agents.news_fetcher import NewsFetcher
from agents.sentiment_agent import SentimentAgent
from agents.signal_agent import SignalAgent
from agents.risk_agent import RiskAgent
from agents.execution_agent import ExecutionAgent
from agents.backtest_agent import BacktestAgent
from config.settings import settings
from utils.logger import get_logger

logger = get_logger("TradingManager")


class TradingManager:
    """
    Orchestrator / Manager Agent.

    Coordinates the specialist agents in a structured pipeline:
      Data → News → Sentiment → Signal → Risk → Execution
    """

    def __init__(self):
        logger.info("=" * 70)
        logger.info("  LLM Trading Agent — Initializing Multi-Agent System")
        logger.info("=" * 70)

        self.market_agent = MarketDataHandler()
        self.news_agent = NewsFetcher()
        self.sentiment_agent = SentimentAgent()
        self.signal_agent = SignalAgent()
        self.risk_agent = RiskAgent()
        self.execution_agent = ExecutionAgent()
        self.backtest_agent = BacktestAgent()

        logger.info("All agents initialized successfully")

    # ------------------------------------------------------------------
    # Full pipeline: Data → News → Sentiment → Signal → Risk → Execute
    # ------------------------------------------------------------------
    def run(self, ticker: str, portfolio_value: float = None) -> dict:
        """
        Execute the full trading pipeline for a single ticker.
        This is the method shown in the live demo.
        """
        portfolio_value = portfolio_value or settings.trading.initial_capital

        logger.info("")
        logger.info("=" * 70)
        logger.info("  PIPELINE START — Ticker: %s | Portfolio: $%.2f", ticker, portfolio_value)
        logger.info("=" * 70)

        # Step 1: Market Data
        logger.info("")
        logger.info("─── STEP 1: Market Data Handler ───")
        market_snapshot = self.market_agent.get_latest_snapshot(ticker)

        # Step 2: News Ingestion
        logger.info("")
        logger.info("─── STEP 2: News Fetcher ───")
        recent_news = self.news_agent.fetch_recent_news(ticker)
        recent_news = self.news_agent.filter_last_24h(recent_news)

        # Step 3: LLM Sentiment Analysis (the "Brain")
        logger.info("")
        logger.info("─── STEP 3: LLM Sentiment Agent (Brain) ───")
        scored_news = self.sentiment_agent.analyze_news_batch(recent_news)
        sentiment_summary = self.sentiment_agent.aggregate_sentiment(scored_news)

        # Step 4: Signal Generation
        logger.info("")
        logger.info("─── STEP 4: Signal Agent ───")
        signal = self.signal_agent.generate_signal(market_snapshot, sentiment_summary)
        signal["ticker"] = ticker

        # Step 5: Risk Management
        logger.info("")
        logger.info("─── STEP 5: Risk Agent ───")
        trade_plan = self.risk_agent.apply_risk_rules(signal, market_snapshot["close"], portfolio_value)
        trade_plan["ticker"] = ticker

        # Step 6: Execution
        logger.info("")
        logger.info("─── STEP 6: Execution Agent ───")
        execution_result = self.execution_agent.execute(ticker, trade_plan)

        # ------------------------------------------------------------------
        # Compile full result
        # ------------------------------------------------------------------
        result = {
            "ticker": ticker,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "market_snapshot": market_snapshot,
            "news_count": len(recent_news),
            "scored_news": scored_news,
            "sentiment_summary": sentiment_summary,
            "signal": signal,
            "trade_plan": trade_plan,
            "execution_result": execution_result,
        }

        # Save sentiment log
        self._save_sentiment_log(scored_news, ticker)

        logger.info("")
        logger.info("=" * 70)
        logger.info("  PIPELINE COMPLETE — Action: %s", execution_result.get("status", "UNKNOWN"))
        logger.info("=" * 70)

        return result

    # ------------------------------------------------------------------
    # Backtest
    # ------------------------------------------------------------------
    def run_backtest(self, ticker: str) -> dict:
        """Run the backtest validation pipeline."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("  BACKTEST START — Ticker: %s | Period: %s", ticker, settings.backtest.lookback_period)
        logger.info("=" * 70)

        result = self.backtest_agent.run_backtest(ticker)

        logger.info("")
        logger.info("=" * 70)
        logger.info("  BACKTEST COMPLETE")
        logger.info("=" * 70)

        return result

    # ------------------------------------------------------------------
    # Demo sequence (matches the required presentation flow)
    # ------------------------------------------------------------------
    def run_demo(self, ticker: str) -> dict:
        """
        Run the full demo sequence for the class presentation.

        Demo flow (matches assignment requirements):
          1. "We fetch latest headlines."
          2. "Our LLM agent scores each headline."
          3. "We aggregate to a 24h conviction score."
          4. "We check if price is above SMA50."
          5. "We run risk filter."
          6. "System outputs Buy / Hold / Sell."
          7. "Here is the backtest equity curve vs SPY."
          8. "Here is one limitation."
        """
        logger.info("")
        logger.info("╔" + "═" * 68 + "╗")
        logger.info("║  LIVE DEMO — LLM-Driven Automated Trading Agent                   ║")
        logger.info("╚" + "═" * 68 + "╝")

        # Run live pipeline
        live_result = self.run(ticker)

        # Run backtest
        backtest_result = self.run_backtest(ticker)

        # Print demo summary
        self._print_demo_summary(live_result, backtest_result)

        return {
            "live_result": live_result,
            "backtest_result": backtest_result,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _save_sentiment_log(self, scored_news: list, ticker: str) -> None:
        """Save scored news to CSV for documentation."""
        import csv
        path = os.path.join("outputs", "sentiment_log.csv")
        os.makedirs("outputs", exist_ok=True)

        with open(path, "w", newline="") as f:
            if scored_news:
                writer = csv.DictWriter(f, fieldnames=scored_news[0].keys())
                writer.writeheader()
                writer.writerows(scored_news)

        logger.info("Sentiment log saved to %s", path)

    def _print_demo_summary(self, live: dict, backtest: dict) -> None:
        """Print a formatted demo summary for the presentation."""
        print("\n")
        print("=" * 70)
        print("  DEMO SUMMARY")
        print("=" * 70)

        # Market
        ms = live["market_snapshot"]
        print(f"\n  ┌─ STEP 1: Market Data ─────────────────────────────────────")
        print(f"  │ Ticker:       {ms['ticker']}")
        print(f"  │ Date:         {ms['date']}")
        print(f"  │ Close:        ${ms['close']:.2f}")
        print(f"  │ SMA20:        ${ms['sma20']:.2f}")
        print(f"  │ SMA50:        ${ms['sma50']:.2f}")
        print(f"  │ Trend Up:     {'Yes' if ms['trend_up'] else 'No'}")

        # News & Sentiment
        ss = live["sentiment_summary"]
        engine = live["scored_news"][0].get("engine", "finbert") if live["scored_news"] else "unknown"
        print(f"\n  ┌─ STEP 2-3: News Ingestion + Sentiment ({engine}) ─────────")
        print(f"  │ Headlines analyzed: {ss['headline_count']}")

        for i, item in enumerate(live["scored_news"], 1):
            label = item["sentiment_label"]
            score = item["conviction_score"]
            headline = item["headline"][:60]
            print(f"  │   {i}. [{label:8s} {score:2d}/10] {headline}")

        print(f"  │")
        print(f"  │ Aggregated signal:  {ss['overall_signal']}")
        print(f"  │ Avg conviction:     {ss['average_conviction']:.1f}/10")
        print(f"  │ Agreement:          {ss['headline_agreement']}")
        print(f"  │ Tradeable:          {ss.get('tradeable', 'N/A')}")

        # Signal & Execution
        sig = live["signal"]
        exe = live["execution_result"]
        print(f"\n  ┌─ STEP 4-6: Signal → Risk → Execution ────────────────────")
        print(f"  │ Signal:       {sig['action']}")
        print(f"  │ Reason:       {sig['reason']}")
        print(f"  │ Execution:    {exe['status']}")

        if exe.get("quantity"):
            print(f"  │ Quantity:     {exe['quantity']} shares")
            print(f"  │ Stop-Loss:    ${exe['stop_loss_price']:.2f}")
            print(f"  │ Take-Profit:  ${exe['take_profit_price']:.2f}")

        # Backtest results table
        if backtest and "metrics" in backtest:
            m = backtest["metrics"]
            print(f"\n  ┌─ STEP 7: Backtest Results ({m['period']}) ─────────────────────")
            print(f"  │")
            print(f"  │  {'Metric':<22s} {'Strategy':>12s} {'SPY B&H':>12s}")
            print(f"  │  {'─'*22} {'─'*12} {'─'*12}")
            print(f"  │  {'Total Return':<22s} {m['total_return_pct']:>11.1f}% {m['benchmark_return_pct']:>11.1f}%")
            print(f"  │  {'Sharpe Ratio':<22s} {m['sharpe_ratio']:>12.2f} {m.get('benchmark_sharpe', 0):>12.2f}")
            bench_dd = m.get('bench_max_drawdown_pct', None)
            bench_dd_str = f"{bench_dd:>11.1f}%" if bench_dd is not None else f"{'N/A':>12s}"
            print(f"  │  {'Max Drawdown':<22s} {m['max_drawdown_pct']:>11.1f}% {bench_dd_str}")
            print(f"  │  {'Win Rate':<22s} {m['win_rate_pct']:>11.0f}% {'N/A':>12s}")
            print(f"  │  {'Total Trades':<22s} {m['total_trades']:>12d} {'1':>12s}")

        # Limitation
        print(f"\n  ┌─ STEP 8: Honest Limitation ───────────────────────────────")
        print(f"  │ The backtest validates the technical/risk engine")
        print(f"  │ rigorously. Historical sentiment replay is a future")
        print(f"  │ extension; the current backtest is conservative and")
        print(f"  │ avoids fabricating unavailable news data.")
        print(f"  │")
        print(f"  │ The system prioritizes risk control and avoids trading")
        print(f"  │ under uncertainty, which reduces drawdowns but can")
        print(f"  │ underperform in trending markets.")

        # Generated files
        print(f"\n  ┌─ Generated Output Files ─────────────────────────────────")
        print(f"  │ outputs/equity_curve.png")
        print(f"  │ outputs/benchmark_vs_strategy.png")
        print(f"  │ outputs/backtest_trades_AAPL.csv")
        print(f"  │ outputs/sentiment_log.csv")
        print(f"  │ outputs/trades.csv")

        print("\n" + "=" * 70)


# ======================================================================
# CLI Entry Point
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="LLM-Driven Automated Trading Agent")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker to analyze")
    parser.add_argument("--backtest-only", action="store_true", help="Run only backtest")
    parser.add_argument("--demo", action="store_true", help="Run full demo sequence")
    parser.add_argument("--portfolio", type=float, default=None, help="Portfolio value")
    args = parser.parse_args()

    manager = TradingManager()

    if args.demo:
        manager.run_demo(args.ticker)
    elif args.backtest_only:
        result = manager.run_backtest(args.ticker)
        print(json.dumps(result["metrics"], indent=2))
    else:
        result = manager.run(args.ticker, portfolio_value=args.portfolio)
        print(json.dumps({
            "execution": result["execution_result"],
            "sentiment": result["sentiment_summary"],
            "signal": result["signal"],
        }, indent=2, default=str))


if __name__ == "__main__":
    main()
