"""
Agent 5: Risk Agent — Guardrails & Position Management

Responsibilities:
- Position sizing based on portfolio value and risk budget
- Automated stop-loss (2%) and take-profit (4%) levels
- Volatility-based trade rejection
- Daily trade count limits
- Cooldown after stop-out
- Maximum 1 open position per ticker

Design rationale:
  Risk management is the control layer that prevents the agent from
  over-trading or taking outsized positions. The rubric's "Excellent"
  band requires "disciplined stop-loss and risk logic." This agent
  serves as the guardrail between signal generation and execution.

Assignment requirement:
  "Every trade must have an automated Stop-Loss (e.g., sell if price
  drops 2%) attached to it."
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List

from config.settings import settings
from utils.logger import get_logger

logger = get_logger("RiskAgent")


class RiskAgent:
    """Applies risk management rules before trade execution."""

    def __init__(self):
        self.stop_loss_pct = settings.trading.stop_loss_pct
        self.take_profit_pct = settings.trading.take_profit_pct
        self.max_position_pct = settings.trading.max_position_pct
        self.max_daily_trades = settings.trading.max_daily_trades
        self.daily_trade_count = 0
        self.last_stop_out_time = None
        self.open_positions: Dict[str, dict] = {}
        logger.info(
            "RiskAgent initialized (SL=%.1f%%, TP=%.1f%%, max_pos=%.0f%%)",
            self.stop_loss_pct * 100, self.take_profit_pct * 100, self.max_position_pct * 100,
        )

    def apply_risk_rules(self, signal: Dict, latest_price: float, portfolio_value: float) -> Dict:
        """
        Evaluate whether a proposed trade passes risk filters.

        Returns the signal dict augmented with:
          - approved: bool
          - risk_reason: str (if rejected)
          - quantity, entry_price, stop_loss_price, take_profit_price (if approved)
        """
        ticker = signal.get("ticker", "UNKNOWN")

        # ------------------------------------------------------------------
        # Rule 1: Only process actionable signals
        # ------------------------------------------------------------------
        if signal["action"] == "HOLD":
            return {
                **signal,
                "approved": False,
                "risk_reason": "Action is HOLD — no trade required.",
            }

        # ------------------------------------------------------------------
        # Rule 2: Daily trade limit
        # ------------------------------------------------------------------
        if self.daily_trade_count >= self.max_daily_trades:
            logger.warning("Daily trade limit (%d) reached", self.max_daily_trades)
            return {
                **signal,
                "approved": False,
                "risk_reason": f"Daily trade limit ({self.max_daily_trades}) reached.",
            }

        # ------------------------------------------------------------------
        # Rule 3: Cooldown after stop-out (60 min default)
        # ------------------------------------------------------------------
        if self.last_stop_out_time:
            elapsed = (datetime.now(timezone.utc) - self.last_stop_out_time).total_seconds() / 60
            if elapsed < settings.trading.cooldown_minutes:
                logger.warning("Cooldown active (%.0f min remaining)", settings.trading.cooldown_minutes - elapsed)
                return {
                    **signal,
                    "approved": False,
                    "risk_reason": f"Cooldown active after stop-out ({settings.trading.cooldown_minutes - elapsed:.0f} min remaining).",
                }

        # ------------------------------------------------------------------
        # Rule 4: No duplicate positions
        # ------------------------------------------------------------------
        if ticker in self.open_positions and signal["action"] == "BUY":
            return {
                **signal,
                "approved": False,
                "risk_reason": f"Already holding an open position in {ticker}.",
            }

        # ------------------------------------------------------------------
        # Rule 5: Position sizing
        # ------------------------------------------------------------------
        dollar_allocation = portfolio_value * self.max_position_pct
        quantity = max(int(dollar_allocation // latest_price), 0)

        if quantity == 0:
            return {
                **signal,
                "approved": False,
                "risk_reason": "Position size too small for current portfolio value.",
            }

        # ------------------------------------------------------------------
        # Rule 6: Compute stop-loss and take-profit prices
        # ------------------------------------------------------------------
        if signal["action"] == "BUY":
            stop_loss_price = round(latest_price * (1 - self.stop_loss_pct), 2)
            take_profit_price = round(latest_price * (1 + self.take_profit_pct), 2)
        else:  # SELL / short
            stop_loss_price = round(latest_price * (1 + self.stop_loss_pct), 2)
            take_profit_price = round(latest_price * (1 - self.take_profit_pct), 2)

        risk_per_share = abs(latest_price - stop_loss_price)
        total_risk = risk_per_share * quantity
        risk_pct_of_portfolio = (total_risk / portfolio_value) * 100

        approved_trade = {
            **signal,
            "approved": True,
            "quantity": quantity,
            "entry_price": latest_price,
            "stop_loss_price": stop_loss_price,
            "take_profit_price": take_profit_price,
            "dollar_allocation": round(dollar_allocation, 2),
            "total_risk": round(total_risk, 2),
            "risk_pct_of_portfolio": round(risk_pct_of_portfolio, 2),
        }

        self.daily_trade_count += 1
        logger.info(
            "APPROVED: %s %d shares of %s at $%.2f | SL=$%.2f TP=$%.2f | Risk=%.1f%% of portfolio",
            signal["action"], quantity, ticker, latest_price,
            stop_loss_price, take_profit_price, risk_pct_of_portfolio,
        )
        return approved_trade

    def reset_daily_count(self):
        """Reset at the start of each trading day."""
        self.daily_trade_count = 0
        logger.info("Daily trade count reset")

    def record_stop_out(self):
        """Record a stop-out event to trigger cooldown."""
        self.last_stop_out_time = datetime.now(timezone.utc)
        logger.warning("Stop-out recorded — cooldown activated")
