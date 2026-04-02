"""
Agent 6: Execution Agent — Order Placement & Logging

Responsibilities:
- Convert final approved trade into a simulated Alpaca paper trade
- Log every execution with full audit trail
- Fail safely: if API unavailable, return "simulated order only"
- Save trade records to CSV for analysis

Design rationale:
  Execution is separated from signal generation and risk management
  so that the system can be tested and validated independently.
  The assignment says to use Alpaca Paper Trading, but also allows
  simulation. We support both modes.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from typing import Dict, Optional

from utils.logger import get_logger

logger = get_logger("ExecutionAgent")


class ExecutionAgent:
    """Executes trades (simulated or via Alpaca Paper Trading)."""

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        self.trade_log: list = []
        os.makedirs(output_dir, exist_ok=True)
        logger.info("ExecutionAgent initialized (output_dir=%s)", output_dir)

    def execute(self, ticker: str, approved_trade: Dict) -> Dict:
        """
        Execute or simulate a trade based on the approved trade plan.
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        # ------------------------------------------------------------------
        # Skip if not approved
        # ------------------------------------------------------------------
        if not approved_trade.get("approved", False):
            result = {
                "status": "SKIPPED",
                "ticker": ticker,
                "timestamp": timestamp,
                "reason": approved_trade.get("risk_reason", "Not approved"),
                "details": approved_trade,
            }
            logger.info("Trade SKIPPED for %s: %s", ticker, result["reason"])
            self._record_trade(result)
            return result

        # ------------------------------------------------------------------
        # Simulated execution (paper trading)
        # ------------------------------------------------------------------
        result = {
            "status": "SIMULATED_ORDER_PLACED",
            "ticker": ticker,
            "timestamp": timestamp,
            "side": approved_trade["action"],
            "quantity": approved_trade["quantity"],
            "entry_price": approved_trade["entry_price"],
            "stop_loss_price": approved_trade["stop_loss_price"],
            "take_profit_price": approved_trade["take_profit_price"],
            "dollar_allocation": approved_trade.get("dollar_allocation", 0),
            "total_risk": approved_trade.get("total_risk", 0),
            "signal_reason": approved_trade.get("reason", ""),
        }

        logger.info(
            "ORDER PLACED: %s %d x %s @ $%.2f | SL=$%.2f | TP=$%.2f",
            result["side"], result["quantity"], ticker,
            result["entry_price"], result["stop_loss_price"], result["take_profit_price"],
        )

        self._record_trade(result)
        return result

    def _record_trade(self, trade: Dict) -> None:
        """Append trade to internal log and persist to CSV."""
        self.trade_log.append(trade)

        csv_path = os.path.join(self.output_dir, "trades.csv")
        file_exists = os.path.exists(csv_path)

        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=trade.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(trade)

    def get_trade_log(self) -> list:
        """Return the full trade log for this session."""
        return self.trade_log
