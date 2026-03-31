"""
Structured logging for the LLM Trading Agent.
Every agent action is logged with timestamp, agent name, and details
for full auditability and reproducibility.
"""

import logging
import sys
from datetime import datetime, timezone


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a structured logger for an agent or module."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(name)-24s | %(levelname)-7s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def log_trade_decision(logger: logging.Logger, decision: dict) -> None:
    """Log a structured trade decision for audit trail."""
    logger.info(
        "TRADE DECISION | ticker=%s action=%s reason=%s approved=%s",
        decision.get("ticker", "N/A"),
        decision.get("action", "N/A"),
        decision.get("reason", "N/A"),
        decision.get("approved", "N/A"),
    )
