"""
Agent 1: Market Data Handler — The "Body" (Price Data)

Responsibilities:
- Fetch 12 months of OHLCV historical data
- Calculate technical indicators (SMA20, SMA50, ATR14, daily returns, volatility)
- Fetch current/latest bar for semi-live demo
- Fallback from Alpaca to yfinance if needed

Design rationale:
  Technical indicators are deterministic calculations. They are computed
  in code, NOT by the LLM, because LLMs are poor calculators (as stated
  in the assignment brief). This clean separation is central to the
  architecture.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import yfinance as yf

from utils.logger import get_logger

logger = get_logger("MarketDataHandler")

# Rate-limit / retry configuration
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2  # doubles each retry: 2s, 4s, 8s
REQUEST_COOLDOWN = 0.5     # minimum gap between API calls (seconds)


class MarketDataHandler:
    """Fetches and enriches OHLCV market data with technical indicators."""

    def __init__(self, period: str = "1y", interval: str = "1d"):
        self.period = period
        self.interval = interval
        self._last_request_time = 0.0
        logger.info("MarketDataHandler initialized (period=%s, interval=%s)", period, interval)

    # ------------------------------------------------------------------
    # Rate-limit aware request throttling
    # ------------------------------------------------------------------
    def _throttle(self) -> None:
        """Enforce minimum gap between API calls to respect rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_COOLDOWN:
            sleep_time = REQUEST_COOLDOWN - elapsed
            logger.debug("Rate-limit throttle: sleeping %.2fs", sleep_time)
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    # ------------------------------------------------------------------
    # Core data fetching with retry and backoff
    # ------------------------------------------------------------------
    def fetch_historical_data(self, ticker: str, period: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch OHLCV data and enrich with technical indicators.
        Primary source: yfinance (free, no key required).

        Includes:
          - Rate-limit throttling between requests
          - Exponential backoff retry (up to 3 attempts)
          - Graceful error handling
        """
        period = period or self.period
        logger.info("Fetching historical data for %s (period=%s)", ticker, period)

        df = pd.DataFrame()
        last_error = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                self._throttle()
                df = yf.download(ticker, period=period, interval=self.interval, auto_adjust=False, progress=False)
                if not df.empty:
                    break  # success
                logger.warning("Empty response for %s (attempt %d/%d)", ticker, attempt, MAX_RETRIES)
            except Exception as e:
                last_error = e
                logger.warning("Fetch attempt %d/%d failed for %s: %s", attempt, MAX_RETRIES, ticker, e)

            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
                logger.info("Retrying in %.0fs...", wait)
                time.sleep(wait)

        if df.empty:
            msg = f"Failed to fetch data for {ticker} after {MAX_RETRIES} attempts"
            if last_error:
                msg += f": {last_error}"
            raise ValueError(msg)

        if df.empty:
            raise ValueError(f"No market data returned for {ticker}")

        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        # Standardize column names
        df.columns = [c.lower() for c in df.columns]

        # Keep only OHLCV
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        df = df[required].copy()

        # ------------------------------------------------------------------
        # Technical indicators (deterministic — no LLM involvement)
        # ------------------------------------------------------------------
        df["sma20"] = df["close"].rolling(20).mean()
        df["sma50"] = df["close"].rolling(50).mean()
        df["daily_return"] = df["close"].pct_change()
        df["volatility_20d"] = df["daily_return"].rolling(20).std()

        # ATR-14 (Average True Range)
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr14"] = true_range.rolling(14).mean()

        df = df.dropna()
        logger.info("Fetched %d rows for %s after indicator computation", len(df), ticker)
        return df

    # ------------------------------------------------------------------
    # Latest snapshot for live/semi-live demo
    # ------------------------------------------------------------------
    def get_latest_snapshot(self, ticker: str) -> dict:
        """Return the most recent market state for the demo pipeline."""
        df = self.fetch_historical_data(ticker, period="3mo")
        latest = df.iloc[-1]

        snapshot = {
            "ticker": ticker,
            "date": str(df.index[-1].date()),
            "close": round(float(latest["close"]), 2),
            "sma20": round(float(latest["sma20"]), 2),
            "sma50": round(float(latest["sma50"]), 2),
            "atr14": round(float(latest["atr14"]), 2),
            "volatility_20d": round(float(latest["volatility_20d"]), 4),
            "daily_return": round(float(latest["daily_return"]), 4),
            "trend_up": bool(latest["close"] > latest["sma50"] and latest["sma20"] > latest["sma50"]),
        }
        logger.info("Latest snapshot for %s: close=%.2f trend_up=%s", ticker, snapshot["close"], snapshot["trend_up"])
        return snapshot
