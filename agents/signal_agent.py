"""
Agent 4: Signal Agent — Decision Fusion

Responsibilities:
- Combine technical indicators (from MarketDataHandler) with LLM sentiment
  (from SentimentAgent) to produce a BUY / SELL / HOLD signal
- Enforce the assignment's core rule: trade ONLY when both technical
  AND sentiment signals align
- Log the reasoning for every decision (explainability)

Assignment requirement (verbatim):
  "The bot should only trade if both conditions align:
   - Technical Indicator: e.g., Stock price is above the 50-day Moving Average
   - LLM Signal: Sentiment Score for the last 24h is > 7/10 (Positive)"
"""

from __future__ import annotations

from typing import Dict

from config.settings import settings
from utils.logger import get_logger

logger = get_logger("SignalAgent")


class SignalAgent:
    """Fuses technical and sentiment signals into a trading decision."""

    def generate_signal(self, market_snapshot: Dict, sentiment_summary: Dict) -> Dict:
        """
        Generate a BUY / SELL / HOLD signal based on dual confirmation.

        Long entry requires:
          1. Close > SMA50 AND SMA20 > SMA50  (uptrend confirmed)
          2. LLM aggregate sentiment is POSITIVE
          3. Average conviction >= threshold (default 7)
          4. At least N headlines analysed (default 2)
          5. No high disagreement among headlines

        Short / exit signal:
          1. Trend is down (Close < SMA50)
          2. Sentiment is NEGATIVE

        Otherwise: HOLD (signals do not align).
        """
        trend_up = market_snapshot.get("trend_up", False)
        overall_signal = sentiment_summary.get("overall_signal", "NEUTRAL")
        avg_conviction = sentiment_summary.get("average_conviction", 0)
        headline_count = sentiment_summary.get("headline_count", 0)
        agreement = sentiment_summary.get("headline_agreement", "no_data")
        tradeable = sentiment_summary.get("tradeable", False)

        reasons = []

        # ------------------------------------------------------------------
        # BUY conditions
        # ------------------------------------------------------------------
        if (
            trend_up
            and overall_signal == "POSITIVE"
            and avg_conviction >= settings.trading.min_conviction
            and headline_count >= settings.trading.min_headlines
            and agreement != "high_disagreement"
        ):
            action = "BUY"
            reasons.append("Uptrend confirmed (Close > SMA50, SMA20 > SMA50).")
            reasons.append(f"24h sentiment is POSITIVE with conviction {avg_conviction:.1f}/10.")
            reasons.append(f"Headline consensus: {agreement} ({headline_count} headlines).")

        # ------------------------------------------------------------------
        # SELL conditions
        # ------------------------------------------------------------------
        elif (
            not trend_up
            and overall_signal == "NEGATIVE"
            and avg_conviction >= 5
        ):
            action = "SELL"
            reasons.append("Downtrend detected (Close < SMA50 or SMA20 < SMA50).")
            reasons.append(f"24h sentiment is NEGATIVE with conviction {avg_conviction:.1f}/10.")

        # ------------------------------------------------------------------
        # HOLD — signals do not align
        # ------------------------------------------------------------------
        else:
            action = "HOLD"
            if not trend_up:
                reasons.append("Technical trend is not bullish.")
            if overall_signal != "POSITIVE":
                reasons.append(f"Sentiment is {overall_signal}, not POSITIVE.")
            if avg_conviction < settings.trading.min_conviction:
                reasons.append(f"Conviction {avg_conviction:.1f} < threshold {settings.trading.min_conviction}.")
            if headline_count < settings.trading.min_headlines:
                reasons.append(f"Only {headline_count} headline(s); need >= {settings.trading.min_headlines}.")
            if agreement == "high_disagreement":
                reasons.append("Headlines show high disagreement — too uncertain to trade.")

        reason_str = " ".join(reasons)
        signal = {
            "action": action,
            "reason": reason_str,
            "trend_up": trend_up,
            "sentiment_signal": overall_signal,
            "conviction": avg_conviction,
            "headline_count": headline_count,
            "agreement": agreement,
        }

        logger.info("Signal: %s — %s", action, reason_str)
        return signal
