"""
Agent 2: News Fetcher — Unstructured Data Ingestion

Responsibilities:
- Fetch recent financial news headlines for a given ticker
- Timestamp and deduplicate headlines
- Filter to last 24 hours for live signal generation
- Support multiple news sources with fallback

Sources (in priority order):
  1. Alpaca News API (included in free tier)
  2. Yahoo Finance RSS / yfinance news
  3. Hardcoded sample headlines for demo fallback

Design rationale:
  News headlines are the unstructured text input that the LLM "Brain"
  will process. This agent only fetches and cleans — it does NOT
  interpret sentiment. That separation is critical for the architecture.
"""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import yfinance as yf

from utils.logger import get_logger

logger = get_logger("NewsFetcher")

# Rate-limit configuration
NEWS_REQUEST_COOLDOWN = 0.5  # seconds between API calls
NEWS_MAX_RETRIES = 2


class NewsFetcher:
    """Fetches, deduplicates, and filters financial news headlines."""

    def __init__(self):
        self._last_request_time = 0.0
        logger.info("NewsFetcher initialized")

    def _throttle(self) -> None:
        """Enforce minimum gap between API calls to respect rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < NEWS_REQUEST_COOLDOWN:
            time.sleep(NEWS_REQUEST_COOLDOWN - elapsed)
        self._last_request_time = time.time()

    # ------------------------------------------------------------------
    # Primary: yfinance news (free, no API key)
    # ------------------------------------------------------------------
    def _fetch_yfinance_news(self, ticker: str) -> List[Dict]:
        """Fetch news from Yahoo Finance via yfinance."""
        logger.info("Fetching yfinance news for %s", ticker)
        try:
            self._throttle()
            stock = yf.Ticker(ticker)
            news_items = stock.news or []
            results = []
            for item in news_items:
                content = item.get("content", {})
                title = content.get("title", "")
                pub_date = content.get("pubDate", "")

                if not title:
                    continue

                # Parse publish date
                try:
                    if pub_date:
                        ts = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
                    else:
                        ts = datetime.now(timezone.utc)
                except (ValueError, TypeError):
                    ts = datetime.now(timezone.utc)

                results.append({
                    "ticker": ticker,
                    "headline": title,
                    "published_at": ts.isoformat(),
                    "source": content.get("provider", {}).get("displayName", "Yahoo Finance"),
                })
            logger.info("Retrieved %d headlines from yfinance for %s", len(results), ticker)
            return results
        except Exception as e:
            logger.warning("yfinance news fetch failed for %s: %s", ticker, e)
            return []

    # ------------------------------------------------------------------
    # Fallback: sample headlines for demo reliability
    # ------------------------------------------------------------------
    def _get_sample_news(self, ticker: str) -> List[Dict]:
        """Hardcoded fallback headlines for demo if live sources fail."""
        now = datetime.now(timezone.utc)
        samples = [
            {
                "ticker": ticker,
                "headline": f"{ticker} reports stronger-than-expected quarterly earnings, beating analyst estimates",
                "published_at": now.isoformat(),
                "source": "Demo-Reuters",
            },
            {
                "ticker": ticker,
                "headline": f"Analysts upgrade {ticker} citing strong demand pipeline and margin expansion",
                "published_at": (now - timedelta(hours=4)).isoformat(),
                "source": "Demo-Bloomberg",
            },
            {
                "ticker": ticker,
                "headline": f"{ticker} announces strategic partnership to expand cloud services",
                "published_at": (now - timedelta(hours=8)).isoformat(),
                "source": "Demo-CNBC",
            },
            {
                "ticker": ticker,
                "headline": f"Regulatory probe into {ticker} data practices raises investor concerns",
                "published_at": (now - timedelta(hours=12)).isoformat(),
                "source": "Demo-WSJ",
            },
            {
                "ticker": ticker,
                "headline": f"{ticker} faces supply chain headwinds amid global semiconductor shortage",
                "published_at": (now - timedelta(hours=18)).isoformat(),
                "source": "Demo-FT",
            },
        ]
        logger.info("Using %d sample headlines for %s (fallback)", len(samples), ticker)
        return samples

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def fetch_recent_news(self, ticker: str, use_live: bool = True) -> List[Dict]:
        """
        Fetch recent headlines. Tries live sources first, falls back to samples.
        """
        headlines = []

        if use_live:
            headlines = self._fetch_yfinance_news(ticker)

        if not headlines:
            logger.info("No live headlines found, using sample fallback for %s", ticker)
            headlines = self._get_sample_news(ticker)

        # Deduplicate by headline hash
        seen = set()
        unique = []
        for item in headlines:
            h = hashlib.md5(item["headline"].lower().encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                unique.append(item)

        logger.info("Total unique headlines for %s: %d", ticker, len(unique))
        return unique

    def filter_last_24h(self, news_items: List[Dict]) -> List[Dict]:
        """Keep only headlines published in the last 24 hours."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        filtered = []

        for item in news_items:
            try:
                ts = datetime.fromisoformat(item["published_at"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts >= cutoff:
                    filtered.append(item)
            except (ValueError, TypeError):
                # If we cannot parse the date, include it to be safe
                filtered.append(item)

        logger.info("Filtered to %d headlines within 24h window", len(filtered))
        return filtered
