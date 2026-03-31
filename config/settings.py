"""
Centralized configuration for the LLM Trading Agent.
All parameters are loaded from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class APIConfig:
    """API credentials and endpoints."""
    # No paid API keys required — FinBERT runs locally, yfinance is free
    alpaca_api_key: str = field(default_factory=lambda: os.getenv("ALPACA_API_KEY", ""))
    alpaca_secret_key: str = field(default_factory=lambda: os.getenv("ALPACA_SECRET_KEY", ""))
    alpaca_base_url: str = field(default_factory=lambda: os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"))


@dataclass
class TradingConfig:
    """Trading strategy parameters."""
    # Technical indicator windows
    sma_short: int = 20
    sma_long: int = 50
    atr_period: int = 14

    # Sentiment thresholds
    min_conviction: float = 7.0
    min_headlines: int = 2
    negative_veto_conviction: float = 8.0

    # Risk management
    stop_loss_pct: float = 0.02       # 2% stop-loss
    take_profit_pct: float = 0.04     # 4% take-profit
    max_position_pct: float = 0.10    # 10% of portfolio per position
    max_daily_trades: int = 3
    cooldown_minutes: int = 60

    # Portfolio
    initial_capital: float = 100_000.0


@dataclass
class BacktestConfig:
    """Backtesting parameters."""
    lookback_period: str = "1y"
    benchmark_ticker: str = "SPY"
    risk_free_rate: float = 0.05  # 5% annualized


@dataclass
class ModelConfig:
    """NLP model configuration."""
    model_name: str = "ProsusAI/finbert"   # Free, local, finance-specific
    # Note: temperature and max_tokens are not used by FinBERT
    # (they are transformer classifier parameters, not generative LLM params)


@dataclass
class Settings:
    """Master settings object."""
    api: APIConfig = field(default_factory=APIConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


# Singleton settings instance
settings = Settings()
