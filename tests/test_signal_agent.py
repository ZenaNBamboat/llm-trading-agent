"""
Tests for SignalAgent — verifies dual-confirmation logic.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.signal_agent import SignalAgent


def test_buy_signal():
    """BUY when trend is up AND sentiment is positive with high conviction."""
    agent = SignalAgent()

    market = {"trend_up": True, "close": 150.0, "sma20": 148.0, "sma50": 145.0}
    sentiment = {
        "overall_signal": "POSITIVE",
        "average_conviction": 8.5,
        "headline_count": 4,
        "headline_agreement": "strong_consensus",
        "tradeable": True,
    }

    signal = agent.generate_signal(market, sentiment)
    assert signal["action"] == "BUY", f"Expected BUY, got {signal['action']}"
    print("PASS: test_buy_signal")


def test_hold_when_trend_down():
    """HOLD when trend is down even if sentiment is positive."""
    agent = SignalAgent()

    market = {"trend_up": False, "close": 140.0, "sma20": 142.0, "sma50": 145.0}
    sentiment = {
        "overall_signal": "POSITIVE",
        "average_conviction": 9.0,
        "headline_count": 5,
        "headline_agreement": "strong_consensus",
        "tradeable": True,
    }

    signal = agent.generate_signal(market, sentiment)
    assert signal["action"] == "HOLD", f"Expected HOLD, got {signal['action']}"
    print("PASS: test_hold_when_trend_down")


def test_hold_when_low_conviction():
    """HOLD when conviction is below threshold."""
    agent = SignalAgent()

    market = {"trend_up": True, "close": 150.0, "sma20": 148.0, "sma50": 145.0}
    sentiment = {
        "overall_signal": "POSITIVE",
        "average_conviction": 4.0,
        "headline_count": 3,
        "headline_agreement": "moderate_agreement",
        "tradeable": False,
    }

    signal = agent.generate_signal(market, sentiment)
    assert signal["action"] == "HOLD", f"Expected HOLD, got {signal['action']}"
    print("PASS: test_hold_when_low_conviction")


def test_hold_on_disagreement():
    """HOLD when headlines show high disagreement."""
    agent = SignalAgent()

    market = {"trend_up": True, "close": 150.0, "sma20": 148.0, "sma50": 145.0}
    sentiment = {
        "overall_signal": "POSITIVE",
        "average_conviction": 8.0,
        "headline_count": 4,
        "headline_agreement": "high_disagreement",
        "tradeable": False,
    }

    signal = agent.generate_signal(market, sentiment)
    assert signal["action"] == "HOLD", f"Expected HOLD, got {signal['action']}"
    print("PASS: test_hold_on_disagreement")


def test_sell_signal():
    """SELL when trend is down AND sentiment is negative."""
    agent = SignalAgent()

    market = {"trend_up": False, "close": 140.0, "sma20": 142.0, "sma50": 145.0}
    sentiment = {
        "overall_signal": "NEGATIVE",
        "average_conviction": 7.0,
        "headline_count": 3,
        "headline_agreement": "strong_consensus",
        "tradeable": True,
    }

    signal = agent.generate_signal(market, sentiment)
    assert signal["action"] == "SELL", f"Expected SELL, got {signal['action']}"
    print("PASS: test_sell_signal")


if __name__ == "__main__":
    test_buy_signal()
    test_hold_when_trend_down()
    test_hold_when_low_conviction()
    test_hold_on_disagreement()
    test_sell_signal()
    print("\nAll SignalAgent tests passed!")
