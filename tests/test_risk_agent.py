"""
Tests for RiskAgent — verifies risk management guardrails.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.risk_agent import RiskAgent


def test_hold_not_approved():
    """HOLD signals should never be approved."""
    agent = RiskAgent()
    signal = {"action": "HOLD", "reason": "No alignment"}
    result = agent.apply_risk_rules(signal, 150.0, 100_000)
    assert result["approved"] is False
    print("PASS: test_hold_not_approved")


def test_buy_approved():
    """Valid BUY signal should be approved with correct SL/TP."""
    agent = RiskAgent()
    signal = {"action": "BUY", "reason": "Dual confirmation", "ticker": "AAPL"}
    result = agent.apply_risk_rules(signal, 150.0, 100_000)

    assert result["approved"] is True
    assert result["quantity"] > 0
    assert result["stop_loss_price"] < 150.0
    assert result["take_profit_price"] > 150.0
    assert result["stop_loss_price"] == round(150.0 * 0.98, 2)  # 2% SL
    assert result["take_profit_price"] == round(150.0 * 1.04, 2)  # 4% TP
    print("PASS: test_buy_approved")


def test_daily_trade_limit():
    """Should reject trades after daily limit reached."""
    agent = RiskAgent()
    agent.daily_trade_count = 3  # Max is 3

    signal = {"action": "BUY", "reason": "Test", "ticker": "AAPL"}
    result = agent.apply_risk_rules(signal, 150.0, 100_000)
    assert result["approved"] is False
    assert "Daily trade limit" in result["risk_reason"]
    print("PASS: test_daily_trade_limit")


def test_position_sizing():
    """Position size should be max 10% of portfolio."""
    agent = RiskAgent()
    signal = {"action": "BUY", "reason": "Test", "ticker": "AAPL"}
    result = agent.apply_risk_rules(signal, 150.0, 100_000)

    max_allocation = 100_000 * 0.10  # $10,000
    max_shares = int(max_allocation // 150.0)
    assert result["quantity"] == max_shares
    print("PASS: test_position_sizing")


if __name__ == "__main__":
    test_hold_not_approved()
    test_buy_approved()
    test_daily_trade_limit()
    test_position_sizing()
    print("\nAll RiskAgent tests passed!")
