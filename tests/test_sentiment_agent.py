"""
Tests for SentimentAgent — verifies aggregation and disagreement logic.
(These tests do not require any API key — FinBERT runs locally.)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.sentiment_agent import SentimentAgent


def test_aggregation_positive_consensus():
    """Strong positive consensus should produce POSITIVE with high conviction."""
    agent = SentimentAgent.__new__(SentimentAgent)

    items = [
        {"sentiment_label": "POSITIVE", "conviction_score": 9},
        {"sentiment_label": "POSITIVE", "conviction_score": 8},
        {"sentiment_label": "POSITIVE", "conviction_score": 7},
    ]
    result = agent.aggregate_sentiment(items)

    assert result["overall_signal"] == "POSITIVE"
    assert result["average_conviction"] >= 7
    assert result["headline_agreement"] == "strong_consensus"
    assert result["tradeable"] is True
    print("PASS: test_aggregation_positive_consensus")


def test_aggregation_negative_consensus():
    """Strong negative consensus should produce NEGATIVE."""
    agent = SentimentAgent.__new__(SentimentAgent)

    items = [
        {"sentiment_label": "NEGATIVE", "conviction_score": 8},
        {"sentiment_label": "NEGATIVE", "conviction_score": 9},
        {"sentiment_label": "NEGATIVE", "conviction_score": 7},
    ]
    result = agent.aggregate_sentiment(items)

    assert result["overall_signal"] == "NEGATIVE"
    assert result["negative_ratio"] >= 0.7
    print("PASS: test_aggregation_negative_consensus")


def test_disagreement_detection():
    """Mixed high-conviction signals should flag high_disagreement."""
    agent = SentimentAgent.__new__(SentimentAgent)

    items = [
        {"sentiment_label": "POSITIVE", "conviction_score": 9},
        {"sentiment_label": "POSITIVE", "conviction_score": 8},
        {"sentiment_label": "NEGATIVE", "conviction_score": 8},
        {"sentiment_label": "NEGATIVE", "conviction_score": 7},
    ]
    result = agent.aggregate_sentiment(items)

    assert result["headline_agreement"] == "high_disagreement"
    assert result["tradeable"] is False
    print("PASS: test_disagreement_detection")


def test_empty_input():
    """Empty input should return NEUTRAL with no tradeability."""
    agent = SentimentAgent.__new__(SentimentAgent)

    result = agent.aggregate_sentiment([])
    assert result["overall_signal"] == "NEUTRAL"
    assert result["headline_count"] == 0
    assert result["tradeable"] is False
    print("PASS: test_empty_input")


def test_low_conviction_not_tradeable():
    """Low conviction should not be tradeable even with consensus."""
    agent = SentimentAgent.__new__(SentimentAgent)

    items = [
        {"sentiment_label": "POSITIVE", "conviction_score": 3},
        {"sentiment_label": "POSITIVE", "conviction_score": 4},
        {"sentiment_label": "NEUTRAL", "conviction_score": 2},
    ]
    result = agent.aggregate_sentiment(items)

    assert result["tradeable"] is False
    print("PASS: test_low_conviction_not_tradeable")


if __name__ == "__main__":
    test_aggregation_positive_consensus()
    test_aggregation_negative_consensus()
    test_disagreement_detection()
    test_empty_input()
    test_low_conviction_not_tradeable()
    print("\nAll SentimentAgent tests passed!")
