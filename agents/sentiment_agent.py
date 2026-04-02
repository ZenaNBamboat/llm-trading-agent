"""
Agent 3: LLM Sentiment Agent — The "Brain"

This is the CORE ML component of the trading agent.

Responsibilities:
- Classify each headline as POSITIVE / NEGATIVE / NEUTRAL
- Assign a conviction score (0–10) derived from model confidence
- Provide a short rationale for each classification
- Aggregate multiple headlines into a 24h conviction summary
- Detect headline disagreement (innovative feature)

Model choice: FinBERT (ProsusAI/finbert)
─────────────────────────────────────────
  FinBERT is a BERT model fine-tuned specifically on financial text
  (financial news, earnings calls, analyst reports). It outperforms
  general-purpose sentiment models on financial language because:

  - "Recalled 2 million vehicles" is NEGATIVE in finance, but a
    general model might score it neutral.
  - "Beat analyst estimates" is strongly POSITIVE in finance but
    meaningless to a general NLP model.

  FinBERT returns calibrated probabilities for positive/negative/neutral,
  which we convert into a conviction score (0–10) using the spread
  between the top class and the runner-up.

  This is FREE, runs locally, and requires NO API key.

Fallback: Financial Lexicon Classifier
──────────────────────────────────────
  If FinBERT cannot be loaded (e.g. no internet on first run), a
  built-in keyword-based financial sentiment classifier activates.
  This uses curated positive/negative financial terms weighted by
  market-impact severity. It ensures the demo NEVER fails.

Design rationale:
  The NLP model is used ONLY for unstructured text interpretation —
  specifically financial headline sentiment classification. It is NOT
  used for numerical forecasting, technical indicator calculation, or
  trade sizing, because those tasks are deterministic and more reliably
  handled in code.

  Allowed models (from assignment):
    "Local (Free/Privacy-focused): Llama-3-8B, DistilBERT (fine-tuned
     for finance like FinBERT), or Mistral via HuggingFace transformers."
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from config.settings import settings
from utils.logger import get_logger

logger = get_logger("SentimentAgent")


# ======================================================================
# Financial Lexicon — curated terms with impact weights
# Used as fallback if FinBERT is unavailable, and also strong on its own
# ======================================================================

POSITIVE_TERMS = {
    # Earnings & revenue (high impact)
    "beat": 3, "beats": 3, "exceeded": 3, "surpassed": 3, "topped": 3,
    "record revenue": 4, "record earnings": 4, "record profit": 4,
    "stronger-than-expected": 3, "better-than-expected": 3,
    "earnings beat": 4, "revenue beat": 4,
    # Analyst actions
    "upgrade": 3, "upgrades": 3, "upgraded": 3, "outperform": 2,
    "buy rating": 3, "price target raised": 3, "raises outlook": 3,
    "strong buy": 3, "overweight": 2,
    # Growth signals
    "growth": 2, "expansion": 2, "margin expansion": 3,
    "strong demand": 3, "demand pipeline": 2,
    "partnership": 2, "strategic partnership": 2,
    "acquisition": 2, "new contract": 2, "major contract": 3,
    # Approvals & positive events
    "approved": 2, "fda approval": 3, "regulatory approval": 3,
    "dividend increase": 2, "buyback": 2, "share repurchase": 2,
    "breakout": 2, "all-time high": 3, "rally": 2,
    "guidance raised": 3, "strong guidance": 3,
}

NEGATIVE_TERMS = {
    # Earnings & revenue misses (high impact)
    "miss": 3, "misses": 3, "missed": 3, "fell short": 3,
    "disappointing": 2, "weaker-than-expected": 3, "below expectations": 3,
    "earnings miss": 4, "revenue miss": 4, "revenue decline": 3,
    # Analyst actions
    "downgrade": 3, "downgrades": 3, "downgraded": 3, "underperform": 2,
    "sell rating": 3, "price target cut": 3, "lowers outlook": 3,
    "underweight": 2, "reduces forecast": 3,
    # Risk signals
    "recall": 3, "recalls": 3, "lawsuit": 2, "sued": 2, "litigation": 2,
    "investigation": 2, "probe": 2, "regulatory probe": 3,
    "fraud": 4, "allegations": 2, "scandal": 3,
    "supply chain": 2, "shortage": 2, "headwinds": 2,
    # Negative events
    "layoffs": 2, "restructuring": 2, "bankruptcy": 4, "default": 4,
    "guidance cut": 3, "warns": 2, "warning": 2, "concern": 1, "concerns": 1,
    "decline": 2, "plunge": 3, "crash": 3, "tumble": 2, "drops": 2,
    "loss": 2, "losses": 2, "deficit": 2,
}


class SentimentAgent:
    """
    Financial sentiment analysis agent.

    Primary engine: FinBERT (ProsusAI/finbert) — free, local, domain-specific.
    Fallback engine: Financial lexicon classifier — works offline, no downloads.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.model.model_name
        self.classifier = None
        self.mode = "lexicon"  # default fallback

        # Try to load FinBERT
        try:
            logger.info("Loading FinBERT model: %s ...", self.model_name)
            from transformers import pipeline as hf_pipeline
            self.classifier = hf_pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                top_k=3,
                truncation=True,
                max_length=512,
            )
            self.mode = "finbert"
            logger.info("FinBERT loaded successfully (local, free, no API key)")
        except Exception as e:
            logger.warning("FinBERT unavailable (%s), using financial lexicon fallback", type(e).__name__)
            logger.info("Financial lexicon classifier activated (works offline, no downloads needed)")

    # ------------------------------------------------------------------
    # Single headline analysis — routes to active engine
    # ------------------------------------------------------------------
    def analyze_headline(self, headline: str) -> Dict:
        """Classify a single headline using the active engine."""
        logger.info("Analyzing [%s]: %s", self.mode, headline[:80])

        if self.mode == "finbert":
            return self._analyze_finbert(headline)
        else:
            return self._analyze_lexicon(headline)

    # ------------------------------------------------------------------
    # Engine 1: FinBERT (primary)
    # ------------------------------------------------------------------
    def _analyze_finbert(self, headline: str) -> Dict:
        """
        Classify using FinBERT.

        Conviction score derivation:
          We use the GAP between the winning class probability and the
          runner-up to measure how confident the model really is.
          - positive=0.95, neutral=0.03 -> gap=0.92 -> conviction=9
          - positive=0.45, neutral=0.35 -> gap=0.10 -> conviction=3
        """
        try:
            results = self.classifier(headline)
            scores = results[0] if results else []

            prob_map = {}
            for item in scores:
                label = item["label"].upper()
                prob_map[label] = item["score"]

            top_label = max(prob_map, key=prob_map.get)
            top_prob = prob_map[top_label]

            sorted_probs = sorted(prob_map.values(), reverse=True)
            runner_up_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
            gap = top_prob - runner_up_prob
            conviction = self._gap_to_conviction(gap, top_prob)

            strength = "strongly" if gap >= 0.40 else "moderately" if gap >= 0.15 else "weakly"
            rationale = f"FinBERT {strength} classifies this as {top_label.lower()} market impact (conf={top_prob:.0%}, gap={gap:.2f})."

            result = {
                "sentiment_label": top_label,
                "conviction_score": conviction,
                "confidence": round(top_prob, 4),
                "prob_positive": round(prob_map.get("POSITIVE", 0.0), 4),
                "prob_negative": round(prob_map.get("NEGATIVE", 0.0), 4),
                "prob_neutral": round(prob_map.get("NEUTRAL", 0.0), 4),
                "rationale": rationale,
                "engine": "finbert",
            }
            logger.info("  -> %s (conviction=%d) %s", top_label, conviction, rationale)
            return result

        except Exception as e:
            logger.error("FinBERT inference error: %s — falling back to lexicon", e)
            return self._analyze_lexicon(headline)

    def _gap_to_conviction(self, gap: float, top_prob: float) -> int:
        """Convert probability gap into conviction score (0–10)."""
        if top_prob < 0.40:
            return 2
        if gap >= 0.80:   return 10
        elif gap >= 0.60: return 9
        elif gap >= 0.45: return 8
        elif gap >= 0.30: return 7
        elif gap >= 0.20: return 6
        elif gap >= 0.12: return 5
        elif gap >= 0.05: return 4
        else:             return 3

    # ------------------------------------------------------------------
    # Engine 2: Financial Lexicon (fallback — works offline)
    # ------------------------------------------------------------------
    def _analyze_lexicon(self, headline: str) -> Dict:
        """
        Classify using a curated financial lexicon.

        Scoring logic:
          - Scan headline for positive and negative financial terms
          - Weight each term by market-impact severity (1–4)
          - Conviction = normalized difference between pos/neg scores
          - Label = direction of the stronger signal
        """
        text = headline.lower()

        pos_score = 0
        neg_score = 0
        pos_matches = []
        neg_matches = []

        for term, weight in POSITIVE_TERMS.items():
            if term in text:
                pos_score += weight
                pos_matches.append(term)

        for term, weight in NEGATIVE_TERMS.items():
            if term in text:
                neg_score += weight
                neg_matches.append(term)

        total = pos_score + neg_score

        if total == 0:
            label = "NEUTRAL"
            conviction = 3
            rationale = "No strong financial signals detected in headline."
        elif pos_score > neg_score:
            label = "POSITIVE"
            dominance = pos_score / total
            # Conviction uses BOTH absolute strength and directional clarity
            # A single weak match scores ~5; multiple strong matches score 8-9
            raw = min(pos_score, 12)  # cap at 12 to prevent runaway scores
            conviction = min(10, max(4, int(2.5 + raw * 0.6 * dominance)))
            terms_str = ", ".join(pos_matches[:3])
            rationale = f"Positive financial signals detected ({terms_str}); strength={pos_score}, dominance={dominance:.0%}."
        elif neg_score > pos_score:
            label = "NEGATIVE"
            dominance = neg_score / total
            raw = min(neg_score, 12)
            conviction = min(10, max(4, int(2.5 + raw * 0.6 * dominance)))
            terms_str = ", ".join(neg_matches[:3])
            rationale = f"Negative financial signals detected ({terms_str}); strength={neg_score}, dominance={dominance:.0%}."
        else:
            label = "NEUTRAL"
            conviction = 3
            rationale = "Mixed financial signals with no clear directional dominance."

        result = {
            "sentiment_label": label,
            "conviction_score": conviction,
            "confidence": round(max(pos_score, neg_score) / max(total, 1), 4),
            "prob_positive": round(pos_score / max(total, 1), 4),
            "prob_negative": round(neg_score / max(total, 1), 4),
            "prob_neutral": round(1.0 - (pos_score + neg_score) / max(total + 3, 1), 4),
            "rationale": rationale,
            "engine": "lexicon",
        }
        logger.info("  -> %s (conviction=%d) %s", label, conviction, rationale)
        return result

    # ------------------------------------------------------------------
    # Batch analysis
    # ------------------------------------------------------------------
    def analyze_news_batch(self, news_items: List[Dict]) -> List[Dict]:
        """Analyze a list of news items, returning enriched results."""
        logger.info("Analyzing batch of %d headlines with %s engine", len(news_items), self.mode)
        outputs = []

        for item in news_items:
            result = self.analyze_headline(item["headline"])
            outputs.append({
                "headline": item["headline"],
                "published_at": item.get("published_at", ""),
                "source": item.get("source", "unknown"),
                "sentiment_label": result["sentiment_label"],
                "conviction_score": result["conviction_score"],
                "confidence": result["confidence"],
                "rationale": result["rationale"],
                "engine": result.get("engine", self.mode),
            })

        return outputs

    # ------------------------------------------------------------------
    # Aggregation with disagreement detection (innovative feature)
    # ------------------------------------------------------------------
    def aggregate_sentiment(self, scored_items: List[Dict]) -> Dict:
        """
        Aggregate individual headline scores into a 24h conviction summary.

        INNOVATIVE FEATURE: Headline Disagreement Filter
        ------------------------------------------------
        Instead of just averaging, we measure agreement across headlines.
        If the same ticker has strongly positive AND strongly negative
        headlines, we flag "high_disagreement" and reduce conviction.
        This prevents trading on contradictory or noisy information.

        Example:
          4 positive headlines, conviction 8 avg -> tradeable
          2 strongly positive + 2 strongly negative -> high uncertainty -> HOLD
        """
        if not scored_items:
            return {
                "headline_count": 0,
                "average_conviction": 0.0,
                "positive_ratio": 0.0,
                "negative_ratio": 0.0,
                "overall_signal": "NEUTRAL",
                "headline_agreement": "no_data",
                "weighted_score": 0.0,
                "tradeable": False,
            }

        score_map = {"POSITIVE": 1, "NEUTRAL": 0, "NEGATIVE": -1}
        n = len(scored_items)

        weighted_sum = sum(
            score_map.get(item["sentiment_label"], 0) * item["conviction_score"]
            for item in scored_items
        )

        avg_conviction = sum(item["conviction_score"] for item in scored_items) / n
        positive_count = sum(1 for item in scored_items if item["sentiment_label"] == "POSITIVE")
        negative_count = sum(1 for item in scored_items if item["sentiment_label"] == "NEGATIVE")
        positive_ratio = positive_count / n
        negative_ratio = negative_count / n

        # Disagreement detection
        high_conviction_positive = sum(
            1 for item in scored_items
            if item["sentiment_label"] == "POSITIVE" and item["conviction_score"] >= 7
        )
        high_conviction_negative = sum(
            1 for item in scored_items
            if item["sentiment_label"] == "NEGATIVE" and item["conviction_score"] >= 7
        )

        if high_conviction_positive > 0 and high_conviction_negative > 0:
            agreement = "high_disagreement"
        elif positive_ratio >= 0.7 or negative_ratio >= 0.7:
            agreement = "strong_consensus"
        else:
            agreement = "moderate_agreement"

        if weighted_sum > 5:
            overall = "POSITIVE"
        elif weighted_sum < -5:
            overall = "NEGATIVE"
        else:
            overall = "NEUTRAL"

        tradeable = (
            agreement != "high_disagreement"
            and avg_conviction >= settings.trading.min_conviction
            and n >= settings.trading.min_headlines
        )

        summary = {
            "headline_count": n,
            "average_conviction": round(avg_conviction, 2),
            "positive_ratio": round(positive_ratio, 2),
            "negative_ratio": round(negative_ratio, 2),
            "overall_signal": overall,
            "headline_agreement": agreement,
            "weighted_score": round(weighted_sum, 2),
            "tradeable": tradeable,
        }

        logger.info(
            "Aggregated: signal=%s conviction=%.1f agreement=%s tradeable=%s",
            overall, avg_conviction, agreement, tradeable,
        )
        return summary
