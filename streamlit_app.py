"""
Streamlit Dashboard for LLM-Driven Trading Agent
─────────────────────────────────────────────────
A visual interface for the class demo. Calls the existing
TradingManager pipeline without modifying any core files.

Run with:
    streamlit run streamlit_app.py
"""

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Trading Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global */
    .stApp { font-family: 'DM Sans', sans-serif; }
    code, .stCode, pre { font-family: 'JetBrains Mono', monospace !important; }

    /* Hero banner */
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        margin-bottom: 1.5rem;
        color: white;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -50%; right: -20%;
        width: 400px; height: 400px;
        background: radial-gradient(circle, rgba(59,130,246,0.15) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero h1 { margin: 0 0 0.25rem 0; font-size: 2rem; font-weight: 700; letter-spacing: -0.5px; }
    .hero p  { margin: 0; opacity: 0.7; font-size: 0.95rem; }

    /* Metric cards */
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        text-align: center;
    }
    .metric-card .label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; color: #64748b; margin-bottom: 0.3rem; }
    .metric-card .value { font-size: 1.8rem; font-weight: 700; color: #0f172a; }
    .metric-card .sub   { font-size: 0.8rem; color: #94a3b8; margin-top: 0.2rem; }

    /* Decision badge */
    .decision-badge {
        display: inline-block;
        padding: 0.6rem 2rem;
        border-radius: 50px;
        font-size: 1.4rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .decision-BUY  { background: #dcfce7; color: #166534; border: 2px solid #22c55e; }
    .decision-SELL { background: #fee2e2; color: #991b1b; border: 2px solid #ef4444; }
    .decision-HOLD { background: #fef3c7; color: #92400e; border: 2px solid #f59e0b; }

    /* Sentiment row */
    .sent-row {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        padding: 0.75rem 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        border-left: 4px solid;
    }
    .sent-POSITIVE { background: #f0fdf4; border-left-color: #22c55e; }
    .sent-NEGATIVE { background: #fef2f2; border-left-color: #ef4444; }
    .sent-NEUTRAL  { background: #f8fafc; border-left-color: #94a3b8; }
    .sent-badge {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 500;
        font-size: 0.8rem;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        white-space: nowrap;
    }
    .sent-badge-POSITIVE { background: #dcfce7; color: #166534; }
    .sent-badge-NEGATIVE { background: #fee2e2; color: #991b1b; }
    .sent-badge-NEUTRAL  { background: #f1f5f9; color: #475569; }
    .sent-headline { font-size: 0.9rem; color: #334155; flex: 1; }
    .sent-rationale { font-size: 0.78rem; color: #64748b; font-style: italic; }

    /* Pipeline step labels */
    .step-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #3b82f6;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }

    /* Guardrails section */
    .guardrail-box {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        font-size: 0.85rem;
        color: #92400e;
    }

    /* Hide streamlit menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Cached agent init ────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_manager():
    """Load TradingManager once, reuse across reruns."""
    from app import TradingManager
    return TradingManager()


# ── Hero ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>📈 LLM-Driven Trading Agent</h1>
    <p>Event-Aware Multi-Agent Trading Copilot &nbsp;·&nbsp; FinBERT Sentiment &nbsp;·&nbsp; SMA Trend Confirmation &nbsp;·&nbsp; Automated Risk Controls</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Configuration")
    ticker = st.text_input("Ticker Symbol", value="AAPL", max_chars=5).upper().strip()
    portfolio = st.number_input("Portfolio Value ($)", value=100_000, step=10_000, min_value=1_000)

    st.markdown("---")
    run_backtest = st.checkbox("Run 12-month backtest", value=True)

    st.markdown("---")
    run_btn = st.button("🚀  Run Agent Pipeline", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("### Architecture")
    st.markdown("""
    ```
    News → FinBERT → Signal
              ↓
    Market → SMA50 → Signal
              ↓
         Risk Agent
              ↓
        BUY / HOLD / SELL
    ```
    """)

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem;color:#94a3b8;'>"
        "No paid APIs · FinBERT runs locally · yfinance for data"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Main content ─────────────────────────────────────────────────────
if run_btn:
    if not ticker or len(ticker) < 1:
        st.error("Please enter a valid ticker symbol.")
        st.stop()

    manager = load_manager()

    # ── STEP 1–6: Live pipeline ──────────────────────────────────────
    with st.status(f"Running agent pipeline for **{ticker}**...", expanded=True) as status:
        st.write("⏳ Fetching market data...")
        try:
            result = manager.run(ticker, portfolio_value=portfolio)
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.stop()
        status.update(label=f"Pipeline complete for {ticker}", state="complete")

    ms = result["market_snapshot"]
    ss = result["sentiment_summary"]
    sig = result["signal"]
    exe = result["execution_result"]
    scored = result["scored_news"]

    # ── Market snapshot metrics ───────────────────────────────────────
    st.markdown('<div class="step-label">Step 1 · Market Data</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Close</div><div class="value">${ms['close']:.2f}</div>
            <div class="sub">{ms['date']}</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="label">SMA 20</div><div class="value">${ms['sma20']:.2f}</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="label">SMA 50</div><div class="value">${ms['sma50']:.2f}</div></div>""", unsafe_allow_html=True)
    with c4:
        trend_icon = "🟢" if ms["trend_up"] else "🔴"
        trend_text = "Uptrend" if ms["trend_up"] else "Downtrend"
        st.markdown(f"""<div class="metric-card">
            <div class="label">Trend</div><div class="value">{trend_icon}</div>
            <div class="sub">{trend_text}</div></div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""<div class="metric-card">
            <div class="label">ATR 14</div><div class="value">${ms.get('atr14', 0):.2f}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Sentiment results ─────────────────────────────────────────────
    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.markdown('<div class="step-label">Steps 2–3 · News Ingestion + FinBERT Sentiment</div>', unsafe_allow_html=True)

        engine = scored[0].get("engine", "finbert") if scored else "unknown"
        st.caption(f"Engine: **{engine}** · {len(scored)} headline(s) analyzed")

        for item in scored:
            label = item["sentiment_label"]
            score = item["conviction_score"]
            headline = item["headline"]
            rationale = item.get("rationale", "")
            st.markdown(f"""
            <div class="sent-row sent-{label}">
                <span class="sent-badge sent-badge-{label}">{label} {score}/10</span>
                <div>
                    <div class="sent-headline">{headline}</div>
                    <div class="sent-rationale">{rationale}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="step-label">Aggregated Conviction</div>', unsafe_allow_html=True)

        ac1, ac2 = st.columns(2)
        with ac1:
            st.markdown(f"""<div class="metric-card">
                <div class="label">Signal</div><div class="value">{ss['overall_signal']}</div></div>""", unsafe_allow_html=True)
        with ac2:
            st.markdown(f"""<div class="metric-card">
                <div class="label">Conviction</div><div class="value">{ss['average_conviction']:.1f}</div>
                <div class="sub">/ 10</div></div>""", unsafe_allow_html=True)

        ac3, ac4 = st.columns(2)
        with ac3:
            agree_icon = {"strong_consensus": "✅", "moderate_agreement": "⚠️", "high_disagreement": "🚫", "no_data": "—"}
            st.markdown(f"""<div class="metric-card">
                <div class="label">Agreement</div>
                <div class="value">{agree_icon.get(ss['headline_agreement'], '?')}</div>
                <div class="sub">{ss['headline_agreement'].replace('_', ' ')}</div></div>""", unsafe_allow_html=True)
        with ac4:
            trd = ss.get("tradeable", False)
            st.markdown(f"""<div class="metric-card">
                <div class="label">Tradeable</div>
                <div class="value">{'✅' if trd else '❌'}</div>
                <div class="sub">{'Yes' if trd else 'No'}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Final decision ────────────────────────────────────────────────
    st.markdown('<div class="step-label">Steps 4–6 · Signal → Risk → Execution</div>', unsafe_allow_html=True)

    dec_col, det_col = st.columns([1, 2])
    with dec_col:
        action = sig["action"]
        st.markdown(
            f'<div style="text-align:center;padding:1.5rem 0;">'
            f'<span class="decision-badge decision-{action}">{action}</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p style='text-align:center;font-size:0.85rem;color:#64748b;'>{exe['status']}</p>",
            unsafe_allow_html=True,
        )

    with det_col:
        st.markdown(f"**Reason:** {sig['reason']}")

        if exe.get("quantity"):
            r1, r2, r3 = st.columns(3)
            r1.metric("Quantity", f"{exe['quantity']} shares")
            r2.metric("Stop-Loss", f"${exe['stop_loss_price']:.2f}")
            r3.metric("Take-Profit", f"${exe['take_profit_price']:.2f}")
        else:
            risk_reason = result["trade_plan"].get("risk_reason", "")
            if risk_reason:
                st.info(f"Risk Agent: {risk_reason}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Guardrails ────────────────────────────────────────────────────
    with st.expander("🛡️ Guardrails & Risk Controls", expanded=False):
        st.markdown("""
        <div class="guardrail-box">
            <strong>Active Guardrails:</strong><br>
            · Conviction threshold ≥ 7/10 required for trade<br>
            · Minimum 2 headlines required<br>
            · Headline Disagreement Filter — vetoes trade if headlines conflict<br>
            · HOLD when technical & sentiment signals do not align<br>
            · Stop-loss 2% · Take-profit 4% · Max position 10%<br>
            · Daily trade limit 3 · Cooldown after stop-out<br>
            · FinBERT used for text only — no LLM for numeric calculations
        </div>
        """, unsafe_allow_html=True)

    # ── Backtest ──────────────────────────────────────────────────────
    if run_backtest:
        st.markdown("---")
        st.markdown('<div class="step-label">Step 7 · 12-Month Backtest vs SPY</div>', unsafe_allow_html=True)

        with st.spinner("Running backtest..."):
            try:
                bt_result = manager.run_backtest(ticker)
                bt_metrics = bt_result["metrics"]
            except Exception as e:
                st.warning(f"Backtest could not run (likely no market data access): {e}")
                bt_metrics = None

        if bt_metrics:
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                delta_color = "normal" if bt_metrics["total_return_pct"] >= 0 else "inverse"
                st.metric("Strategy Return", f"{bt_metrics['total_return_pct']:.1f}%")
            with m2:
                st.metric("Benchmark (SPY)", f"{bt_metrics['benchmark_return_pct']:.1f}%")
            with m3:
                st.metric("Sharpe Ratio", f"{bt_metrics['sharpe_ratio']:.2f}")
            with m4:
                st.metric("Max Drawdown", f"{bt_metrics['max_drawdown_pct']:.1f}%")
            with m5:
                st.metric("Win Rate", f"{bt_metrics['win_rate_pct']:.0f}%")

            # Results comparison table
            comp_df = pd.DataFrame({
                "Metric": ["Total Return", "Sharpe Ratio", "Max Drawdown", "Win Rate", "Total Trades"],
                "Strategy": [
                    f"{bt_metrics['total_return_pct']:.1f}%",
                    f"{bt_metrics['sharpe_ratio']:.2f}",
                    f"{bt_metrics['max_drawdown_pct']:.1f}%",
                    f"{bt_metrics['win_rate_pct']:.0f}%",
                    str(bt_metrics["total_trades"]),
                ],
                "SPY Buy & Hold": [
                    f"{bt_metrics['benchmark_return_pct']:.1f}%",
                    f"{bt_metrics.get('benchmark_sharpe', 'N/A')}",
                    f"{bt_metrics.get('bench_max_drawdown_pct', 'N/A')}%",
                    "N/A",
                    "1",
                ],
            })
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # Show equity curve image
        curve_path = Path("outputs/equity_curve.png")
        if curve_path.exists():
            st.image(str(curve_path), caption="Equity Curve vs SPY Benchmark", use_container_width=True)

        bench_path = Path("outputs/benchmark_vs_strategy.png")
        if bench_path.exists():
            st.image(str(bench_path), caption="Return Comparison", use_container_width=True)

    # ── Limitation ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="step-label">Step 8 · Honest Limitation</div>', unsafe_allow_html=True)
    st.markdown(
        "> The backtest validates the technical/risk engine rigorously. "
        "Historical sentiment replay is a future extension; the current "
        "backtest is conservative and avoids fabricating unavailable news data. "
        "The system prioritizes risk control and avoids trading under uncertainty, "
        "which reduces drawdowns but can underperform in trending markets."
    )

else:
    # ── Landing state ─────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)

    l1, l2, l3 = st.columns(3)
    with l1:
        st.markdown("""
        <div class="metric-card">
            <div class="label">Sentiment Engine</div>
            <div class="value" style="font-size:1.2rem;">FinBERT</div>
            <div class="sub">Free · Local · Domain-specific</div>
        </div>""", unsafe_allow_html=True)
    with l2:
        st.markdown("""
        <div class="metric-card">
            <div class="label">Architecture</div>
            <div class="value" style="font-size:1.2rem;">7 Agents</div>
            <div class="sub">Multi-agent orchestration</div>
        </div>""", unsafe_allow_html=True)
    with l3:
        st.markdown("""
        <div class="metric-card">
            <div class="label">Paid APIs</div>
            <div class="value" style="font-size:1.2rem;">Zero</div>
            <div class="sub">100% free stack</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 Enter a ticker and click **Run Agent Pipeline** to start the demo.")

    # Show existing outputs if available
    curve_path = Path("outputs/equity_curve.png")
    if curve_path.exists():
        with st.expander("📊 Previous backtest results", expanded=False):
            st.image(str(curve_path), use_container_width=True)
