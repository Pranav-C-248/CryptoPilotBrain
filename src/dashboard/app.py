import streamlit as st
from styles import apply_global_style
import os
import sys
from datetime import datetime, timezone
import traceback

# Project Path Setup
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

# Internal Imports
from src.dashboard.shared.state import init_session_state
from src.dashboard.shared.database import get_db, AuditLog, QueuedTrade
from src.dashboard.components.cards import render_signal_card
from src.dashboard.components.charts import render_candlestick_chart
from src.tools.binance_client import BinancePublicClient
from src.tools.indicators import add_indicators


# Page Configuration
st.set_page_config(
    page_title="CryptoPilot",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Apply Global UI Styling
apply_global_style()


# Initialize Session State
init_session_state()


# Page Title
st.title("CryptoPilot Dashboard")
st.markdown("""
<div class="card" style="font-size:13px; color:#64748b;">
AI Trading Dashboard
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("## CryptoPilot")
    st.caption("AI-Powered Trading System")
    st.markdown("---")

    st.markdown("### Market Selection")
    symbol = st.selectbox(
        "Trading Pair",
        options=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        index=["BTC/USDT", "ETH/USDT", "SOL/USDT"].index(
            st.session_state["current_symbol"]
        ),
        label_visibility="collapsed",
    )

    if symbol != st.session_state["current_symbol"]:
        st.session_state["current_symbol"] = symbol
        st.session_state["last_market_data"] = None

    st.markdown("---")
    st.markdown("### System Status")
    st.success("Engine Online")
    st.info("Agent Cycle: 4h")

    st.markdown("---")
    st.caption("v1.0.0")


# Fetch Market Data
if st.session_state["last_market_data"] is None:
    with st.spinner("Fetching market data..."):
        try:
            client = BinancePublicClient()
            df = client.get_historical_klines(
                st.session_state["current_symbol"], "4h", limit=100
            )
            df = add_indicators(df)
            st.session_state["last_market_data"] = df
        except Exception as e:
            st.error(f"Error fetching market data: {e}")


# Fetch Latest Signals from Database
db_gen = get_db()
db = next(db_gen)

try:
    latest_log = (
        db.query(AuditLog)
        .filter(AuditLog.symbol == st.session_state["current_symbol"])
        .order_by(AuditLog.id.desc())
        .first()
    )

    latest_trade = (
        db.query(QueuedTrade)
        .filter(
            QueuedTrade.asset_name == st.session_state["current_symbol"]
        )
        .order_by(QueuedTrade.id.desc())
        .first()
    )
finally:
    db.close()


# Top Metrics Section (Market Data Based)
if st.session_state["last_market_data"] is not None:
    df = st.session_state["last_market_data"]

    if len(df) >= 2:
        last_close = df["close"].iloc[-1]
        prev_close = df["close"].iloc[-2]
        pct_change = ((last_close - prev_close) / prev_close) * 100 if prev_close != 0 else 0
    else:
        last_close = df["close"].iloc[-1]
        pct_change = 0

    high_val = df["high"].iloc[-1]
    low_val = df["low"].iloc[-1]

    price_str = f"${last_close:,.2f}"
    change_color = "#22c55e" if pct_change >= 0 else "#ef4444"
    change_str = f"{pct_change:+.2f}%"
    high_str = f"${high_val:,.2f}"
    low_str = f"${low_val:,.2f}"

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="metric-title">Current Price</div>
            <div class="metric-value">{price_str}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="card">
            <div class="metric-title">Change</div>
            <div class="metric-value" style="color:{change_color};">
                {change_str}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="card">
            <div class="metric-title">High</div>
            <div class="metric-value">{high_str}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="card">
            <div class="metric-title">Low</div>
            <div class="metric-value">{low_str}</div>
        </div>
        """, unsafe_allow_html=True)


# Main Layout
col_left, col_right = st.columns([2, 1])


# Market Chart Section
with col_left:
    st.subheader("Market Chart")

    if st.session_state["last_market_data"] is not None:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        fig = render_candlestick_chart(
            st.session_state["last_market_data"],
            title=st.session_state["current_symbol"],
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False}
        )

        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Market data not available")


# AI Analysis Section
with col_right:
    st.subheader("AI Analysis")

    if latest_log:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        signal_data = {
            "signal": latest_log.action,
            "confidence": latest_log.confidence,
        }

        render_signal_card(signal_data)

        # Timestamp (safe)
        timestamp_str = (
            latest_log.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')
            if latest_log.timestamp else "N/A"
        )

        st.markdown(f"""
        <div style="margin-bottom:10px; font-size:12px; color:#94a3b8;">
            Timestamp: {timestamp_str}
        </div>
        """, unsafe_allow_html=True)

        # Analyst Reasoning
        st.markdown(f"""
        <div class="card">
            <div class="metric-title">Analyst Reasoning</div>
            <div style="margin-top:8px; font-size:14px; color:#cbd5f5; line-height:1.5;">
                {latest_log.explanation}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Trade Proposal
        if latest_trade and latest_trade.signal != "HOLD":
            st.markdown(f"""
            <div class="card">
                <div class="metric-title">Trade Proposal</div>

                <div style="margin-top:10px;">
                    <p style="margin:4px 0;"><b>Status:</b> {latest_trade.status}</p>
                    <p style="margin:4px 0;"><b>Position Size:</b> ${latest_trade.position_size:.2f}</p>
                    <p style="margin:4px 0;"><b>Target:</b> ${latest_trade.target:.4f}</p>
                    <p style="margin:4px 0;"><b>Stop Loss:</b> ${latest_trade.stop_loss:.4f}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="card">
            No analysis available yet. Waiting for agent cycle.
        </div>
        """, unsafe_allow_html=True)