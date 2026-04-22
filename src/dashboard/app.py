import streamlit as st
import os
import sys
from datetime import datetime, timezone
import traceback

# Add project root to sys.path so imports work correctly
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from src.dashboard.shared.state import init_session_state
from src.dashboard.shared.database import get_db, AuditLog, QueuedTrade
from src.dashboard.components.cards import render_metric_card, render_signal_card
from src.dashboard.components.charts import render_candlestick_chart
from src.tools.binance_client import BinancePublicClient
from src.tools.indicators import add_indicators

# Page Configuration
st.set_page_config(
    page_title="CryptoPilot - Markets",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State
init_session_state()

# Main UI
st.title("CryptoPilot: Markets 📈")

# Sidebar
with st.sidebar:
    st.header("Controls")
    symbol = st.selectbox(
        "Select Asset",
        options=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        index=["BTC/USDT", "ETH/USDT", "SOL/USDT"].index(st.session_state['current_symbol'])
    )
    if symbol != st.session_state['current_symbol']:
        st.session_state['current_symbol'] = symbol
        st.session_state['last_market_data'] = None
        
    st.info("The Data Agent runs automatically in the background every 4 hours.")

# Auto-Fetch Market Data (4h timeframe)
if st.session_state['last_market_data'] is None:
    with st.spinner("Fetching latest 4h market data..."):
        try:
            client = BinancePublicClient()
            df = client.get_historical_klines(st.session_state['current_symbol'], "4h", limit=100)
            df = add_indicators(df)
            st.session_state['last_market_data'] = df
        except Exception as e:
            st.error(f"Failed to fetch market data: {e}")

# Fetch Latest DB Signals
db_gen = get_db()
db = next(db_gen)
try:
    latest_log = db.query(AuditLog).filter(AuditLog.symbol == st.session_state['current_symbol']).order_by(AuditLog.id.desc()).first()
    latest_trade = db.query(QueuedTrade).filter(QueuedTrade.asset_name == st.session_state['current_symbol']).order_by(QueuedTrade.id.desc()).first()
finally:
    db.close()

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Live Chart (4h): {st.session_state['current_symbol']}")
    if st.session_state['last_market_data'] is not None:
        fig = render_candlestick_chart(st.session_state['last_market_data'], title=st.session_state['current_symbol'])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Market data not available.")

with col2:
    st.subheader("Latest AI Analysis")
    if latest_log:
        signal_data = {
            'signal': latest_log.action,
            'confidence': latest_log.confidence
        }
        render_signal_card(signal_data)
        
        st.markdown(f"**Timestamp:** {latest_log.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        st.markdown("### Analyst Reasoning")
        st.info(latest_log.explanation)
        
        if latest_trade and latest_trade.signal != "HOLD":
            st.markdown("### Trade Proposal")
            st.write(f"**Status:** {latest_trade.status}")
            st.write(f"**Size:** ${latest_trade.position_size:.2f}")
            st.write(f"**Target:** ${latest_trade.target:.4f}")
            st.write(f"**Stop Loss:** ${latest_trade.stop_loss:.4f}")
            
    else:
        st.write("No analysis logged yet for this asset. Please wait for the Data Agent's first cycle.")
