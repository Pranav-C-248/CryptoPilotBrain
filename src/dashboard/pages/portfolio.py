import streamlit as st
import os
import sys
import pandas as pd
from datetime import datetime, timezone
import plotly.express as px
import plotly.graph_objects as go

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, project_root)

from src.dashboard.shared.database import get_db, Portfolio, Position, TradeLedger

# Page Configuration
st.set_page_config(
    page_title="CryptoPilot - Portfolio",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.dashboard.shared.theme import inject_theme_css
inject_theme_css()

st.title("CryptoPilot: Portfolio")

# --- Database Fetching & Initialization ---
db_gen = get_db()
db = next(db_gen)

# Initialize portfolio if it doesn't exist
portfolio = db.query(Portfolio).first()
if not portfolio:
    portfolio = Portfolio(balance=10000.0)
    db.add(portfolio)
    db.commit()
    db.refresh(portfolio)

# Fetch data
positions = db.query(Position).all()
trades = db.query(TradeLedger).order_by(TradeLedger.timestamp.desc()).all()

# Close DB session for read
db.close()

# --- Calculations ---
total_realized_pnl = sum(trade.realized_pnl for trade in trades if trade.realized_pnl is not None)
available_balance = portfolio.balance

# If we had live prices, we could calculate unrealized P&L here
# For now, we'll just show the position size and average price

# --- Layout: Top Metrics ---
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Available Balance", value=f"${available_balance:,.2f}")

with col2:
    pnl_pct = (total_realized_pnl / 10000.0) * 100
    if total_realized_pnl > 0:
        st.metric("Total Realized P&L", f"+${total_realized_pnl:,.2f}")
        st.markdown(f'<p style="color:#0ecb81; font-size:0.875rem; margin-top:-15px;">↑ {pnl_pct:.2f}%</p>', unsafe_allow_html=True)
    elif total_realized_pnl < 0:
        st.metric("Total Realized P&L", f"-${abs(total_realized_pnl):,.2f}")
        st.markdown(f'<p style="color:#f6465d; font-size:0.875rem; margin-top:-15px;">↓ {abs(pnl_pct):.2f}%</p>', unsafe_allow_html=True)
    else:
        st.metric("Total Realized P&L", "$0.00")
        st.markdown('<p style="color:#848e9c; font-size:0.875rem; margin-top:-15px;">− 0.00%</p>', unsafe_allow_html=True)

with col3:
    active_positions_count = len([p for p in positions if p.quantity > 0])
    st.metric(label="Active Positions", value=active_positions_count)

st.markdown("---")

# --- Layout: Current Positions ---
st.subheader("Current Positions")
if positions:
    # Convert to DataFrame for display
    pos_data = []
    for p in positions:
        if p.quantity > 0:
            pos_data.append({
                "Symbol": p.symbol,
                "Quantity": p.quantity,
                "Average Entry Price": f"${p.average_price:,.2f}",
                "Total Value (at Entry)": f"${(p.quantity * p.average_price):,.2f}",
                "Last Updated": p.updated_at.strftime("%Y-%m-%d %H:%M:%S")
            })
    
    if pos_data:
        df_positions = pd.DataFrame(pos_data)
        st.dataframe(df_positions, use_container_width=True, hide_index=True)
    else:
        st.info("No active positions currently held.")
else:
    st.info("No positions recorded yet.")

st.markdown("---")

# --- Layout: Trade History ---
st.subheader("Trade Ledger")
if trades:
    trade_data = []
    for t in trades:
        trade_data.append({
            "Time": t.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "Symbol": t.symbol,
            "Action": t.action,
            "Quantity": t.quantity,
            "Price": f"${t.price:,.2f}",
            "Fee": f"${t.fee:,.4f}",
            "Realized P&L": f"${t.realized_pnl:,.2f}" if t.realized_pnl is not None else "-"
        })
    df_trades = pd.DataFrame(trade_data)
    
    # Optional styling for actions
    def style_action(val):
        color = '#0ecb81' if val == 'BUY' else '#f6465d'
        return f'color: {color}'
        
    styled_df = df_trades.style.map(style_action, subset=['Action'])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
else:
    st.info("No trades executed yet.")
