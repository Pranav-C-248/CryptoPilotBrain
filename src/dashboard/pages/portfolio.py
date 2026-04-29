import streamlit as st
import os
import sys
import pandas as pd
from datetime import datetime, timezone

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, project_root)

from styles import apply_global_style
from src.dashboard.shared.database import get_db, Portfolio, Position, TradeLedger

# Page Configuration
st.set_page_config(
    page_title="CryptoPilot - Portfolio",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply Global Style
apply_global_style()

# Page Title
st.title("Portfolio")
st.markdown("""
<div class="card" style="font-size:13px; color:#64748b;">
AI Trading Dashboard
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Database Fetching & Initialization
db_gen = get_db()
db = next(db_gen)

portfolio = db.query(Portfolio).first()
if not portfolio:
    portfolio = Portfolio(balance=10000.0)
    db.add(portfolio)
    db.commit()
    db.refresh(portfolio)

positions = db.query(Position).all()
trades = db.query(TradeLedger).order_by(TradeLedger.timestamp.desc()).all()

db.close()

# Calculations
total_realized_pnl = sum(
    trade.realized_pnl for trade in trades if trade.realized_pnl is not None
)

available_balance = portfolio.balance
active_positions_count = len([p for p in positions if p.quantity > 0])

# Top Metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="card">
        <div class="metric-title">Available Balance</div>
        <div class="metric-value">${available_balance:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    pnl_color = "#22c55e" if total_realized_pnl >= 0 else "#ef4444"

    st.markdown(f"""
    <div class="card">
        <div class="metric-title">Total Realized P&L</div>
        <div class="metric-value" style="color:{pnl_color};">
            ${total_realized_pnl:,.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="card">
        <div class="metric-title">Active Positions</div>
        <div class="metric-value">{active_positions_count}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Current Positions
st.subheader("Current Positions")

if positions:
    pos_data = []

    for p in positions:
        if p.quantity > 0:
            pos_data.append({
                "Symbol": p.symbol,
                "Quantity": p.quantity,
                "Avg Price": f"${p.average_price:,.2f}",
                "Value": f"${(p.quantity * p.average_price):,.2f}",
                "Last Updated": p.updated_at.strftime("%Y-%m-%d %H:%M:%S")
            })

    if pos_data:
        df_positions = pd.DataFrame(pos_data)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.dataframe(df_positions, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card">No active positions currently held.</div>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="card">No positions recorded yet.</div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Trade History
st.subheader("Trade History")

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
            "P&L": f"${t.realized_pnl:,.2f}" if t.realized_pnl is not None else "-"
        })

    df_trades = pd.DataFrame(trade_data)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.dataframe(df_trades, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="card">No trades executed yet.</div>
    """, unsafe_allow_html=True)