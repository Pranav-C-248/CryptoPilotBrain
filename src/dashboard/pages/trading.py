import streamlit as st
import os
import sys
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, project_root)

from styles import apply_global_style
from src.dashboard.shared.database import get_db, QueuedTrade

# Page Configuration
st.set_page_config(
    page_title="CryptoPilot - Trading",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply Global Style
apply_global_style()

# Page Title
st.title("Trading Queue")
st.markdown("""
<div class="card" style="font-size:13px; color:#64748b;">
AI Trading Dashboard
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

db_gen = get_db()
db = next(db_gen)

# Action Handlers
def update_trade(trade_id, action, entry, exit_cond, tp, sl):
    trade = db.query(QueuedTrade).filter(QueuedTrade.id == trade_id).first()
    if not trade:
        return
        
    if action == "approve":
        trade.entry_condition = entry
        trade.exit_condition = exit_cond
        trade.target = tp
        trade.stop_loss = sl
        trade.status = "UNENTERED"
        st.success(f"Trade {trade_id} approved and added to active queue!")
    elif action == "reject":
        trade.status = "REJECTED"
        st.warning(f"Trade {trade_id} rejected.")
        
    db.commit()

# Fetch Trades
pending_trades = db.query(QueuedTrade).filter(QueuedTrade.status == "PENDING_APPROVAL").all()
active_trades = db.query(QueuedTrade).filter(
    QueuedTrade.status.in_(["UNENTERED", "ENTERED"])
).order_by(QueuedTrade.id.desc()).all()

# Pending Approvals
st.subheader("Pending Approvals")

if pending_trades:
    st.markdown(f"""
    <div class="card">
        {len(pending_trades)} trades pending approval
    </div>
    """, unsafe_allow_html=True)

    for trade in pending_trades:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-title">{trade.signal} {trade.asset_name}</div>
        <div style="font-size:13px; color:#94a3b8;">
            Position Size: ${trade.position_size:.2f} | Timeframe: {trade.timeframe}
        </div>
        """, unsafe_allow_html=True)

        with st.form(key=f"trade_form_{trade.id}"):

            col1, col2 = st.columns(2)

            with col1:
                new_entry = st.text_input(
                    "Entry Condition",
                    value=trade.entry_condition,
                    key=f"entry_{trade.id}"
                )
                new_tp = st.number_input(
                    "Target Price",
                    value=float(trade.target),
                    format="%.4f",
                    key=f"tp_{trade.id}"
                )

            with col2:
                new_exit = st.text_input(
                    "Exit Condition",
                    value=trade.exit_condition,
                    key=f"exit_{trade.id}"
                )
                new_sl = st.number_input(
                    "Stop Loss",
                    value=float(trade.stop_loss),
                    format="%.4f",
                    key=f"sl_{trade.id}"
                )

            btn_col1, btn_col2 = st.columns(2)

            with btn_col1:
                approve = st.form_submit_button("Approve", use_container_width=True)

            with btn_col2:
                reject = st.form_submit_button("Reject", use_container_width=True)

            if approve:
                update_trade(trade.id, "approve", new_entry, new_exit, new_tp, new_sl)
                st.rerun()

            if reject:
                update_trade(trade.id, "reject", new_entry, new_exit, new_tp, new_sl)
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="card">No trades pending approval</div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Active Queue
st.subheader("Active Engine Queue")

if active_trades:
    queue_data = []

    for t in active_trades:
        queue_data.append({
            "ID": t.id,
            "Asset": t.asset_name,
            "Signal": t.signal,
            "Status": t.status,
            "Entry Condition": t.entry_condition,
            "Exit Condition": t.exit_condition,
            "Target": f"${t.target:,.4f}" if t.target else "-",
            "Stop Loss": f"${t.stop_loss:,.4f}" if t.stop_loss else "-",
            "Entry Price": f"${t.entry_price:,.4f}" if t.entry_price else "Waiting"
        })

    df_queue = pd.DataFrame(queue_data)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.dataframe(df_queue, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="card">The engine queue is currently empty</div>
    """, unsafe_allow_html=True)

db.close()