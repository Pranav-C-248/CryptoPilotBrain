import streamlit as st
import os
import sys
import json

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from styles import apply_global_style
from src.dashboard.shared.database import get_db, AuditLog

# Page Configuration
st.set_page_config(
    page_title="CryptoPilot - Explainable AI",
    layout="wide"
)

# Apply Global Style
apply_global_style()

# Page Title
st.title("Explainable AI")
st.markdown("""
<div class="card" style="font-size:13px; color:#64748b;">
AI Trading Dashboard
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div class="card">
Detailed breakdown of the AI's internal reasoning process
</div>
""", unsafe_allow_html=True)


# Fetch Logs
@st.cache_data(ttl=10)
def fetch_latest_logs():
    db_gen = get_db()
    db = next(db_gen)
    return db.query(AuditLog).order_by(AuditLog.timestamp.desc()).limit(10).all()

logs = fetch_latest_logs()


# Main UI
if logs:
    selected_log_id = st.selectbox(
        "Select Decision",
        options=[log.id for log in logs],
        format_func=lambda x: next(
            f"{l.timestamp.strftime('%Y-%m-%d %H:%M:%S')} | {l.symbol} | {l.action}"
            for l in logs if l.id == x
        )
    )

    selected_log = next(l for l in logs if l.id == selected_log_id)

    col1, col2 = st.columns([1, 2])

    # Decision Summary
    with col1:
        st.subheader("Decision Summary")

        confidence = f"{selected_log.confidence * 100:.2f}%"
        color = "#22c55e" if selected_log.action == "BUY" else "#ef4444" if selected_log.action == "SELL" else "#f59e0b"

        st.markdown(f"""
        <div class="card">
            <p><b>Asset:</b> {selected_log.symbol}</p>
            <p><b>Action:</b> <span style="color:{color};">{selected_log.action}</span></p>
            <p><b>Confidence:</b> {confidence}</p>
            <p><b>Price:</b> ${selected_log.price}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="card">
            <div class="metric-title">Explanation</div>
            <div style="margin-top:8px; font-size:14px; color:#cbd5f5; line-height:1.5;">
                {selected_log.explanation}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Internal Monologue
    with col2:
        st.subheader("Internal Analysis")

        try:
            factors = selected_log.factors
            if isinstance(factors, str):
                factors = json.loads(factors)

            monologue = factors.get("internal_monologue", "No monologue available.")

            st.markdown(f"""
            <div class="card">
                <div class="metric-title">Internal Monologue</div>
                <pre style="margin-top:10px; color:#e2e8f0; font-size:13px;">
{monologue}
                </pre>
            </div>
            """, unsafe_allow_html=True)

        except Exception:
            st.markdown("""
            <div class="card">Failed to parse factors for this log</div>
            """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="card">No AI decisions recorded yet</div>
    """, unsafe_allow_html=True)