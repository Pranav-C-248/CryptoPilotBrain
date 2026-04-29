import streamlit as st
import pandas as pd
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from styles import apply_global_style
from src.dashboard.shared.database import get_db, AuditLog

# Page Configuration
st.set_page_config(
    page_title="CryptoPilot - Audit Trail",
    layout="wide"
)

# Apply Global Style
apply_global_style()

# Page Title
st.title("Audit Trail")
st.markdown("""
<div class="card" style="font-size:13px; color:#64748b;">
AI Trading Dashboard
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div class="card">
History of all AI trading decisions
</div>
""", unsafe_allow_html=True)


# Fetch Audit Logs
@st.cache_data(ttl=10)
def fetch_audit_logs():
    db_gen = get_db()
    db = next(db_gen)
    logs = db.query(AuditLog).order_by(AuditLog.timestamp.desc()).all()
    
    data = []
    for log in logs:
        data.append({
            "Timestamp": log.timestamp,
            "Symbol": log.symbol,
            "Action": log.action,
            "Confidence": round(log.confidence, 2),
            "Price": log.price,
            "Explanation": log.explanation
        })
    return pd.DataFrame(data)


df_logs = fetch_audit_logs()


# Audit Table
st.subheader("Decision History")

if not df_logs.empty:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.dataframe(
        df_logs,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Timestamp": st.column_config.DatetimeColumn(
                "Date & Time",
                format="D MMM YYYY, h:mm a"
            ),
            "Action": st.column_config.TextColumn("Signal"),
            "Confidence": st.column_config.ProgressColumn(
                "Confidence",
                min_value=0,
                max_value=1,
                format="%.2f"
            )
        }
    )

    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="card">No audit logs found</div>
    """, unsafe_allow_html=True)