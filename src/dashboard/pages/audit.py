import streamlit as st
import pandas as pd
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from src.dashboard.shared.database import get_db, AuditLog

st.set_page_config(page_title="CryptoPilot - Audit Trail", page_icon="", layout="wide")

from src.dashboard.shared.theme import inject_theme_css
inject_theme_css()

st.title("Audit Trail")
st.markdown("Raw logs of all AI signals, analysis, and actions.")

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

if not df_logs.empty:
    st.dataframe(
        df_logs,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Timestamp": st.column_config.DatetimeColumn("Date & Time", format="D MMM YYYY, h:mm a"),
            "Action": st.column_config.TextColumn("Signal"),
            "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.2f")
        }
    )
else:
    st.info("No audit logs found. Run an AI analysis to generate logs.")
