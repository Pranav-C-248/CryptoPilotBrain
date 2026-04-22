import streamlit as st
import os
import sys
import json

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from src.dashboard.shared.database import get_db, AuditLog

st.set_page_config(page_title="CryptoPilot - Explainable AI", page_icon="", layout="wide")

from src.dashboard.shared.theme import inject_theme_css
inject_theme_css()

st.title("Explainable AI (XAI) 🧠")
st.markdown("Detailed breakdown of the AI's internal reasoning process.")

@st.cache_data(ttl=10)
def fetch_latest_logs():
    db_gen = get_db()
    db = next(db_gen)
    return db.query(AuditLog).order_by(AuditLog.timestamp.desc()).limit(10).all()

logs = fetch_latest_logs()

if logs:
    selected_log_id = st.selectbox(
        "Select Decision to Explain",
        options=[log.id for log in logs],
        format_func=lambda x: next(f"{l.timestamp.strftime('%Y-%m-%d %H:%M:%S')} | {l.symbol} | {l.action}" for l in logs if l.id == x)
    )
    
    selected_log = next(l for l in logs if l.id == selected_log_id)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Decision Summary")
        st.write(f"**Asset:** {selected_log.symbol}")
        st.write(f"**Action:** {selected_log.action}")
        st.write(f"**Confidence:** {selected_log.confidence:.2%}")
        st.write(f"**Price at execution:** ${selected_log.price}")
        st.info(selected_log.explanation)
        
    with col2:
        st.subheader("Internal Monologue")
        try:
            factors = selected_log.factors
            if isinstance(factors, str):
                factors = json.loads(factors)
                
            monologue = factors.get("internal_monologue", "No monologue available.")
            st.code(monologue, language="text")
        except Exception as e:
            st.error("Failed to parse factors for this log.")
            
else:
    st.info("No AI decisions recorded yet.")
