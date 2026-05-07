import streamlit as st
import os
import sys
import json
import html

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from src.dashboard.shared.database import get_db, AuditLog

st.set_page_config(page_title="CryptoPilot - Explainable AI", page_icon="", layout="wide")

from src.dashboard.shared.theme import inject_theme_css
inject_theme_css()

st.title("Explainable AI (XAI)")
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
                
            import re
            
            def format_monologue(text):
                text = html.escape(text)
                # Highlight keywords
                text = re.sub(r'\b(BUY|LONG|BULLISH)\b', r'<span style="color:#0ecb81; font-weight:bold;">\1</span>', text)
                text = re.sub(r'\b(SELL|SHORT|BEARISH)\b', r'<span style="color:#f6465d; font-weight:bold;">\1</span>', text)
                text = re.sub(r'\b(HOLD|NEUTRAL)\b', r'<span style="color:#eab308; font-weight:bold;">\1</span>', text)
                text = re.sub(r'\b(RSI|MACD|EMA|SMA|ATR|Volume)\b', r'<span style="color:#3b82f6; font-weight:bold;">\1</span>', text)
                text = re.sub(r'\b(Support|Resistance)\b', r'<span style="color:#a855f7; font-weight:bold;">\1</span>', text)
                return text

            monologue = factors.get("internal_monologue", "No monologue available.")
            monologue_html = format_monologue(monologue)
            
            st.markdown(
                f'''<div style="white-space: pre-wrap; font-family: 'Inter', sans-serif; line-height: 1.6; background: linear-gradient(145deg, rgba(20,20,20,0.6), rgba(10,10,10,0.8)); padding: 20px; border-radius: 10px; border-left: 4px solid #3b82f6; box-shadow: 0 4px 6px rgba(0,0,0,0.3); font-size: 0.95rem; color: #e2e8f0;">{monologue_html}</div>''',
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error("Failed to parse factors for this log.")
            
else:
    st.info("No AI decisions recorded yet.")
