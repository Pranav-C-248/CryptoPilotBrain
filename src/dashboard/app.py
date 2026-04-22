import streamlit as st
import os
import sys
from datetime import datetime, timezone
import traceback

# Add project root to sys.path so imports work correctly
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from src.dashboard.shared.state import init_session_state
from src.dashboard.shared.database import get_db, AuditLog
from src.dashboard.components.cards import render_metric_card, render_signal_card, render_risk_card
from src.dashboard.components.charts import render_candlestick_chart

from src.tools.binance_client import BinancePublicClient
from src.tools.indicators import add_indicators
from src.agents.analyst import AnalystAgent, MarketDataProcessor
from src.agents.risk_manager import RiskManagerAgent
from src.core.knowledge_base import TradingKnowledgeBase

# Page Configuration
st.set_page_config(
    page_title="CryptoPilot - Markets",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State
init_session_state()

def run_analysis():
    st.session_state['last_analysis_time'] = datetime.now(timezone.utc)
    
    with st.spinner("Fetching Market Data..."):
        client = BinancePublicClient()
        df = client.get_historical_klines(st.session_state['current_symbol'], "1h", limit=100)
        df = add_indicators(df)
        st.session_state['last_market_data'] = df
        
    with st.spinner("Analyzing Market Context (AnalystAgent)..."):
        # The agent needs context packet
        packet = MarketDataProcessor.get_context_packet(df, len(df)-1, window=55)
        if not packet:
            st.error("Not enough data to analyze.")
            return
            
        packet['asset_name'] = st.session_state['current_symbol']
        
        try:
            kb = TradingKnowledgeBase()
            analyst = AnalystAgent(knowledge_base=kb)
            # Default sentiment score 0.5 (neutral)
            sentiment_score = 0.5 
            signal = analyst.analyze(market_data=packet, sentiment_score=sentiment_score)
            st.session_state['last_signal'] = signal
        except Exception as e:
            st.error(f"AnalystAgent Error: {str(e)}")
            st.error(traceback.format_exc())
            return
            
    with st.spinner("Evaluating Risk (RiskManagerAgent)..."):
        try:
            rm = RiskManagerAgent(total_equity=10000.0) # Dummy 10k portfolio
            risk_assessment = rm.evaluate(
                analyst_signal=signal,
                market_data=packet,
                sentiment_score=sentiment_score,
                current_portfolio=[]
            )
            st.session_state['last_risk_assessment'] = risk_assessment
            
            # Log to Database
            db_gen = get_db()
            db = next(db_gen)
            audit_log = AuditLog(
                symbol=st.session_state['current_symbol'],
                action=risk_assessment.signal,
                confidence=signal.confidence,
                price=packet['snapshot']['close'],
                factors={"internal_monologue": signal.internal_monologue},
                explanation=signal.reasoning,
            )
            db.add(audit_log)
            db.commit()
            db.refresh(audit_log)
        except Exception as e:
            st.error(f"RiskManagerAgent Error: {str(e)}")
            st.error(traceback.format_exc())
            return

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
        st.session_state['last_signal'] = None
        st.session_state['last_risk_assessment'] = None
        st.session_state['last_market_data'] = None
        
    st.button("Run AI Analysis", on_click=run_analysis, use_container_width=True)

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Live Chart: {st.session_state['current_symbol']}")
    if st.session_state['last_market_data'] is not None:
        fig = render_candlestick_chart(st.session_state['last_market_data'], title=st.session_state['current_symbol'])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Click 'Run AI Analysis' to fetch data and generate signals.")

with col2:
    st.subheader("AI Trading Signal")
    if st.session_state['last_risk_assessment'] is not None:
        # We display the final signal from RiskManager
        signal_data = {
            'signal': st.session_state['last_risk_assessment'].signal,
            'confidence': st.session_state['last_signal'].confidence
        }
        render_signal_card(signal_data)
        render_risk_card(st.session_state['last_risk_assessment'])
        
        st.markdown("---")
        st.markdown("### Analyst Reasoning")
        st.info(st.session_state['last_signal'].reasoning)
    else:
        st.write("No analysis generated yet.")
