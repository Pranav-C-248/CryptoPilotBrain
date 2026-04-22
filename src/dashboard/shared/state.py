import streamlit as st

def init_session_state():
    if 'current_symbol' not in st.session_state:
        st.session_state['current_symbol'] = 'BTC/USDT'
    
    if 'last_signal' not in st.session_state:
        st.session_state['last_signal'] = None

    if 'last_risk_assessment' not in st.session_state:
        st.session_state['last_risk_assessment'] = None
    
    if 'last_market_data' not in st.session_state:
        st.session_state['last_market_data'] = None

    if 'last_analysis_time' not in st.session_state:
        st.session_state['last_analysis_time'] = None
