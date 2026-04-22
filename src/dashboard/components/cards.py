import streamlit as st

def render_metric_card(title: str, value: str, delta: str = None, key=None):
    """Renders a styled metric card."""
    # Since Streamlit columns handle layout, we just use the built-in st.metric 
    # but we could wrap it in custom CSS if needed.
    st.metric(label=title, value=value, delta=delta)

def render_signal_card(signal_data: dict):
    """Renders a card highlighting the trading signal."""
    signal = signal_data.get('signal', 'HOLD')
    confidence = signal_data.get('confidence', 0.0)
    
    color = "#0ecb81" if signal == "BUY" else "#f6465d" if signal == "SELL" else "#fcd535"
    
    st.markdown(f"""
    <div style="background-color: #161b22; padding: 20px; border-radius: 10px; border-left: 5px solid {color};">
        <h3 style="margin-top: 0; color: {color};">{signal}</h3>
        <p style="margin-bottom: 5px; color: #848e9c;">Confidence: <strong>{confidence:.0%}</strong></p>
    </div>
    """, unsafe_allow_html=True)

def render_risk_card(risk_data):
    """Renders the risk manager's assessment."""
    st.markdown(f"""
    <div style="background-color: #161b22; padding: 20px; border-radius: 10px; margin-top: 15px;">
        <h4 style="margin-top: 0;">Risk Management</h4>
        <div style="display: grid; grid-template-columns: 1fr 1fr;">
            <div>
                <p style="color: #848e9c; margin-bottom: 2px;">Stop Loss</p>
                <p style="font-size: 18px; margin-top: 0;">${risk_data.stop_loss:.2f}</p>
            </div>
            <div>
                <p style="color: #848e9c; margin-bottom: 2px;">Take Profit Target</p>
                <p style="font-size: 18px; margin-top: 0;">${risk_data.target:.2f}</p>
            </div>
            <div>
                <p style="color: #848e9c; margin-bottom: 2px;">Position Size</p>
                <p style="font-size: 18px; margin-top: 0;">${risk_data.position_size:.2f}</p>
            </div>
            <div>
                <p style="color: #848e9c; margin-bottom: 2px;">Valid Until</p>
                <p style="font-size: 16px; margin-top: 0;">{risk_data.valid_until}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
