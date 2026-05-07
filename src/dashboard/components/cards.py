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
    
    color = "#05b169" if signal == "BUY" else "#cf202f" if signal == "SELL" else "#f4b000"
    
    st.markdown(f"""
    <div style="background-color: #16181c; padding: 32px; border-radius: 24px; border-left: 5px solid {color}; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);">
        <h3 style="margin-top: 0; color: {color}; font-weight: 400;">{signal}</h3>
        <p style="margin-bottom: 5px; color: #a8acb3;">Confidence: <strong style="color: #ffffff;">{confidence:.0%}</strong></p>
    </div>
    """, unsafe_allow_html=True)

def render_risk_card(risk_data):
    """Renders the risk manager's assessment."""
    st.markdown(f"""
    <div style="background-color: #16181c; padding: 32px; border-radius: 24px; margin-top: 15px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);">
        <h4 style="margin-top: 0; font-weight: 400;">Risk Management</h4>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
            <div>
                <p style="color: #a8acb3; margin-bottom: 2px;">Stop Loss</p>
                <p style="font-family: 'JetBrains Mono', monospace; font-size: 18px; margin-top: 0; color: #ffffff;">${risk_data.stop_loss:.2f}</p>
            </div>
            <div>
                <p style="color: #a8acb3; margin-bottom: 2px;">Take Profit Target</p>
                <p style="font-family: 'JetBrains Mono', monospace; font-size: 18px; margin-top: 0; color: #ffffff;">${risk_data.target:.2f}</p>
            </div>
            <div>
                <p style="color: #a8acb3; margin-bottom: 2px;">Position Size</p>
                <p style="font-family: 'JetBrains Mono', monospace; font-size: 18px; margin-top: 0; color: #ffffff;">${risk_data.position_size:.2f}</p>
            </div>
            <div>
                <p style="color: #a8acb3; margin-bottom: 2px;">Valid Until</p>
                <p style="font-size: 16px; margin-top: 0; color: #ffffff;">{risk_data.valid_until}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
