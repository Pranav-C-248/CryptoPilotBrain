import streamlit as st

def render_metric_card(title: str, value: str, delta: str = None, key=None):
    """Renders a styled metric card."""
    # Since Streamlit columns handle layout, we just use the built-in st.metric 
    # but we could wrap it in custom CSS if needed.
    st.metric(label=title, value=value, delta=delta)

def render_signal_card(signal_data):
    import streamlit as st

    signal = signal_data.get("signal", "N/A")
    confidence = signal_data.get("confidence", 0)

    # Color logic
    if signal == "BUY":
        color = "#22c55e"
    elif signal == "SELL":
        color = "#ef4444"
    else:
        color = "#f59e0b"

    st.markdown(f"""
    <div class="card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            
            <div>
                <div class="metric-title">AI Signal</div>
                <div style="font-size:24px; font-weight:600; color:{color};">
                    {signal}
                </div>
            </div>

            <div style="text-align:right;">
                <div class="metric-title">Confidence</div>
                <div style="font-size:20px; font-weight:500;">
                    {confidence}%
                </div>
            </div>

        </div>
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
