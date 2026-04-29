def apply_global_style():
    import streamlit as st

    st.markdown("""
    <style>

    /* Page padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* Card */
    .card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 14px;
        padding: 16px 18px;
        margin-bottom: 16px;
        backdrop-filter: blur(10px);
    }

    /* Metric Title */
    .metric-title {
        font-size: 12px;
        color: #94a3b8;
        margin-bottom: 4px;
    }

    /* Metric Value */
    .metric-value {
        font-size: 22px;
        font-weight: 600;
        color: #e2e8f0;
    }

    /* Section spacing */
    .section-gap {
        margin-top: 10px;
    }

    /* Table fix */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Remove top padding gap */
    header {visibility: hidden;}
    footer {visibility: hidden;}

    </style>
    """, unsafe_allow_html=True)

