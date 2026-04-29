def apply_global_style():
    import streamlit as st

    st.markdown("""
    <style>

    /* Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Background */
    body {
        background: radial-gradient(circle at top left, #0f172a, #020617);
    }

    /* Page padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a, #020617);
        border-right: 1px solid rgba(255,255,255,0.05);
        padding-top: 10px;
    }

    /* Card */
    .card {
        position: relative;
        background: linear-gradient(145deg, #0f172a, #1e293b);
        border-radius: 16px;
        padding: 18px 20px;
        margin-bottom: 18px;
        border: 1px solid rgba(255,255,255,0.04);

        box-shadow: 
            0 20px 40px rgba(0,0,0,0.45),
            inset 0 1px 0 rgba(255,255,255,0.05);

        transition: all 0.2s ease-in-out;
    }

    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 24px 50px rgba(0,0,0,0.6);
    }

    /* Card glow */
    .card::before {
        content: "";
        position: absolute;
        inset: 0;
        border-radius: 16px;
        padding: 1px;
        background: linear-gradient(45deg, #3b82f6, #8b5cf6);
        -webkit-mask: 
            linear-gradient(#000 0 0) content-box, 
            linear-gradient(#000 0 0);
        -webkit-mask-composite: xor;
        mask-composite: exclude;
        opacity: 0.25;
        pointer-events: none;
    }

    /* Headings */
    h1 {
        font-size: 32px;
        font-weight: 700;
        color: #e5e7eb;
    }

    h2, h3 {
        font-weight: 600;
        color: #d1d5db;
    }

    /* Metrics */
    .metric-title {
        font-size: 12px;
        color: #9ca3af;
        margin-bottom: 4px;
    }

    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #ffffff;
    }

    /* Text */
    .text-body {
        font-size: 14px;
        line-height: 1.6;
        color: #cbd5f5;
    }

    /* Buttons */
    .stButton button {
        border-radius: 8px;
        padding: 6px 10px;
        font-weight: 500;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        color: white;
        border: none;
        transition: all 0.2s ease;
    }

    .stButton button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }

    /* Tables */
    [data-testid="stDataFrame"] {
        background: transparent;
    }

    [data-testid="stDataFrame"] table {
        border-radius: 12px;
        overflow: hidden;
    }

    [data-testid="stDataFrame"] th {
        background: #0f172a;
        color: #9ca3af;
        font-weight: 500;
    }

    [data-testid="stDataFrame"] td {
        background: #020617;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }

    /* Accent colors */
    :root {
        --accent-blue: #3b82f6;
        --accent-purple: #8b5cf6;
        --accent-green: #22c55e;
        --accent-red: #ef4444;
    }

    /* Hide Streamlit default header/footer */
    header {visibility: hidden;}
    footer {visibility: hidden;}

    </style>
    """, unsafe_allow_html=True)