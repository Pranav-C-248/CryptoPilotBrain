import streamlit as st

def inject_theme_css():
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Typography & Colors */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0a0b0d !important;
        color: #ffffff !important;
    }
    
    .stApp {
        background-color: #0a0b0d !important;
    }
    
    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 400 !important;
        color: #ffffff !important;
    }
    h1 {
        letter-spacing: -1.5px !important;
    }
    h2, h3 {
        letter-spacing: -0.5px !important;
    }

    /* Tabular / Number Data */
    div[data-testid="stMetricValue"], .stDataFrame {
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 500 !important;
    }

    /* Buttons (Coinbase Pill CTA) */
    .stButton > button {
        background-color: #0052ff !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 100px !important; /* Pill shape */
        font-weight: 600 !important;
        padding: 0.5rem 1.5rem !important;
        box-shadow: none !important;
        transition: background-color 0.2s ease !important;
    }
    .stButton > button:hover {
        background-color: #003ecc !important;
        color: #ffffff !important;
    }

    /* Metric Cards / Standard Containers */
    div[data-testid="stMetric"], div[data-testid="stExpander"] {
        background-color: #16181c !important;
        border: none !important;
        border-radius: 24px !important; /* Cards are 24px */
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04) !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0a0b0d !important;
        border-right: 1px solid #16181c !important;
    }
    
    /* Code Blocks */
    code {
        font-family: 'JetBrains Mono', monospace !important;
        color: #a8acb3 !important;
        background-color: #16181c !important;
        border-radius: 4px;
        padding: 2px 4px;
    }
    
    /* Links */
    a {
        color: #0052ff !important;
        text-decoration: none !important;
        transition: color 0.2s ease !important;
    }
    a:hover {
        text-decoration: underline !important;
    }

    /* Success text / Badges */
    .st-emotion-cache-1kyxreq, .st-emotion-cache-16idsys {
        color: #05b169 !important;
    }

    /* Dividers */
    hr {
        border-bottom-color: #16181c !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
