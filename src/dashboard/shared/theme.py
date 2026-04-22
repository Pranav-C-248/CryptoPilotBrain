import streamlit as st

def inject_theme_css():
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Anton&family=Fira+Code:wght@400;500;600&family=Rubik:wght@400;500;600;700&display=swap');

    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Rubik', sans-serif;
    }
    
    /* Headings */
    h1 {
        font-family: 'Anton', sans-serif !important;
        letter-spacing: 1px;
    }
    h2, h3, h4, h5, h6 {
        font-family: 'Rubik', sans-serif !important;
        font-weight: 500;
    }

    /* Buttons */
    .stButton > button {
        background-color: #79628c !important;
        color: #ffffff !important;
        border: 1px solid #584674 !important;
        border-radius: 13px !important;
        text-transform: uppercase !important;
        font-weight: 600 !important;
        letter-spacing: 0.2px !important;
        box-shadow: rgba(0, 0, 0, 0.4) 0px 2px 4px 0px inset !important;
        transition: all 0.2s ease-in-out !important;
        padding: 0.5rem 1rem !important;
    }
    .stButton > button:hover {
        box-shadow: rgba(0, 0, 0, 0.18) 0px 0.5rem 1.5rem !important;
        background-color: #6a5fc1 !important;
        color: #ffffff !important;
        border-color: #6a5fc1 !important;
    }

    /* Metric Cards / Standard Containers */
    div[data-testid="stMetric"], div[data-testid="stExpander"] {
        background-color: #150f23 !important;
        border: 1px solid #362d59 !important;
        border-radius: 8px !important;
        box-shadow: rgba(0, 0, 0, 0.1) 0px 10px 15px -3px !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        border-right: 1px solid #362d59 !important;
    }
    
    /* Code Blocks */
    code {
        font-family: 'Fira Code', monospace !important;
        color: #dcdcaa !important;
        background-color: rgba(255, 255, 255, 0.05) !important;
        border-radius: 4px;
        padding: 2px 4px;
    }
    
    /* Links */
    a {
        color: #6a5fc1 !important;
        text-decoration: underline !important;
        transition: color 0.2s ease !important;
    }
    a:hover {
        color: #ffffff !important;
    }

    /* Success text / Badges */
    .st-emotion-cache-1kyxreq, .st-emotion-cache-16idsys {
        color: #c2ef4e !important;
    }

    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
