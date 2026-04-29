import streamlit as st
import pandas as pd
from datetime import datetime
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from styles import apply_global_style
from src.agents.sentiment_agent import SentimentAgent

# Page Configuration
st.set_page_config(
    page_title="CryptoPilot - News",
    layout="wide"
)

# Apply Global Style
apply_global_style()

# Page Title
st.title("Crypto News & Sentiment")
st.markdown("""
<div class="card" style="font-size:13px; color:#64748b;">
AI Trading Dashboard
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div class="card">
Latest top 5 news analyzed with AI sentiment
</div>
""", unsafe_allow_html=True)


# Fetch News
@st.cache_data(ttl=600)
def fetch_top_5_news():
    agent = SentimentAgent()
    articles = agent.fetch_and_analyze_news(limit=5)
    return pd.DataFrame(articles)


with st.spinner("Fetching and analyzing latest news..."):
    df_news = fetch_top_5_news()

    if not df_news.empty:

        # Market Sentiment Overview
        st.subheader("Market Sentiment Overview")

        avg_sentiment = df_news['sentiment_score'].mean()

        sentiment_label = "Neutral"
        sentiment_color = "#f59e0b"

        if avg_sentiment > 0.6:
            sentiment_label = "Bullish"
            sentiment_color = "#22c55e"
        elif avg_sentiment < 0.4:
            sentiment_label = "Bearish"
            sentiment_color = "#ef4444"

        st.markdown(f"""
        <div class="card">
            <div class="metric-title">Overall Sentiment</div>
            <div class="metric-value" style="color:{sentiment_color};">
                {avg_sentiment:.2f}
            </div>
            <div style="font-size:13px; color:#94a3b8;">
                {sentiment_label}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # News Articles
        st.subheader("Top News")

        for _, row in df_news.iterrows():

            sentiment_color = "#22c55e" if row['sentiment_score'] >= 0.5 else "#ef4444"

            st.markdown(f"""
            <div class="card">

                <div style="font-size:16px; font-weight:600;">
                    <a href="{row['link']}" target="_blank" style="text-decoration:none; color:#e2e8f0;">
                        {row['title']}
                    </a>
                </div>

                <div style="font-size:12px; color:#94a3b8; margin-top:4px;">
                    Published: {row['published']}
                </div>

                <div style="margin-top:10px; padding:10px; border-radius:6px; background:rgba(255,255,255,0.03);">

                    <div style="font-size:13px; font-weight:500; color:{sentiment_color};">
                        Sentiment Score: {row['sentiment_score']:.2f}
                    </div>

                    <div style="margin-top:6px; font-size:14px; color:#cbd5f5; line-height:1.5;">
                        {row['reasoning']}
                    </div>

                </div>

            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="card">No news articles available</div>
        """, unsafe_allow_html=True)