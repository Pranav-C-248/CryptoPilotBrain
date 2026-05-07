import streamlit as st
import pandas as pd
from datetime import datetime
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

st.set_page_config(page_title="CryptoPilot - News", page_icon="", layout="wide")

from src.dashboard.shared.theme import inject_theme_css
from src.agents.sentiment_agent import SentimentAgent

inject_theme_css()

st.title("Crypto News & Sentiment")
st.markdown("Latest top 5 news from CoinDesk, analyzed by AI.")

@st.cache_data(ttl=600)
def fetch_top_5_news():
    agent = SentimentAgent()
    articles = agent.fetch_and_analyze_news(limit=5)
    return pd.DataFrame(articles)

with st.spinner("Fetching and analyzing latest news..."):
    df_news = fetch_top_5_news()
    
    if not df_news.empty:
        # Display sentiment overview
        st.subheader("Market Sentiment Overview")
        avg_sentiment = df_news['sentiment_score'].mean()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Sentiment", f"{avg_sentiment:.2f}")
            if avg_sentiment > 0.6:
                st.markdown('<p style="color:#05b169; font-size:0.875rem; margin-top:-15px;">↑ Bullish</p>', unsafe_allow_html=True)
            elif avg_sentiment < 0.4:
                st.markdown('<p style="color:#cf202f; font-size:0.875rem; margin-top:-15px;">↓ Bearish</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p style="color:#a8acb3; font-size:0.875rem; margin-top:-15px;">− Neutral</p>', unsafe_allow_html=True)
            
        st.markdown("---")
        
        # Display news cards
        for _, row in df_news.iterrows():
            if row['sentiment_score'] > 0.6:
                sentiment_color = "#05b169"
            elif row['sentiment_score'] < 0.4:
                sentiment_color = "#cf202f"
            else:
                sentiment_color = "#f4b000"
            st.markdown(f"""
            <div style="background-color: #16181c; padding: 24px; border-radius: 24px; margin-bottom: 15px; border-left: 4px solid {sentiment_color}; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);">
                <h4 style="margin-top: 0;"><a href="{row['link']}" target="_blank" style="color: #ffffff; text-decoration: none;">{row['title']}</a></h4>
                <p style="color: #a8acb3; font-size: 14px; margin-bottom: 5px;">Published: {row['published']}</p>
                <div style="background-color: #0a0b0d; padding: 16px; border-radius: 12px; margin-top: 16px;">
                    <strong style="color: {sentiment_color}; font-family: 'JetBrains Mono', monospace;">AI Sentiment Score: {row['sentiment_score']:.2f}</strong>
                    <p style="color: #eef0f3; margin-top: 8px; font-size: 14px; line-height: 1.5;">{row['reasoning']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No news articles available at the moment.")
