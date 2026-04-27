import streamlit as st
import pandas as pd
from datetime import datetime
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

st.set_page_config(page_title="CryptoPilot - News", page_icon="📰", layout="wide")

from src.dashboard.shared.theme import inject_theme_css
from src.agents.sentiment_agent import SentimentAgent

inject_theme_css()

st.title("Crypto News & Sentiment 📰")
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
            sentiment_label = "Neutral"
            if avg_sentiment > 0.6:
                sentiment_label = "Bullish"
            elif avg_sentiment < 0.4:
                sentiment_label = "Bearish"
            st.metric("Overall Sentiment", f"{avg_sentiment:.2f}", sentiment_label)
            
        st.markdown("---")
        
        # Display news cards
        for _, row in df_news.iterrows():
            sentiment_color = "#0ecb81" if row['sentiment_score'] >= 0.5 else "#f6465d"
            st.markdown(f"""
            <div style="background-color: #161b22; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid {sentiment_color};">
                <h4><a href="{row['link']}" target="_blank" style="color: #ffffff; text-decoration: none;">{row['title']}</a></h4>
                <p style="color: #848e9c; font-size: 14px; margin-bottom: 5px;">Published: {row['published']}</p>
                <div style="background-color: #0b0e11; padding: 10px; border-radius: 4px; margin-top: 10px;">
                    <strong style="color: {sentiment_color};">AI Sentiment Score: {row['sentiment_score']:.2f}</strong>
                    <p style="color: #e2e8f0; margin-top: 5px; font-size: 14px; line-height: 1.4;">{row['reasoning']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No news articles available at the moment.")
