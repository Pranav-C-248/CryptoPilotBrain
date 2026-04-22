import streamlit as st
import feedparser
import pandas as pd
from datetime import datetime
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

st.set_page_config(page_title="CryptoPilot - News", page_icon="📰", layout="wide")

st.title("Crypto News & Sentiment 📰")
st.markdown("Latest news from CoinDesk.")

@st.cache_data(ttl=600)
def fetch_coindesk_news():
    url = "https://www.coindesk.com/arc/outboundfeeds/rss/"
    feed = feedparser.parse(url)
    
    articles = []
    for entry in feed.entries[:20]:
        articles.append({
            "title": entry.title,
            "link": entry.link,
            "published": entry.published,
            # Assign dummy sentiment for now
            "sentiment_score": 0.5
        })
    return pd.DataFrame(articles)

with st.spinner("Fetching latest news..."):
    df_news = fetch_coindesk_news()
    
    if not df_news.empty:
        # Display sentiment overview
        st.subheader("Market Sentiment Overview")
        avg_sentiment = df_news['sentiment_score'].mean()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Sentiment", f"{avg_sentiment:.2f}", "Neutral")
            
        st.markdown("---")
        
        # Display news cards
        for _, row in df_news.iterrows():
            st.markdown(f"""
            <div style="background-color: #161b22; padding: 15px; border-radius: 8px; margin-bottom: 10px;">
                <h4><a href="{row['link']}" target="_blank" style="color: #0ecb81; text-decoration: none;">{row['title']}</a></h4>
                <p style="color: #848e9c; font-size: 14px;">Published: {row['published']} | Est. Sentiment: {row['sentiment_score']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No news articles available at the moment.")
