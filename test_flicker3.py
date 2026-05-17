import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time

st.set_page_config(layout="wide")

if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame({
        'open_time': pd.date_range(start='1/1/2026', periods=100, freq='h'),
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 105,
        'low': np.random.randn(100).cumsum() + 95,
        'close': np.random.randn(100).cumsum() + 100
    })

if 'fig' not in st.session_state:
    df = st.session_state.df
    fig = go.Figure(data=[go.Candlestick(x=df['open_time'], open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
    fig.update_layout(uirevision='constant')
    st.session_state.fig = fig

@st.fragment(run_every="3s")
def render():
    st.session_state.df.iloc[-1, 4] += np.random.randn() # change close price
    df = st.session_state.df
    
    # Mutate existing figure
    fig = st.session_state.fig
    fig.data[0].x = df['open_time']
    fig.data[0].open = df['open']
    fig.data[0].high = df['high']
    fig.data[0].low = df['low']
    fig.data[0].close = df['close']
    
    st.plotly_chart(fig, use_container_width=True, key="my_chart")

render()
