import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def render_candlestick_chart(df: pd.DataFrame, title: str = "Price Chart"):
    """
    Renders an interactive Plotly candlestick chart with Bollinger Bands and RSI.
    """
    if df.empty:
        return go.Figure()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=(title, 'RSI (14)')
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df['close_time'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#0ecb81',
            decreasing_line_color='#f6465d'
        ),
        row=1, col=1
    )

    # EMA 20
    if 'ema20' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['close_time'], y=df['ema20'], line=dict(color='yellow', width=1), name='EMA 20'),
            row=1, col=1
        )

    # Bollinger Bands
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['close_time'], y=df['bb_upper'], line=dict(color='rgba(255,255,255,0.2)', width=1, dash='dash'), name='BB Upper'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['close_time'], y=df['bb_lower'], line=dict(color='rgba(255,255,255,0.2)', width=1, dash='dash'), name='BB Lower', fill='tonexty', fillcolor='rgba(255,255,255,0.05)'),
            row=1, col=1
        )

    # RSI
    if 'rsi' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['close_time'], y=df['rsi'], line=dict(color='#8A2BE2', width=1), name='RSI'),
            row=2, col=1
        )
        # RSI Overbought/Oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # Layout styling for Dark Theme
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b0e11",
        plot_bgcolor="#0b0e11",
        xaxis_rangeslider_visible=False,
        height=600,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Grid lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#161b22')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#161b22')

    return fig
