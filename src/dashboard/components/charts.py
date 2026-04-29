import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def render_candlestick_chart(df: pd.DataFrame, title: str = "Price Chart"):
    """
    Enhanced Trading Dashboard Chart:
    - Candlestick
    - EMA
    - Bollinger Bands
    - RSI Panel
    - Clean SaaS Dark Theme
    """

    if df.empty:
        return go.Figure()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.72, 0.28]
    )

    # CANDLESTICK 
    fig.add_trace(
        go.Candlestick(
            x=df['close_time'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#22c55e',
            decreasing_line_color='#ef4444',
            showlegend=False
        ),
        row=1, col=1
    )

    # EMA 20
    if 'ema20' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['close_time'],
                y=df['ema20'],
                line=dict(color='#6366f1', width=2),
                name='EMA 20'
            ),
            row=1, col=1
        )

    # BOLLINGER BANDS
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['close_time'],
                y=df['bb_upper'],
                line=dict(color='rgba(255,255,255,0.2)', width=1, dash='dot'),
                name='BB Upper'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=df['close_time'],
                y=df['bb_lower'],
                line=dict(color='rgba(255,255,255,0.2)', width=1, dash='dot'),
                fill='tonexty',
                fillcolor='rgba(255,255,255,0.05)',
                name='BB Lower'
            ),
            row=1, col=1
        )

    # RSI
    if 'rsi' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['close_time'],
                y=df['rsi'],
                line=dict(color='#8b5cf6', width=2),
                name='RSI'
            ),
            row=2, col=1
        )

        # RSI Levels
        fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#22c55e", row=2, col=1)

    # LAYOUT (SaaS Style) 
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",

        height=600,
        margin=dict(l=10, r=10, t=10, b=10),

        font=dict(color="#e2e8f0"),

        hovermode="x unified",

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)"
        )
    )

    # AXES STYLING
    fig.update_xaxes(
        showgrid=False,
        rangeslider_visible=False
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.05)"
    )

    return fig