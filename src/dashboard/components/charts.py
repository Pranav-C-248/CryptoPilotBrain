import pandas as pd
import numpy as np

def render_candlestick_chart(df: pd.DataFrame, title: str = "Price Chart"):
    """
    Renders an interactive TradingView Lightweight Chart configuration with Bollinger Bands and RSI.
    Returns a list of dictionaries compatible with streamlit-lightweight-charts.
    """
    if df.empty:
        return []

    # Ensure time is formatted correctly for lightweight charts (unix timestamp in seconds)
    df = df.copy()
    
    # Robustly convert datetime to unix timestamp in seconds, regardless of numpy resolution
    epoch = pd.Timestamp("1970-01-01")
    df['time'] = ((pd.to_datetime(df['open_time']).dt.tz_localize(None) - epoch) // pd.Timedelta('1s')).astype(int)

    # Main Price Chart Config
    main_chart = {
        "chart": {
            "height": 450,
            "layout": {
                "background": {"type": "solid", "color": "#0b0e11"},
                "textColor": "rgba(255, 255, 255, 0.9)"
            },
            "grid": {
                "vertLines": {"color": "#161b22"},
                "horzLines": {"color": "#161b22"}
            },
            "crosshair": {
                "mode": 0
            },
            "watermark": {
                "color": "rgba(255, 255, 255, 0.05)",
                "visible": True,
                "text": title,
                "fontSize": 48,
                "horzAlign": "center",
                "vertAlign": "center"
            },
            "handleScroll": {
                "mouseWheel": True,
                "pressedMouseMove": True,
                "horzTouchDrag": True,
                "vertTouchDrag": True
            },
            "handleScale": {
                "axisPressedMouseMove": {"time": True, "price": True},
                "mouseWheel": True,
                "pinch": True
            },
            "timeScale": {
                "rightOffset": 5,
                "barSpacing": 15,
                "fixLeftEdge": False,
                "lockVisibleTimeRangeOnResize": True,
                "rightBarStaysOnScroll": True,
                "borderVisible": False,
                "visible": True,
                "timeVisible": True,
                "secondsVisible": False,
                "shiftVisibleRangeOnNewBar": True
            }
        },
        "series": []
    }

    # Candlestick Series
    main_chart["series"].append({
        "type": "Candlestick",
        "data": df[['time', 'open', 'high', 'low', 'close']].to_dict('records'),
        "options": {
            "upColor": "#0ecb81",
            "downColor": "#f6465d",
            "borderVisible": False,
            "wickUpColor": "#0ecb81",
            "wickDownColor": "#f6465d",
            "priceFormat": {
                "type": 'price',
                "precision": 2,
                "minMove": 0.01,
            }
        }
    })

    # EMA 20 Series
    if 'ema20' in df.columns:
        ema_data = df[['time', 'ema20']].rename(columns={'ema20': 'value'}).dropna().to_dict('records')
        main_chart["series"].append({
            "type": "Line",
            "data": ema_data,
            "options": {
                "color": "yellow",
                "lineWidth": 1,
                "title": "EMA 20"
            }
        })

    # Bollinger Bands Series
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        bb_upper_data = df[['time', 'bb_upper']].rename(columns={'bb_upper': 'value'}).dropna().to_dict('records')
        bb_lower_data = df[['time', 'bb_lower']].rename(columns={'bb_lower': 'value'}).dropna().to_dict('records')
        
        main_chart["series"].append({
            "type": "Line",
            "data": bb_upper_data,
            "options": {
                "color": "rgba(255,255,255,0.3)",
                "lineWidth": 1,
                "lineStyle": 2, # dashed
                "title": "BB Upper"
            }
        })
        main_chart["series"].append({
            "type": "Line",
            "data": bb_lower_data,
            "options": {
                "color": "rgba(255,255,255,0.3)",
                "lineWidth": 1,
                "lineStyle": 2, # dashed
                "title": "BB Lower"
            }
        })

    # Sub-pane: RSI Chart Config
    rsi_chart = {
        "chart": {
            "height": 150,
            "layout": {
                "background": {"type": "solid", "color": "#0b0e11"},
                "textColor": "rgba(255, 255, 255, 0.9)"
            },
            "grid": {
                "vertLines": {"color": "#161b22"},
                "horzLines": {"color": "#161b22"}
            }
        },
        "series": []
    }

    if 'rsi' in df.columns:
        rsi_data = df[['time', 'rsi']].rename(columns={'rsi': 'value'}).dropna().to_dict('records')
        rsi_chart["series"].append({
            "type": "Line",
            "data": rsi_data,
            "options": {
                "color": "#8A2BE2",
                "lineWidth": 2,
                "title": "RSI 14"
            }
        })
        
        # Add baseline for RSI 30 and 70
        rsi_chart["series"].append({
            "type": "Baseline",
            "data": rsi_data,
            "options": {
                "baseValue": {"type": "price", "price": 50},
                "topLineColor": "rgba(255, 0, 0, 0)", 
                "bottomLineColor": "rgba(0, 255, 0, 0)",
                "baseLineColor": "rgba(255, 255, 255, 0.2)",
                "baseLineStyle": 2,
                "lineWidth": 0,
            }
        })

    return [main_chart, rsi_chart]
