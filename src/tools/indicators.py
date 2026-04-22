import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.trend import EMAIndicator

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds required technical indicators to the OHLCV DataFrame:
    rsi, ema20, bb_upper, bb_lower, stochastic
    """
    if df.empty or len(df) < 55:
        return df

    # RSI (14)
    rsi_indicator = RSIIndicator(close=df['close'], window=14)
    df['rsi'] = rsi_indicator.rsi()

    # EMA (20)
    ema_20 = EMAIndicator(close=df['close'], window=20)
    df['ema20'] = ema_20.ema_indicator()

    # Bollinger Bands (20, 2)
    bb_indicator = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb_indicator.bollinger_hband()
    df['bb_lower'] = bb_indicator.bollinger_lband()

    # Stochastic Oscillator (14, 3, 3)
    stoch = StochasticOscillator(
        high=df['high'], 
        low=df['low'], 
        close=df['close'], 
        window=14, 
        smooth_window=3
    )
    df['stochastic'] = stoch.stoch()

    return df
