import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.trend import EMAIndicator

import numpy as np


class MarketDataProcessor:

    @staticmethod
    def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds required technical indicators to the OHLCV DataFrame.

        Computed columns:
            rsi         — RSI (14)
            ema9        — EMA (9)   | fast   — MA Twist & Convergence
            ema20       — EMA (20)  | medium — existing anchor, MA Twist
            ema50       — EMA (50)  | slow   — MA Twist & Convergence
            ema200      — EMA (200) | macro  — 200 MA Macro Pullback
            bb_upper    — Bollinger Band upper (20, 2σ)
            bb_lower    — Bollinger Band lower (20, 2σ)
            stochastic  — Stochastic %K (14, 3)
        """
        if df.empty or len(df) < 200:
            return df

        # RSI (14)
        df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()

        # EMAs
        df['ema9']   = EMAIndicator(close=df['close'], window=9).ema_indicator()
        df['ema20']  = EMAIndicator(close=df['close'], window=20).ema_indicator()
        df['ema50']  = EMAIndicator(close=df['close'], window=50).ema_indicator()
        df['ema200'] = EMAIndicator(close=df['close'], window=200).ema_indicator()

        # Bollinger Bands (20, 2σ)
        bb = BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()

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

    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_atr(df: pd.DataFrame, current_idx: int, period: int = 14) -> float:
        """
        Computes Average True Range over `period` candles ending at current_idx.
        True Range = max(high-low, |high-prev_close|, |low-prev_close|)
        """
        start  = max(0, current_idx - period + 1)
        slice_ = df.iloc[start: current_idx + 1]

        true_ranges = []
        for i in range(1, len(slice_)):
            high       = slice_.iloc[i]['high']
            low        = slice_.iloc[i]['low']
            prev_close = slice_.iloc[i - 1]['close']
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            true_ranges.append(tr)

        return (
            float(np.mean(true_ranges)) if true_ranges
            else float(slice_.iloc[-1]['high'] - slice_.iloc[-1]['low'])
        )

    @staticmethod
    def _detect_structure(current_row, history_10: pd.DataFrame) -> str:
        """
        Proper HH/HL (Uptrend) vs LH/LL (Downtrend) vs neither (Sideways).
        Splits the 10-candle window into first and second half and compares
        swing highs and lows between halves.
        """
        mid         = len(history_10) // 2
        first_half  = history_10.iloc[:mid]
        second_half = history_10.iloc[mid:]

        first_high  = first_half['high'].max()
        first_low   = first_half['low'].min()
        second_high = second_half['high'].max()
        second_low  = second_half['low'].min()

        hh = second_high > first_high   # Higher High
        hl = second_low  > first_low    # Higher Low
        lh = second_high < first_high   # Lower High
        ll = second_low  < first_low    # Lower Low

        if hh and hl:
            return "Uptrend"
        elif lh and ll:
            return "Downtrend"
        else:
            return "Sideways/Ranging"

    @staticmethod
    def _detect_ma_twist(
        ema9: float, ema20: float, ema50: float,
        history_10: pd.DataFrame
    ) -> dict:
        """
        Detects the MA Twist & Convergence state for the current candle.

        Convergence   — the three EMAs are within a tight band of each other
                        (max spread < 1.5% of ema50), signalling the 'twist' moment.
        Stack order   — whether the EMAs are fanned out in bullish or bearish order
                        after a convergence.

        Returns a dict with:
            converging  (bool)   — True if the three EMAs are tightly clustered
            stack_order (str)    — "Bullish" | "Bearish" | "Mixed"
            spread_pct  (float)  — (max - min) of the 3 EMAs as % of ema50
            ema9_slope  (str)    — "Increasing" | "Decreasing"
            ema50_slope (str)    — "Increasing" | "Decreasing"
        """
        spread     = max(ema9, ema20, ema50) - min(ema9, ema20, ema50)
        spread_pct = round((spread / ema50) * 100, 3) if ema50 > 0 else 0.0

        # Convergence threshold: spread < 1.5% of the slow EMA
        converging = spread_pct < 1.5

        # Stack order — bullish = fast > medium > slow
        if ema9 > ema20 > ema50:
            stack_order = "Bullish"
        elif ema9 < ema20 < ema50:
            stack_order = "Bearish"
        else:
            stack_order = "Mixed"

        ema9_slope  = "Increasing" if ema9  > float(history_10['ema9'].mean())  else "Decreasing"
        ema50_slope = "Increasing" if ema50 > float(history_10['ema50'].mean()) else "Decreasing"

        return {
            "converging":  converging,
            "stack_order": stack_order,
            "spread_pct":  spread_pct,
            "ema9_slope":  ema9_slope,
            "ema50_slope": ema50_slope,
        }

    @staticmethod
    def _detect_ema200_context(
        price: float, ema200: float, history_10: pd.DataFrame
    ) -> dict:
        """
        Computes EMA(200) proximity and slope context for the
        200 MA Macro Pullback strategy.

        proximity_pct — how far price is from the 200 EMA as a % of ema200.
                        Negative = price is below the 200 EMA.
        slope         — direction of the 200 EMA over the last 10 candles.
        price_rel     — "Price Above EMA200" | "Price Below EMA200"
        near_band     — True if price is within 2% of the 200 EMA (entry zone).
        """
        proximity_pct = round(((price - ema200) / ema200) * 100, 3) if ema200 > 0 else 0.0
        slope         = "Increasing" if ema200 > float(history_10['ema200'].mean()) else "Decreasing"
        price_rel     = "Price Above EMA200" if price >= ema200 else "Price Below EMA200"
        near_band     = abs(proximity_pct) <= 2.0

        return {
            "value":        round(ema200, 2),
            "proximity_pct": proximity_pct,
            "slope":        slope,
            "price_rel":    price_rel,
            "near_band":    near_band,
        }

    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def get_context_packet(df: pd.DataFrame, current_idx: int, window: int = 200) -> dict | None:
        """
        Extracts and calculates all market context fields from a dataframe slice.

        Window is 200 to support EMA(200) for the 200 MA Macro Pullback strategy.
        Caller is responsible for ensuring at least 200 rows of data are fetched.

        Expected columns (produced by add_indicators):
            open, high, low, close, volume,
            rsi, ema9, ema20, ema50, ema200,
            bb_upper, bb_lower, stochastic (or stoch_k)
        """
        if current_idx < window:
            return None

        if 'rsi' not in df.columns:
            df = MarketDataProcessor.add_indicators(df)

        current_row = df.iloc[current_idx]
        prev_row    = df.iloc[current_idx - 1]
        history_200 = df.iloc[current_idx - 200 + 1 : current_idx + 1]
        history_55  = df.iloc[current_idx - 55  + 1 : current_idx + 1]
        history_50  = df.iloc[current_idx - 50  + 1 : current_idx + 1]
        history_20  = df.iloc[current_idx - 20  + 1 : current_idx + 1]
        history_10  = df.iloc[current_idx - 10  + 1 : current_idx + 1]

        price = float(current_row['close'])

        # ── 1. Snapshot ───────────────────────────────────────────────────────
        snapshot = {
            "open":  float(current_row['open']),
            "high":  float(current_row['high']),
            "low":   float(current_row['low']),
            "close": price
        }

        # ── 2. Trends ─────────────────────────────────────────────────────────
        pct_change_50 = ((price - history_50.iloc[0]['close']) / history_50.iloc[0]['close']) * 100
        pct_change_10 = ((price - history_10.iloc[0]['close']) / history_10.iloc[0]['close']) * 100
        structure     = MarketDataProcessor._detect_structure(current_row, history_10)

        # ── 3. RSI ────────────────────────────────────────────────────────────
        rsi_current = float(current_row['rsi'])
        rsi_prev    = float(prev_row['rsi'])
        rsi_avg_50  = float(history_50['rsi'].mean())
        rsi_slope   = "Increasing" if rsi_current > float(history_10['rsi'].mean()) else "Decreasing"

        # ── 4. Bollinger Bands ────────────────────────────────────────────────
        bb_upper = float(current_row['bb_upper'])
        bb_lower = float(current_row['bb_lower'])

        if price > bb_upper:
            bb_pos = "Above Upper Band"
        elif price < bb_lower:
            bb_pos = "Below Lower Band"
        elif price > (bb_upper + bb_lower) / 2:
            bb_pos = "Upper Half"
        else:
            bb_pos = "Lower Half"

        # ── 5. EMA (20) — existing anchor ────────────────────────────────────
        ema_20    = float(current_row['ema20'])
        ema_slope = "Increasing" if ema_20 > float(history_10['ema20'].mean()) else "Decreasing"
        ema_rel   = "Price Above EMA" if price > ema_20 else "Price Below EMA"

        # ── 6. EMA Multi (9 / 20 / 50) — MA Twist & Convergence ──────────────
        ema_9  = float(current_row['ema9'])
        ema_50 = float(current_row['ema50'])
        ma_twist = MarketDataProcessor._detect_ma_twist(ema_9, ema_20, ema_50, history_10)

        # ── 7. EMA (200) — 200 MA Macro Pullback ─────────────────────────────
        ema_200  = float(current_row['ema200'])
        ma_200   = MarketDataProcessor._detect_ema200_context(price, ema_200, history_10)

        # ── 8. Levels — Donchian channels for Turtle strategies ───────────────
        high_50 = float(history_50['high'].max())
        low_50  = float(history_50['low'].min())
        high_20 = float(history_20['high'].max())
        low_20  = float(history_20['low'].min())
        high_55 = float(history_55['high'].max())
        low_55  = float(history_55['low'].min())

        # ── 9. Volume ─────────────────────────────────────────────────────────
        vol_avg  = float(history_50['volume'].mean())
        is_spike = bool(current_row['volume'] > vol_avg * 1.5)

        # ── 10. ATR ───────────────────────────────────────────────────────────
        atr = MarketDataProcessor._compute_atr(df, current_idx, period=14)

        # ── 11. Stochastic ────────────────────────────────────────────────────
        stoch_col = 'stochastic' if 'stochastic' in df.columns else 'stoch_k'
        stoch_val = float(current_row[stoch_col]) if stoch_col in df.columns else 50.0

        return {
            "snapshot": snapshot,

            "trends": {
                "pct_change_50":    round(pct_change_50, 2),
                "pct_change_10":    round(pct_change_10, 2),
                "market_structure": structure
            },

            "rsi": {
                "current": round(rsi_current, 2),
                "prev":    round(rsi_prev, 2),
                "avg_50":  round(rsi_avg_50, 2),
                "trend":   rsi_slope
            },

            "bands": {
                "upper":          round(bb_upper, 2),
                "lower":          round(bb_lower, 2),
                "position_label": bb_pos
            },

            # EMA(20) — kept at top level for backward compatibility
            # with the existing analyst prompt and all original strategies
            "ema": {
                "value":     round(ema_20, 2),
                "price_rel": ema_rel,
                "slope":     ema_slope
            },

            # EMA(9 / 20 / 50) — for MA Twist & Convergence strategy
            "ema_multi": {
                "ema9":        round(ema_9,  2),
                "ema20":       round(ema_20, 2),
                "ema50":       round(ema_50, 2),
                "converging":  ma_twist["converging"],
                "stack_order": ma_twist["stack_order"],
                "spread_pct":  ma_twist["spread_pct"],
                "ema9_slope":  ma_twist["ema9_slope"],
                "ema50_slope": ma_twist["ema50_slope"],
            },

            # EMA(200) — for 200 MA Macro Pullback strategy
            "ema200": {
                "value":         ma_200["value"],
                "proximity_pct": ma_200["proximity_pct"],
                "slope":         ma_200["slope"],
                "price_rel":     ma_200["price_rel"],
                "near_band":     ma_200["near_band"],
            },

            "levels": {
                "high_50": round(high_50, 2),
                "low_50":  round(low_50,  2),
                "high_20": round(high_20, 2),
                "low_20":  round(low_20,  2),
                "high_55": round(high_55, 2),
                "low_55":  round(low_55,  2)
            },

            "volume": {
                "current":  round(float(current_row['volume']), 2),
                "avg_50":   round(vol_avg, 2),
                "is_spike": is_spike
            },

            "volatility": {
                "atr": round(atr, 2)
            },

            "stochastic": {
                "current": round(stoch_val, 2)
            }
        }