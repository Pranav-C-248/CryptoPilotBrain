import pandas as pd
import numpy as np
import warnings
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import EMAIndicator, ADXIndicator


class MarketDataProcessor:

    @staticmethod
    def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds all technical indicators to the OHLCV DataFrame.

        Computed columns:
            rsi          — RSI (14)        | all strategies
            rsi5         — RSI (5)         | BB + RSI Extremes (hyper-sensitive trigger)
            ema9         — EMA (9)         | MA Twist & Convergence (fast)
            ema20        — EMA (20)        | all strategies (anchor)
            ema50        — EMA (50)        | MA Twist & Convergence (slow)
            ema200       — EMA (200)       | 200 MA Macro Pullback
            bb_upper     — BB (20, 2σ)     | Volatility Breakout, Bottom BB, VWAP proxy
            bb_lower     — BB (20, 2σ)
            bb_mid       — BB midline      | Bottom BB Mean Reversion exit target
            bb40_upper   — BB (40, 2σ)     | BB + RSI Extremes
            bb40_lower   — BB (40, 2σ)
            bb15_upper   — BB (20, 1.5σ)  | VWAP + Stochastic Micro-Reversion
            bb15_lower   — BB (20, 1.5σ)
            stochastic   — Stochastic (7,4,3) | VWAP + Stochastic
            adx          — ADX (14)        | BB + RSI Extremes non-trending gate

        Requires at least 200 rows. Returns df unchanged with a warning if not met.
        """
        if df.empty or len(df) < 200:
            warnings.warn(
                f"add_indicators: need ≥200 rows, got {len(df)}. "
                "No indicators added — signals will not be produced."
            )
            return df

        close = df['close']
        high  = df['high']
        low   = df['low']

        # ── RSI ───────────────────────────────────────────────────────────────
        df['rsi']  = RSIIndicator(close=close, window=14).rsi()
        df['rsi5'] = RSIIndicator(close=close, window=5).rsi()

        # ── EMAs ──────────────────────────────────────────────────────────────
        df['ema9']   = EMAIndicator(close=close, window=9).ema_indicator()
        df['ema20']  = EMAIndicator(close=close, window=20).ema_indicator()
        df['ema50']  = EMAIndicator(close=close, window=50).ema_indicator()
        df['ema200'] = EMAIndicator(close=close, window=200).ema_indicator()

        # ── Bollinger Bands ───────────────────────────────────────────────────
        # Standard: BB(20, 2σ) — primary band for most strategies
        bb20 = BollingerBands(close=close, window=20, window_dev=2)
        df['bb_upper'] = bb20.bollinger_hband()
        df['bb_lower'] = bb20.bollinger_lband()
        df['bb_mid']   = bb20.bollinger_mavg()   # (upper + lower) / 2 = 20-period SMA

        # Wide: BB(40, 2σ) — BB + RSI Extremes strategy
        bb40 = BollingerBands(close=close, window=40, window_dev=2)
        df['bb40_upper'] = bb40.bollinger_hband()
        df['bb40_lower'] = bb40.bollinger_lband()

                
        # ── ADX (14) — BB + RSI Extremes non-trending gate ────────────────────
        adx_indicator = ADXIndicator(high=high, low=low, close=close, window=14)
        df['adx'] = adx_indicator.adx()

        # Add atr 
        from ta.volatility import AverageTrueRange
        df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
        df['atr_avg_5'] = df['atr'].rolling(5).mean().shift(1)
        df['atr_expanding'] = df['atr'] > df['atr_avg_5']
        
        # ── Volume Spikes ───────────────────────────────────────────────────────
        vol_avg_50 = df['volume'].rolling(50).mean().shift(1)
        vol_avg_20 = df['volume'].rolling(20).mean().shift(1)
        df['is_spike'] = df['volume'] > (vol_avg_50 * 1.5)
        df['is_capitulation_spike'] = df['volume'] > (vol_avg_20 * 2.0)

        return df

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _detect_structure(current_row, history_10: pd.DataFrame) -> str:
        """
        HH/HL → Uptrend, LH/LL → Downtrend, else Sideways/Ranging.
        Splits the 10-candle window into two halves and compares swing
        highs and lows between them.
        """
        mid         = len(history_10) // 2
        first_half  = history_10.iloc[:mid]
        second_half = history_10.iloc[mid:]

        first_high  = first_half['high'].max()
        first_low   = first_half['low'].min()
        second_high = second_half['high'].max()
        second_low  = second_half['low'].min()

        hh = second_high > first_high
        hl = second_low  > first_low
        lh = second_high < first_high
        ll = second_low  < first_low

        if hh and hl:
            return "Uptrend"
        elif lh and ll:
            return "Downtrend"
        else:
            return "Sideways/Ranging"

    @staticmethod
    def _detect_ma_twist(
        ema9: float, ema20: float, ema50: float,
        history_5_ema9: float, history_5_ema50: float
    ) -> dict:
        """
        Detects the MA Twist & Convergence state.

        Convergence  — all three EMAs within 1.5% spread of ema50 (the 'twist' moment)
        Stack order  — whether EMAs have re-fanned in bullish or bearish order
        Slopes       — computed as ema[-1] vs ema[-5] for a true directional delta,
                       not a mean comparison

        Args:
            ema9, ema20, ema50         — current EMA values
            history_5_ema9/ema50       — EMA values 5 candles ago (for slope)
        """
        spread     = max(ema9, ema20, ema50) - min(ema9, ema20, ema50)
        spread_pct = round((spread / ema50) * 100, 3) if ema50 > 0 else 0.0
        converging = spread_pct < 1.5

        if ema9 > ema20 > ema50:
            stack_order = "Bullish"
        elif ema9 < ema20 < ema50:
            stack_order = "Bearish"
        else:
            stack_order = "Mixed"

        # True slope: current value vs value 5 candles ago
        ema9_slope  = "Increasing" if ema9  > history_5_ema9  else "Decreasing"
        ema50_slope = "Increasing" if ema50 > history_5_ema50 else "Decreasing"

        return {
            "converging":  converging,
            "stack_order": stack_order,
            "spread_pct":  spread_pct,
            "ema9_slope":  ema9_slope,
            "ema50_slope": ema50_slope,
        }

    @staticmethod
    def _detect_ema200_context(
        price: float,
        ema200_current: float,
        ema200_5ago: float
    ) -> dict:
        """
        EMA(200) proximity and slope for the 200 MA Macro Pullback strategy.

        proximity_pct — price distance from EMA200 as % (negative = below)
        slope         — true directional delta: ema200[-1] vs ema200[-5]
        near_band     — True if price is within 2% of EMA200 (entry zone)
        """
        proximity_pct = round(((price - ema200_current) / ema200_current) * 100, 3) if ema200_current > 0 else 0.0
        slope         = "Increasing" if ema200_current > ema200_5ago else "Decreasing"
        price_rel     = "Price Above EMA200" if price >= ema200_current else "Price Below EMA200"
        near_band     = abs(proximity_pct) <= 2.0

        return {
            "value":         round(ema200_current, 2),
            "proximity_pct": proximity_pct,
            "slope":         slope,
            "price_rel":     price_rel,
            "near_band":     near_band,
        }

    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def get_context_packet(df: pd.DataFrame, current_idx: int, window: int = 200) -> dict | None:
        """
        Extracts and calculates all market context fields from a dataframe slice.

        Window is 200 to support EMA(200) for the 200 MA Macro Pullback strategy.
        Caller must ensure at least 200 rows are fetched from the data source.

        Expected columns (all produced by add_indicators):
            open, high, low, close, volume,
            rsi, rsi5,
            ema9, ema20, ema50, ema200,
            bb_upper, bb_lower, bb_mid,
            bb40_upper, bb40_lower, adx
        """
        if current_idx < window:
            return None

        if 'ema200' not in df.columns:
            df = MarketDataProcessor.add_indicators(df)

        current_row  = df.iloc[current_idx]
        prev_row     = df.iloc[current_idx - 1]
        # Lookback slices
        history_10  = df.iloc[current_idx - 10  + 1 : current_idx + 1]
        history_20  = df.iloc[current_idx - 20  + 1 : current_idx + 1]
        history_50  = df.iloc[current_idx - 50  + 1 : current_idx + 1]
        history_55  = df.iloc[current_idx - 55  + 1 : current_idx + 1]
        # Row 5 candles ago — for true EMA slope deltas
        row_5ago    = df.iloc[current_idx - 5]

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

        # ── 3. RSI (14) — all strategies ──────────────────────────────────────
        rsi_current = float(current_row['rsi'])
        rsi_prev    = float(prev_row['rsi'])
        rsi_avg_50  = float(history_50['rsi'].mean())
        rsi_slope   = "Increasing" if rsi_current > float(row_5ago['rsi']) else "Decreasing"

        # ── 4. RSI (5) — BB + RSI Extremes only ───────────────────────────────
        rsi5_current = float(current_row['rsi5'])
        rsi5_prev    = float(prev_row['rsi5'])
        rsi5_slope   = "Increasing" if rsi5_current > float(row_5ago['rsi5']) else "Decreasing"

        # ── 5. Bollinger Bands (20, 2σ) — primary ─────────────────────────────
        bb_upper = float(current_row['bb_upper'])
        bb_lower = float(current_row['bb_lower'])
        bb_mid   = float(current_row['bb_mid'])

        if price > bb_upper:
            bb_pos = "Above Upper Band"
        elif price < bb_lower:
            bb_pos = "Below Lower Band"
        elif price > bb_mid:
            bb_pos = "Upper Half"
        else:
            bb_pos = "Lower Half"

        # ── 6. Bollinger Bands (40, 2σ) — BB + RSI Extremes ───────────────────
        bb40_upper = float(current_row['bb40_upper'])
        bb40_lower = float(current_row['bb40_lower'])

        if price > bb40_upper:
            bb40_pos = "Above Upper Band"
        elif price < bb40_lower:
            bb40_pos = "Below Lower Band"
        elif price > (bb40_upper + bb40_lower) / 2:
            bb40_pos = "Upper Half"
        else:
            bb40_pos = "Lower Half"
        

        # ── 8. EMA (20) — primary anchor ──────────────────────────────────────
        ema_20    = float(current_row['ema20'])
        ema_slope = "Increasing" if ema_20 > float(row_5ago['ema20']) else "Decreasing"
        ema_rel   = "Price Above EMA" if price > ema_20 else "Price Below EMA"

        # ── 9. EMA Multi (9/20/50) — MA Twist & Convergence ───────────────────
        ema_9  = float(current_row['ema9'])
        ema_50 = float(current_row['ema50'])
        ma_twist = MarketDataProcessor._detect_ma_twist(
            ema9=ema_9, ema20=ema_20, ema50=ema_50,
            history_5_ema9=float(row_5ago['ema9']),
            history_5_ema50=float(row_5ago['ema50'])
        )

        # ── 10. EMA (200) — 200 MA Macro Pullback ─────────────────────────────
        ema_200  = float(current_row['ema200'])
        ma_200   = MarketDataProcessor._detect_ema200_context(
            price=price,
            ema200_current=ema_200,
            ema200_5ago=float(row_5ago['ema200'])
        )

        # ── 11. ADX (14) — BB + RSI Extremes non-trending gate ────────────────
        adx_value     = float(current_row['adx'])
        adx_non_trend = adx_value < 25   # True = non-trending, strategy is valid

        # ── 12. Donchian Levels ────────────────────────────────────────────────
        high_10 = float(history_10['high'].max())
        low_10  = float(history_10['low'].min())
        high_20 = float(history_20['high'].max())
        low_20  = float(history_20['low'].min())
        high_50 = float(history_50['high'].max())
        low_50  = float(history_50['low'].min())
        high_55 = float(history_55['high'].max())
        low_55  = float(history_55['low'].min())

        # ── 13. Volume ────────────────────────────────────────────────────────
        vol_current = float(current_row['volume'])
        vol_avg_50  = float(history_50['volume'].mean())
        vol_avg_20  = float(history_20['volume'].mean())
        # Standard spike: 1.5x 50-period avg (Volatility Breakout, Turtle)
        is_spike = bool(vol_current > vol_avg_50 * 1.5)
        # Capitulation spike: 2x 20-period avg (200 MA Macro Pullback)
        is_capitulation_spike = bool(vol_current > vol_avg_20 * 2.0)

        # ── 14. ATR + expansion flag ───────────────────────────────────────────
        atr           = float(current_row['atr'])
        atr_avg_5     = float(current_row['atr_avg_5'])
        atr_expanding = bool(current_row['atr_expanding'])


        return {
            "snapshot": snapshot,

            "trends": {
                "pct_change_50":    round(pct_change_50, 2),
                "pct_change_10":    round(pct_change_10, 2),
                "market_structure": structure
            },

            # RSI(14) — general momentum across all strategies
            "rsi": {
                "current": round(rsi_current, 2),
                "prev":    round(rsi_prev, 2),
                "avg_50":  round(rsi_avg_50, 2),
                "trend":   rsi_slope
            },

            # RSI(5) — hyper-sensitive trigger for BB + RSI Extremes
            "rsi5": {
                "current": round(rsi5_current, 2),
                "prev":    round(rsi5_prev, 2),
                "trend":   rsi5_slope
            },

            # BB(20, 2σ) — primary bands
            "bands": {
                "upper":          round(bb_upper, 2),
                "mid":            round(bb_mid,   2),
                "lower":          round(bb_lower, 2),
                "position_label": bb_pos
            },

            # BB(40, 2σ) — BB + RSI Extremes
            "bands40": {
                "upper":          round(bb40_upper, 2),
                "lower":          round(bb40_lower, 2),
                "position_label": bb40_pos
            },

            # BB(20, 1.5σ) — VWAP + Stochastic Micro-Reversion
            

            # EMA(20) — kept at top level for backward compatibility
            "ema": {
                "value":     round(ema_20, 2),
                "price_rel": ema_rel,
                "slope":     ema_slope
            },

            # EMA(9/20/50) — MA Twist & Convergence
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

            # EMA(200) — 200 MA Macro Pullback
            "ema200": {
                "value":         ma_200["value"],
                "proximity_pct": ma_200["proximity_pct"],
                "slope":         ma_200["slope"],
                "price_rel":     ma_200["price_rel"],
                "near_band":     ma_200["near_band"],
            },

            # ADX(14) — non-trending gate for BB + RSI Extremes
            "adx": {
                "value":          round(adx_value, 2),
                "non_trending":   adx_non_trend    # True when ADX < 25
            },

            "levels": {
                "high_10": round(high_10, 2),
                "low_10":  round(low_10,  2),
                "high_20": round(high_20, 2),
                "low_20":  round(low_20,  2),
                "high_50": round(high_50, 2),
                "low_50":  round(low_50,  2),
                "high_55": round(high_55, 2),
                "low_55":  round(low_55,  2)
            },

            "volume": {
                "current":              round(vol_current, 2),
                "avg_50":               round(vol_avg_50,  2),
                "avg_20":               round(vol_avg_20,  2),
                "is_spike":             is_spike,
                "is_capitulation_spike": is_capitulation_spike
            },

            "volatility": {
                "atr":          round(atr, 2),
                "atr_avg_5":    round(atr_avg_5, 2),
                "atr_expanding": atr_expanding
            },

            
        }