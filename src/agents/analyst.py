import json
from datetime import datetime, timezone, timedelta
from dotenv import dotenv_values
from google import genai
from google.genai import types
import pandas as pd
import numpy as np
import requests
import re
from langchain_ollama import ChatOllama
from src.schema.models import AnalystSignal
from src.core.knowledge_base import TradingKnowledgeBase


class AnalystAgent:
    def __init__(self, knowledge_base: TradingKnowledgeBase, timeframe_hours: int = 4):
        self.kb = knowledge_base
        self.timeframe_hours = timeframe_hours

        config = dotenv_values(".env")
        self.client = genai.Client(api_key=config["gemini_key"])
        self.model_name = "gemini-2.5-flash"
        self.ollama_llm = ChatOllama(model="gemma4:e2b", format="json", temperature=0)

    def _detect_regime(self, market_data: dict) -> str:
        structure = market_data['trends']['market_structure'].lower()
        atr       = market_data['volatility']['atr']
        price     = market_data['snapshot']['close']
        vol_spike = market_data['volume']['is_spike']
        atr_pct   = atr / price if price > 0 else 0

        if "sideways" in structure or "ranging" in structure:
            if atr_pct > 0.02 or vol_spike:
                return "Volatile"
            return "Sideways"
        elif "trend" in structure or "bull" in structure or "bear" in structure:
            if vol_spike:
                return "Volatile"
            return "Trending"
        return None

    def _build_rsi_narrative(self, market_data: dict) -> tuple:
        rsi_current = market_data['rsi']['current']
        rsi_prev    = market_data['rsi'].get('prev', None)
        rsi_avg     = market_data['rsi']['avg_50']
        rsi_trend   = market_data['rsi']['trend']

        if rsi_prev is not None and rsi_prev < 30 and rsi_current > 30:
            narrative = (
                f"*** RSI was oversold last candle ({rsi_prev}) and has now exited "
                f"the oversold zone ({rsi_current}). Oversold EXIT confirmed. ***"
            )
        elif rsi_prev is not None and rsi_prev > 70 and rsi_current < 70:
            narrative = (
                f"*** RSI was overbought last candle ({rsi_prev}) and has now exited "
                f"the overbought zone ({rsi_current}). Overbought EXIT confirmed. ***"
            )
        else:
            narrative = ""

        return rsi_current, rsi_prev, rsi_avg, rsi_trend, narrative

    def _build_bb_narrative(self, market_data: dict) -> str:
        close    = market_data['snapshot']['close']
        bb_lower = market_data['bands']['lower']
        bb_upper = market_data['bands']['upper']

        if close <= bb_lower * 1.01:
            return (
                f"*** Price ({close}) is at or near BB Lower ({bb_lower}) — "
                f"potential bounce zone. Band breach/touch confirmed. ***"
            )
        elif close >= bb_upper * 0.99:
            return (
                f"*** Price ({close}) is at or near BB Upper ({bb_upper}) — "
                f"potential rejection zone. Band breach/touch confirmed. ***"
            )
        else:
            return f"Price ({close}) is inside the bands (L: {bb_lower}, U: {bb_upper})."

    def _compute_valid_till(self) -> str:
        expiry = datetime.now(timezone.utc) + timedelta(hours=self.timeframe_hours)
        return expiry.strftime("%Y-%m-%dT%H:%M:%SZ")

    def analyze(self, market_data: dict, sentiment_score: float) -> AnalystSignal:
        regime     = self._detect_regime(market_data)
        valid_till = self._compute_valid_till()
        asset_name = market_data.get("asset_name", "UNKNOWN")

        # ── Derived variables ─────────────────────────────────────────────────
        structure  = market_data['trends']['market_structure']
        atr        = market_data['volatility']['atr']
        price      = market_data['snapshot']['close']
        atr_pct    = atr / price if price > 0 else 0
        stoch      = market_data['stochastic']['current']
        regime_str = regime.lower() if regime else "any"

        rsi_current, rsi_prev, rsi_avg, rsi_trend, rsi_narrative = self._build_rsi_narrative(market_data)
        bb_narrative = self._build_bb_narrative(market_data)

        rsi_label   = "overbought" if rsi_current > 70 else "oversold" if rsi_current < 30 else "neutral momentum"
        vol_spike   = "high volume breakout" if market_data['volume']['is_spike'] else ""
        ema_rel     = market_data['ema']['price_rel'].lower()
        bb_pos      = market_data['bands']['position_label'].lower()
        atr_label   = "high volatility atr expansion" if atr_pct > 0.02 else "low volatility consolidation"
        stoch_label = (
            "stochastic oversold exhaustion" if stoch < 20
            else "stochastic overbought exhaustion" if stoch > 80
            else ""
        )

        query = (
            f"{regime_str} market "
            f"{rsi_label} rsi {rsi_trend.lower()} "
            f"{ema_rel} "
            f"{bb_pos} bollinger band "
            f"{atr_label} "
            f"{vol_spike} "
            f"{stoch_label} "
            f"entry confirmation breakout momentum mean reversion scalping"
        ).strip()

        # ── KB retrieval ──────────────────────────────────────────────────────
        strategies: list[dict] = self.kb.get_relevant_strategies(
            query=query,
            k=3,
            score_threshold=0.30,
            regime_filter=regime
        )
        strategy_context = json.dumps(strategies, indent=2) if strategies else None

        # ── Turtle levels — only shown if Trending regime (Turtle strategies) ─
        levels     = market_data.get('levels', {})
        high_20    = levels.get('high_20', 'N/A')
        low_20     = levels.get('low_20',  'N/A')
        high_55    = levels.get('high_55', 'N/A')
        low_55     = levels.get('low_55',  'N/A')
        high_50    = levels.get('high_50', 'N/A')
        low_50     = levels.get('low_50',  'N/A')

        # ── Prompts ───────────────────────────────────────────────────────────
        system_msg = """You are an Institutional Grade Quantitative Analyst.

YOUR THINKING PROCESS — follow these steps IN ORDER before producing any output:

STEP 1 — REGIME CHECK:
Identify the market regime (Trending / Volatile / Sideways). State it explicitly.

STEP 2 — STRATEGY SELECTION:
Review each strategy candidate and understand its theory. For each one ask:
  a) Does its regime match the current regime?
  b) Does the current market trigger its entry_primary condition?
  c) Does anything in its conflicts_with list describe the current market? If yes, DISCARD it.
Select the best-fit strategy. If none fit, write "fallback".
IMPORTANT: The packet contains pre-computed *** annotations that explicitly confirm or deny
key conditions. These are authoritative — trust them over your own raw number interpretation.

STEP 3 — CONFLUENCE CHECK:
List every entry_confirmation condition from the chosen strategy.
For each condition: does the current market data satisfy it? YES or NO.
If ADX is not present, infer non-trending confirmation from market_structure being
"Sideways/Ranging" combined with the ATR/Price% shown in the packet. Do not reject
a strategy solely because ADX is absent.

STEP 4 — SIGNAL DECISION:
- All or most confirmations met → consider BUY or SELL
- Fewer than half met → HOLD
- Any hard conflict present → HOLD
- Sideways regime → default to HOLD unless confluence is clear

STEP 5 — SYNTHESIZE CONDITIONS:
Write:
  - entry_condition: readable if-condition using only indicators in the packet.
    Format: "rsi_prev < 30 and rsi_current > 30 and price <= bb_lower * 1.01"
  - exit_condition: readable if-condition for when to exit.
    Format: "rsi > 70 or price < stop_loss or price > target"

STEP 6 — WRITE YOUR OUTPUT:
Populate JSON in this exact order:
  1. internal_monologue — full Step 1–4 reasoning, cite at least 3 metric values
  2. strategy_used — strategy name or "fallback"
  3. reasoning — concise 2–3 sentence summary
  4. entry_condition — from Step 5
  5. exit_condition — from Step 5
  6. signal — BUY, SELL, or HOLD
  7. confidence — float 0.0–1.0

RULES:
- confidence > 0.8 only if ALL entry_confirmation conditions are met
- confidence > 0.5 never allowed for Sideways regime
- signal must be derivable from internal_monologue
- Trust *** annotated lines — they are pre-computed facts, not interpretations
- For Turtle strategies: 20-day and 55-day breakout levels are provided in LEVELS section
- For VWAP+Stochastic: stochastic value is provided in MOMENTUM section
- Output valid JSON only. No markdown fences."""

        user_msg = f"""ASSET: {asset_name}
VALID_TILL: {valid_till}

--- MARKET CONTEXT PACKET ---
SNAPSHOT: {market_data['snapshot']}
STRUCTURE: {structure} (50-Day: {market_data['trends']['pct_change_50']}%, 10-Day: {market_data['trends']['pct_change_10']}%)

TECHNICAL OVERLAYS:
- RSI: Current {rsi_current} | Prev: {rsi_prev} | Avg_50: {rsi_avg} | Trend: {rsi_trend}
  {rsi_narrative}
- EMA(20): Value {market_data['ema']['value']} | Slope: {market_data['ema']['slope']} | Relation: {market_data['ema']['price_rel']}
- BOLLINGER: Pos: {market_data['bands']['position_label']} (U: {market_data['bands']['upper']}, L: {market_data['bands']['lower']})
  {bb_narrative}

LEVELS:
- 50-Candle: High {high_50} | Low {low_50}
- 20-Candle: High {high_20} | Low {low_20}  ← Turtle System 1 breakout reference
- 55-Candle: High {high_55} | Low {low_55}  ← Turtle System 2 breakout reference

MOMENTUM:
- Stochastic Oscillator: {stoch} {'← oversold exhaustion zone' if stoch < 20 else '← overbought exhaustion zone' if stoch > 80 else ''}

LIQUIDITY & VOLATILITY:
- VOLUME: Current {market_data['volume']['current']} | Spike: {market_data['volume']['is_spike']} (Avg: {market_data['volume']['avg_50']})
- ATR: {atr} | ATR/Price: {round(atr_pct * 100, 3)}% ({'high — volatile' if atr_pct > 0.02 else 'low — non-trending confirmed'})

SENTIMENT: {sentiment_score} (0.5=Neutral)

--- STRATEGY CANDIDATES ---
{strategy_context if strategy_context else "No strategies retrieved. Use standard institutional mean-reversion or trend-following logic."}

Now follow Steps 1 through 6 from your instructions and produce the JSON output."""

        # ── LLM call ──────────────────────────────────────────────────────────
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_msg,
                config=types.GenerateContentConfig(
                    system_instruction=system_msg,
                    temperature=0,
                    response_mime_type="application/json"
                )
            )
            parsed = json.loads(response.text)
        except Exception as e:
            print(f"Gemini API failed: {e}. Falling back to local Ollama (gemma4:e2b)...")
            try:
                response = self.ollama_llm.invoke([("system", system_msg), ("human", user_msg)])
                clean_content = self._clean_json_response(response.content)
                parsed = json.loads(clean_content)
            except Exception as ollama_e:
                print(f"Ollama fallback also failed: {ollama_e}")
                raise e # Raise the original exception if fallback fails

        try:
            parsed["asset_name"] = asset_name
            parsed["valid_till"] = valid_till
            return AnalystSignal.model_validate(parsed)
        except Exception as e:
            print(f"--- DEBUG: LLM Output parsing or validation failed ---")
            raise e

    def _clean_json_response(self, content: str) -> str:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
        if match:
            return match.group(1)
        return content.strip()


# ══════════════════════════════════════════════════════════════════════════════


class MarketDataProcessor:

    @staticmethod
    def _compute_atr(df: pd.DataFrame, current_idx: int, period: int = 14) -> float:
        """
        Computes Average True Range over `period` candles ending at current_idx.
        True Range = max(high-low, |high-prev_close|, |low-prev_close|)
        """
        start = max(0, current_idx - period + 1)
        slice_ = df.iloc[start: current_idx + 1]

        true_ranges = []
        for i in range(1, len(slice_)):
            high      = slice_.iloc[i]['high']
            low       = slice_.iloc[i]['low']
            prev_close= slice_.iloc[i - 1]['close']
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            true_ranges.append(tr)

        return float(np.mean(true_ranges)) if true_ranges else float(slice_.iloc[-1]['high'] - slice_.iloc[-1]['low'])

    @staticmethod
    def _detect_structure(current_row, history_10: pd.DataFrame) -> str:
        """
        Proper HH/HL (Uptrend) vs LH/LL (Downtrend) vs neither (Sideways).
        Splits the 10-candle window into first and second half and compares
        swing highs and lows between halves.
        """
        mid = len(history_10) // 2
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
    def get_context_packet(df: pd.DataFrame, current_idx: int, window: int = 55) -> dict | None:
        """
        Extracts and calculates all market context fields from a dataframe slice.
        Window is 55 to support Turtle System 2 (55-day lookback).
        Expects columns: open, high, low, close, volume, rsi, bb_upper, bb_lower,
                         ema20, stochastic (or stoch_k)
        """
        if current_idx < window:
            return None

        current_row = df.iloc[current_idx]
        prev_row    = df.iloc[current_idx - 1]
        history_55  = df.iloc[current_idx - 55 + 1 : current_idx + 1]
        history_50  = df.iloc[current_idx - 50 + 1 : current_idx + 1]
        history_20  = df.iloc[current_idx - 20 + 1 : current_idx + 1]
        history_10  = df.iloc[current_idx - 10 + 1 : current_idx + 1]

        price = current_row['close']

        # 1. Snapshot
        snapshot = {
            "open":  float(current_row['open']),
            "high":  float(current_row['high']),
            "low":   float(current_row['low']),
            "close": float(price)
        }

        # 2. Trends
        pct_change_50 = ((price - history_50.iloc[0]['close']) / history_50.iloc[0]['close']) * 100
        pct_change_10 = ((price - history_10.iloc[0]['close']) / history_10.iloc[0]['close']) * 100
        structure     = MarketDataProcessor._detect_structure(current_row, history_10)

        # 3. RSI
        rsi_current = float(current_row['rsi'])
        rsi_prev    = float(prev_row['rsi'])
        rsi_avg_50  = float(history_50['rsi'].mean())
        rsi_slope   = "Increasing" if rsi_current > float(history_10['rsi'].mean()) else "Decreasing"

        # 4. Bollinger Bands
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

        # 5. EMA 20
        ema_20    = float(current_row['ema20'])
        ema_slope = "Increasing" if ema_20 > float(history_10['ema20'].mean()) else "Decreasing"
        ema_rel   = "Price Above EMA" if price > ema_20 else "Price Below EMA"

        # 6. Levels — 50, 20, 55 candle highs/lows for all strategies
        high_50 = float(history_50['high'].max())
        low_50  = float(history_50['low'].min())
        high_20 = float(history_20['high'].max())
        low_20  = float(history_20['low'].min())
        high_55 = float(history_55['high'].max())
        low_55  = float(history_55['low'].min())

        # 7. Volume
        vol_avg  = float(history_50['volume'].mean())
        is_spike = bool(current_row['volume'] > vol_avg * 1.5)

        # 8. ATR — computed from true range, not mapped from stddev
        atr = MarketDataProcessor._compute_atr(df, current_idx, period=14)

        # 9. Stochastic — supports VWAP+Stochastic strategy
        stoch_col = 'stochastic' if 'stochastic' in df.columns else 'stoch_k'
        stoch_val = float(current_row[stoch_col]) if stoch_col in df.columns else 50.0

        return {
            "snapshot": snapshot,
            "trends": {
                "pct_change_50":   round(pct_change_50, 2),
                "pct_change_10":   round(pct_change_10, 2),
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
            "ema": {
                "value":    round(ema_20, 2),
                "price_rel": ema_rel,
                "slope":    ema_slope
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
                "current": round(float(current_row['volume']), 2),
                "avg_50":  round(vol_avg, 2),
                "is_spike": is_spike
            },
            "volatility": {
                "atr": round(atr, 2)
            },
            "stochastic": {
                "current": round(stoch_val, 2)
            }
        }