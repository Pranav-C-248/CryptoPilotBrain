import json
import re
from langchain_ollama import ChatOllama
from src.schema.models import AnalystSignal
from src.core.knowledge_base import TradingKnowledgeBase
from datetime import datetime, timezone, timedelta
import google.generativeai as genai
from dotenv import dotenv_values


class AnalystAgent:
    def __init__(self, knowledge_base: TradingKnowledgeBase, timeframe_hours: int = 4):
        self.kb = knowledge_base
        self.timeframe_hours = timeframe_hours
        config = dotenv_values(".env")
        genai.configure(api_key=config["gemini_key"])
        self.llm = genai.GenerativeModel(
            model_name="gemini-1.5-pro"
        )

    def _detect_regime(self, market_data: dict) -> str:
        structure = market_data['trends']['market_structure'].lower()
        atr       = market_data['volatility']['atr']
        price     = market_data['snapshot']['close']
        vol_spike = market_data['volume']['is_spike']

        # ATR as % of price — crypto ATR is an absolute dollar value
        atr_pct = atr / price if price > 0 else 0

        if "sideways" in structure or "ranging" in structure:
            if atr_pct > 0.02 or vol_spike:   # 2% of price threshold
                return "Volatile"
            return "Sideways"
        elif "trend" in structure or "bull" in structure or "bear" in structure:
            if vol_spike:
                return "Volatile"
            return "Trending"
        return None

    def _compute_valid_till(self) -> str:
        """Deterministically compute valid_till as an ISO UTC string."""
        expiry = datetime.now(timezone.utc) + timedelta(hours=self.timeframe_hours)
        return expiry.strftime("%Y-%m-%dT%H:%M:%SZ")

    def analyze(self, market_data: dict, sentiment_score: float) -> AnalystSignal:
        regime     = self._detect_regime(market_data)
        valid_till = self._compute_valid_till()

        structure   = market_data['trends']['market_structure']
        rsi         = market_data['rsi']['current']
        rsi_label   = "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral momentum"
        vol_spike   = "high volume breakout" if market_data['volume']['is_spike'] else ""
        ema_rel     = market_data['ema']['price_rel'].lower()
        bb_pos      = market_data['bands']['position_label'].lower()
        rsi_trend   = market_data['rsi']['trend'].lower()
        atr         = market_data['volatility']['atr']
        price       = market_data['snapshot']['close']
        atr_pct     = atr / price if price > 0 else 0
        atr_label   = "high volatility atr expansion" if atr_pct > 0.02 else "low volatility consolidation"
        stoch       = market_data['stochastic']['current']
        stoch_label = (
            "stochastic oversold exhaustion" if stoch < 20
            else "stochastic overbought exhaustion" if stoch > 80
            else ""
        )
        regime_str  = regime.lower() if regime else "any"

        query = (
            f"{regime_str} market "
            f"{rsi_label} rsi {rsi_trend} "
            f"{ema_rel} "
            f"{bb_pos} bollinger band "
            f"{atr_label} "
            f"{vol_spike} "
            f"{stoch_label} "
            f"entry confirmation breakout momentum mean reversion scalping"
        ).strip()

        strategies: list[dict] = self.kb.get_relevant_strategies(
            query=query,
            k=3,
            score_threshold=0.30,
            regime_filter=regime
        )

        strategy_context = json.dumps(strategies, indent=2) if strategies else None

        system_msg = """You are an Institutional Grade Quantitative Analyst.

YOUR THINKING PROCESS — follow these steps IN ORDER before producing any output:

STEP 1 — REGIME CHECK:
Identify the market regime (Trending / Volatile / Sideways). State it explicitly.

STEP 2 — STRATEGY SELECTION:
Review each strategy candidate. For each one ask:
  a) Does its regime match the current regime?
  b) Does the current market trigger its entry_primary condition?
  c) Does anything in its conflicts_with list describe the current market? If yes, DISCARD it.
Select the single best-fit strategy. If blending multiple, they must share the same regime and have no conflicts with each other.
If none fit, write "fallback" and use standard institutional logic.

STEP 3 — CONFLUENCE CHECK:
List every entry_confirmation condition from the chosen strategy.
For each condition: does the current market data satisfy it? YES or NO.
Count how many are satisfied.

STEP 4 — SIGNAL DECISION:
- If ADX is not present in the market data, infer non-trending confirmation from market_structure 
  being "Sideways/Ranging" combined with low ATR relative to price. Do not reject a strategy 
  solely because ADX is absent.
- All or most confirmations met → consider BUY or SELL
- Fewer than half met → HOLD
- Any hard conflict present → HOLD


STEP 5 — SYNTHESIZE CONDITIONS:
Based on the chosen strategy and your confluence analysis, write:
  - entry_condition: A readable if-condition string describing exactly when to enter.
    Use only indicators present in the market data.
    Format example: "rsi < 30 and price < bb_lower and volume_spike = true"
  - exit_condition: A readable if-condition string describing when to exit.
    Format example: "rsi > 70 or price < stop_loss or price > target"
  If you blended two strategies, merge their conditions logically with AND/OR.
  Keep both strings concise and machine-readable.

STEP 6 — WRITE YOUR OUTPUT:
Populate the JSON fields in this exact order:
  1. internal_monologue: Your full Step 1-4 reasoning written out. Must cite at least 3 specific metric values.
  2. strategy_used: The name of the chosen strategy, or "fallback".
  3. reasoning: A concise 2-3 sentence summary of why you chose this signal.
  4. entry_condition: The string you wrote in Step 5.
  5. exit_condition: The string you wrote in Step 5.
  6. signal: BUY, SELL, or HOLD — must be consistent with your internal_monologue.
  7. confidence: A float 0.0-1.0 reflecting how many confirmations were satisfied.
  8. valid_till: Copy this value exactly as given: {VALID_TILL_PLACEHOLDER}

RULES:
- Never assign confidence > 0.8 unless ALL entry_confirmation conditions are met.
- Never assign confidence > 0.5 for a Sideways regime.
- signal must always be derivable from internal_monologue. If they contradict, fix the signal.
- entry_condition and exit_condition must only reference indicators visible in the market data packet.
- valid_till must be copied exactly — do not modify it.
- Output valid JSON only. No markdown fences."""

        # Inject the computed valid_till into the system prompt
        system_msg = system_msg.replace("{VALID_TILL_PLACEHOLDER}", valid_till)

        user_msg = f"""--- MARKET CONTEXT PACKET ---
SNAPSHOT: {market_data['snapshot']}
STRUCTURE: {market_data['trends']['market_structure']} (50-Day: {market_data['trends']['pct_change_50']}%, 10-Day: {market_data['trends']['pct_change_10']}%)

TECHNICAL OVERLAYS:
- RSI: Current {market_data['rsi']['current']} | Avg_50: {market_data['rsi']['avg_50']} | Trend: {market_data['rsi']['trend']}
- EMA(20): Value {market_data['ema']['value']} | Slope: {market_data['ema']['slope']} | Relation: {market_data['ema']['price_rel']}
- BOLLINGER: Pos: {market_data['bands']['position_label']} (U: {market_data['bands']['upper']}, L: {market_data['bands']['lower']})
- LEVELS: 50-Candle High: {market_data['levels']['high_50']} | Low: {market_data['levels']['low_50']}

LIQUIDITY & VOLATILITY:
- VOLUME: Current {market_data['volume']['current']} | Spike: {market_data['volume']['is_spike']} (Avg: {market_data['volume']['avg_50']})
- ATR: {market_data['volatility']['atr']}

SENTIMENT: {sentiment_score} (0.5=Neutral)

--- STRATEGY CANDIDATES ---
{strategy_context if strategy_context else "No strategies retrieved. Use standard institutional mean-reversion or trend-following logic."}

Now follow Steps 1 through 6 from your instructions and produce the JSON output."""

        full_prompt = f"""
                {system_msg}

                {user_msg}
                """

        response = self.llm.generate_content(
            full_prompt,
            generation_config={
                "temperature": 0,
                "response_mime_type": "application/json"
            }
        )

        raw_output = response.text
        clean_content = self._clean_json_response(raw_output)

        try:
            # Parse JSON into dict first
            parsed = json.loads(clean_content)

            # Inject asset_name from market_data
            parsed["asset_name"] = market_data.get("asset_name")

            # Validate with Pydantic model
            return AnalystSignal.model_validate(parsed)

        except Exception as e:
            print(f"--- DEBUG: LLM Output was: {clean_content} ---")
            raise e

    def _clean_json_response(self, content: str) -> str:
        content = content.strip()

        # Remove markdown if present
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
        if match:
            content = match.group(1)

        # Remove trailing commas (Gemini sometimes does this)
        content = re.sub(r",\s*}", "}", content)
        content = re.sub(r",\s*]", "]", content)

        return content.strip()

import pandas as pd
import numpy as np

class MarketDataProcessor:
    @staticmethod
    def get_context_packet(df, current_idx, window=50):
        """
        Extracts and calculates all 9 points of context from a dataframe slice.
        """
        # Ensure we have enough history
        if current_idx < window:
            return None
        
        # Get the current row and the window slice
        current_row = df.iloc[current_idx]
        history_50 = df.iloc[current_idx - window + 1 : current_idx + 1]
        history_10 = df.iloc[current_idx - 10 + 1 : current_idx + 1]

        # 1. Snapshot
        snapshot = {
            "open": current_row['open'],
            "high": current_row['high'],
            "low": current_row['low'],
            "close": current_row['close']
        }

        # 2. Trends
        pct_change_50 = ((current_row['close'] - history_50.iloc[0]['close']) / history_50.iloc[0]['close']) * 100
        pct_change_10 = ((current_row['close'] - history_10.iloc[0]['close']) / history_10.iloc[0]['close']) * 100
        
        # Market Structure Logic
        # Simplification: Higher Highs/Lows vs Lower Highs/Lows
        if current_row['high'] > history_10['high'].mean() and current_row['low'] > history_10['low'].mean():
            structure = "Uptrend"
        elif current_row['low'] < history_10['low'].mean() and current_row['high'] < history_10['high'].mean():
            structure = "Downtrend"
        else:
            structure = "Sideways/Ranging"

        # 3. RSI Metrics
        rsi_avg = history_50['rsi'].mean()
        rsi_slope = "Increasing" if current_row['rsi'] > history_10['rsi'].mean() else "Decreasing"

        # 4. Bollinger Bands
        bb_upper = current_row['bb_upper']
        bb_lower = current_row['bb_lower']
        price = current_row['close']
        
        if price > bb_upper: bb_pos = "Above Upper Band"
        elif price < bb_lower: bb_pos = "Below Lower Band"
        elif price > (bb_upper + bb_lower)/2: bb_pos = "Upper Half"
        else: bb_pos = "Lower Half"

        # 5. EMA 20
        ema_20 = current_row['ema20']
        ema_slope = "Increasing" if ema_20 > history_10['ema20'].mean() else "Decreasing"
        ema_rel = "Price Above EMA" if price > ema_20 else "Price Below EMA"

        # 6. Levels
        high_50 = history_50['high'].max()
        low_50 = history_50['low'].min()

        # 7. Volume
        vol_avg = history_50['volume'].mean()
        is_spike = current_row['volume'] > (vol_avg * 1.5)

        # 8. Volatility
        # Using your existing volatility column (usually Standard Deviation or ATR)
        curr_volatility = current_row['volatility']

        # Construct the final Packet
        return {
            "snapshot": snapshot,
            "trends": {
                "pct_change_50": round(pct_change_50, 2),
                "pct_change_10": round(pct_change_10, 2),
                "market_structure": structure
            },
            "rsi": {
                "current": round(current_row['rsi'], 2),
                "avg_50": round(rsi_avg, 2),
                "trend": rsi_slope
            },
            "bands": {
                "upper": round(bb_upper, 2),
                "lower": round(bb_lower, 2),
                "position_label": bb_pos
            },
            "ema": {
                "value": round(ema_20, 2),
                "price_rel": ema_rel,
                "slope": ema_slope
            },
            "levels": {
                "high_50": round(high_50, 2),
                "low_50": round(low_50, 2)
            },
            "volume": {
                "current": round(current_row['volume'], 2),
                "avg_50": round(vol_avg, 2),
                "is_spike": is_spike
            },
            "volatility": {
                "atr": round(curr_volatility, 2) # Mapping your vol to ATR field
            }
        }