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
                f"*** RSI(14) was oversold last candle ({rsi_prev}) and has now exited "
                f"the oversold zone ({rsi_current}). Oversold EXIT confirmed. ***"
            )
        elif rsi_prev is not None and rsi_prev > 70 and rsi_current < 70:
            narrative = (
                f"*** RSI(14) was overbought last candle ({rsi_prev}) and has now exited "
                f"the overbought zone ({rsi_current}). Overbought EXIT confirmed. ***"
            )
        else:
            narrative = ""

        return rsi_current, rsi_prev, rsi_avg, rsi_trend, narrative

    def _build_rsi5_narrative(self, market_data: dict) -> tuple:
        rsi5_current = market_data['rsi5']['current']
        rsi5_prev    = market_data['rsi5']['prev']
        rsi5_trend   = market_data['rsi5']['trend']

        if rsi5_prev < 30 and rsi5_current > 30:
            narrative = (
                f"*** RSI(5) was oversold last candle ({rsi5_prev}) and has now exited "
                f"the oversold zone ({rsi5_current}). Hyper-sensitive oversold EXIT confirmed. ***"
            )
        elif rsi5_prev > 70 and rsi5_current < 70:
            narrative = (
                f"*** RSI(5) was overbought last candle ({rsi5_prev}) and has now exited "
                f"the overbought zone ({rsi5_current}). Hyper-sensitive overbought EXIT confirmed. ***"
            )
        else:
            narrative = ""

        return rsi5_current, rsi5_prev, rsi5_trend, narrative

    def _build_bb_narrative(self, market_data: dict) -> str:
        close    = market_data['snapshot']['close']
        bb_lower = market_data['bands']['lower']
        bb_upper = market_data['bands']['upper']
        bb_mid   = market_data['bands']['mid']

        if close <= bb_lower * 1.01:
            return (
                f"*** Price ({close}) is at or near BB(20) Lower ({bb_lower}) — "
                f"potential bounce zone. Band breach/touch confirmed. ***"
            )
        elif close >= bb_upper * 0.99:
            return (
                f"*** Price ({close}) is at or near BB(20) Upper ({bb_upper}) — "
                f"potential rejection zone. Band breach/touch confirmed. ***"
            )
        else:
            return (
                f"Price ({close}) is inside BB(20) bands "
                f"(L: {bb_lower} | Mid: {bb_mid} | U: {bb_upper})."
            )

    def _build_bb40_narrative(self, market_data: dict) -> str:
        close      = market_data['snapshot']['close']
        bb40_lower = market_data['bands40']['lower']
        bb40_upper = market_data['bands40']['upper']

        if close <= bb40_lower * 1.01:
            return (
                f"*** Price ({close}) is at or near BB(40) Lower ({bb40_lower}) — "
                f"wide-band statistical extreme confirmed. ***"
            )
        elif close >= bb40_upper * 0.99:
            return (
                f"*** Price ({close}) is at or near BB(40) Upper ({bb40_upper}) — "
                f"wide-band statistical extreme confirmed. ***"
            )
        else:
            return (
                f"Price ({close}) is inside BB(40) bands "
                f"(L: {bb40_lower} | U: {bb40_upper})."
            )

    def _build_query(self, market_data: dict, regime_str: str) -> str:
        price        = market_data['snapshot']['close']

        # ── 1. Regime phrase ──────────────────────────────────────────────────
        regime_phrases = {
            "trending": "macro trend extension with directional momentum and sustained institutional capital flow",
            "sideways": "range-bound oscillation with no directional momentum predictably cycling between statistical boundaries",
            "volatile": "kinetic energy release from compression with explosive liquidity cascade and elevated participation",
        }
        regime_phrase = regime_phrases.get(
            regime_str,
            "indeterminate market structure with mixed momentum signals"
        )

        # ── 2. Momentum phrase ────────────────────────────────────────────────
        rsi_current  = market_data['rsi']['current']
        rsi5_current = market_data['rsi5']['current']
        rsi5_prev    = market_data['rsi5']['prev']
        bb_pos       = market_data['bands']['position_label'].lower()
        bb40_pos     = market_data['bands40']['position_label'].lower()

        if rsi_current > 70:
            rsi14_phrase = "overbought momentum exhaustion approaching reversal threshold"
        elif rsi_current < 30:
            rsi14_phrase = "oversold fear-driven retail capitulation at statistical extreme"
        else:
            rsi14_phrase = "neutral momentum with no extreme oscillator reading"

        if rsi5_prev < 30 and rsi5_current > 30:
            rsi5_phrase = "hyper-sensitive oscillator exiting oversold zone confirming snap-back momentum"
        elif rsi5_prev > 70 and rsi5_current < 70:
            rsi5_phrase = "hyper-sensitive oscillator exiting overbought zone confirming exhaustion reversal"
        elif rsi5_current < 30:
            rsi5_phrase = "hyper-sensitive oscillator deep in oversold extreme"
        elif rsi5_current > 70:
            rsi5_phrase = "hyper-sensitive oscillator deep in overbought extreme"
        else:
            rsi5_phrase = ""

        bb20_phrases = {
            "below lower band": "price at maximum negative standard deviation volatility overextension bounce zone",
            "above upper band": "price at maximum positive standard deviation volatility overextension rejection zone",
            "upper half":       "price consolidating in upper bollinger half above mean",
            "lower half":       "price consolidating in lower bollinger half below mean",
        }
        bb40_phrases = {
            "below lower band": "price breaching wide-band lower boundary confirming statistical extreme in non-trending environment",
            "above upper band": "price breaching wide-band upper boundary confirming statistical extreme in non-trending environment",
            "upper half":       "price in upper half of wide statistical range",
            "lower half":       "price in lower half of wide statistical range",
        }
        bb20_phrase = bb20_phrases.get(bb_pos, "")
        bb40_phrase = bb40_phrases.get(bb40_pos, "")

        momentum_phrase = " ".join(filter(None, [
            rsi14_phrase, rsi5_phrase, bb20_phrase, bb40_phrase
        ]))

        # ── 3. Volatility + volume phrase ─────────────────────────────────────
        atr_expanding   = market_data['volatility']['atr_expanding']
        is_spike        = market_data['volume']['is_spike']
        is_capitulation = market_data['volume']['is_capitulation_spike']

        if atr_expanding and is_capitulation:
            vol_phrase = "ATR expansion confirmed with capitulation volume spike institutional accumulation demand zone"
        elif atr_expanding and is_spike:
            vol_phrase = "ATR expansion with high volume breakout surge in market participation"
        elif atr_expanding:
            vol_phrase = "ATR expanding increasing volatility momentum building"
        elif is_spike:
            vol_phrase = "volume spike with stable ATR possible absorption at structural boundary"
        else:
            vol_phrase = "low volatility consolidation compression ATR contraction tight range"

        # ── 4. EMA state phrase ───────────────────────────────────────────────
        ema_rel      = market_data['ema']['price_rel'].lower()
        ema_slope    = market_data['ema']['slope'].lower()
        converging   = market_data['ema_multi']['converging']
        stack_order  = market_data['ema_multi']['stack_order'].lower()
        ema200_rel   = market_data['ema200']['price_rel'].lower()
        ema200_near  = market_data['ema200']['near_band']
        ema200_slope = market_data['ema200']['slope'].lower()

        if "above" in ema_rel and ema_slope == "increasing":
            ema20_phrase = "price above rising moving average confirming bullish trend alignment"
        elif "below" in ema_rel and ema_slope == "decreasing":
            ema20_phrase = "price below declining moving average confirming bearish trend alignment"
        elif "above" in ema_rel and ema_slope == "decreasing":
            ema20_phrase = "price above but moving average curling downward early trend deterioration"
        else:
            ema20_phrase = "price below but moving average flattening potential base formation"

        if converging and stack_order == "bullish":
            ema_multi_phrase = "fast medium slow moving averages converging and re-stacking bullish synchronized algorithmic launchpad trend continuation"
        elif converging and stack_order == "bearish":
            ema_multi_phrase = "fast medium slow moving averages converging and re-stacking bearish synchronized algorithmic launchpad trend continuation"
        elif converging:
            ema_multi_phrase = "moving averages in equilibrium convergence twist consolidation fair value alignment"
        elif stack_order == "bullish":
            ema_multi_phrase = "moving averages fanned out in bullish order mature uptrend"
        elif stack_order == "bearish":
            ema_multi_phrase = "moving averages fanned out in bearish order mature downtrend"
        else:
            ema_multi_phrase = "moving averages mixed order choppy non-directional"

        if ema200_near and ema200_slope == "increasing" and "below" in ema200_rel:
            ema200_phrase = "price approaching upward sloping 200 moving average maximum institutional demand smart money accumulation zone"
        elif ema200_near and ema200_slope == "increasing" and "above" in ema200_rel:
            ema200_phrase = "price near upward sloping 200 moving average long-term fair value support"
        elif "above" in ema200_rel and ema200_slope == "increasing":
            ema200_phrase = "price above rising 200 moving average macro bull regime confirmed"
        else:
            ema200_phrase = "price below or distant from 200 moving average macro trend not confirmed"

        ema_phrase = " ".join(filter(None, [
            ema20_phrase, ema_multi_phrase, ema200_phrase
        ]))

        # ── Assemble ──────────────────────────────────────────────────────────
        return (
            f"{regime_phrase}. "
            f"{momentum_phrase}. "
            f"{vol_phrase}. "
            f"{ema_phrase}."
        )

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
        regime_str = regime.lower() if regime else "any"

        # Narratives
        rsi_current, rsi_prev, rsi_avg, rsi_trend, rsi_narrative   = self._build_rsi_narrative(market_data)
        rsi5_current, rsi5_prev, rsi5_trend, rsi5_narrative         = self._build_rsi5_narrative(market_data)
        bb_narrative                                                 = self._build_bb_narrative(market_data)
        bb40_narrative                                               = self._build_bb40_narrative(market_data)

        # Query
        query = self._build_query(market_data, regime_str)

        # ── KB retrieval ──────────────────────────────────────────────────────
        strategies: list[dict] = self.kb.get_relevant_strategies(
            query=query,
            k=3,
            score_threshold=0.30,
            regime_filter=regime
        )
        strategy_context = json.dumps(strategies, indent=2) if strategies else None

        # ── Packet field extraction ───────────────────────────────────────────
        levels  = market_data.get('levels', {})
        volume  = market_data.get('volume', {})
        ema     = market_data.get('ema', {})
        ema_m   = market_data.get('ema_multi', {})
        ema200  = market_data.get('ema200', {})
        bands   = market_data.get('bands', {})
        bands40 = market_data.get('bands40', {})
        adx     = market_data.get('adx', {})
        vol     = market_data.get('volatility', {})     
        
        # ── Prompts ───────────────────────────────────────────────────────────
        system_msg = """You are an Institutional Grade Quantitative Analyst.

YOUR THINKING PROCESS — follow these steps IN ORDER before producing any output:

STEP 1 — REGIME CHECK:
Identify the market regime (Trending / Volatile / Sideways). State it explicitly.
Base this on market_structure, ATR/Price%, and volume spike flag.

STEP 2 — STRATEGY SELECTION:
Review each strategy candidate from the STRATEGY CANDIDATES section.
For each one ask:
a) Does its regime tag match the current regime?
b) Does the current market satisfy its entry_primary condition?
c) Does anything in its conflicts_with list describe the current market? If yes, DISCARD it.
Select the single best-fit strategy. If none fit cleanly, write "fallback".

STEP 3 — CONFLUENCE CHECK:
List every entry_confirmation condition from the chosen strategy.
For each condition state explicitly: does the current packet data satisfy it? YES or NO.
Count satisfied conditions vs total. Be precise — cite actual values from the packet.

STEP 4 — SIGNAL DECISION:
- All or most confirmations met → consider BUY or SELL
- Fewer than half met → HOLD
- Any hard conflict present → HOLD
- Sideways regime → HOLD unless confluence is unambiguous and strong

STEP 5 — SYNTHESIZE CONDITIONS:
Write:
- entry_condition: a readable if-statement using only indicator names visible in the packet.
    Example format: "rsi_prev < 30 and rsi_current > 30 and price <= bb_lower * 1.01"
- exit_condition: a readable if-statement for when to exit the trade.
    Example format: "price < stop_loss or price > take_profit_target"

STEP 6 — WRITE YOUR OUTPUT:
Populate JSON in this exact order:
1. internal_monologue — full Step 1–4 reasoning, cite at least 4 metric values from the packet
2. strategy_used      — exact strategy name from the candidates, or "fallback"
3. reasoning          — concise 2–3 sentence summary of why this signal was chosen
4. entry_condition    — from Step 5
5. exit_condition     — from Step 5
6. signal             — BUY, SELL, or HOLD
7. confidence         — float 0.0–1.0

RULES:
- confidence > 0.8 only if ALL entry_confirmation conditions are met
- confidence > 0.6 only if the majority of entry_confirmation conditions are met
- signal must be directly derivable from internal_monologue — no contradictions
- Do not favour or penalise any strategy based on its name — evaluate purely on conditions
- Output valid JSON only. No markdown fences. No commentary outside the JSON."""

        user_msg = f"""ASSET: {asset_name}
VALID_TILL: {valid_till}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MARKET CONTEXT PACKET
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[SNAPSHOT]
Open: {market_data['snapshot']['open']} | High: {market_data['snapshot']['high']} | Low: {market_data['snapshot']['low']} | Close: {price}

[STRUCTURE]
Market Structure : {structure}
50-Candle Change : {market_data['trends']['pct_change_50']}%
10-Candle Change : {market_data['trends']['pct_change_10']}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MEAN REVERSION INDICATORS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[RSI — 14 Period]
Current : {rsi_current} | Prev: {rsi_prev} | 50-Avg: {rsi_avg} | Trend: {rsi_trend}
{rsi_narrative}

[RSI — 5 Period]
Current : {rsi5_current} | Prev: {rsi5_prev} | Trend: {rsi5_trend}
{rsi5_narrative}

[BOLLINGER BANDS — 20 Period, 2σ]
Position : {bands.get('position_label', 'N/A')} | Upper: {bands.get('upper', 'N/A')} | Mid: {bands.get('mid', 'N/A')} | Lower: {bands.get('lower', 'N/A')}
{bb_narrative}

[BOLLINGER BANDS — 40 Period, 2σ]
Position : {bands40.get('position_label', 'N/A')} | Upper: {bands40.get('upper', 'N/A')} | Lower: {bands40.get('lower', 'N/A')}
{bb40_narrative}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TREND STRENGTH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[ADX — 14 Period]
Value        : {adx.get('value', 'N/A')} | Non-Trending (ADX < 25): {adx.get('non_trending', 'N/A')}

[VOLATILITY]
ATR          : {atr} | ATR/Price: {round(atr_pct * 100, 3)}% ({'high — expansion confirmed' if atr_pct > 0.02 else 'low — compression'})
ATR Avg(5)   : {vol.get('atr_avg_5', 'N/A')} | Expanding: {vol.get('atr_expanding', 'N/A')}

[VOLUME]
Current      : {volume.get('current', 'N/A')} | 50-Avg: {volume.get('avg_50', 'N/A')} | 20-Avg: {volume.get('avg_20', 'N/A')}
Spike (1.5x) : {volume.get('is_spike', 'N/A')}
Capitulation (2x 20-Avg): {volume.get('is_capitulation_spike', 'N/A')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EMA STATE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[EMA — 20 Period  |  Primary Anchor]
Value        : {ema.get('value', 'N/A')} | Slope: {ema.get('slope', 'N/A')} | Price Relation: {ema.get('price_rel', 'N/A')}

[EMA MULTI — 9 / 20 / 50 Period]
EMA9         : {ema_m.get('ema9',  'N/A')} | Slope: {ema_m.get('ema9_slope',  'N/A')}
EMA20        : {ema_m.get('ema20', 'N/A')}
EMA50        : {ema_m.get('ema50', 'N/A')} | Slope: {ema_m.get('ema50_slope', 'N/A')}
Converging   : {ema_m.get('converging', 'N/A')} | Spread: {ema_m.get('spread_pct', 'N/A')}% | Stack Order: {ema_m.get('stack_order', 'N/A')}

[EMA — 200 Period  |  Macro Anchor]
Value        : {ema200.get('value', 'N/A')} | Slope: {ema200.get('slope', 'N/A')}
Price Rel    : {ema200.get('price_rel', 'N/A')} | Proximity: {ema200.get('proximity_pct', 'N/A')}% | Near Band (±2%): {ema200.get('near_band', 'N/A')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TREND FOLLOWING LEVELS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[DONCHIAN CHANNELS]
10-Candle    : High {levels.get('high_10', 'N/A')} | Low {levels.get('low_10', 'N/A')}
20-Candle    : High {levels.get('high_20', 'N/A')} | Low {levels.get('low_20', 'N/A')}
50-Candle    : High {levels.get('high_50', 'N/A')} | Low {levels.get('low_50', 'N/A')}
55-Candle    : High {levels.get('high_55', 'N/A')} | Low {levels.get('low_55', 'N/A')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MARKET SENTIMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Sentiment Score: {sentiment_score} (0.0=Extreme Fear | 0.5=Neutral | 1.0=Extreme Greed)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRATEGY CANDIDATES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{strategy_context if strategy_context else "No strategies retrieved. Apply standard institutional logic based on regime and confluence of indicators above."}

Now follow Steps 1 through 6 from your instructions and produce the JSON output."""
        # ── LLM call ──────────────────────────────────────────────────────────
        try:
            # temporarily disabled for testing purposes
            raise Exception("Gemini API Disabled")
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

