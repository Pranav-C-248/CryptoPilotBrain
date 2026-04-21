import json
from datetime import datetime, timezone, timedelta
from dotenv import dotenv_values
from google import genai
from google.genai import types

from src.schema.models import AnalystSignal
from src.core.knowledge_base import TradingKnowledgeBase

class AnalystAgent:
    def __init__(self, knowledge_base: TradingKnowledgeBase, timeframe_hours: int = 4):
        self.kb = knowledge_base
        self.timeframe_hours = timeframe_hours
        
        config = dotenv_values(".env")
        
        # New SDK initialization
        self.client = genai.Client(api_key=config["gemini_key"])
        
        # 2.5-flash is the current standard for the free tier
        self.model_name = "gemini-2.5-flash"

    def _detect_regime(self, market_data: dict) -> str:
        structure = market_data['trends']['market_structure'].lower()
        atr       = market_data['volatility']['atr']
        price     = market_data['snapshot']['close']
        vol_spike = market_data['volume']['is_spike']

        atr_pct = atr / price if price > 0 else 0

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
        structure    = market_data['trends']['market_structure']
        atr          = market_data['volatility']['atr']
        price        = market_data['snapshot']['close']
        atr_pct      = atr / price if price > 0 else 0
        stoch        = market_data['stochastic']['current']
        regime_str   = regime.lower() if regime else "any"

        rsi_current, rsi_prev, rsi_avg, rsi_trend, rsi_narrative = self._build_rsi_narrative(market_data)
        bb_narrative = self._build_bb_narrative(market_data)

        rsi_label  = "overbought" if rsi_current > 70 else "oversold" if rsi_current < 30 else "neutral momentum"
        vol_spike  = "high volume breakout" if market_data['volume']['is_spike'] else ""
        ema_rel    = market_data['ema']['price_rel'].lower()
        bb_pos     = market_data['bands']['position_label'].lower()
        atr_label  = "high volatility atr expansion" if atr_pct > 0.02 else "low volatility consolidation"
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
    - LEVELS: 50-Candle High: {market_data['levels']['high_50']} | Low: {market_data['levels']['low_50']}

    LIQUIDITY & VOLATILITY:
    - VOLUME: Current {market_data['volume']['current']} | Spike: {market_data['volume']['is_spike']} (Avg: {market_data['volume']['avg_50']})
    - ATR: {atr} | ATR/Price: {round(atr_pct * 100, 3)}% ({'high — volatile' if atr_pct > 0.02 else 'low — non-trending confirmed'})

    SENTIMENT: {sentiment_score} (0.5=Neutral)

    --- STRATEGY CANDIDATES ---
    {strategy_context if strategy_context else "No strategies retrieved. Use standard institutional mean-reversion or trend-following logic."}

    Now follow Steps 1 through 6 from your instructions and produce the JSON output."""

        # ── LLM call ──────────────────────────────────────────────────────────
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=user_msg,
            config=types.GenerateContentConfig(
                system_instruction=system_msg,
                temperature=0,
                response_mime_type="application/json"
            )
        )

        try:
            parsed = json.loads(response.text)
            # Python owns these fields — always overwrite regardless of LLM output
            parsed["asset_name"] = asset_name
            parsed["valid_till"] = valid_till    # ← THIS WAS MISSING
            return AnalystSignal.model_validate(parsed)
        except Exception as e:
            print(f"--- DEBUG: LLM Output was: {response.text} ---")
            raise e
    
