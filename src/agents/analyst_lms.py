import json
import re
from datetime import datetime, timezone, timedelta
from dotenv import dotenv_values
from google import genai
from google.genai import types
from langchain_openai import ChatOpenAI
from src.schema.models import AnalystSignal
from src.core.knowledge_base_lms import TradingKnowledgeBase


class AnalystAgent:
    def __init__(self, knowledge_base: TradingKnowledgeBase, timeframe_hours: int = 4):
        self.kb = knowledge_base
        self.timeframe_hours = timeframe_hours

        config = dotenv_values(".env")
        self.client = genai.Client(api_key=config["gemini_key"])
        self.model_name = "gemini-2.5-flash"
        self.lmstudio_llm = ChatOpenAI(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            model="gemma4:e2b",
            temperature=0,
        )

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
            k=2,
            score_threshold=0.40,
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
        system_msg = """<role>You are an Institutional Grade Quantitative Analyst.</role>

<instructions>
Follow these steps IN ORDER before producing any output.

<step n="1" name="REGIME_CHECK">
Identify the market regime. It will be provided as a pre-computed value in the data packet.
Verify it against market_structure, ATR/Price%, and volume spike flag.
State the regime explicitly: Trending, Volatile, or Sideways.
</step>

<step n="2" name="STRATEGY_SELECTION">
Review each strategy candidate from the strategy_candidates section.
For each one ask:
  a) REGIME CHECK — the strategy's regime field has three sub-fields:
       - primary     : the regime this strategy is designed for
       - also_valid  : additional regimes where it is still valid
       - forbidden   : regimes where it must NEVER be used
     The strategy is eligible if the current regime matches primary OR appears in also_valid.
     If the current regime appears in forbidden, DISCARD the strategy immediately.
  b) Does the current market satisfy its entry_primary condition?
  c) CONFLICT CHECK — evaluate each expression in conflicts_structured against the current
     packet values (e.g. "atr_expanding == false" means check the atr_expanding field directly).
     If ANY expression evaluates to True for the current market, DISCARD the strategy.
Select the single best-fit strategy. If none fit cleanly, write "fallback".
</step>

<step n="3" name="CONFLUENCE_CHECK">
List every entry_confirmation condition from the chosen strategy.
For each condition:
  - If entry_confirmation_structured is not null for that condition, evaluate the structured
    expression directly against the current packet values and state YES or NO with the value cited.
  - If entry_confirmation_structured is null for that condition, evaluate the prose description
    using the packet data as best you can and state YES or NO with your reasoning.
Count satisfied conditions vs total. Be precise — cite actual numeric values from the packet.
</step>

<step n="4" name="SIGNAL_DECISION">
Decision rules:
- All or most confirmations met → consider BUY
- Fewer than half met → HOLD
- Any hard conflict present → HOLD
- Sideways regime → HOLD unless confluence is unambiguous and strong
</step>

<step n="5" name="SYNTHESIZE_CONDITIONS">
Write entry_condition and exit_condition as pure logical mathematical expressions.
CRITICAL: DO NOT write conversational English, explanations, or sentences in these fields. If you cannot form a valid mathematical condition, output exactly "False" or "HOLD".
You MUST use ONLY the following variable names (these are the exact column names available at execution time):

<available_variables>
price, open, high, low, close, volume,
rsi, rsi5, ema9, ema20, ema50, ema200,
bb_upper, bb_lower, bb_mid, bb40_upper, bb40_lower,
adx, atr, atr_avg_5, atr_expanding,
high_10, low_10, high_20, low_20, high_50, low_50, high_55, low_55
</available_variables>

To reference the previous candle's value, the backtester does NOT have _prev columns.
So use comparisons against thresholds only (e.g. "rsi < 30" not "rsi_prev < 30").

<formatting_rules>
- Use standard Python operators: <, >, <=, >=, ==, and, or
- Example format only (DO NOT COPY THIS EXACT LOGIC): <entry_condition>price > ema20 and adx > 25</entry_condition>
- You MUST derive the exact conditions directly from the specific rules of the chosen strategy!
</formatting_rules>
</step>

<step n="6" name="WRITE_OUTPUT">
Produce a JSON object with exactly these 7 keys in this order.
Do NOT include asset_name or valid_till — they are injected automatically.

<output_schema>
{
  "internal_monologue": "Full Step 1-4 reasoning. Cite at least 4 metric values from the packet.",
  "strategy_used": "Exact strategy name from the candidates, or fallback",
  "reasoning": "Concise 2-3 sentence summary of why this signal was chosen",
  "entry_condition": "From Step 5 — uses only available_variables",
  "exit_condition": "From Step 5 — uses only available_variables",
  "signal": "BUY or HOLD",
  "confidence": 0.0
}
</output_schema>
</step>
</instructions>

<rules>
- confidence > 0.8 only if ALL entry_confirmation conditions are met
- confidence > 0.6 only if the majority of entry_confirmation conditions are met
- signal must be directly derivable from internal_monologue — no contradictions
- Do not favour or penalise any strategy based on its name — evaluate purely on conditions
- Output valid JSON only. No markdown fences. No commentary outside the JSON.
</rules>"""

        user_msg = f"""<market_data asset="{asset_name}" valid_till="{valid_till}">

<regime detected="{regime_str}" />

<snapshot>
  <open>{market_data['snapshot']['open']}</open>
  <high>{market_data['snapshot']['high']}</high>
  <low>{market_data['snapshot']['low']}</low>
  <close>{price}</close>
</snapshot>

<structure>
  <market_structure>{structure}</market_structure>
  <pct_change_50>{market_data['trends']['pct_change_50']}%</pct_change_50>
  <pct_change_10>{market_data['trends']['pct_change_10']}%</pct_change_10>
</structure>

<indicators type="mean_reversion">
  <rsi period="14">
    <current>{rsi_current}</current>
    <prev>{rsi_prev}</prev>
    <avg_50>{rsi_avg}</avg_50>
    <trend>{rsi_trend}</trend>
    <narrative>{rsi_narrative}</narrative>
  </rsi>

  <rsi period="5">
    <current>{rsi5_current}</current>
    <prev>{rsi5_prev}</prev>
    <trend>{rsi5_trend}</trend>
    <narrative>{rsi5_narrative}</narrative>
  </rsi>

  <bollinger_bands period="20" std="2">
    <position>{bands.get('position_label', 'N/A')}</position>
    <upper>{bands.get('upper', 'N/A')}</upper>
    <mid>{bands.get('mid', 'N/A')}</mid>
    <lower>{bands.get('lower', 'N/A')}</lower>
    <narrative>{bb_narrative}</narrative>
  </bollinger_bands>

  <bollinger_bands period="40" std="2">
    <position>{bands40.get('position_label', 'N/A')}</position>
    <upper>{bands40.get('upper', 'N/A')}</upper>
    <lower>{bands40.get('lower', 'N/A')}</lower>
    <narrative>{bb40_narrative}</narrative>
  </bollinger_bands>
</indicators>

<indicators type="trend_strength">
  <adx period="14">
    <value>{adx.get('value', 'N/A')}</value>
    <non_trending>{adx.get('non_trending', 'N/A')}</non_trending>
  </adx>

  <volatility>
    <atr>{atr}</atr>
    <atr_price_pct>{round(atr_pct * 100, 3)}%</atr_price_pct>
    <atr_interpretation>{'high — expansion confirmed' if atr_pct > 0.02 else 'low — compression'}</atr_interpretation>
    <atr_avg_5>{vol.get('atr_avg_5', 'N/A')}</atr_avg_5>
    <atr_expanding>{vol.get('atr_expanding', 'N/A')}</atr_expanding>
  </volatility>

  <volume>
    <current>{volume.get('current', 'N/A')}</current>
    <avg_50>{volume.get('avg_50', 'N/A')}</avg_50>
    <avg_20>{volume.get('avg_20', 'N/A')}</avg_20>
    <is_spike_1_5x>{volume.get('is_spike', 'N/A')}</is_spike_1_5x>
    <is_capitulation_2x>{volume.get('is_capitulation_spike', 'N/A')}</is_capitulation_2x>
  </volume>
</indicators>

<indicators type="ema_state">
  <ema period="20" role="primary_anchor">
    <value>{ema.get('value', 'N/A')}</value>
    <slope>{ema.get('slope', 'N/A')}</slope>
    <price_rel>{ema.get('price_rel', 'N/A')}</price_rel>
  </ema>

  <ema_multi periods="9/20/50">
    <ema9>{ema_m.get('ema9', 'N/A')}</ema9>
    <ema9_slope>{ema_m.get('ema9_slope', 'N/A')}</ema9_slope>
    <ema20>{ema_m.get('ema20', 'N/A')}</ema20>
    <ema50>{ema_m.get('ema50', 'N/A')}</ema50>
    <ema50_slope>{ema_m.get('ema50_slope', 'N/A')}</ema50_slope>
    <converging>{ema_m.get('converging', 'N/A')}</converging>
    <spread_pct>{ema_m.get('spread_pct', 'N/A')}%</spread_pct>
    <stack_order>{ema_m.get('stack_order', 'N/A')}</stack_order>
  </ema_multi>

  <ema period="200" role="macro_anchor">
    <value>{ema200.get('value', 'N/A')}</value>
    <slope>{ema200.get('slope', 'N/A')}</slope>
    <price_rel>{ema200.get('price_rel', 'N/A')}</price_rel>
    <proximity_pct>{ema200.get('proximity_pct', 'N/A')}%</proximity_pct>
    <near_band_2pct>{ema200.get('near_band', 'N/A')}</near_band_2pct>
  </ema>
</indicators>

<donchian_channels>
  <channel period="10"><high>{levels.get('high_10', 'N/A')}</high><low>{levels.get('low_10', 'N/A')}</low></channel>
  <channel period="20"><high>{levels.get('high_20', 'N/A')}</high><low>{levels.get('low_20', 'N/A')}</low></channel>
  <channel period="50"><high>{levels.get('high_50', 'N/A')}</high><low>{levels.get('low_50', 'N/A')}</low></channel>
  <channel period="55"><high>{levels.get('high_55', 'N/A')}</high><low>{levels.get('low_55', 'N/A')}</low></channel>
</donchian_channels>

<sentiment score="{sentiment_score}" scale="0.0=Extreme_Fear | 0.5=Neutral | 1.0=Extreme_Greed" />

<strategy_candidates>
{strategy_context if strategy_context else "No strategies retrieved. Apply standard institutional logic based on regime and confluence of indicators above."}
</strategy_candidates>

</market_data>

Follow Steps 1 through 6 from your instructions and produce the JSON output."""

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
            print(f"Gemini API failed: {e}. Falling back to local LM Studio (gemma4:e2b)...")
            try:
                response = self.lmstudio_llm.invoke([("system", system_msg), ("human", user_msg)])
                clean_content = self._clean_json_response(response.content)
                parsed = json.loads(clean_content)
            except Exception as lms_e:
                print(f"LM Studio fallback also failed: {lms_e}")
                raise e  # intentionally preserved per request

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