import json
import os
from datetime import datetime, timezone, timedelta
from pydantic import BaseModel
from src.schema.models import AnalystSignal


# ── JSON-derived data ─────────────────────────────────────────────────────────
# Single source of truth for fields that live in the strategies JSON.
# risk_profile  : loaded per strategy — eliminates drift with operational_meta
# SIDEWAYS_ALLOWED: derived from regime.primary / also_valid — any strategy
#                   whose regime includes Sideways (and doesn't forbid it)
#                   is automatically exempt from the sideways veto.

_STRATEGIES_JSON_PATH = "data/strategies/strategies_processed.json"


def _load_strategy_data(path: str) -> tuple[dict[str, str], set[str]]:
    """
    Reads strategies_processed.json and returns:
      risk_profiles   : {strategy_name: "Aggressive" | "Conservative"}
      sideways_allowed: set of strategy names valid in a Sideways regime
    """
    risk_profiles: dict[str, str] = {}
    sideways_allowed: set[str]    = set()

    if not os.path.exists(path):
        print(f"[RiskManager] WARNING: strategies JSON not found at '{path}'. "
              "All profiles will default to Conservative.")
        return risk_profiles, sideways_allowed

    with open(path, "r") as f:
        strategies: list[dict] = json.load(f)

    for s in strategies:
        name    = s["metadata"]["name"]
        regime  = s["metadata"]["regime"]          # dict: primary / also_valid / forbidden
        profile = s["operational_meta"]["risk_profile"]

        risk_profiles[name] = profile

        primary    = regime.get("primary", "")
        also_valid = regime.get("also_valid", [])
        forbidden  = regime.get("forbidden", [])

        # A strategy is sideways-allowed if Sideways appears in its valid regimes
        # AND is not explicitly forbidden (the latter is a safety guard).
        if "Sideways" in ([primary] + also_valid) and "Sideways" not in forbidden:
            sideways_allowed.add(name)

    return risk_profiles, sideways_allowed


_RISK_PROFILES, SIDEWAYS_ALLOWED = _load_strategy_data(_STRATEGIES_JSON_PATH)


# ── Output Schema ─────────────────────────────────────────────────────────────

class RiskAssessment(BaseModel):
    asset_name: str
    signal: str
    entry_condition: str
    exit_condition: str
    target: float
    stop_loss: float
    position_size: float
    timeframe: str
    audit_summary: str
    valid_until: str


# ── Strategy Registry ─────────────────────────────────────────────────────────

STRATEGY_REGISTRY = {
    # ── Aggressive — fixed R:R targets ───────────────────────────────────────
    "Volatility Breakout Scalping": {
        "fixed_target": True,
        "rsi_veto":     True,
        "guards":       [],
    },
    "Bollinger Bands + RSI Extremes": {
        "fixed_target": True,
        "rsi_veto":     False,   # RSI extremes ARE the entry signal — veto contradictory
        "guards":       [],
    },
    "Bottom Bollinger Band Mean Reversion": {
        "fixed_target": True,
        "rsi_veto":     True,
        "guards":       [],
    },

    # ── Aggressive — volatile regime ──────────────────────────────────────────
    "ATR Expansion Breakout": {
        "fixed_target": True,
        "rsi_veto":     True,
        "guards":       ["atr_expanding"],
    },
    "Donchian Volatility Range Breach": {
        "fixed_target": True,
        "rsi_veto":     True,
        "guards":       ["atr_expanding"],
    },

    # ── Conservative — Turtle: fixed R:R target ────────────────────────────────
    "Turtle Strategy (System 1 - 20-Day Breakout)": {
        "fixed_target": True,
        "rsi_veto":     True,
        "guards":       [],
    },
    "Turtle Strategy (System 2 - 55-Day Macro Breakout)": {
        "fixed_target": True,
        "rsi_veto":     True,
        "guards":       [],
    },

    # ── Conservative — trending, explicit take-profit logic ───────────────────
    "200 MA Macro Pullback Accumulation": {
        "fixed_target": True,
        "rsi_veto":     True,
        "guards":       ["ema200_upsloping"],
    },
    "MA Twist & Convergence Continuation": {
        "fixed_target": True,
        "rsi_veto":     True,
        "guards":       ["ma_converging"],
    },

    # ── Conservative — sideways regime ────────────────────────────────────────
    "RSI Divergence Fade": {
        "fixed_target": True,
        "rsi_veto":     False,   # RSI divergence IS the entry signal — veto contradictory
        "guards":       [],
    },
    "Donchian Range Oscillation": {
        "fixed_target": True,
        "rsi_veto":     False,   # RSI overbought/oversold IS the confirmation signal
        "guards":       [],
    },
    "EMA20 Mean Reversion": {
        "fixed_target": True,
        "rsi_veto":     True,
        "guards":       [],
    },

    # ── Fallback ──────────────────────────────────────────────────────────────
    "fallback": {
        "fixed_target": True,
        "rsi_veto":     True,
        "guards":       [],
    },
}


# ── Strategies exempt from the sentiment veto ─────────────────────────────────
# These strategies use extreme sentiment as a mandatory entry condition,
# not a risk signal. Vetoing them on sentiment directly contradicts their logic.
SENTIMENT_VETO_EXEMPT = {
    "200 MA Macro Pullback Accumulation",
    "Bottom Bollinger Band Mean Reversion",
    "RSI Divergence Fade",
    "EMA20 Mean Reversion",
}


# ── Risk Manager ──────────────────────────────────────────────────────────────

class RiskManagerAgent:
    """
    Strategy-aware, deterministic capital preservation layer.
    Conservative and Aggressive strategies are governed by different rule sets.
    No LLM. Every decision is auditable Python logic.
    """

    # ── Global Parameters ─────────────────────────────────────────────────────
    MAX_CONCURRENT          = 3
    CONFIDENCE_FLOOR        = 0.50
    RSI_OVERBOUGHT_VETO     = 83
    RSI_OVERSOLD_VETO       = 17
    SENTIMENT_BULL_FLOOR    = 0.22
    SENTIMENT_BEAR_CEIL     = 0.78
    CONFIDENCE_LEVEL_SENSITIVITY = 1.0
    MIN_STOP_PCT = 0.003    # Stop distance floor: 0.3% of entry price
    MIN_RISK_REWARD = 1.5   # Hard floor: reward must be ≥ 1.5× risk (1:1.5)

    # ── Aggressive Strategy Parameters ───────────────────────────────────────
    AGG_BASE_RISK_PCT       = 0.02
    AGG_MAX_POSITION_PCT    = 0.10
    AGG_MAX_LOSS_PCT        = 0.02
    AGG_ATR_STOP_MULT       = 1.5
    AGG_RISK_REWARD         = 2.0
    AGG_ATR_VOLATILITY_CAP  = 0.08

    # ── Conservative Strategy Parameters ─────────────────────────────────────
    CON_RISK_PCT            = 0.02
    CON_MAX_POSITION_PCT    = 0.15
    CON_ATR_STOP_MULT       = 2.0
    CON_ATR_VOLATILITY_CAP  = 0.15
    CON_RISK_REWARD         = 2.0   # default R:R for Conservative trending strategies
    CON_SIDEWAYS_RR         = 1.5   # tighter R:R for mean reversion strategies

    def __init__(self, total_equity: float, timeframe_hours: int = 4):
        self.total_equity    = total_equity
        self.timeframe_hours = timeframe_hours

    # ── Public Entry Point ────────────────────────────────────────────────────

    def evaluate(
        self,
        analyst_signal: AnalystSignal,
        market_data: dict,
        sentiment_score: float,
        current_portfolio: list,
        available_balance: float = None,
    ) -> RiskAssessment:

        if available_balance is not None:
            self.total_equity = available_balance

        entry_price   = market_data['snapshot']['close']
        atr           = market_data['volatility']['atr']
        atr_expanding = market_data['volatility']['atr_expanding']
        rsi           = market_data['rsi']['current']
        regime        = market_data['trends']['market_structure'].lower()
        pct_change_50 = market_data['trends']['pct_change_50']
        adx_value     = market_data['adx']['value']
        ema200_slope  = market_data['ema200']['slope']
        ma_converging = market_data['ema_multi']['converging']
        ma_stack      = market_data['ema_multi']['stack_order']

        # After
        strategy_meta = STRATEGY_REGISTRY.get(
            analyst_signal.strategy_used,
            STRATEGY_REGISTRY["fallback"]
        )
        # profile is sourced from the JSON to keep it in sync with operational_meta.
        # Falls back to "Conservative" for unknown strategies or if JSON is unavailable.
        profile      = _RISK_PROFILES.get(analyst_signal.strategy_used, "Conservative")
        fixed_target = strategy_meta["fixed_target"]
        rsi_veto     = strategy_meta["rsi_veto"]
        guards       = strategy_meta["guards"]

        # Bundle all context the gauntlet needs to evaluate guards
        guard_context = {
            "atr_expanding":  atr_expanding,
            "ema200_slope":   ema200_slope,
            "ma_converging":  ma_converging,
            "ma_stack":       ma_stack,
        }

        signal, veto_reasons = self._run_veto_gauntlet(
            analyst_signal  = analyst_signal,
            rsi             = rsi,
            atr             = atr,
            price           = entry_price,
            sentiment       = sentiment_score,
            portfolio       = current_portfolio,
            regime          = regime,
            profile         = profile,
            rsi_veto        = rsi_veto,
            adx_value       = adx_value,
            guards          = guards,
            guard_context   = guard_context,
        )

        stop_loss, target = self._calculate_levels(
            signal        = signal,
            entry_price   = entry_price,
            atr           = atr,
            profile       = profile,
            fixed_target  = fixed_target,
            strategy_name = analyst_signal.strategy_used,
            pct_change_50 = pct_change_50,
            vetoed        = bool(veto_reasons),
            confidence    = analyst_signal.confidence,
        )
        position_size = self._calculate_position_size(
            signal         = signal,
            confidence     = analyst_signal.confidence,
            atr            = atr,
            price          = entry_price,
            open_positions = len(current_portfolio),
            profile        = profile,
            vetoed         = bool(veto_reasons), 
        )
        valid_until   = self._compute_valid_until()
        audit_summary = self._build_audit_summary(
            final_signal   = signal,
            analyst_signal = analyst_signal,
            rsi            = rsi,
            atr            = atr,
            sentiment      = sentiment_score,
            entry          = entry_price,
            stop_loss      = stop_loss,
            target         = target,
            position_size  = position_size,
            profile        = profile,
            veto_reasons   = veto_reasons,
            adx_value      = adx_value,
            ema200_slope   = ema200_slope,
            ma_converging  = ma_converging,
            ma_stack       = ma_stack,
            vetoed         = bool(veto_reasons),
        )

        return RiskAssessment(
            asset_name      = analyst_signal.asset_name,
            signal          = signal,
            entry_condition = analyst_signal.entry_condition,
            exit_condition  = analyst_signal.exit_condition,
            target          = round(target, 4),
            stop_loss       = round(stop_loss, 4),
            position_size   = round(position_size, 2),
            timeframe       = f"{self.timeframe_hours}h",
            audit_summary   = audit_summary,
            valid_until     = valid_until,
        )

    # ── Veto Gauntlet ─────────────────────────────────────────────────────────

    def _run_veto_gauntlet(
        self,
        analyst_signal: AnalystSignal,
        rsi: float,
        atr: float,
        price: float,
        sentiment: float,
        portfolio: list,
        regime: str,
        profile: str,
        rsi_veto: bool,
        adx_value: float,
        guards: list[str],
        guard_context: dict,
    ) -> tuple[str, list[str]]:

        sig    = analyst_signal.signal
        vetoes = []

        # 1. Analyst HOLD passthrough — no further checks needed
        if sig == "HOLD":
            vetoes.append("Analyst issued HOLD.")
            return "HOLD", vetoes

        # 2. Portfolio saturation — cheapest check, highest frequency rejection
        if len(portfolio) >= self.MAX_CONCURRENT:
            vetoes.append(
                f"Exposure veto: {len(portfolio)} open positions "
                f"hits max cap {self.MAX_CONCURRENT}."
            )

        # 3. Global confidence floor
        if analyst_signal.confidence < self.CONFIDENCE_FLOOR:
            vetoes.append(
                f"Confidence veto: {analyst_signal.confidence:.2f} "
                f"below floor {self.CONFIDENCE_FLOOR}."
            )

        # 4. RSI extreme veto — skipped for strategies where RSI extremes are
        #    the entry signal itself (rsi_veto=False in registry)
        if rsi_veto:
            if sig == "BUY" and rsi > self.RSI_OVERBOUGHT_VETO:
                vetoes.append(
                    f"RSI veto: BUY at RSI(14)={rsi:.1f} is critically "
                    f"overbought (>{self.RSI_OVERBOUGHT_VETO})."
                )
            if sig == "SELL" and rsi < self.RSI_OVERSOLD_VETO:
                vetoes.append(
                    f"RSI veto: SELL at RSI(14)={rsi:.1f} is critically "
                    f"oversold (<{self.RSI_OVERSOLD_VETO})."
                )

        # 5. Sentiment conflict — exempt strategies that use extreme sentiment
        #    as a mandatory entry condition rather than a risk signal
        if analyst_signal.strategy_used not in SENTIMENT_VETO_EXEMPT:
            if sig == "BUY" and sentiment < self.SENTIMENT_BULL_FLOOR:
                vetoes.append(
                    f"Sentiment veto: BUY into extreme fear "
                    f"(sentiment={sentiment:.2f})."
                )
            if sig == "SELL" and sentiment > self.SENTIMENT_BEAR_CEIL:
                vetoes.append(
                    f"Sentiment veto: SELL into extreme greed "
                    f"(sentiment={sentiment:.2f})."
                )

        # 6. ATR volatility cap — profile-aware
        atr_pct = atr / price if price > 0 else 0
        atr_cap = (
            self.AGG_ATR_VOLATILITY_CAP if profile == "Aggressive"
            else self.CON_ATR_VOLATILITY_CAP
        )
        if atr_pct > atr_cap:
            vetoes.append(
                f"Volatility veto: ATR/price={atr_pct:.3f} exceeds "
                f"{profile} cap {atr_cap}."
            )

        # 7. Sideways regime + Conservative strategy veto.
        #    Exempt strategies explicitly designed for sideways conditions.
        if ("sideways" in regime or "ranging" in regime) and profile == "Conservative":
            if analyst_signal.strategy_used not in SIDEWAYS_ALLOWED:
                vetoes.append(
                    f"Regime veto: Conservative trending strategy "
                    f"'{analyst_signal.strategy_used}' triggered in Sideways regime. "
                    f"Forced HOLD."
                )

        # 8. Registry-driven strategy-specific guards.
        #    Each guard key maps to a condition that must be True for the
        #    strategy to be valid. Failures are surfaced with a clear reason.
        guard_failures = self._evaluate_guards(
            guards        = guards,
            guard_context = guard_context,
            strategy_name = analyst_signal.strategy_used,
        )
        vetoes.extend(guard_failures)

        if vetoes:
            return "HOLD", vetoes

        return sig, vetoes

    # ── Registry-Driven Guard Evaluator ───────────────────────────────────────

    def _evaluate_guards(
        self,
        guards: list[str],
        guard_context: dict,
        strategy_name: str,
    ) -> list[str]:
        """
        Evaluates strategy-specific guards defined in the registry.
        Returns a list of veto strings for any guard that fails.
        Adding a new strategy-specific condition only requires:
          1. Adding the guard key to the registry entry.
          2. Adding its evaluation clause here.
        No other methods need to be touched.
        """
        failures = []

        for guard in guards:

            if guard == "ema200_upsloping":
                # 200 MA Macro Pullback: EMA(200) must slope upward
                # to confirm the asset is in a macro bull regime.
                if guard_context["ema200_slope"] != "Increasing":
                    failures.append(
                        f"Guard fail [{strategy_name}]: EMA(200) slope is "
                        f"'{guard_context['ema200_slope']}' — bull regime not confirmed."
                    )

            elif guard == "ma_converging":
                # MA Twist: EMAs must be actively converging for the
                # twist setup to be valid. A diverged stack means the
                # twist has already fired or hasn't formed yet.
                if not guard_context["ma_converging"]:
                    failures.append(
                        f"Guard fail [{strategy_name}]: EMAs not converging "
                        f"(stack='{guard_context['ma_stack']}'). Twist condition not met."
                    )

            elif guard == "atr_expanding":
                # ATR Expansion Breakout / Donchian Volatility Range Breach:
                # ATR must be actively expanding — a breakout without expanding
                # ATR is a low-conviction range wobble, not a volatility event.
                if not guard_context["atr_expanding"]:
                    failures.append(
                        f"Guard fail [{strategy_name}]: ATR is not expanding. "
                        f"No volatility regime shift confirmed."
                    )

        return failures

    # ── Levels ────────────────────────────────────────────────────────────────

    # def _calculate_levels(
    #     self,
    #     signal: str,
    #     entry_price: float,
    #     atr: float,
    #     profile: str,
    #     fixed_target: bool,
    #     strategy_name: str,
    #     pct_change_50: float,
    #     vetoed: bool,
    # ) -> tuple[float, float]:
    #     """
    #     Returns (stop_loss, target). If vetoed or signal is HOLD, returns (0.0, 0.0).

    #     Aggressive strategies  : stop = 1.5×ATR | target = 2× stop distance (1:2 R:R)
    #     Turtle strategies      : stop = 2×ATR   | target = 0.0 (profits run)
    #     200 MA Pullback        : stop = 2×ATR   | target = entry ± 50% of 50-candle drop
    #     MA Twist               : stop = 2×ATR   | target = 2× stop distance (proxy)
    #     Conservative Sideways  : stop = 2×ATR   | target = 1.5× stop distance (1:1.5 R:R)
    #     """
    #     if signal == "HOLD" or entry_price == 0 or vetoed:
    #         return 0.0, 0.0

    #     if profile == "Conservative":
    #         stop_dist = self.CON_ATR_STOP_MULT * atr

    #         if strategy_name == "200 MA Macro Pullback Accumulation":
    #             # Target = 50% recovery of the 50-candle macro drop.
    #             # pct_change_50 is negative during a pullback so abs() is used.
    #             macro_drop_usd = abs(pct_change_50 / 100) * entry_price
    #             target_dist    = macro_drop_usd * 0.50

    #         elif strategy_name == "MA Twist & Convergence Continuation":
    #             # Proxy for "EMAs begin fanning out" — not directly computable
    #             # from a single snapshot, so 2× stop distance is used.
    #             target_dist = 2.0 * stop_dist

    #         elif not fixed_target:
    #             # Turtle strategies — profits run, no fixed target
    #             target_dist = 0.0

    #         else:
    #             # General Conservative fixed-target (Sideways mean reversion).
    #             # Tighter R:R than Aggressive because mean reversion targets
    #             # a known structural level (EMA20 / BB midline), not open air.
    #             target_dist = self.CON_SIDEWAYS_RR * stop_dist

    #     else:  # Aggressive
    #         stop_dist   = self.AGG_ATR_STOP_MULT * atr
    #         target_dist = self.AGG_RISK_REWARD * stop_dist

    #     if signal == "BUY":
    #         stop_loss = entry_price - stop_dist
    #         target    = (entry_price + target_dist) if fixed_target else 0.0
    #     else:  # SELL
    #         stop_loss = entry_price + stop_dist
    #         target    = (entry_price - target_dist) if fixed_target else 0.0

    #     return stop_loss, target

    def _calculate_levels(
            self,
            signal: str,
            entry_price: float,
            atr: float,
            profile: str,
            fixed_target: bool,
            strategy_name: str,
            pct_change_50: float,
            vetoed: bool,
            confidence: float,          
        ) -> tuple[float, float]:
        """
        Every strategy now gets a fixed target derived from its risk-reward ratio.
        Stop widens and target shrinks proportionally as confidence falls.
        At confidence=1.0 → no adjustment.
        At confidence=0.5 (floor) → stop 50% wider, target 50% smaller.
        """
        if signal == "HOLD" or entry_price == 0 or vetoed:
            return 0.0, 0.0

        # ── Confidence scalars ────────────────────────────────────────────────────
        # Stop: wider when less confident — inverse of confidence
        stop_conf_mult   = 1.0 + (1.0 - confidence) * self.CONFIDENCE_LEVEL_SENSITIVITY
        # Target: smaller when less confident — direct scale
        target_conf_mult = confidence

        if profile == "Conservative":
            base_stop_dist = self.CON_ATR_STOP_MULT * atr
            stop_dist = base_stop_dist * stop_conf_mult

            if strategy_name == "200 MA Macro Pullback Accumulation":
                macro_drop_usd = abs(pct_change_50 / 100) * entry_price
                target_dist    = (macro_drop_usd * 0.50) * target_conf_mult

            elif strategy_name == "MA Twist & Convergence Continuation":
                # Proxy target based on BASE stop distance — not the widened one
                target_dist = (2.0 * base_stop_dist) * target_conf_mult

            elif strategy_name in (
                "Donchian Range Oscillation",
                "RSI Divergence Fade",
                "EMA20 Mean Reversion",
            ):
                # Sideways mean reversion — tighter R:R targeting known
                # structural levels (EMA20, BB midline, Donchian midline).
                target_dist = (self.CON_SIDEWAYS_RR * base_stop_dist) * target_conf_mult

            else:
                # All other Conservative strategies (incl. Turtle) — fixed R:R target.
                target_dist = (self.CON_RISK_REWARD * base_stop_dist) * target_conf_mult

        else:  # Aggressive
            base_stop_dist = self.AGG_ATR_STOP_MULT * atr
            stop_dist      = base_stop_dist * stop_conf_mult
            target_dist    = (self.AGG_RISK_REWARD * base_stop_dist) * target_conf_mult

        # ── Minimum stop clamp ─────────────────────────────────────────────────────
        min_stop = entry_price * self.MIN_STOP_PCT
        stop_dist = max(stop_dist, min_stop)

        # ── Enforce minimum R:R floor (1:1.5) ─────────────────────────────────────
        # After all confidence scaling, guarantee that for every dollar risked
        # (stop_dist), the reward (target_dist) is at least 1.5× that risk.
        # This prevents low-confidence scaling from creating bad R:R trades.
        min_target_dist = self.MIN_RISK_REWARD * stop_dist
        if target_dist < min_target_dist:
            target_dist = min_target_dist

        if signal == "BUY":
            stop_loss = entry_price - stop_dist
            target    = entry_price + target_dist
        else:
            stop_loss = entry_price + stop_dist
            target    = entry_price - target_dist

        return stop_loss, target

    # ── Position Sizing ───────────────────────────────────────────────────────

    # def _calculate_position_size(
    #     self,
    #     signal: str,
    #     confidence: float,
    #     atr: float,
    #     price: float,
    #     open_positions: int,
    #     profile: str,
    # ) -> float:
    #     """
    #     Both profiles: position_size = max_loss_usd / stop_distance × price
    #     Aggressive   : scaled by confidence + exposure penalty + hard cap
    #     Conservative : Turtle 2% ATR unit rule — mechanical, no confidence scaling
    #     """
    #     if signal == "HOLD" or price == 0 or atr == 0:
    #         return 0.0

    #     if profile == "Conservative":
    #         stop_dist    = self.CON_ATR_STOP_MULT * atr
    #         max_loss_usd = self.total_equity * self.CON_RISK_PCT
    #         raw_units    = max_loss_usd / stop_dist
    #         raw_notional = raw_units * price
    #         hard_cap     = self.total_equity * self.CON_MAX_POSITION_PCT
    #         return min(raw_notional, hard_cap)

    #     else:  # Aggressive
    #         stop_dist    = self.AGG_ATR_STOP_MULT * atr
    #         max_loss_usd = self.total_equity * self.AGG_MAX_LOSS_PCT
    #         raw_units    = max_loss_usd / stop_dist
    #         raw_notional = raw_units * price

    #         confidence_scalar = confidence
    #         exposure_scalar   = max(0.45, 1.0 - (open_positions * 0.15))

    #         sized    = raw_notional * confidence_scalar * exposure_scalar
    #         hard_cap = self.total_equity * self.AGG_MAX_POSITION_PCT
    #         return min(sized, hard_cap)


    def _calculate_position_size(
        self,
        signal: str,
        confidence: float,
        atr: float,
        price: float,
        open_positions: int,
        profile: str,
        vetoed: bool,
    ) -> float:
        if signal == "HOLD" or price == 0 or atr == 0 or vetoed:
            return 0.0

        # ── Minimum stop clamp applied here too for sizing consistency ────────────
        stop_dist = max(
            (self.CON_ATR_STOP_MULT if profile == "Conservative" else self.AGG_ATR_STOP_MULT) * atr,
            price * self.MIN_STOP_PCT
        )

        if profile == "Conservative":
            max_loss_usd = self.total_equity * self.CON_RISK_PCT
            raw_units    = max_loss_usd / stop_dist
            raw_notional = raw_units * price

            # Mild confidence scalar: 0.70 at floor (0.50) → 1.0 at full (1.0).
            # Conservative stays mechanical but not completely blind to conviction.
            confidence_scalar = 0.70 + (confidence - self.CONFIDENCE_FLOOR) * 0.60
            sized    = raw_notional * confidence_scalar
            hard_cap = self.total_equity * self.CON_MAX_POSITION_PCT
            return min(sized, hard_cap)

        else:  # Aggressive
            max_loss_usd      = self.total_equity * self.AGG_MAX_LOSS_PCT
            raw_units         = max_loss_usd / stop_dist
            raw_notional      = raw_units * price
            confidence_scalar = confidence
            exposure_scalar   = max(0.45, 1.0 - (open_positions * 0.15))
            sized             = raw_notional * confidence_scalar * exposure_scalar
            hard_cap          = self.total_equity * self.AGG_MAX_POSITION_PCT
            return min(sized, hard_cap)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _compute_valid_until(self) -> str:
        expiry = datetime.now(timezone.utc) + timedelta(hours=self.timeframe_hours)
        return expiry.strftime("%Y-%m-%dT%H:%M:%SZ")

    def _build_audit_summary(
        self,
        final_signal: str,
        analyst_signal: AnalystSignal,
        rsi: float,
        atr: float,
        sentiment: float,
        entry: float,
        stop_loss: float,
        target: float,
        position_size: float,
        profile: str,
        veto_reasons: list[str],
        adx_value: float,
        ema200_slope: str,
        ma_converging: bool,
        ma_stack: str,
        vetoed: bool,
    ) -> str:
        if vetoed:
            target_str    = "N/A"
            stop_loss_str = "N/A"
            size_str      = "N/A"
        else:
            target_str    = f"{target:.4f}"
            stop_loss_str = f"{stop_loss:.4f}"
            size_str      = f"${position_size:.2f}"

        lines = [
            f"[{profile.upper()}] {analyst_signal.asset_name} | "
            f"ANALYST: {analyst_signal.signal} → FINAL: {final_signal}",
            f"STRATEGY: {analyst_signal.strategy_used} | "
            f"CONF: {analyst_signal.confidence:.2f}",
            f"ENTRY: {entry} | SL: {stop_loss_str} | "
            f"TP: {target_str} | SIZE: {size_str}",
            f"RSI(14): {rsi:.1f} | ATR: {atr} | "
            f"SENTIMENT: {sentiment:.2f} | ADX: {adx_value:.1f}",
            f"EMA200 Slope: {ema200_slope} | "
            f"MA Converging: {ma_converging} | MA Stack: {ma_stack}",
        ]

        if veto_reasons:
            lines.append("VETOES: " + " | ".join(veto_reasons))
        else:
            lines.append("STATUS: All checks passed.")

        return "\n".join(lines)