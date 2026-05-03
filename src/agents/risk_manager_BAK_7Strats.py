from datetime import datetime, timezone, timedelta
from pydantic import BaseModel
from src.schema.models import AnalystSignal


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
        "profile":      "Aggressive",
        "fixed_target": True,
        "rsi_veto":     True,    # RSI(14) extreme veto applies
    },
    "Bollinger Bands + RSI Extremes": {
        "profile":      "Aggressive",
        "fixed_target": True,
        "rsi_veto":     False,   # RSI extremes ARE the entry signal — veto contradictory
    },
    "Bottom Bollinger Band Mean Reversion": {
        "profile":      "Aggressive",
        "fixed_target": True,
        "rsi_veto":     True,
    },

    # ── Conservative — Turtle: profits run, no fixed target ──────────────────
    "Turtle Strategy (System 1 - 20-Day Breakout)": {
        "profile":      "Conservative",
        "fixed_target": False,
        "rsi_veto":     True,
    },
    "Turtle Strategy (System 2 - 55-Day Macro Breakout)": {
        "profile":      "Conservative",
        "fixed_target": False,
        "rsi_veto":     True,
    },

    # ── Conservative — new strategies with explicit take-profit logic ─────────
    "200 MA Macro Pullback Accumulation": {
        "profile":      "Conservative",
        "fixed_target": True,    # target = entry + 50% of macro drop
        "rsi_veto":     True,
    },
    "MA Twist & Convergence Continuation": {
        "profile":      "Conservative",
        "fixed_target": True,    # target = when EMAs begin fanning out again
        "rsi_veto":     True,
    },

    # ── Fallback ──────────────────────────────────────────────────────────────
    "fallback": {
        "profile":      "Conservative",
        "fixed_target": True,
        "rsi_veto":     True,
    },
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
    ) -> RiskAssessment:

        entry_price = market_data['snapshot']['close']
        atr         = market_data['volatility']['atr']
        rsi         = market_data['rsi']['current']
        regime      = market_data['trends']['market_structure'].lower()

        # New packet fields used by veto and level logic
        pct_change_50 = market_data['trends']['pct_change_50']
        adx_value     = market_data['adx']['value']
        ema200_slope  = market_data['ema200']['slope']
        ma_converging = market_data['ema_multi']['converging']
        ma_stack      = market_data['ema_multi']['stack_order']

        strategy_meta = STRATEGY_REGISTRY.get(
            analyst_signal.strategy_used,
            STRATEGY_REGISTRY["fallback"]
        )
        profile      = strategy_meta["profile"]
        fixed_target = strategy_meta["fixed_target"]
        rsi_veto     = strategy_meta["rsi_veto"]

        signal, veto_reasons = self._run_veto_gauntlet(
            analyst_signal, rsi, atr, entry_price,
            sentiment_score, current_portfolio, regime,
            profile, rsi_veto, adx_value, ema200_slope,
            ma_converging, ma_stack
        )

        stop_loss, target = self._calculate_levels(
            signal, entry_price, atr, profile,
            fixed_target, analyst_signal.strategy_used,
            pct_change_50
        )
        position_size = self._calculate_position_size(
            signal, analyst_signal.confidence, atr,
            entry_price, len(current_portfolio), profile
        )
        valid_until   = self._compute_valid_until()
        audit_summary = self._build_audit_summary(
            signal, analyst_signal, rsi, atr, sentiment_score,
            entry_price, stop_loss, target, position_size,
            profile, veto_reasons, adx_value, ema200_slope,
            ma_converging, ma_stack
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
        signal_obj: AnalystSignal,
        rsi: float,
        atr: float,
        price: float,
        sentiment: float,
        portfolio: list,
        regime: str,
        profile: str,
        rsi_veto: bool,
        adx_value: float,
        ema200_slope: str,
        ma_converging: bool,
        ma_stack: str,
    ) -> tuple[str, list[str]]:

        sig    = signal_obj.signal
        vetoes = []

        # 1. Analyst HOLD passthrough
        if sig == "HOLD":
            vetoes.append("Analyst issued HOLD.")
            return "HOLD", vetoes

        # 2. Global confidence floor
        if signal_obj.confidence < self.CONFIDENCE_FLOOR:
            vetoes.append(
                f"Confidence veto: {signal_obj.confidence:.2f} "
                f"below floor {self.CONFIDENCE_FLOOR}."
            )

        # 3. Portfolio saturation
        if len(portfolio) >= self.MAX_CONCURRENT:
            vetoes.append(
                f"Exposure veto: {len(portfolio)} open positions "
                f"hits max cap {self.MAX_CONCURRENT}."
            )

        # 4. RSI extreme veto — skipped for strategies where extremes are the signal
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

        # 5. Sentiment conflict
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

        # 7. Sideways regime + Conservative strategy
        if ("sideways" in regime or "ranging" in regime) and profile == "Conservative":
            vetoes.append(
                f"Regime veto: Conservative strategy in Sideways regime. Forced HOLD."
            )

        # 8. 200 MA Macro Pullback specific guard —
        #    EMA(200) must be upward sloping to confirm bull regime
        if signal_obj.strategy_used == "200 MA Macro Pullback Accumulation":
            if ema200_slope != "Increasing":
                vetoes.append(
                    f"200MA veto: EMA(200) slope is {ema200_slope} — "
                    f"not a bull regime. Entry condition not met."
                )

        # 9. MA Twist specific guard —
        #    MAs must actually be converging for the twist to be valid
        if signal_obj.strategy_used == "MA Twist & Convergence Continuation":
            if not ma_converging:
                vetoes.append(
                    f"MA Twist veto: EMAs are not converging "
                    f"(stack={ma_stack}). Twist condition not met."
                )

        if vetoes:
            return "HOLD", vetoes

        return sig, vetoes

    # ── Levels ────────────────────────────────────────────────────────────────

    def _calculate_levels(
        self,
        signal: str,
        entry_price: float,
        atr: float,
        profile: str,
        fixed_target: bool,
        strategy_name: str,
        pct_change_50: float,
    ) -> tuple[float, float]:
        """
        Aggressive  : stop = 1.5×ATR | target = 2× stop distance (1:2 R:R)
        Turtle      : stop = 2×ATR   | target = 0 (profits run)
        200 MA      : stop = 2×ATR   | target = entry + 50% of 50-candle macro drop
        MA Twist    : stop = 2×ATR   | target = 2× stop distance as proxy
                      for "EMAs begin fanning out" — not directly computable
        """
        if signal == "HOLD" or entry_price == 0:
            return 0.0, 0.0

        if profile == "Conservative":
            stop_dist = self.CON_ATR_STOP_MULT * atr

            if strategy_name == "200 MA Macro Pullback Accumulation":
                # Target = 50% recovery of the 50-candle macro drop
                # pct_change_50 is negative during a pullback
                macro_drop_usd = abs(pct_change_50 / 100) * entry_price
                target_dist    = macro_drop_usd * 0.50

            elif strategy_name == "MA Twist & Convergence Continuation":
                # Proxy: 2× stop distance — represents the next trend leg
                target_dist = 2.0 * stop_dist

            else:
                # Turtle — profits run, no fixed target
                target_dist = 0.0

        else:  # Aggressive
            stop_dist   = self.AGG_ATR_STOP_MULT * atr
            target_dist = self.AGG_RISK_REWARD * stop_dist

        if signal == "BUY":
            stop_loss = entry_price - stop_dist
            target    = (entry_price + target_dist) if fixed_target else 0.0
        else:  # SELL
            stop_loss = entry_price + stop_dist
            target    = (entry_price - target_dist) if fixed_target else 0.0

        return stop_loss, target

    # ── Position Sizing ───────────────────────────────────────────────────────

    def _calculate_position_size(
        self,
        signal: str,
        confidence: float,
        atr: float,
        price: float,
        open_positions: int,
        profile: str,
    ) -> float:
        """
        Both profiles: position_size = max_loss_usd / stop_distance × price
        Aggressive   : scaled by confidence + exposure penalty + hard cap
        Conservative : Turtle 2% ATR unit rule — mechanical, no confidence scaling
        """
        if signal == "HOLD" or price == 0 or atr == 0:
            return 0.0

        if profile == "Conservative":
            stop_dist    = self.CON_ATR_STOP_MULT * atr
            max_loss_usd = self.total_equity * self.CON_RISK_PCT
            raw_units    = max_loss_usd / stop_dist
            raw_notional = raw_units * price
            hard_cap     = self.total_equity * self.CON_MAX_POSITION_PCT
            return min(raw_notional, hard_cap)

        else:  # Aggressive
            stop_dist    = self.AGG_ATR_STOP_MULT * atr
            max_loss_usd = self.total_equity * self.AGG_MAX_LOSS_PCT
            raw_units    = max_loss_usd / stop_dist
            raw_notional = raw_units * price

            confidence_scalar = confidence
            exposure_scalar   = max(0.45, 1.0 - (open_positions * 0.15))

            sized    = raw_notional * confidence_scalar * exposure_scalar
            hard_cap = self.total_equity * self.AGG_MAX_POSITION_PCT
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
    ) -> str:
        target_str = f"{target:.4f}" if target > 0 else "OPEN (profits run)"

        lines = [
            f"[{profile.upper()}] {analyst_signal.asset_name} | "
            f"ANALYST: {analyst_signal.signal} → FINAL: {final_signal}",
            f"STRATEGY: {analyst_signal.strategy_used} | "
            f"CONF: {analyst_signal.confidence:.2f}",
            f"ENTRY: {entry} | SL: {stop_loss:.4f} | "
            f"TP: {target_str} | SIZE: ${position_size:.2f}",
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