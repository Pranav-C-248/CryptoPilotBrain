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
    position_size: float       # USD notional
    timeframe: str
    audit_summary: str
    valid_until: str           # ISO UTC string


# ── Strategy Registry ─────────────────────────────────────────────────────────
# Maps strategy name → risk profile and behaviour flags

STRATEGY_REGISTRY = {
    "Volatility Breakout Scalping":               {"profile": "Aggressive", "fixed_target": True},
    "Pure Momentum Scalping":                     {"profile": "Aggressive", "fixed_target": True},
    "Bollinger Bands + RSI Extremes":             {"profile": "Aggressive", "fixed_target": True},
    "VWAP and Stochastic Oscillator Micro-Reversion": {"profile": "Aggressive", "fixed_target": True},
    "Turtle Strategy (System 1 - 20-Day Breakout)":   {"profile": "Conservative", "fixed_target": False},
    "Turtle Strategy (System 2 - 55-Day Macro Breakout)": {"profile": "Conservative", "fixed_target": False},
    "fallback":                                   {"profile": "Conservative", "fixed_target": True},
}


# ── Risk Manager ──────────────────────────────────────────────────────────────

class RiskManagerAgent:
    """
    Strategy-aware, deterministic capital preservation layer.
    Conservative and Aggressive strategies are governed by different rule sets.
    No LLM. Every decision is auditable Python logic.
    """

    # ── Global Parameters ─────────────────────────────────────────────────────
    MAX_CONCURRENT          = 3      # Hard cap on concurrent open positions
    CONFIDENCE_FLOOR        = 0.50   # Global floor — below this, no trade ever
    RSI_OVERBOUGHT_VETO     = 83     # BUY above this RSI → vetoed
    RSI_OVERSOLD_VETO       = 17     # SELL below this RSI → vetoed
    SENTIMENT_BULL_FLOOR    = 0.22   # BUY with sentiment below this → vetoed
    SENTIMENT_BEAR_CEIL     = 0.78   # SELL with sentiment above this → vetoed

    # ── Aggressive Strategy Parameters ───────────────────────────────────────
    AGG_BASE_RISK_PCT       = 0.02   # 2% equity risked per trade
    AGG_MAX_POSITION_PCT    = 0.10   # 10% equity hard cap per position
    AGG_MAX_LOSS_PCT        = 0.02   # Max loss per trade = 2% equity
    AGG_ATR_STOP_MULT       = 1.5    # Stop = 1.5× ATR from entry
    AGG_RISK_REWARD         = 2.0    # Target = 2× stop distance (1:2 R:R)
    AGG_ATR_VOLATILITY_CAP  = 0.08   # ATR/price above this → too erratic for scalping

    # ── Conservative Strategy Parameters (Turtle logic) ───────────────────────
    CON_RISK_PCT            = 0.02   # Turtle's own 2% rule
    CON_MAX_POSITION_PCT    = 0.15   # Turtle holds larger positions, longer duration
    CON_ATR_STOP_MULT       = 2.0    # Turtle uses 2N stop
    CON_ATR_VOLATILITY_CAP  = 0.15   # Macro strategies tolerate higher ATR

    def __init__(self, total_equity: float, timeframe_hours: int = 4):
        self.total_equity = total_equity
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

        strategy_meta = STRATEGY_REGISTRY.get(
            analyst_signal.strategy_used,
            STRATEGY_REGISTRY["fallback"]
        )
        profile      = strategy_meta["profile"]
        fixed_target = strategy_meta["fixed_target"]

        signal, veto_reasons = self._run_veto_gauntlet(
            analyst_signal, rsi, atr, entry_price,
            sentiment_score, current_portfolio, regime, profile
        )

        stop_loss, target = self._calculate_levels(
            signal, entry_price, atr, profile, fixed_target
        )
        position_size = self._calculate_position_size(
            signal, analyst_signal.confidence, atr,
            entry_price, len(current_portfolio), profile
        )
        valid_until   = self._compute_valid_until()
        audit_summary = self._build_audit_summary(
            signal, analyst_signal, rsi, atr, sentiment_score,
            entry_price, stop_loss, target, position_size,
            profile, veto_reasons
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
    ) -> tuple[str, list[str]]:

        sig    = signal_obj.signal
        vetoes = []

        # 1. Analyst HOLD passthrough — nothing to evaluate
        if sig == "HOLD":
            vetoes.append("Analyst issued HOLD.")
            return "HOLD", vetoes

        # 2. Global confidence floor
        if signal_obj.confidence < self.CONFIDENCE_FLOOR:
            vetoes.append(
                f"Confidence veto: {signal_obj.confidence:.2f} below floor {self.CONFIDENCE_FLOOR}."
            )

        # 3. Portfolio saturation
        if len(portfolio) >= self.MAX_CONCURRENT:
            vetoes.append(
                f"Exposure veto: {len(portfolio)} open positions hits max cap {self.MAX_CONCURRENT}."
            )

        # 4. RSI extremes
        if sig == "BUY" and rsi > self.RSI_OVERBOUGHT_VETO:
            vetoes.append(
                f"RSI veto: BUY at RSI={rsi:.1f} is critically overbought (>{self.RSI_OVERBOUGHT_VETO})."
            )
        if sig == "SELL" and rsi < self.RSI_OVERSOLD_VETO:
            vetoes.append(
                f"RSI veto: SELL at RSI={rsi:.1f} is critically oversold (<{self.RSI_OVERSOLD_VETO})."
            )

        # 5. Sentiment conflict
        if sig == "BUY" and sentiment < self.SENTIMENT_BULL_FLOOR:
            vetoes.append(
                f"Sentiment veto: BUY into extreme fear (sentiment={sentiment:.2f})."
            )
        if sig == "SELL" and sentiment > self.SENTIMENT_BEAR_CEIL:
            vetoes.append(
                f"Sentiment veto: SELL into extreme greed (sentiment={sentiment:.2f})."
            )

        # 6. ATR volatility cap — profile-aware
        atr_pct = atr / price if price > 0 else 0
        atr_cap = self.AGG_ATR_VOLATILITY_CAP if profile == "Aggressive" else self.CON_ATR_VOLATILITY_CAP
        if atr_pct > atr_cap:
            vetoes.append(
                f"Volatility veto: ATR/price={atr_pct:.3f} exceeds {profile} cap {atr_cap}."
            )

        # 7. Sideways regime + wrong strategy type
        # BB+RSI is Aggressive + Sideways — allow it through.
        # Conservative strategies in Sideways → veto.
        if ("sideways" in regime or "ranging" in regime) and profile == "Conservative":
            vetoes.append(
                f"Regime veto: Conservative strategy triggered in Sideways regime. Forced HOLD."
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
    ) -> tuple[float, float]:
        """
        Aggressive: stop = 1.5×ATR, target = 3×ATR (1:2 R:R on 1.5× base)
        Conservative (Turtle): stop = 2×ATR, target = 0 (profits run, no fixed target)
        """
        if signal == "HOLD" or entry_price == 0:
            return 0.0, 0.0

        if profile == "Conservative":
            stop_dist = self.CON_ATR_STOP_MULT * atr
            target_dist = 0.0  # Turtle lets profits run — no fixed target
        else:
            stop_dist   = self.AGG_ATR_STOP_MULT * atr
            target_dist = self.AGG_RISK_REWARD * stop_dist

        if signal == "BUY":
            stop_loss = entry_price - stop_dist
            target    = (entry_price + target_dist) if fixed_target else 0.0
        else:
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
        Both profiles use: position_size = max_loss_usd / stop_distance × price
        Aggressive: scaled by confidence + exposure penalty + hard cap
        Conservative: Turtle's 2% ATR unit rule, no confidence scaling
                      (Turtle sizing is mechanical, not confidence-weighted)
        """
        if signal == "HOLD" or price == 0 or atr == 0:
            return 0.0

        if profile == "Conservative":
            stop_dist    = self.CON_ATR_STOP_MULT * atr
            max_loss_usd = self.total_equity * self.CON_RISK_PCT
            raw_units    = max_loss_usd / stop_dist
            raw_notional = raw_units * price
            hard_cap     = self.total_equity * self.CON_MAX_POSITION_PCT
            # Turtle sizing is mechanical — no confidence or exposure scaling
            return min(raw_notional, hard_cap)

        else:  # Aggressive
            stop_dist    = self.AGG_ATR_STOP_MULT * atr
            max_loss_usd = self.total_equity * self.AGG_MAX_LOSS_PCT
            raw_units    = max_loss_usd / stop_dist
            raw_notional = raw_units * price

            # Confidence scalar: more sure → larger size
            confidence_scalar = confidence  # already >= 0.50 post-veto

            # Exposure penalty: each open position reduces size by 15%
            exposure_scalar = max(0.45, 1.0 - (open_positions * 0.15))

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
    ) -> str:
        target_str = f"{target:.4f}" if target > 0 else "OPEN (profits run)"
        lines = [
            f"[{profile.upper()}] {analyst_signal.asset_name} | "
            f"ANALYST: {analyst_signal.signal} → FINAL: {final_signal}",
            f"STRATEGY: {analyst_signal.strategy_used} | CONF: {analyst_signal.confidence:.2f}",
            f"ENTRY: {entry} | SL: {stop_loss:.4f} | TP: {target_str} | SIZE: ${position_size:.2f}",
            f"RSI: {rsi:.1f} | ATR: {atr} | SENTIMENT: {sentiment:.2f}",
        ]
        if veto_reasons:
            lines.append("VETOES: " + " | ".join(veto_reasons))
        else:
            lines.append("STATUS: All checks passed.")
        return "\n".join(lines)