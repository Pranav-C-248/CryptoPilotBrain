import sys
import os
import json
from datetime import datetime, timezone

sys.path.append(os.path.join(os.getcwd()))

from src.core.knowledge_base import TradingKnowledgeBase
from src.agents.analyst import AnalystAgent
from src.agents.risk_manager import RiskManagerAgent

# ══════════════════════════════════════════════════════════════════════════════
#  SCENARIO TEST — Bollinger Bands + RSI Extremes
#
#  Pipeline under test:
#    market_data → AnalystAgent → AnalystSignal
#                                      ↓
#                              RiskManagerAgent → RiskAssessment
#
#  Both layers are tested independently and jointly.
#  Analyst correctness: right signal, right strategy, right vocabulary.
#  Risk correctness:    signal preserved, stop below entry, target above entry,
#                       position sized > 0, valid_until well-formed.
# ══════════════════════════════════════════════════════════════════════════════

# ── Injected Market Data ──────────────────────────────────────────────────────

BB_RSI_SCENARIO = {
    "asset_name": "BTCUSDT",
    "snapshot": {
        "open":  41800.00,
        "high":  42050.00,
        "low":   41500.00,
        "close": 41950.00,      # Price back inside lower BB after breach
    },
    "trends": {
        "pct_change_50": -0.8,
        "pct_change_10": -0.3,
        "market_structure": "Sideways/Ranging"  # Critical — triggers Sideways regime
    },
    "rsi": {
        "current": 33.0,        # Just crossed back above 30 — oversold exit signal
        "avg_50":  49.5,
        "trend":   "Increasing",
        "prev":    28.0         # Was below 30 last candle — oversold confirmed
    },
    "bands": {
        "upper":          44200.00,
        "lower":          41800.00, # Close at/just above lower band — bounce confirmed
        "position_label": "Lower Half"
    },
    "ema": {
        "value":     42900.00,
        "price_rel": "Price Below EMA",
        "slope":     "Decreasing"
    },
    "levels": {
        "high_50": 44500.00,
        "low_50":  41200.00,
        "high_20": 44100.00,
        "low_20":  41500.00,
        "high_55": 45200.00,
        "low_55":  40800.00,
    },
    "volume": {
        "current":  1850.00,
        "avg_50":   2100.00,
        "is_spike": False       # No spike — keeps regime Sideways, not Volatile
    },
    "volatility": {
        "atr": 280.00           # ATR/price = 280/41950 ≈ 0.67% — non-trending confirmed
    },
    "stochastic": {
        "current": 22.0
    },
}

SENTIMENT_SCORE  = 0.48        # Mildly bearish — not extreme enough to veto (floor 0.22)
ASSET_NAME       = "BTC/USDT"
TOTAL_EQUITY     = 100_000.00  # Notional portfolio for risk sizing calculations
TIMEFRAME_HOURS  = 4

# ── Expected Outcomes — Analyst ───────────────────────────────────────────────

EXPECTED_SIGNAL   = "BUY"
EXPECTED_STRATEGY = "Bollinger Bands + RSI Extremes"
CONFIDENCE_MIN    = 0.40
CONFIDENCE_MAX    = 0.80  # Sideways prompt cap is 0.5; >0.5 triggers warning, not failure

# ── Expected Outcomes — Risk Manager ─────────────────────────────────────────
#
#  BB+RSI is Aggressive in STRATEGY_REGISTRY. Aggressive rules:
#    stop_loss  = entry - (1.5 × ATR) = 41950 - 420 = 41530
#    target     = entry + (2 × stop_dist) = 41950 + 840 = 42790
#    position   = (equity × 2%) / stop_dist × price × confidence × exposure_scalar
#               hard cap = equity × 10% = $10,000
#
#  Vetoes that must NOT fire on this scenario:
#    confidence >= 0.50        → passes if analyst is confident enough
#    RSI 33  < veto_high (83)  → no RSI veto
#    sentiment 0.48 > floor (0.22) → no sentiment veto
#    ATR/price 0.67% < cap (8%)   → no ATR veto
#    portfolio empty              → no saturation veto
#    Sideways + Aggressive        → regime veto only hits Conservative strategies

RISK_EXPECTED_SIGNAL      = "BUY"
RISK_STOP_MUST_BE_BELOW   = 41950.00  # stop_loss < entry_price
RISK_TARGET_MUST_BE_ABOVE = 41950.00  # target > entry_price (fixed_target=True)
RISK_POSITION_MIN_USD     = 100.00    # Must size something meaningful


# ── Helpers ───────────────────────────────────────────────────────────────────

def section(title: str):
    print(f"\n{'─'*62}")
    print(f"  {title}")
    print(f"{'─'*62}")

def ok(msg):   print(f"  ✅  {msg}")
def fail(msg): print(f"  ❌  FAIL — {msg}")
def warn(msg): print(f"  ⚠️   WARN — {msg}")
def info(msg): print(f"  ℹ️   {msg}")


# ── Narrative Builders ────────────────────────────────────────────────────────
# Python pre-computes transition conditions so the LLM reads conclusions,
# not raw numbers it might misread in isolation.

def _build_rsi_narrative(market_data: dict) -> str:
    rsi_curr = market_data["rsi"]["current"]
    rsi_prev = market_data["rsi"]["prev"]
    if rsi_prev < 30 and rsi_curr > 30:
        return (
            f"*** RSI was oversold last candle ({rsi_prev}) and has now exited the oversold "
            f"zone ({rsi_curr}). Oversold EXIT confirmed. ***"
        )
    return f"RSI current: {rsi_curr} | prev: {rsi_prev} (no oversold exit detected)"


def _build_bb_narrative(market_data: dict) -> str:
    close    = market_data["snapshot"]["close"]
    bb_lower = market_data["bands"]["lower"]
    if close <= bb_lower * 1.01:
        return (
            f"*** Price ({close}) is at or near BB Lower ({bb_lower}) — potential bounce zone. "
            f"Band breach/touch confirmed. ***"
        )
    return f"Price ({close}) is not near BB Lower ({bb_lower})."


# ── Main Test ─────────────────────────────────────────────────────────────────

def run_bb_rsi_scenario_test():
    print("\n" + "═"*62)
    print("   SCENARIO TEST — Bollinger Bands + RSI Extremes")
    print("   BB Lower Band Bounce + RSI Oversold Exit → Expect BUY")
    print(f"   {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("═"*62)

    failures = []
    warnings = []
    signal   = None   # AnalystSignal
    risk     = None   # RiskAssessment

    # ── Step 1: Infrastructure ────────────────────────────────────────────────
    section("STEP 1 — Infrastructure")

    try:
        kb = TradingKnowledgeBase()
        ok("TradingKnowledgeBase loaded")
    except Exception as e:
        fail(f"KnowledgeBase failed to load: {e}")
        return

    try:
        analyst = AnalystAgent(knowledge_base=kb, timeframe_hours=TIMEFRAME_HOURS)
        ok("AnalystAgent initialized")
    except Exception as e:
        fail(f"AnalystAgent init failed: {e}")
        return

    try:
        risk_manager = RiskManagerAgent(
            total_equity=TOTAL_EQUITY,
            timeframe_hours=TIMEFRAME_HOURS
        )
        ok(f"RiskManagerAgent initialized  (equity=${TOTAL_EQUITY:,.0f})")
    except Exception as e:
        fail(f"RiskManagerAgent init failed: {e}")
        return

    # ── Step 2: KB Sanity ─────────────────────────────────────────────────────
    section("STEP 2 — KB Retrieval Sanity Check")
    info("Querying KB with Sideways regime filter...")

    try:
        strategies = kb.get_relevant_strategies(
            query="Sideways ranging oversold rsi below lower bollinger band bounce entry exit strategy",
            k=3,
            score_threshold=0.3,
            regime_filter="Sideways"
        )
        if not strategies:
            msg = "KB returned no strategies for Sideways — BB+RSI unavailable to Analyst"
            fail(msg); failures.append(msg)
        else:
            names = [s['metadata']['name'] for s in strategies]
            info(f"Retrieved: {names}")
            if EXPECTED_STRATEGY in names:
                ok(f"'{EXPECTED_STRATEGY}' present in KB results")
            else:
                msg = f"'{EXPECTED_STRATEGY}' NOT retrieved. Got: {names}"
                fail(msg); failures.append(msg)
    except Exception as e:
        msg = f"KB retrieval failed: {e}"
        fail(msg); failures.append(msg)

    # ── Step 3: Regime Detection ──────────────────────────────────────────────
    section("STEP 3 — Regime Detection Check")
    info("Verifying injected data produces Sideways regime...")

    detected_regime = analyst._detect_regime(BB_RSI_SCENARIO)
    info(f"market_structure : {BB_RSI_SCENARIO['trends']['market_structure']}")
    info(f"is_spike         : {BB_RSI_SCENARIO['volume']['is_spike']}")
    info(f"atr              : {BB_RSI_SCENARIO['volatility']['atr']}")
    info(f"Detected regime  : {detected_regime}")

    if detected_regime == "Sideways":
        ok("Regime correctly detected as Sideways")
    elif detected_regime == "Volatile":
        msg = "Regime detected as Volatile — check ATR threshold or is_spike flag"
        fail(msg); failures.append(msg)
    else:
        msg = f"Unexpected regime: '{detected_regime}'"
        fail(msg); failures.append(msg)

    if failures:
        _print_summary(failures, warnings, signal, risk)
        return

    # ── Step 4: Analyst LLM Inference ────────────────────────────────────────
    section("STEP 4 — Analyst LLM Inference")

    rsi_narrative = _build_rsi_narrative(BB_RSI_SCENARIO)
    bb_narrative  = _build_bb_narrative(BB_RSI_SCENARIO)
    atr_pct       = BB_RSI_SCENARIO["volatility"]["atr"] / BB_RSI_SCENARIO["snapshot"]["close"]

    info("Pre-computed narrative annotations (injected into analyst context):")
    info(f"  RSI : {rsi_narrative}")
    info(f"  BB  : {bb_narrative}")
    info(f"  ATR : {BB_RSI_SCENARIO['volatility']['atr']} | "
         f"ATR/Price: {atr_pct*100:.3f}% (low — non-trending confirmed)")
    print()
    info("Raw scenario values:")
    info(f"  market_structure : {BB_RSI_SCENARIO['trends']['market_structure']}")
    info(f"  close            : {BB_RSI_SCENARIO['snapshot']['close']}")
    info(f"  bb_lower         : {BB_RSI_SCENARIO['bands']['lower']}")
    info(f"  rsi_current      : {BB_RSI_SCENARIO['rsi']['current']}")
    info(f"  rsi_prev         : {BB_RSI_SCENARIO['rsi']['prev']}")
    info(f"  volume_spike     : {BB_RSI_SCENARIO['volume']['is_spike']}")
    info(f"  sentiment        : {SENTIMENT_SCORE}")
    print()

    try:
        signal = analyst.analyze(BB_RSI_SCENARIO, SENTIMENT_SCORE)
        ok("Analyst returned a valid AnalystSignal")
        print()
        info(f"  signal        : {signal.signal}")
        info(f"  strategy_used : {signal.strategy_used}")
        info(f"  confidence    : {signal.confidence:.2f}")
        info(f"  valid_till    : {signal.valid_till}")
        info(f"  entry_cond    : {signal.entry_condition}")
        info(f"  exit_cond     : {signal.exit_condition}")
        print(f"\n  REASONING:\n    {signal.reasoning}")
        print(f"\n  INTERNAL MONOLOGUE:")
        for line in signal.internal_monologue.split("."):
            line = line.strip()
            if line:
                print(f"    {line}.")
    except Exception as e:
        msg = f"Analyst inference failed: {e}"
        fail(msg); failures.append(msg)
        _print_summary(failures, warnings, signal, risk)
        return

    # ── Step 5: Analyst Assertions ────────────────────────────────────────────
    section("STEP 5 — Analyst Assertions")

    # A1: Signal must be BUY
    if signal.signal == EXPECTED_SIGNAL:
        ok(f"Signal is BUY ✓")
    else:
        msg = (
            f"Signal expected '{EXPECTED_SIGNAL}', got '{signal.signal}'. "
            f"Model may have missed the BB bounce or RSI exit transition."
        )
        fail(msg); failures.append(msg)

    # A2: Strategy must be BB+RSI Extremes
    if signal.strategy_used == EXPECTED_STRATEGY:
        ok(f"Strategy is '{EXPECTED_STRATEGY}' ✓")
    else:
        msg = (
            f"Strategy expected '{EXPECTED_STRATEGY}', got '{signal.strategy_used}'. "
            f"Wrong strategy selected despite Sideways regime filter."
        )
        fail(msg); failures.append(msg)

    # A3: Confidence in range
    if CONFIDENCE_MIN <= signal.confidence <= CONFIDENCE_MAX:
        ok(f"Confidence {signal.confidence:.2f} within [{CONFIDENCE_MIN}, {CONFIDENCE_MAX}] ✓")
    else:
        msg = (
            f"Confidence {signal.confidence:.2f} out of bounds "
            f"[{CONFIDENCE_MIN}, {CONFIDENCE_MAX}]."
        )
        fail(msg); failures.append(msg)

    # A4: Sideways confidence cap — warn not fail
    if signal.confidence > 0.50:
        w = (
            f"Confidence {signal.confidence:.2f} exceeds Sideways cap of 0.50. "
            f"Model may be ignoring the Sideways constraint in the prompt."
        )
        warn(w); warnings.append(w)
    else:
        ok(f"Confidence respects Sideways cap of 0.50 ✓")

    # A5: entry_condition vocabulary
    entry_lower = signal.entry_condition.lower()
    if any(kw in entry_lower for kw in ["bb", "bollinger", "rsi", "lower band", "oversold"]):
        ok("entry_condition references BB/RSI vocabulary ✓")
    else:
        msg = (
            f"entry_condition missing BB/RSI vocabulary. "
            f"Got: '{signal.entry_condition}'"
        )
        fail(msg); failures.append(msg)

    # A6: exit_condition vocabulary
    exit_lower = signal.exit_condition.lower()
    if any(kw in exit_lower for kw in ["bb", "bollinger", "rsi", "upper band", "overbought", "stop"]):
        ok("exit_condition references BB/RSI vocabulary ✓")
    else:
        msg = (
            f"exit_condition missing BB/RSI vocabulary. "
            f"Got: '{signal.exit_condition}'"
        )
        fail(msg); failures.append(msg)

    # A7: valid_till ISO format
    if signal.valid_till and "T" in signal.valid_till and "Z" in signal.valid_till:
        ok(f"valid_till is valid ISO UTC: {signal.valid_till} ✓")
    else:
        msg = f"valid_till malformed or empty: '{signal.valid_till}'"
        fail(msg); failures.append(msg)

    # ── Step 6: Risk Manager Evaluation ──────────────────────────────────────
    section("STEP 6 — Risk Manager Evaluation")
    info("Passing AnalystSignal into RiskManagerAgent...")
    info(f"  Analyst signal    : {signal.signal}")
    info(f"  Analyst strategy  : {signal.strategy_used}")
    info(f"  Analyst confidence: {signal.confidence:.2f}")
    info(f"  Portfolio state   : empty (0 open positions)")
    info(f"  Total equity      : ${TOTAL_EQUITY:,.0f}")
    print()

    try:
        risk = risk_manager.evaluate(
            analyst_signal    = signal,
            market_data       = BB_RSI_SCENARIO,
            sentiment_score   = SENTIMENT_SCORE,
            current_portfolio = [],
        )
        ok("RiskManagerAgent returned a valid RiskAssessment")
        print()
        info(f"  signal        : {risk.signal}")
        info(f"  stop_loss     : {risk.stop_loss}")
        info(f"  target        : {risk.target}")
        info(f"  position_size : ${risk.position_size:,.2f}")
        info(f"  timeframe     : {risk.timeframe}")
        info(f"  valid_until   : {risk.valid_until}")
        info(f"  entry_cond    : {risk.entry_condition}")
        info(f"  exit_cond     : {risk.exit_condition}")
        print(f"\n  AUDIT SUMMARY:")
        for line in risk.audit_summary.split("\n"):
            print(f"    {line}")
    except Exception as e:
        msg = f"Risk Manager evaluation failed: {e}"
        fail(msg); failures.append(msg)
        _print_summary(failures, warnings, signal, risk)
        return

    # ── Step 7: Risk Manager Assertions ───────────────────────────────────────
    section("STEP 7 — Risk Manager Assertions")

    entry_price = BB_RSI_SCENARIO["snapshot"]["close"]   # 41950.00
    atr         = BB_RSI_SCENARIO["volatility"]["atr"]   # 280.00
    stop_dist   = 1.5 * atr                              # 420.00
    expected_sl = entry_price - stop_dist                # 41530.00
    expected_tp = entry_price + (2.0 * stop_dist)        # 42790.00

    # R1: Risk must not veto a valid BUY
    if risk.signal == RISK_EXPECTED_SIGNAL:
        ok("Risk signal is BUY — veto gauntlet passed ✓")
    else:
        msg = (
            f"Risk returned '{risk.signal}', expected 'BUY'. "
            f"A veto fired unexpectedly. Check audit_summary for the reason."
        )
        fail(msg); failures.append(msg)

    # R2: Stop loss below entry (BUY direction)
    if risk.stop_loss < RISK_STOP_MUST_BE_BELOW:
        ok(f"stop_loss {risk.stop_loss} < entry {entry_price}  "
           f"(expected ~{expected_sl:.2f}) ✓")
    else:
        msg = (
            f"stop_loss {risk.stop_loss} is NOT below entry {entry_price}. "
            f"Aggressive BUY: should be entry - 1.5×ATR = {expected_sl:.2f}."
        )
        fail(msg); failures.append(msg)

    # R3: Target above entry (fixed_target=True for BB+RSI Aggressive)
    if risk.target > RISK_TARGET_MUST_BE_ABOVE:
        ok(f"target {risk.target} > entry {entry_price}  "
           f"(expected ~{expected_tp:.2f}) ✓")
    else:
        msg = (
            f"target {risk.target} is NOT above entry {entry_price}. "
            f"Aggressive fixed-target: should be entry + 2×stop_dist = {expected_tp:.2f}."
        )
        fail(msg); failures.append(msg)

    # R4: Position size is meaningful
    if risk.position_size >= RISK_POSITION_MIN_USD:
        pct = (risk.position_size / TOTAL_EQUITY) * 100
        ok(f"position_size ${risk.position_size:,.2f} ({pct:.1f}% equity) is meaningful ✓")
    else:
        msg = (
            f"position_size ${risk.position_size:.2f} is below "
            f"minimum ${RISK_POSITION_MIN_USD}."
        )
        fail(msg); failures.append(msg)

    # R5: Position size respects hard cap (10% equity)
    hard_cap = TOTAL_EQUITY * 0.10
    if risk.position_size <= hard_cap:
        ok(f"position_size ${risk.position_size:,.2f} ≤ hard cap ${hard_cap:,.0f} ✓")
    else:
        msg = (
            f"position_size ${risk.position_size:,.2f} exceeds Aggressive "
            f"hard cap ${hard_cap:,.0f} (10% equity)."
        )
        fail(msg); failures.append(msg)

    # R6: Risk must not flip direction
    if signal.signal != "HOLD":
        if risk.signal == "HOLD" or risk.signal == signal.signal:
            ok(f"Direction not flipped  "
               f"(Analyst: {signal.signal} → Risk: {risk.signal}) ✓")
        else:
            msg = (
                f"Direction FLIPPED: Analyst '{signal.signal}' → "
                f"Risk '{risk.signal}'. This must never happen."
            )
            fail(msg); failures.append(msg)

    # R7: valid_until ISO format
    if risk.valid_until and "T" in risk.valid_until and "Z" in risk.valid_until:
        ok(f"valid_until is valid ISO UTC: {risk.valid_until} ✓")
    else:
        msg = f"valid_until malformed or empty: '{risk.valid_until}'"
        fail(msg); failures.append(msg)

    # R8: Stop-loss directional sanity
    if risk.signal == "BUY" and risk.stop_loss >= entry_price:
        msg = (
            f"Directional error: BUY stop_loss {risk.stop_loss} "
            f">= entry {entry_price}"
        )
        fail(msg); failures.append(msg)
    elif risk.signal == "SELL" and risk.stop_loss <= entry_price:
        msg = (
            f"Directional error: SELL stop_loss {risk.stop_loss} "
            f"<= entry {entry_price}"
        )
        fail(msg); failures.append(msg)
    elif risk.signal != "HOLD":
        ok(f"Stop-loss direction correct for {risk.signal} ✓")

    # ── Step 8: Log Output ────────────────────────────────────────────────────
    section("STEP 8 — Log Output")

    os.makedirs("./logs", exist_ok=True)
    log_path = "./logs/scenario_bb_rsi_test.json"

    log_entry = {
        "test_run":       datetime.utcnow().isoformat() + "Z",
        "scenario":       "BB+RSI Extremes — Lower Band Bounce",
        "asset":          ASSET_NAME,
        "total_equity":   TOTAL_EQUITY,
        "injected_data":  BB_RSI_SCENARIO,
        "sentiment":      SENTIMENT_SCORE,
        "analyst_output": signal.model_dump() if signal else None,
        "risk_output":    risk.model_dump() if risk else None,
        "analyst_assertions": {
            "signal_correct":           signal.signal == EXPECTED_SIGNAL,
            "strategy_correct":         signal.strategy_used == EXPECTED_STRATEGY,
            "confidence_in_range":      CONFIDENCE_MIN <= signal.confidence <= CONFIDENCE_MAX,
            "confidence_cap_respected": signal.confidence <= 0.50,
            "entry_vocabulary_ok":      any(
                kw in signal.entry_condition.lower()
                for kw in ["bb", "bollinger", "rsi", "lower band", "oversold"]
            ),
            "exit_vocabulary_ok":       any(
                kw in signal.exit_condition.lower()
                for kw in ["bb", "bollinger", "rsi", "upper band", "overbought", "stop"]
            ),
            "valid_till_ok":            bool(
                signal.valid_till
                and "T" in signal.valid_till
                and "Z" in signal.valid_till
            ),
        },
        "risk_assertions": {
            "signal_preserved_as_buy":  risk.signal == RISK_EXPECTED_SIGNAL,
            "stop_below_entry":         risk.stop_loss < entry_price,
            "target_above_entry":       risk.target > entry_price,
            "position_size_meaningful": risk.position_size >= RISK_POSITION_MIN_USD,
            "position_size_within_cap": risk.position_size <= TOTAL_EQUITY * 0.10,
            "direction_not_flipped":    risk.signal in ("HOLD", signal.signal),
            "valid_until_ok":           bool(
                risk.valid_until
                and "T" in risk.valid_until
                and "Z" in risk.valid_until
            ),
            "stop_direction_correct":   (
                (risk.signal == "BUY"  and risk.stop_loss < entry_price)
                or (risk.signal == "SELL" and risk.stop_loss > entry_price)
                or risk.signal == "HOLD"
            ),
        },
        "failures": failures,
        "warnings": warnings,
        "passed":   len(failures) == 0,
    }

    with open(log_path, "w") as f:
        json.dump(log_entry, f, indent=2)

    ok(f"Full output written to {log_path}")

    _print_summary(failures, warnings, signal, risk)


# ── Summary ───────────────────────────────────────────────────────────────────

def _print_summary(failures, warnings, signal, risk):
    print("\n" + "═"*62)

    if not failures:
        print("   ✅  ALL ASSERTIONS PASSED — full pipeline correct")
    else:
        print(f"   ❌  {len(failures)} ASSERTION(S) FAILED")
        for i, f in enumerate(failures, 1):
            print(f"   {i}. {f}")

    if warnings:
        print(f"\n   ⚠️   {len(warnings)} WARNING(S):")
        for w in warnings:
            print(f"   → {w}")

    if signal:
        print(
            f"\n   ANALYST  : {signal.signal} | "
            f"{signal.strategy_used} | "
            f"conf={signal.confidence:.2f}"
        )
    if risk:
        print(
            f"   RISK MGR : {risk.signal} | "
            f"SL={risk.stop_loss:.2f} | "
            f"TP={risk.target:.2f} | "
            f"Size=${risk.position_size:,.2f}"
        )

    print("═"*62 + "\n")


if __name__ == "__main__":
    run_bb_rsi_scenario_test()