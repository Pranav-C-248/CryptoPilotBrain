import sys
import os
import json
from datetime import datetime ,timedelta,timezone ,time

sys.path.append(os.path.join(os.getcwd()))

from src.core.knowledge_base import TradingKnowledgeBase
from src.agents.analyst import AnalystAgent

# ══════════════════════════════════════════════════════════════════════════════
#  STRATEGY SCENARIO TEST — Bollinger Bands + RSI Extremes
#  Tests whether the Analyst correctly identifies and fires a BUY signal
#  when presented with a textbook BB lower band bounce + RSI oversold exit.
# ══════════════════════════════════════════════════════════════════════════════

# ── Injected Market Data ──────────────────────────────────────────────────────
#
#  Scenario narrative:
#  BTC has been trading sideways for several candles in a well-defined range.
#  Price pushed below the lower Bollinger Band during the previous candle,
#  with RSI dropping to 26 (oversold). On the current candle, price has
#  snapped back INSIDE the lower band, and RSI has exited the oversold zone
#  at 33. ATR is low confirming a non-trending environment. Volume is flat —
#  no spike, so regime stays Sideways, not Volatile.
#  ADX is implicitly confirmed via Sideways/Ranging market structure + low ATR.
#
#  Expected behaviour:
#  → Regime detected as Sideways
#  → BB + RSI Extremes retrieved from KB
#  → Entry primary satisfied: price was below bb_lower, now back inside
#  → RSI confirmation satisfied: RSI crossed below 30 then exited back above
#  → Signal: BUY
#  → Strategy: "Bollinger Bands + RSI Extremes"
#  → Confidence: between 0.50 and 0.80 (Sideways hard cap at 0.5 per prompt rules,
#                but we allow up to 0.80 here in case model reasons both
#                confirmations as met — strict cap enforcement is a prompt concern)

BB_RSI_SCENARIO = {
    "asset_name":"BTCUSDT",
    "snapshot": {
        "open":  41800.00,
        "high":  42050.00,
        "low":   41500.00,
        "close": 41950.00       # Price back inside lower BB after breach
    },
    "trends": {
        "pct_change_50": -0.8,  # Essentially flat over 50 candles
        "pct_change_10": -0.3,  # Flat over 10 candles too
        "market_structure": "Sideways/Ranging"  # Critical — triggers Sideways regime
    },
    "rsi": {
        "current": 33.0,        # Just crossed back above 30 — oversold exit signal
        "avg_50":  49.5,        # Long-term RSI average is neutral — confirms ranging
        "trend":   "Increasing" ,# RSI is recovering upward from the oversold dip
        "prev" : 28.0
    },
    "bands": {
        "upper":          44200.00,
        "lower":          41800.00, # Close is at/just above lower band — bounce confirmed
        "position_label": "Lower Half"
    },
    "ema": {
        "value":     42900.00,
        "price_rel": "Price Below EMA", # Price below midpoint — consistent with ranging low
        "slope":     "Decreasing"       # Flat/slight decline, not trending
    },
    "levels": {
        "high_50": 44500.00,
        "low_50":  41200.00,
        "high_20": 44100.00,
        "low_20":  41500.00,
        "high_55": 45200.00,
        "low_55":  40800.00
    },
    "volume": {
        "current":  1850.00,
        "avg_50":   2100.00,
        "is_spike": False       # No volume spike — keeps regime Sideways, not Volatile
    },
    "volatility": {
        "atr": 280.00           # Low ATR relative to price — confirms non-trending env
                                # ATR/price = 280/41950 ≈ 0.0067 — well below any cap
    },
    "stochastic": {
        "current": 22.0         # Stochastic also in oversold territory — supporting confluence
    }
}

SENTIMENT_SCORE = 0.48   # Slightly below neutral — mildly bearish sentiment
                          # Not extreme enough to trigger sentiment veto (floor is 0.22)
ASSET_NAME      = "BTC/USDT"

# ── Expected Outcomes ─────────────────────────────────────────────────────────

EXPECTED_SIGNAL   = "BUY"
EXPECTED_STRATEGY = "Bollinger Bands + RSI Extremes"
CONFIDENCE_MIN    = 0.40   # Floor — model should have at least some conviction
CONFIDENCE_MAX    = 0.80   # Ceiling — Sideways prompt rule caps at 0.5, but
                            # we test up to 0.8 to catch misconfigured prompts
                            # A value above 0.5 is noted as a prompt compliance warning

# ── Helpers ───────────────────────────────────────────────────────────────────

def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")

def ok(msg):   print(f"  ✅  {msg}")
def fail(msg): print(f"  ❌  FAIL — {msg}")
def warn(msg): print(f"  ⚠️   WARN — {msg}")
def info(msg): print(f"  ℹ️   {msg}")

# ── Main Test ─────────────────────────────────────────────────────────────────

def run_bb_rsi_scenario_test():
    print("\n" + "═"*60)
    print("   SCENARIO TEST — Bollinger Bands + RSI Extremes")
    print("   BB Lower Band Bounce + RSI Oversold Exit → Expect BUY")
    print(f"   {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("═"*60)

    failures = []
    warnings = []
    signal   = None

    # ── Step 1: Infrastructure ────────────────────────────────────────────────
    section("STEP 1 — Infrastructure")

    try:
        kb = TradingKnowledgeBase()
        ok("TradingKnowledgeBase loaded")
    except Exception as e:
        fail(f"KnowledgeBase failed to load: {e}")
        return

    try:
        analyst = AnalystAgent(knowledge_base=kb, timeframe_hours=4)
        ok("AnalystAgent initialized")
    except Exception as e:
        fail(f"AnalystAgent init failed: {e}")
        return

    # ── Step 2: KB Sanity — Does BB+RSI surface for Sideways? ─────────────────
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
            msg = "KB returned no strategies for Sideways regime — BB+RSI will not be available to the Analyst"
            fail(msg)
            failures.append(msg)
        else:
            names = [s['metadata']['name'] for s in strategies]
            info(f"Retrieved: {names}")

            if EXPECTED_STRATEGY in names:
                ok(f"'{EXPECTED_STRATEGY}' is present in KB results")
            else:
                msg = f"'{EXPECTED_STRATEGY}' was NOT retrieved — Analyst cannot select it. Got: {names}"
                fail(msg)
                failures.append(msg)

    except Exception as e:
        msg = f"KB retrieval failed: {e}"
        fail(msg)
        failures.append(msg)

    # ── Step 3: Regime Detection Check ───────────────────────────────────────
    section("STEP 3 — Regime Detection Check")
    info("Verifying that injected data produces Sideways regime...")

    # Call the private method directly to verify deterministic logic
    detected_regime = analyst._detect_regime(BB_RSI_SCENARIO)
    info(f"market_structure : {BB_RSI_SCENARIO['trends']['market_structure']}")
    info(f"is_spike         : {BB_RSI_SCENARIO['volume']['is_spike']}")
    info(f"atr              : {BB_RSI_SCENARIO['volatility']['atr']}")
    info(f"Detected regime  : {detected_regime}")

    if detected_regime == "Sideways":
        ok("Regime correctly detected as Sideways")
    elif detected_regime == "Volatile":
        msg = f"Regime detected as Volatile instead of Sideways — check ATR threshold or is_spike flag"
        fail(msg)
        failures.append(msg)
    else:
        msg = f"Unexpected regime: '{detected_regime}' — market_structure keyword not matching"
        fail(msg)
        failures.append(msg)

    # Abort early if regime is wrong — rest of test is meaningless
    if failures:
        _print_summary(failures, warnings, signal)
        return

    # ── Step 4: Analyst LLM Call ──────────────────────────────────────────────
    section("STEP 4 — Analyst LLM Inference")
    info("Scenario conditions being passed to Analyst:")
    info(f"  market_structure : {BB_RSI_SCENARIO['trends']['market_structure']}")
    info(f"  close            : {BB_RSI_SCENARIO['snapshot']['close']}")
    info(f"  bb_lower         : {BB_RSI_SCENARIO['bands']['lower']}")
    info(f"  rsi_current      : {BB_RSI_SCENARIO['rsi']['current']} (just exited <30)")
    info(f"  rsi_trend        : {BB_RSI_SCENARIO['rsi']['trend']}")
    info(f"  volume_spike     : {BB_RSI_SCENARIO['volume']['is_spike']}")
    info(f"  atr              : {BB_RSI_SCENARIO['volatility']['atr']}")
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
        print(f"\n  INTERNAL MONOLOGUE:\n")
        for line in signal.internal_monologue.split("."):
            line = line.strip()
            if line:
                print(f"    {line}.")
    except Exception as e:
        msg = f"Analyst inference failed: {e}"
        fail(msg)
        failures.append(msg)
        _print_summary(failures, warnings, signal)
        return

    # ── Step 5: Strict Assertions ─────────────────────────────────────────────
    section("STEP 5 — Strict Assertions")

    # Assertion 1: Signal must be BUY
    if signal.signal == EXPECTED_SIGNAL:
        ok(f"Signal is BUY ✓")
    else:
        msg = (
            f"Signal assertion failed: expected '{EXPECTED_SIGNAL}', "
            f"got '{signal.signal}'. "
            f"The model either missed the BB bounce, failed the RSI exit check, "
            f"or incorrectly defaulted to HOLD in Sideways regime."
        )
        fail(msg)
        failures.append(msg)

    # Assertion 2: Strategy must be BB+RSI Extremes
    if signal.strategy_used == EXPECTED_STRATEGY:
        ok(f"Strategy is '{EXPECTED_STRATEGY}' ✓")
    else:
        msg = (
            f"Strategy assertion failed: expected '{EXPECTED_STRATEGY}', "
            f"got '{signal.strategy_used}'. "
            f"The model selected a different strategy despite Sideways regime filter."
        )
        fail(msg)
        failures.append(msg)

    # Assertion 3: Confidence must be within range
    if CONFIDENCE_MIN <= signal.confidence <= CONFIDENCE_MAX:
        ok(f"Confidence {signal.confidence:.2f} is within bounds [{CONFIDENCE_MIN}, {CONFIDENCE_MAX}] ✓")
    else:
        msg = (
            f"Confidence out of bounds: {signal.confidence:.2f} "
            f"not in [{CONFIDENCE_MIN}, {CONFIDENCE_MAX}]."
        )
        fail(msg)
        failures.append(msg)

    # Assertion 4: Sideways confidence prompt rule — warn if above 0.5
    if signal.confidence > 0.50:
        w = (
            f"Confidence {signal.confidence:.2f} exceeds Sideways cap of 0.50. "
            f"Prompt rule states confidence > 0.5 is never allowed for Sideways. "
            f"Model may be ignoring this constraint."
        )
        warn(w)
        warnings.append(w)
    else:
        ok(f"Confidence correctly respects Sideways cap of 0.50 ✓")

    # Assertion 5: entry_condition must reference BB or RSI
    entry_lower = signal.entry_condition.lower()
    if any(kw in entry_lower for kw in ["bb", "bollinger", "rsi", "lower band", "oversold"]):
        ok("entry_condition references relevant BB/RSI indicators ✓")
    else:
        msg = (
            f"entry_condition does not reference BB or RSI indicators. "
            f"Got: '{signal.entry_condition}'. "
            f"Model may have synthesized conditions from the wrong strategy."
        )
        fail(msg)
        failures.append(msg)

    # Assertion 6: exit_condition must reference BB or RSI
    exit_lower = signal.exit_condition.lower()
    if any(kw in exit_lower for kw in ["bb", "bollinger", "rsi", "upper band", "overbought", "stop"]):
        ok("exit_condition references relevant BB/RSI indicators ✓")
    else:
        msg = (
            f"exit_condition does not reference BB or RSI indicators. "
            f"Got: '{signal.exit_condition}'. "
        )
        fail(msg)
        failures.append(msg)

    # Assertion 7: valid_till must be a non-empty ISO string
    if signal.valid_till and "T" in signal.valid_till and "Z" in signal.valid_till:
        ok(f"valid_till is a valid ISO UTC string: {signal.valid_till} ✓")
    else:
        msg = f"valid_till is malformed or empty: '{signal.valid_till}'"
        fail(msg)
        failures.append(msg)

    # ── Step 6: Log Output ────────────────────────────────────────────────────
    section("STEP 6 — Log Output")

    os.makedirs("./logs", exist_ok=True)
    log_path = "./logs/scenario_bb_rsi_test.json"

    log_entry = {
        "test_run":        datetime.utcnow().isoformat() + "Z",
        "scenario":        "BB+RSI Extremes — Lower Band Bounce",
        "asset":           ASSET_NAME,
        "injected_data":   BB_RSI_SCENARIO,
        "sentiment":       SENTIMENT_SCORE,
        "analyst_output":  signal.model_dump() if signal else None,
        "assertions": {
            "signal_correct":    signal.signal == EXPECTED_SIGNAL if signal else False,
            "strategy_correct":  signal.strategy_used == EXPECTED_STRATEGY if signal else False,
            "confidence_in_range": CONFIDENCE_MIN <= signal.confidence <= CONFIDENCE_MAX if signal else False,
            "confidence_cap_respected": signal.confidence <= 0.50 if signal else False,
        },
        "failures":  failures,
        "warnings":  warnings,
        "passed":    len(failures) == 0
    }

    with open(log_path, "w") as f:
        json.dump(log_entry, f, indent=2)

    ok(f"Full output written to {log_path}")

    # ── Final Summary ─────────────────────────────────────────────────────────
    _print_summary(failures, warnings, signal)


def _print_summary(failures, warnings, signal):
    print("\n" + "═"*60)
    if not failures:
        print("   ✅  ALL ASSERTIONS PASSED — BB+RSI strategy fires correctly")
    else:
        print(f"   ❌  {len(failures)} ASSERTION(S) FAILED")
        for i, f in enumerate(failures, 1):
            print(f"   {i}. {f}")

    if warnings:
        print(f"\n   ⚠️  {len(warnings)} WARNING(S):")
        for w in warnings:
            print(f"   → {w}")

    if signal:
        print(f"\n   MODEL VERDICT: {signal.signal} | "
              f"Strategy: {signal.strategy_used} | "
              f"Confidence: {signal.confidence:.2f}")
    print("═"*60 + "\n")


if __name__ == "__main__":
    run_bb_rsi_scenario_test()