import sys
import os
import json
from datetime import datetime

sys.path.append(os.path.join(os.getcwd()))

from src.core.knowledge_base import TradingKnowledgeBase
from src.agents.analyst import AnalystAgent
from src.agents.risk_manager import RiskManagerAgent

# ── Mock Market Data ──────────────────────────────────────────────────────────
# Simulates a 4H BTC snapshot in a Bullish Trending regime
# RSI neutral, volume spike present, price above EMA — should trigger Turtle or Momentum

BTC_MARKET_DATA = {
    "asset_name" : "BTCUSDT",
    "snapshot": {
        "open":  67100.00,
        "high":  67800.00,
        "low":   66950.00,
        "close": 67450.25
    },
    "trends": {
        "pct_change_50": 12.4,
        "pct_change_10": 3.1,
        "market_structure": "Uptrend"
    },
    "rsi": {
        "current": 62.5,
        "avg_50":  54.2,
        "trend":   "Increasing"
    },
    "bands": {
        "upper":          69200.00,
        "lower":          64800.00,
        "position_label": "Upper Half"
    },
    "ema": {
        "value":     66100.00,
        "price_rel": "Price Above EMA",
        "slope":     "Increasing"
    },
    "levels": {
        "high_50": 68100.00,
        "low_50":  59200.00,
        "high_20": 67800.00,
        "low_20":  63400.00,
        "high_55": 70100.00,
        "low_55":  55000.00
    },
    "volume": {
        "current": 3850.00,
        "avg_50":  2100.00,
        "is_spike": True
    },
    "volatility": {
        "atr": 680.00
    },
    "stochastic": {
        "current": 72.0
    }
}

SENTIMENT_SCORE = 0.65   # Mildly bullish sentiment
ASSET_NAME      = "BTC/USDT"
TOTAL_EQUITY    = 10_000.0

# ── Helpers ───────────────────────────────────────────────────────────────────

def section(title: str):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")

def ok(msg):  print(f"  ✅  {msg}")
def err(msg): print(f"  ❌  {msg}")
def info(msg):print(f"  ℹ️   {msg}")

# ── Test Runner ───────────────────────────────────────────────────────────────

def run_sanity_test():
    print("\n" + "═"*55)
    print("   AGENTIC CRYPTO TRADER — SANITY TEST")
    print(f"   {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("═"*55)

    analyst      = None
    risk_manager = None
    signal       = None
    assessment   = None

    # ── PHASE 1: Infrastructure ───────────────────────────────────────────────
    section("PHASE 1 — Infrastructure")

    try:
        kb = TradingKnowledgeBase(
            json_path="data/strategies/strategies_processed.json",
            db_dir="database/chroma_db"
        )
        ok("TradingKnowledgeBase loaded")
    except Exception as e:
        err(f"KnowledgeBase failed: {e}")
        return

    try:
        analyst = AnalystAgent(knowledge_base=kb, timeframe_hours=4)
        ok("AnalystAgent initialized (gemma4:e4b)")
    except Exception as e:
        err(f"AnalystAgent init failed: {e}")
        return

    try:
        risk_manager = RiskManagerAgent(total_equity=TOTAL_EQUITY, timeframe_hours=4)
        ok(f"RiskManagerAgent initialized (equity=${TOTAL_EQUITY:,.2f})")
    except Exception as e:
        err(f"RiskManagerAgent init failed: {e}")
        return

    # ── PHASE 2: Knowledge Base Retrieval ────────────────────────────────────
    section("PHASE 2 — Knowledge Base Retrieval")

    try:
        strategies = kb.get_relevant_strategies(
            query="Uptrend bullish momentum above ema high volume breakout entry exit strategy",
            k=3,
            score_threshold=0.45,
            regime_filter="Trending"
        )
        if strategies:
            ok(f"Retrieved {len(strategies)} strategy/strategies")
            for s in strategies:
                info(f"  → {s['metadata']['name']} [{s['metadata']['regime']}]")
        else:
            err("No strategies returned — check ChromaDB or score_threshold")
            return
    except Exception as e:
        err(f"KB retrieval failed: {e}")
        return

    # ── PHASE 3: Analyst Agent ────────────────────────────────────────────────
    section("PHASE 3 — Analyst Agent")
    info(f"Asset       : {ASSET_NAME}")
    info(f"Close Price : ${BTC_MARKET_DATA['snapshot']['close']:,.2f}")
    info(f"RSI         : {BTC_MARKET_DATA['rsi']['current']}")
    info(f"Volume Spike: {BTC_MARKET_DATA['volume']['is_spike']}")
    info(f"Sentiment   : {SENTIMENT_SCORE} (0.5=Neutral)")

    try:
        signal = analyst.analyze(BTC_MARKET_DATA, SENTIMENT_SCORE)
        ok(f"Signal        : {signal.signal}")
        ok(f"Strategy Used : {signal.strategy_used}")
        ok(f"Confidence    : {signal.confidence:.2f}")
        ok(f"Valid Till    : {signal.valid_till}")
        info(f"Entry Cond  : {signal.entry_condition}")
        info(f"Exit Cond   : {signal.exit_condition}")
        info(f"Reasoning   : {signal.reasoning}")
    except Exception as e:
        err(f"Analyst failed: {e}")
        return

    # ── PHASE 4: Risk Manager ─────────────────────────────────────────────────
    section("PHASE 4 — Risk Manager")

    try:
        assessment = risk_manager.evaluate(
            analyst_signal=signal,
            market_data=BTC_MARKET_DATA,
            sentiment_score=SENTIMENT_SCORE,
            current_portfolio=[]       # empty portfolio for sanity test
        )

        ok(f"Final Signal  : {assessment.signal}")
        ok(f"Position Size : ${assessment.position_size:,.2f}")
        ok(f"Stop Loss     : ${assessment.stop_loss:,.4f}")
        tp_str = f"${assessment.target:,.4f}" if assessment.target > 0 else "OPEN (profits run)"
        ok(f"Take Profit   : {tp_str}")
        ok(f"Valid Until   : {assessment.valid_until}")
        info(f"Timeframe   : {assessment.timeframe}")
        info(f"Entry Cond  : {assessment.entry_condition}")
        info(f"Exit Cond   : {assessment.exit_condition}")
        print(f"\n  AUDIT SUMMARY:\n")
        for line in assessment.audit_summary.split("\n"):
            print(f"    {line}")

    except Exception as e:
        err(f"Risk Manager failed: {e}")
        return

    # ── PHASE 5: Signal Consistency Check ────────────────────────────────────
    section("PHASE 5 — Consistency Checks")

    checks_passed = True

    # Check 1: Signal is a valid value
    if signal.signal in ("BUY", "SELL", "HOLD"):
        ok(f"Analyst signal is valid: {signal.signal}")
    else:
        err(f"Invalid analyst signal: {signal.signal}")
        checks_passed = False

    # Check 2: Risk manager signal is valid
    if assessment.signal in ("BUY", "SELL", "HOLD"):
        ok(f"Risk signal is valid: {assessment.signal}")
    else:
        err(f"Invalid risk signal: {assessment.signal}")
        checks_passed = False

    # Check 3: If HOLD, position size should be zero
    if assessment.signal == "HOLD" and assessment.position_size != 0.0:
        err(f"HOLD signal but position_size={assessment.position_size} — should be 0.0")
        checks_passed = False
    else:
        ok("Position size consistent with signal")

    # Check 4: Stop loss is on the correct side of entry
    entry = BTC_MARKET_DATA['snapshot']['close']
    if assessment.signal == "BUY" and assessment.stop_loss >= entry:
        err(f"BUY stop_loss {assessment.stop_loss} >= entry {entry}")
        checks_passed = False
    elif assessment.signal == "SELL" and assessment.stop_loss <= entry:
        err(f"SELL stop_loss {assessment.stop_loss} <= entry {entry}")
        checks_passed = False
    elif assessment.signal != "HOLD":
        ok("Stop loss is on the correct side of entry price")

    # Check 5: Risk manager never flipped direction
    direction_map = {"BUY": 1, "SELL": -1, "HOLD": 0}
    a_dir = direction_map.get(signal.signal, 0)
    r_dir = direction_map.get(assessment.signal, 0)
    if a_dir != 0 and r_dir != 0 and a_dir != r_dir:
        err(f"Risk Manager flipped direction: {signal.signal} → {assessment.signal}")
        checks_passed = False
    else:
        ok("Risk Manager did not flip signal direction")

    # Check 6: valid_till and valid_until are both present and non-empty
    if signal.valid_till and assessment.valid_until:
        ok("valid_till and valid_until are populated")
    else:
        err("Missing valid_till or valid_until")
        checks_passed = False

    # ── PHASE 6: Log Output ───────────────────────────────────────────────────
    section("PHASE 6 — Log Output")

    os.makedirs("./logs", exist_ok=True)
    log_path = "./logs/sanity_test.json"

    log_entry = {
        "test_run":    datetime.utcnow().isoformat() + "Z",
        "asset":       ASSET_NAME,
        "market_data": BTC_MARKET_DATA,
        "sentiment":   SENTIMENT_SCORE,
        "analyst_output": signal.model_dump(),
        "risk_output":    assessment.model_dump(),
        "checks_passed":  checks_passed
    }

    with open(log_path, "w") as f:
        json.dump(log_entry, f, indent=2)

    ok(f"Full output written to {log_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═"*55)
    if checks_passed:
        print("   ✅  ALL CHECKS PASSED — System is functional")
    else:
        print("   ⚠️   SOME CHECKS FAILED — Review output above")
    print("═"*55 + "\n")


run_sanity_test()