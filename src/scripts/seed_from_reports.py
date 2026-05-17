import os
import sys
import re
import glob
import json
from datetime import datetime, timezone

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from src.dashboard.shared.database import get_db, AuditLog, QueuedTrade, Portfolio, Position, TradeLedger

def parse_float(s):
    if s is None or s == "N/A" or s == "None":
        return 0.0
    s = str(s).replace("$", "").replace(",", "").strip()
    try:
        return float(s)
    except:
        return 0.0

def process_reports():
    print("Clearing database...")
    db = next(get_db())
    db.query(AuditLog).delete()
    db.query(QueuedTrade).delete()
    db.query(TradeLedger).delete()
    db.query(Position).delete()
    db.query(Portfolio).delete()
    db.commit()

    report_dir = os.path.join(project_root, "Report", "1YearRevised")
    symbols = ["BTC", "ETH", "SOL"]
    
    valid_prefixes = ["2026-03", "2026-04", "2026-05"]

    initial_balance_set = False

    for symbol in symbols:
        print(f"Processing {symbol}...")
        sym_dir = os.path.join(report_dir, symbol)
        if not os.path.exists(sym_dir):
            continue
            
        audit_files = glob.glob(os.path.join(sym_dir, "backtest_*_trade.txt"))
        # We need the one WITHOUT _trade for audit logs
        all_txts = glob.glob(os.path.join(sym_dir, "backtest_*.txt"))
        trade_files = [f for f in all_txts if "_trade" in f]
        main_files = [f for f in all_txts if "_trade" not in f]

        if not main_files or not trade_files:
            print(f"Missing logs for {symbol}")
            continue
            
        # Try to load the original llm_cache from the 1YEAR folder
        cache_file_path = os.path.join(project_root, "Report", "1YEAR", symbol, f"llm_cache_{symbol}USDT.json")
        llm_cache = {}
        if os.path.exists(cache_file_path):
            with open(cache_file_path, "r") as cf:
                try:
                    llm_cache = json.load(cf)
                except:
                    pass

        main_file = main_files[0]
        trade_file = trade_files[0]
        db_symbol = f"{symbol}/USDT"

        # --- Parse Main Audit Logs ---
        with open(main_file, "r") as f:
            content = f.read()

        blocks = content.split("--------------------------------------------------")
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            
            # Extract timestamp
            ts_match = re.search(r"\[(.*?)\] Price:", block)
            if not ts_match:
                continue
            ts_str = ts_match.group(1)
            
            # Check if it falls in Mar, Apr, May 2026
            if not any(ts_str.startswith(vp) for vp in valid_prefixes):
                continue

            try:
                dt = datetime.fromisoformat(ts_str)
            except:
                continue

            # Extract price and balance
            price_match = re.search(r"Price: \$([\d\.]+)", block)
            balance_match = re.search(r"Balance: \$([\d\.]+)", block)
            
            price = parse_float(price_match.group(1)) if price_match else 0.0
            balance = parse_float(balance_match.group(1)) if balance_match else 0.0

            if not initial_balance_set and balance > 0:
                db.add(Portfolio(balance=balance, updated_at=dt))
                initial_balance_set = True

            # Extract Analyst Signal
            signal_match = re.search(r"Analyst Signal:\s*(\w+)", block)
            signal = signal_match.group(1) if signal_match else "HOLD"

            # Extract PnL
            rpnl_match = re.search(r"Realized PnL:\s*([-\$\d\.,]+)", block)
            upnl_match = re.search(r"Unrealized PnL:\s*([-\$\d\.,]+)", block)
            rpnl = parse_float(rpnl_match.group(1)) if rpnl_match else 0.0
            upnl = parse_float(upnl_match.group(1)) if upnl_match else 0.0
            total_pnl = rpnl + upnl

            # Extract Reasoning
            reasoning_match = re.search(r"Reasoning:\s*(.*?)(?:\n|$)", block)
            reasoning = reasoning_match.group(1) if reasoning_match else ""

            # Extract Confidence
            conf_match = re.search(r"CONF:\s*([\d\.]+)", block)
            confidence = parse_float(conf_match.group(1)) if conf_match else 0.0

            factors = {}
            
            # Lookup internal monologue in the cache using the original timestamp string
            monologue = reasoning
            if ts_str in llm_cache:
                try:
                    monologue = llm_cache[ts_str]["signal"]["internal_monologue"]
                except KeyError:
                    pass
            
            factors["internal_monologue"] = monologue
            
            rsi_match = re.search(r"RSI\(14\):\s*([\d\.]+)", block)
            if rsi_match: factors["RSI(14)"] = parse_float(rsi_match.group(1))
            atr_match = re.search(r"ATR:\s*([\d\.]+)", block)
            if atr_match: factors["ATR"] = parse_float(atr_match.group(1))

            audit = AuditLog(
                timestamp=dt,
                symbol=db_symbol,
                action=signal,
                confidence=confidence,
                price=price,
                factors=factors,
                explanation=reasoning,
                profit_loss=total_pnl
            )
            db.add(audit)

        # --- Parse Trade Logs ---
        with open(trade_file, "r") as f:
            t_content = f.read()

        t_blocks = t_content.split("============================================================")
        
        # We need to map ENTRY to EXIT to keep track of QueuedTrade
        # Wait, since the DB model has QueuedTrade for the dashboard, let's just make one for each TRADE OPENED
        
        # State tracker to match CLOSED to OPENED (simple stack or latest)
        active_queued_trade = None

        for t_block in t_blocks:
            t_block = t_block.strip()
            if not t_block:
                continue

            if ">>> TRADE OPENED" in t_block:
                ts_match = re.search(r"Time:\s*(.*)", t_block)
                if not ts_match: continue
                ts_str = ts_match.group(1).strip()
                if not any(ts_str.startswith(vp) for vp in valid_prefixes):
                    continue
                
                try:
                    dt = datetime.fromisoformat(ts_str)
                except: continue

                direction_match = re.search(r"Direction:\s*(.*)", t_block)
                direction = direction_match.group(1).strip() if direction_match else "BUY"
                
                ep_match = re.search(r"Entry Price:\s*([-\$\d\.,]+)", t_block)
                price = parse_float(ep_match.group(1)) if ep_match else 0.0
                
                ps_match = re.search(r"Position Size:\s*([-\$\d\.,]+)", t_block)
                pos_size = parse_float(ps_match.group(1)) if ps_match else 0.0
                
                t_match = re.search(r"Target:\s*([-\$\d\.,]+)", t_block)
                target = parse_float(t_match.group(1)) if t_match else 0.0
                
                sl_match = re.search(r"Stop Loss:\s*([-\$\d\.,]+)", t_block)
                stop_loss = parse_float(sl_match.group(1)) if sl_match else 0.0
                
                entry_c = re.search(r"Entry Cond:\s*(.*)", t_block)
                entry_cond = entry_c.group(1).strip() if entry_c else ""
                
                exit_c = re.search(r"Exit Cond:\s*(.*)", t_block)
                exit_cond = exit_c.group(1).strip() if exit_c else ""
                
                qty = pos_size / price if price > 0 else 0

                db.add(TradeLedger(
                    timestamp=dt, symbol=db_symbol, action=direction,
                    quantity=qty, price=price, fee=pos_size*0.001
                ))

                qt = QueuedTrade(
                    asset_name=db_symbol, signal=direction,
                    entry_condition=entry_cond, exit_condition=exit_cond,
                    target=target, stop_loss=stop_loss, position_size=pos_size, timeframe="4h",
                    status="ENTERED", entry_price=price, entry_time=dt
                )
                db.add(qt)
                active_queued_trade = qt

            elif "<<< TRADE CLOSED" in t_block:
                ts_match = re.search(r"Exit Time:\s*(.*)", t_block)
                if not ts_match: continue
                ts_str = ts_match.group(1).strip()
                if not any(ts_str.startswith(vp) for vp in valid_prefixes):
                    continue
                
                try:
                    dt = datetime.fromisoformat(ts_str)
                except: continue

                direction = "BUY"
                if active_queued_trade:
                    direction = active_queued_trade.signal
                
                ep_match = re.search(r"Exit Price:\s*([-\$\d\.,]+)", t_block)
                exit_price = parse_float(ep_match.group(1)) if ep_match else 0.0
                
                pnl_match = re.search(r"PnL:\s*([-\$\d\.,]+)", t_block)
                pnl = parse_float(pnl_match.group(1)) if pnl_match else 0.0

                # If direction was BUY, close is SELL
                close_action = "SELL" if direction == "BUY" else "BUY"
                
                qty = 0
                if active_queued_trade:
                    qty = active_queued_trade.position_size / active_queued_trade.entry_price if active_queued_trade.entry_price > 0 else 0
                    
                    active_queued_trade.status = "CLOSED"
                    active_queued_trade.exit_price = exit_price
                    active_queued_trade.exit_time = dt
                    active_queued_trade.realized_pnl = pnl

                db.add(TradeLedger(
                    timestamp=dt, symbol=db_symbol, action=close_action,
                    quantity=qty, price=exit_price, fee=qty*exit_price*0.001, realized_pnl=pnl
                ))

    db.commit()
    db.close()
    print("Successfully populated database with reports data!")

if __name__ == "__main__":
    process_reports()
