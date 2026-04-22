import os
import sys
import random
from datetime import datetime, timezone, timedelta

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from src.dashboard.shared.database import get_db, AuditLog, QueuedTrade, Portfolio, Position, TradeLedger, Base, engine

def seed_database():
    print("Seeding database with 1 week of demo data...")
    db = next(get_db())
    
    # We won't wipe the DB, but let's clear existing mock data if any
    db.query(AuditLog).delete()
    db.query(QueuedTrade).delete()
    db.query(TradeLedger).delete()
    db.query(Position).delete()
    db.query(Portfolio).delete()
    
    now = datetime.now(timezone.utc)
    one_week_ago = now - timedelta(days=7)
    
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    current_prices = {"BTC/USDT": 65000.0, "ETH/USDT": 3500.0, "SOL/USDT": 150.0}
    
    # 1. Setup Portfolio
    portfolio = Portfolio(balance=10542.50, updated_at=now)
    db.add(portfolio)
    
    # 2. Generate Audit Logs (6 times a day for 7 days = 42 logs per symbol)
    print("Generating Audit Logs...")
    for symbol in symbols:
        price = current_prices[symbol] * 0.9 # Start 10% lower 1 week ago
        for i in range(42):
            log_time = one_week_ago + timedelta(hours=4*i)
            # Random walk price
            price = price * (1 + random.uniform(-0.02, 0.02))
            
            action = random.choices(["HOLD", "BUY", "SELL"], weights=[0.7, 0.15, 0.15])[0]
            confidence = random.uniform(0.5, 0.95) if action != "HOLD" else random.uniform(0.1, 0.5)
            
            explanation = "Market is chopping sideways. No clear edge."
            if action == "BUY":
                explanation = "RSI oversold on 4h. MACD crossover confirmed. Strong bullish divergence."
            elif action == "SELL":
                explanation = "Price hit major resistance block. Volume declining. Bearish divergence on RSI."
                
            audit = AuditLog(
                timestamp=log_time,
                symbol=symbol,
                action=action,
                confidence=confidence,
                price=price,
                factors={"internal_monologue": f"Thinking about {action}ing {symbol}..."},
                explanation=explanation
            )
            db.add(audit)

    # 3. Generate Historical Trades (Trade Ledger)
    print("Generating Trade Ledger & Positions...")
    for symbol in symbols:
        # Simulate a completed trade
        buy_time = one_week_ago + timedelta(days=random.uniform(1, 3))
        sell_time = buy_time + timedelta(days=random.uniform(1, 2))
        
        buy_price = current_prices[symbol] * 0.95
        sell_price = buy_price * 1.05 # 5% profit
        qty = 1000.0 / buy_price # $1000 position
        
        buy_trade = TradeLedger(
            timestamp=buy_time, symbol=symbol, action="BUY", 
            quantity=qty, price=buy_price, fee=1.0
        )
        sell_trade = TradeLedger(
            timestamp=sell_time, symbol=symbol, action="SELL", 
            quantity=qty, price=sell_price, fee=1.05, realized_pnl=50.0
        )
        db.add(buy_trade)
        db.add(sell_trade)
        
        # Simulate an active position
        if random.random() > 0.5:
            pos_buy_time = now - timedelta(hours=random.uniform(12, 48))
            pos_buy_price = current_prices[symbol] * random.uniform(0.98, 1.02)
            pos_qty = 1500.0 / pos_buy_price
            
            db.add(TradeLedger(
                timestamp=pos_buy_time, symbol=symbol, action="BUY",
                quantity=pos_qty, price=pos_buy_price, fee=1.5
            ))
            
            db.add(Position(
                symbol=symbol, quantity=pos_qty, average_price=pos_buy_price, updated_at=now
            ))
            
            # Active QueuedTrade (ENTERED)
            db.add(QueuedTrade(
                asset_name=symbol, signal="BUY", entry_condition="RSI < 30", exit_condition="MACD < 0",
                target=pos_buy_price * 1.1, stop_loss=pos_buy_price * 0.9, position_size=1500.0, timeframe="4h",
                status="ENTERED", entry_price=pos_buy_price, entry_time=pos_buy_time
            ))

    # 4. Generate some PENDING_APPROVAL and UNENTERED trades
    print("Generating Queued Trades...")
    db.add(QueuedTrade(
        asset_name="BTC/USDT", signal="BUY", entry_condition="price > 66000", exit_condition="price < 64000",
        target=68000, stop_loss=63500, position_size=2000.0, timeframe="4h", status="PENDING_APPROVAL"
    ))
    db.add(QueuedTrade(
        asset_name="ETH/USDT", signal="SELL", entry_condition="RSI > 75", exit_condition="RSI < 50",
        target=3200, stop_loss=3650, position_size=1000.0, timeframe="4h", status="UNENTERED"
    ))

    db.commit()
    db.close()
    print("Successfully populated demo data!")

if __name__ == "__main__":
    seed_database()
