import argparse
import subprocess
import time
import os
import sys
import traceback
from datetime import datetime, timezone
import schedule

project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.tools.binance_client import BinancePublicClient
from src.tools.indicators import add_indicators
from src.agents.analyst import AnalystAgent, MarketDataProcessor
from src.agents.risk_manager import RiskManagerAgent
from src.core.knowledge_base import TradingKnowledgeBase
from src.dashboard.shared.database import get_db, AuditLog

def run_dashboard():
    """Starts the Streamlit dashboard."""
    print("Starting CryptoPilot Dashboard...")
    app_path = os.path.join(project_root, "src", "dashboard", "app.py")
    subprocess.run(["streamlit", "run", app_path])

def background_task():
    """The task that runs periodically in the background worker."""
    print(f"\n[{datetime.now(timezone.utc).isoformat()}] Running background AI analysis...")
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    
    try:
        client = BinancePublicClient()
        kb = TradingKnowledgeBase()
        analyst = AnalystAgent(knowledge_base=kb)
        rm = RiskManagerAgent(total_equity=10000.0)
        db_gen = get_db()
        db = next(db_gen)
        
        for symbol in symbols:
            print(f"Analyzing {symbol}...")
            df = client.get_historical_klines(symbol, "1h", limit=100)
            df = add_indicators(df)
            
            packet = MarketDataProcessor.get_context_packet(df, len(df)-1, window=55)
            if not packet:
                print(f"Not enough data for {symbol}.")
                continue
                
            packet['asset_name'] = symbol
            
            # Run Agents
            signal = analyst.analyze(market_data=packet, sentiment_score=0.5)
            risk_assessment = rm.evaluate(
                analyst_signal=signal,
                market_data=packet,
                sentiment_score=0.5,
                current_portfolio=[]
            )
            
            # Log to Database
            audit_log = AuditLog(
                symbol=symbol,
                action=risk_assessment.signal,
                confidence=signal.confidence,
                price=packet['snapshot']['close'],
                factors={"internal_monologue": signal.internal_monologue},
                explanation=signal.reasoning,
            )
            db.add(audit_log)
            db.commit()
            print(f"Successfully logged {risk_assessment.signal} signal for {symbol}.")
            
    except Exception as e:
        print(f"Background worker error: {str(e)}")
        traceback.print_exc()

def run_worker(interval_minutes=60):
    """Starts the background worker polling the markets."""
    print(f"Starting CryptoPilot Background Worker (polling every {interval_minutes} minutes)...")
    
    # Run once immediately
    background_task()
    
    # Schedule to run periodically
    schedule.every(interval_minutes).minutes.do(background_task)
    
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CryptoPilot CLI")
    parser.add_argument("mode", choices=["dashboard", "worker"], help="Run mode: 'dashboard' or 'worker'")
    parser.add_argument("--interval", type=int, default=60, help="Polling interval in minutes for worker mode")
    
    args = parser.parse_args()
    
    if args.mode == "dashboard":
        run_dashboard()
    elif args.mode == "worker":
        run_worker(interval_minutes=args.interval)
