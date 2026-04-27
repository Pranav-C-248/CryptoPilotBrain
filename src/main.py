import argparse
import subprocess
import time
import os
import sys
import traceback
from datetime import datetime, timezone
import schedule
import threading

project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.tools.binance_client import BinancePublicClient
from src.tools.indicators import add_indicators
from src.agents.analyst import AnalystAgent, MarketDataProcessor
from src.agents.risk_manager import RiskManagerAgent
from src.core.knowledge_base import TradingKnowledgeBase
from src.dashboard.shared.database import get_db, AuditLog
from src.core.paper_trader import PaperTradingEngine

from src.agents.data_agent import MainDataAgent

# Global Paper Trading Engine instance
engine = PaperTradingEngine()

def run_dashboard():
    """Starts the Streamlit dashboard."""
    print("Starting CryptoPilot Dashboard...")
    app_path = os.path.join(project_root, "src", "dashboard", "app.py")
    subprocess.run(["streamlit", "run", app_path])

def background_task():
    """The task that runs periodically in the background worker."""
    agent = MainDataAgent()
    agent.run_analysis_cycle()

def run_worker(interval_minutes=240):
    """Starts the background worker polling the markets."""
    print(f"Starting CryptoPilot Background Worker (polling every {interval_minutes} minutes)...")
    
    # Run once immediately
    background_task()
    
    # Schedule to run periodically
    schedule.every(interval_minutes).minutes.do(background_task)
    
    while True:
        schedule.run_pending()
        time.sleep(1)

def engine_loop():
    """Runs the Paper Trading Engine periodically to process the queue."""
    print("Starting Paper Trading Engine Loop...")
    while True:
        try:
            engine.process_queue()
        except Exception as e:
            print(f"Engine loop error: {e}")
        time.sleep(60) # Run every minute

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CryptoPilot CLI")
    parser.add_argument("mode", choices=["dashboard", "worker"], help="Run mode: 'dashboard' or 'worker'")
    parser.add_argument("--interval", type=int, default=60, help="Polling interval in minutes for worker mode")
    
    args = parser.parse_args()
    
    # Start the Paper Trading Engine in a background daemon thread
    t = threading.Thread(target=engine_loop, daemon=True)
    t.start()
    
    if args.mode == "dashboard":
        run_dashboard()
    elif args.mode == "worker":
        run_dashboard()
        run_worker(interval_minutes=args.interval)
