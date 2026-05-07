import os
import sys
import traceback
from datetime import datetime, timezone

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from src.tools.binance_client import BinancePublicClient
from src.tools.indicators import MarketDataProcessor
from src.agents.analyst_lms import AnalystAgent
from src.agents.risk_manager import RiskManagerAgent
from src.core.knowledge_base_lms import TradingKnowledgeBase
from src.dashboard.shared.database import get_db, AuditLog
from src.core.paper_trader import PaperTradingEngine
from src.agents.sentiment_agent import SentimentAgent

class MainDataAgent:
    """
    Main orchestrator agent that fetches market data, runs the Analyst and Risk Manager,
    and passes approved trades to the Paper Trading Engine.
    """
    def __init__(self):
        self.client = BinancePublicClient()
        self.kb = TradingKnowledgeBase()
        self.analyst = AnalystAgent(knowledge_base=self.kb)
        self.rm = RiskManagerAgent(total_equity=10000.0, timeframe_hours=4)
        self.engine = PaperTradingEngine()
        self.sentiment_agent = SentimentAgent()
        self.symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

    def run_analysis_cycle(self):
        print(f"\n[{datetime.now(timezone.utc).isoformat()}] MainDataAgent starting 4h analysis cycle...")
        
        db_gen = get_db()
        db = next(db_gen)
        
        try:
            for symbol in self.symbols:
                self.run_for_symbol(symbol, db)
        except Exception as e:
            print(f"MainDataAgent error: {str(e)}")
            traceback.print_exc()
        finally:
            db.close()
            print(f"[{datetime.now(timezone.utc).isoformat()}] MainDataAgent cycle complete.")

    def run_for_symbol(self, symbol: str, db=None):
        should_close_db = False
        if db is None:
            db = next(get_db())
            should_close_db = True
            
        try:
            print(f"Analyzing {symbol} (4h)...")
            df = self.client.get_historical_klines(symbol, "4h", limit=250)
            packet = MarketDataProcessor.get_context_packet(df, len(df)-1, window=200)
            if not packet:
                print(f"Not enough data for {symbol}.")
                return
                
            packet['asset_name'] = symbol
            
            # 0. Fetch Sentiment
            sentiment_score = 0.5
            try:
                sentiment_score = self.sentiment_agent.get_aggregate_sentiment(limit=5)
                print(f"Sentiment score for {symbol} is {sentiment_score}")
            except Exception as e:
                print(f"Failed to fetch sentiment, defaulting to 0.5. Error: {e}")

            # 1. Analyst Agent
            signal = self.analyst.analyze(market_data=packet, sentiment_score=sentiment_score)
            
            # 2. Risk Manager Agent
            risk_assessment = self.rm.evaluate(
                analyst_signal=signal,
                market_data=packet,
                sentiment_score=sentiment_score,
                current_portfolio=[]
            )
            
            # 3. Log to Database
            audit_log = AuditLog(
                symbol=symbol,
                action=risk_assessment.signal,
                confidence=signal.confidence,
                price=packet['snapshot']['close'],
                factors={"internal_monologue": getattr(signal, 'internal_monologue', "N/A")},
                explanation=signal.reasoning,
            )
            db.add(audit_log)
            db.commit()
            print(f"Successfully logged {risk_assessment.signal} signal for {symbol}.")
            
            # 4. Pass to Paper Trading Engine
            if risk_assessment.signal in ["BUY", "SELL"]:
                self.engine.add_to_queue(risk_assessment, symbol, risk_assessment.signal)
        finally:
            if should_close_db:
                db.close()
