import pandas as pd
import sys
import os
import json
import logging
from datetime import datetime
from tqdm import tqdm

# Ensure imports work from src
sys.path.append(os.path.join(os.getcwd()))

from src.core.knowledge_base import TradingKnowledgeBase
from src.agents.analyst import AnalystAgent
from src.agents.risk_manager import RiskManagerAgent

# --- SETUP LOGGING ---
os.makedirs("logs", exist_ok=True)
log_filename = f"logs/backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout) # Prints to console AND file
    ]
)
logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(self, csv_path, initial_balance=10000):
        logger.info("Initializing Backtest Engine...")
        self.df = pd.read_csv(csv_path)
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.portfolio = [] 
        self.trade_history = [] # List of dicts for detailed stats
        
        # Initialize Agents
        kb = TradingKnowledgeBase(pdf_dir="./data/strategies", db_dir="./database/chroma_db")
        self.analyst = AnalystAgent(knowledge_base=kb)
        self.risk_mgr = RiskManagerAgent(total_equity=initial_balance)
        logger.info(f"Agents ready. Data loaded: {len(self.df)} rows.")

    def run(self, limit=20):
        logger.info(f"--- Starting Verbose Backtest: {limit} Candles ---")
        
        for i in range(len(self.df.head(limit))):
            row = self.df.iloc[i]
            timestamp = row.get('timestamp', f'Index_{i}')
            
            print(f"\n{'='*60}")
            print(f"STEP {i+1}/{limit} | TIMESTAMP: {timestamp}")
            print(f"{'='*60}")

            # 1. Market Data Snapshot
            market_data = {
                "symbol": "BTC/USDT",
                "price": float(row['close']),
                "rsi": float(row.get('rsi', 50)),
                "trend": "bullish" if row['close'] > row.get('open', 0) else "bearish",
                "high": float(row['high']),
                "low": float(row['low'])
            }
            logger.info(f"MARKET STATE: Price: {market_data['price']} | RSI: {market_data['rsi']} | Trend: {market_data['trend']}")

            # 2. Portfolio Management (Check SL/TP)
            self._manage_open_positions(market_data['price'], timestamp)

            # 3. Agentic Cycle
            try:
                print("\n[Analyst] Thinking...")
                signal = self.analyst.analyze(market_data, sentiment_score=0.5)
                
                logger.info(f"ANALYST THOUGHTS: {signal.internal_monologue}")
                logger.info(f"ANALYST DECISION: {signal.signal} | Confidence: {signal.confidence}")

                print("\n[Risk Manager] Auditing...")
                verdict = self.risk_mgr.evaluate_trade(signal, market_data, self.portfolio)
                
                logger.info(f"RISK VERDICT: {'APPROVED' if verdict.is_approved else 'REJECTED'}")
                logger.info(f"RISK REASONING: {verdict.risk_monologue}")

                # 4. Execution
                if verdict.is_approved and signal.signal.upper() == "BUY":
                    self._execute_buy(market_data['price'], verdict, timestamp)
                elif signal.signal == "SELL" and self.portfolio:
                    self._execute_sell_all(market_data['price'], timestamp)
                
            except Exception as e:
                logger.error(f"Cycle failed at {timestamp}: {str(e)}")
                continue

        self._print_results()

    def _execute_buy(self, price, verdict, timestamp):
        trade = {
            "entry_time": timestamp,
            "entry_price": price,
            "size": verdict.final_position_size,
            "sl": verdict.stop_loss_price,
            "tp": verdict.take_profit_price
        }
        self.portfolio.append(trade)
        self.balance -= verdict.final_position_size
        logger.info(f"💸 EXECUTION: BOUGHT at {price}. Position Size: ${trade['size']:.2f}")

    def _execute_sell_all(self, price, timestamp):
        # Implementation for manual Sell signals from the Analyst
        for trade in self.portfolio[:]:
            pnl = ((price - trade['entry_price']) / trade['entry_price']) * trade['size']
            self.balance += (trade['size'] + pnl)
            self._log_trade("MANUAL_SELL", trade, price, pnl, timestamp)
            self.portfolio.remove(trade)

    def _manage_open_positions(self, current_price, timestamp):
        if not self.portfolio:
            return
        
        logger.info(f"Checking {len(self.portfolio)} open positions...")
        for trade in self.portfolio[:]:
            # Stop Loss
            if current_price <= trade['sl']:
                pnl = ((current_price - trade['entry_price']) / trade['entry_price']) * trade['size']
                self.balance += (trade['size'] + pnl)
                self._log_trade("STOP_LOSS", trade, current_price, pnl, timestamp)
                self.portfolio.remove(trade)
            
            # Take Profit
            elif current_price >= trade['tp']:
                pnl = ((current_price - trade['entry_price']) / trade['entry_price']) * trade['size']
                self.balance += (trade['size'] + pnl)
                self._log_trade("TAKE_PROFIT", trade, current_price, pnl, timestamp)
                self.portfolio.remove(trade)

    def _log_trade(self, exit_type, trade, exit_price, pnl, timestamp):
        result = "WIN" if pnl > 0 else "LOSS"
        logger.warning(f"🎯 EXIT [{exit_type}]: {result} | PnL: ${pnl:.2f} | Final Balance: ${self.balance:.2f}")
        self.trade_history.append({
            "exit_type": exit_type,
            "pnl": pnl,
            "timestamp": timestamp
        })

    def _print_results(self):
        roi = ((self.balance - self.initial_balance) / self.initial_balance) * 100
        wins = len([t for t in self.trade_history if t['pnl'] > 0])
        losses = len([t for t in self.trade_history if t['pnl'] <= 0])
        
        print("\n" + "#"*60)
        print(f"📊 BACKTEST FINAL REPORT")
        print("#"*60)
        print(f"Initial Balance:  ${self.initial_balance}")
        print(f"Final Balance:    ${self.balance:.2f}")
        print(f"Total ROI:        {roi:.2f}%")
        print(f"Total Trades:     {len(self.trade_history)}")
        print(f"Win Rate:         {(wins/len(self.trade_history)*100 if self.trade_history else 0):.2f}% ({wins}W / {losses}L)")
        print(f"Full log saved to: {log_filename}")
        print("#"*60)

if __name__ == "__main__":
    # Ensure your data folder and file path are correct
    tester = BacktestEngine(csv_path="crypto_data/BTCUSDT_4h_indicators.csv") 
    tester.run(limit=15)