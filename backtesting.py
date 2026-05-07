import os
import sys
import ast
import json
import time
import argparse
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure imports work from src
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.core.knowledge_base import TradingKnowledgeBase
from src.agents.analyst import AnalystAgent
from src.tools.indicators import MarketDataProcessor
from src.agents.risk_manager import RiskManagerAgent
from src.schema.models import AnalystSignal
from src.agents.risk_manager import RiskAssessment

CACHE_FILE = os.path.join(project_root, "tests", "llm_cache.json")
os.makedirs(os.path.join(project_root, "tests", "logs"), exist_ok=True)

class BacktestEngine:
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.portfolio = [] # Active trades
        self.trade_history = [] # Closed trades
        self.equity_curve = [] # Track balance over time
        
        self.kb = TradingKnowledgeBase()
        self.analyst = AnalystAgent(knowledge_base=self.kb)
        self.risk_mgr = RiskManagerAgent(total_equity=initial_balance)
        self.cache = self._load_cache()
        self.log_file = os.path.join(project_root, "tests", "logs", f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
        # Initialize log file
        with open(self.log_file, "w") as f:
            f.write(f"=== BACKTEST START: {datetime.now(timezone.utc).isoformat()} ===\n")
        
    def _load_cache(self):
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        return {}
        
    def _save_cache(self):
        with open(CACHE_FILE, "w") as f:
            json.dump(self.cache, f, indent=2)

    def _log_step(self, timestamp, price, signal, verdict):
        realized_pnl = sum(t['pnl'] for t in self.trade_history)
        
        unrealized_pnl = 0
        for trade in self.portfolio:
            unrealized_pnl += ((price - trade['entry_price']) / trade['entry_price']) * trade['position_size']
            
        with open(self.log_file, "a") as f:
            f.write(f"\n[{timestamp}] Price: ${price:.2f} | Balance: ${self.balance:.2f}\n")
            f.write(f"Realized PnL: ${realized_pnl:.2f} | Unrealized PnL: ${unrealized_pnl:.2f}\n")
            if signal:
                f.write(f"Analyst Signal: {signal.signal} | Reasoning: {signal.reasoning}\n")
            if verdict:
                f.write(f"Risk Verdict: {'APPROVED' if verdict.is_approved else 'REJECTED'} | Size: ${verdict.position_size if hasattr(verdict, 'position_size') else 0:.2f}\n")
                f.write(f"Risk Monologue: {verdict.risk_monologue}\n")
            f.write("-" * 50 + "\n")



    def _evaluate_condition(self, condition_str: str, locals_dict: dict) -> bool:
        """Evaluates a simple comparison condition string safely using AST parsing.
        
        Supports expressions like 'RSI < 70', 'price > bb_lower * 1.01 and rsi < 30',
        using only variables present in locals_dict.
        """
        if not condition_str or condition_str.lower() in ("true", "none", "n/a"):
            return True
        if condition_str.lower() in ("false",):
            return False

        try:
            tree = ast.parse(condition_str, mode='eval')
            result = self._safe_eval_node(tree.body, locals_dict)
            return bool(result)
        except Exception:
            return False

    def _safe_eval_node(self, node: ast.AST, variables: dict):
        """Recursively evaluate an AST node, allowing only safe operations."""
        # Numeric / string / boolean constants
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, bool, str)):
                return node.value
            raise ValueError(f"Disallowed constant type: {type(node.value)}")

        # Variable lookup
        if isinstance(node, ast.Name):
            var_id = node.id.lower()
            if var_id == 'true':
                return True
            if var_id == 'false':
                return False
            if var_id in variables:
                return variables[var_id]
            if node.id in variables:
                return variables[node.id]
            raise NameError(f"Unknown variable: {node.id}")

        # Unary operators: -x, +x, not x
        if isinstance(node, ast.UnaryOp):
            operand = self._safe_eval_node(node.operand, variables)
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.Not):
                return not operand
            raise ValueError(f"Disallowed unary op: {type(node.op).__name__}")

        # Binary arithmetic: +, -, *, /, //
        if isinstance(node, ast.BinOp):
            left = self._safe_eval_node(node.left, variables)
            right = self._safe_eval_node(node.right, variables)
            ops = {
                ast.Add: lambda a, b: a + b,
                ast.Sub: lambda a, b: a - b,
                ast.Mult: lambda a, b: a * b,
                ast.Div: lambda a, b: a / b,
                ast.FloorDiv: lambda a, b: a // b,
            }
            op_func = ops.get(type(node.op))
            if op_func is None:
                raise ValueError(f"Disallowed binary op: {type(node.op).__name__}")
            return op_func(left, right)

        # Comparisons: <, >, <=, >=, ==, !=
        if isinstance(node, ast.Compare):
            left = self._safe_eval_node(node.left, variables)
            cmp_ops = {
                ast.Lt: lambda a, b: a < b,
                ast.LtE: lambda a, b: a <= b,
                ast.Gt: lambda a, b: a > b,
                ast.GtE: lambda a, b: a >= b,
                ast.Eq: lambda a, b: a == b,
                ast.NotEq: lambda a, b: a != b,
            }
            # Chained comparisons: a < b < c  →  a < b and b < c
            for op, comparator in zip(node.ops, node.comparators):
                right = self._safe_eval_node(comparator, variables)
                op_func = cmp_ops.get(type(op))
                if op_func is None:
                    raise ValueError(f"Disallowed comparison: {type(op).__name__}")
                if not op_func(left, right):
                    return False
                left = right
            return True

        # Boolean operators: and, or
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return all(self._safe_eval_node(v, variables) for v in node.values)
            if isinstance(node.op, ast.Or):
                return any(self._safe_eval_node(v, variables) for v in node.values)
            raise ValueError(f"Disallowed bool op: {type(node.op).__name__}")

        raise ValueError(f"Disallowed AST node: {type(node).__name__}")

    def run(self, df: pd.DataFrame, limit: int = None):
        print("\n--- Starting Backtest Simulation ---")
        window = 200
        total_steps = len(df) - window
        print(window,total_steps)
        if limit:
            total_steps = min(total_steps, limit)
            
        for i in range(window, window + total_steps):
            row = df.iloc[i]
            timestamp = row['open_time'].isoformat()
            price = row['close']
            
            # Record equity
            self.equity_curve.append({
                "timestamp": timestamp,
                "balance": self.balance,
                "price": price
            })
            
            print(f"\nStep {i-window+1}/{total_steps} | {timestamp} | Price: ${price:.2f}")
            
            # Build execution namespace for evaluating conditions
            locals_dict = {col: row[col] for col in df.columns}
            locals_dict['price'] = price
            
            # Add donchian levels required by prompts
            locals_dict['high_10'] = df['high'].iloc[i-9:i+1].max()
            locals_dict['low_10'] = df['low'].iloc[i-9:i+1].min()
            locals_dict['high_20'] = df['high'].iloc[i-19:i+1].max()
            locals_dict['low_20'] = df['low'].iloc[i-19:i+1].min()
            locals_dict['high_50'] = df['high'].iloc[i-49:i+1].max()
            locals_dict['low_50'] = df['low'].iloc[i-49:i+1].min()
            locals_dict['high_55'] = df['high'].iloc[i-54:i+1].max()
            locals_dict['low_55'] = df['low'].iloc[i-54:i+1].min()
            
            # Make case-insensitive by adding uppercase versions of all keys
            locals_dict.update({k.upper(): v for k, v in locals_dict.items() if isinstance(k, str)})
            
            # 1. Check Exits for Open Trades
            # print("1 step")
            for trade in self.portfolio[:]:
                exit_triggered = False
                exit_reason = ""
                
                if price <= trade['stop_loss']:
                    exit_triggered, exit_reason = True, "Stop Loss"
                elif price >= trade['target']:
                    exit_triggered, exit_reason = True, "Target"
                elif self._evaluate_condition(trade['exit_condition'], locals_dict):
                    exit_triggered, exit_reason = True, "Dynamic Condition"
                    
                if exit_triggered:
                    pnl = ((price - trade['entry_price']) / trade['entry_price']) * trade['position_size']
                    self.balance += trade['position_size'] + pnl
                    trade['exit_price'] = price
                    trade['exit_time'] = timestamp
                    trade['exit_reason'] = exit_reason
                    trade['pnl'] = pnl
                    self.trade_history.append(trade)
                    self.portfolio.remove(trade)
                    print(f"  -> EXIT [{exit_reason}]: PnL ${pnl:.2f} | Balance: ${self.balance:.2f}")
            
            # 2. Get Context Packet
            # print("2 step")
            packet = MarketDataProcessor.get_context_packet(df, i, window=window)
            if not packet: continue
            packet['asset_name'] = "BTC/USDT"
            
            # 3. Cache / Agent Cycle
            print("3 step")
            if timestamp in self.cache:
                print("  -> Using Cached LLM Output")
                cache_data = self.cache[timestamp]
                signal = AnalystSignal.model_validate(cache_data['signal'])
                verdict = RiskAssessment.model_validate(cache_data['risk'])
            else:
                try:
                    signal = self.analyst.analyze(market_data=packet, sentiment_score=0.5)
                    verdict = self.risk_mgr.evaluate(
                        analyst_signal=signal,
                        market_data=packet,
                        sentiment_score=0.5,
                        current_portfolio=self.portfolio
                    )
                    print(signal,verdict)
                    
                    self.cache[timestamp] = {
                        "signal": signal.model_dump(),
                        "risk": verdict.model_dump()
                    }
                    self._save_cache()
                    
                except Exception as e:
                    print(f"  -> API Error at {timestamp}: {e}")
                    # Save progress and sleep
                    self._save_cache()
                    time.sleep(10)
                    continue

            # 4. Entry Execution
            # print("4 step")
            if verdict.signal == "BUY":
                # We only take ONE position at a time for simplicity in this backtest
                if len(self.portfolio) == 0:
                    pos_size = verdict.position_size
                    if pos_size > self.balance:
                        pos_size = self.balance # Constrain to balance
                        
                    if pos_size > 0:
                        self.balance -= pos_size
                        trade = {
                            "asset_name": verdict.asset_name,
                            "entry_time": timestamp,
                            "entry_price": price,
                            "position_size": pos_size,
                            "target": verdict.target,
                            "stop_loss": verdict.stop_loss,
                            "entry_condition": signal.entry_condition,
                            "exit_condition": signal.exit_condition
                        }
                        self.portfolio.append(trade)
                        print(f"  -> ENTRY [BUY]: Size ${pos_size:.2f} @ ${price:.2f}")

            # 5. Step Logging
            if 'signal' in locals() and 'verdict' in locals():
                self._log_step(timestamp, price, signal, verdict)
            else:
                self._log_step(timestamp, price, None, None)

        # Close all open positions at end of test to realize PnL
        # print("ur mum")
        if self.portfolio:
            final_price = df.iloc[window + total_steps - 1]['close']
            final_time = df.iloc[window + total_steps - 1]['open_time'].isoformat()
            for trade in self.portfolio[:]:
                pnl = ((final_price - trade['entry_price']) / trade['entry_price']) * trade['position_size']
                self.balance += trade['position_size'] + pnl
                trade['exit_price'] = final_price
                trade['exit_time'] = final_time
                trade['exit_reason'] = "End of Backtest"
                trade['pnl'] = pnl
                self.trade_history.append(trade)
            self.portfolio.clear()

        # Final record
        self.equity_curve.append({
            "timestamp": df.iloc[window + total_steps - 1]['open_time'].isoformat(),
            "balance": self.balance,
            "price": df.iloc[window + total_steps - 1]['close']
        })

    def print_metrics(self):
        roi = ((self.balance - self.initial_balance) / self.initial_balance) * 100
        wins = [t for t in self.trade_history if t['pnl'] > 0]
        losses = [t for t in self.trade_history if t['pnl'] <= 0]
        win_rate = (len(wins) / len(self.trade_history) * 100) if self.trade_history else 0
        
        gross_profit = sum(t['pnl'] for t in wins)
        gross_loss = abs(sum(t['pnl'] for t in losses))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

        print("\n" + "="*50)
        print("📊 BACKTEST FINAL REPORT")
        print("="*50)
        print(f"Initial Balance:  ${self.initial_balance:,.2f}")
        print(f"Final Balance:    ${self.balance:,.2f}")
        print(f"Net Profit:       ${(self.balance - self.initial_balance):,.2f}")
        print(f"Total ROI:        {roi:.2f}%")
        print(f"Total Trades:     {len(self.trade_history)}")
        print(f"Win Rate:         {win_rate:.2f}% ({len(wins)}W / {len(losses)}L)")
        print(f"Profit Factor:    {profit_factor:.2f}")
        print("="*50)

        # Save metrics to table
        metrics_df = pd.DataFrame([{
            "Initial Balance": self.initial_balance,
            "Final Balance": self.balance,
            "Net Profit": self.balance - self.initial_balance,
            "Total ROI %": roi,
            "Total Trades": len(self.trade_history),
            "Win Rate %": win_rate,
            "Profit Factor": profit_factor
        }])
        metrics_df.to_csv(os.path.join(project_root, "tests", "metrics.csv"), index=False)
        print("Metrics saved to tests/metrics.csv")

    def generate_graphs(self, df: pd.DataFrame):
        print("Generating graphs...")
        
        # 1. Equity Curve
        eq_df = pd.DataFrame(self.equity_curve)
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=eq_df['timestamp'], y=eq_df['balance'], mode='lines', name='Balance ($)', line=dict(color='#c2ef4e', width=2)))
        fig1.update_layout(title="Equity Curve", template="plotly_dark", plot_bgcolor="#1f1633", paper_bgcolor="#150f23")
        fig1.write_html(os.path.join(project_root, "tests", "equity_curve.html"))
        fig1.write_image(os.path.join(project_root, "tests", "equity_curve.png"))

        # 2. Trade Chart (Candlestick + Entries/Exits)
        # We plot the entire range that was tested
        start_ts = eq_df['timestamp'].iloc[0]
        end_ts = eq_df['timestamp'].iloc[-1]
        mask = (df['open_time'] >= start_ts) & (df['open_time'] <= end_ts)
        chart_df = df.loc[mask].copy()
        
        fig2 = go.Figure(data=[go.Candlestick(x=chart_df['open_time'],
                open=chart_df['open'], high=chart_df['high'],
                low=chart_df['low'], close=chart_df['close'],
                name='Price')])

        # Add trades
        for trade in self.trade_history:
            # Entry (Green Triangle Up)
            fig2.add_trace(go.Scatter(
                x=[trade['entry_time']], y=[trade['entry_price']],
                mode='markers', marker=dict(symbol='triangle-up', size=12, color='#0ecb81'),
                name=f"BUY", hovertext=f"Size: ${trade['position_size']:.2f}"
            ))
            # Exit (Red Triangle Down or Circle depending on reason)
            color = '#f6465d' if trade['pnl'] < 0 else '#0ecb81'
            fig2.add_trace(go.Scatter(
                x=[trade['exit_time']], y=[trade['exit_price']],
                mode='markers', marker=dict(symbol='triangle-down', size=12, color=color),
                name=f"EXIT ({trade['exit_reason']})", hovertext=f"PnL: ${trade['pnl']:.2f}"
            ))
            
        fig2.update_layout(title="Trade Executions", template="plotly_dark", xaxis_rangeslider_visible=False, plot_bgcolor="#1f1633", paper_bgcolor="#150f23")
        fig2.write_html(os.path.join(project_root, "tests", "trade_chart.html"))
        fig2.write_image(os.path.join(project_root, "tests", "trade_chart.png"))
        print(f"Graphs saved to tests/equity_curve.html/png and tests/trade_chart.html/png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Historical Backtest")
    parser.add_argument("--limit", type=int, default=None, help="Number of candles to process. Omit to run full dataset.")
    parser.add_argument("--data", type=str, default="tests/BTCUSDT_4h_historical.csv", help="Path to historical data CSV")
    args = parser.parse_args()

    engine = BacktestEngine(initial_balance=10000)
    
    # 1. Fetch & prep data
    csv_path = os.path.join(project_root, args.data)
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} does not exist. Please run 'python scripts/download_historical_data.py' first.")
        sys.exit(1)
        
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    df['open_time'] = pd.to_datetime(df['open_time'], format='ISO8601', utc=True)
    df['close_time'] = pd.to_datetime(df['close_time'], format='ISO8601', utc=True)
    
    # 2. Run simulation
    # print(df.head(10))
    engine.run(df, limit=args.limit)
    
    # 3. Results
    engine.print_metrics()
    engine.generate_graphs(df)
