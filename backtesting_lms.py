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
import html
import re

# Ensure imports work from src
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.core.knowledge_base_lms import TradingKnowledgeBase
from src.agents.analyst_lms import AnalystAgent
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
        log_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(project_root, "tests", "logs", f"backtest_{log_ts}.txt")
        self.trade_log_file = os.path.join(project_root, "tests", "logs", f"backtest_{log_ts}_trade.txt")
        
        # Initialize log files
        start_header = f"=== BACKTEST START: {datetime.now(timezone.utc).isoformat()} ===\n"
        with open(self.log_file, "w") as f:
            f.write(start_header)
        with open(self.trade_log_file, "w") as f:
            f.write(start_header)
        
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
                f.write(f"Risk Verdict| Size: ${verdict.position_size if hasattr(verdict, 'position_size') else 0:.2f}\n")
                f.write(f"Risk audit summary: {verdict.audit_summary}\n")
            f.write("-" * 50 + "\n")

    def _log_trade(self, action: str, trade: dict, current_price: float = None):
        """Log detailed trade execution info to the log file."""
        with open(self.trade_log_file, "a") as f:
            f.write("\n" + "=" * 60 + "\n")
            if action == "ENTRY":
                f.write(f">>> TRADE OPENED [{trade.get('asset_name', 'N/A')}]\n")
                f.write(f"    Time:          {trade['entry_time']}\n")
                f.write(f"    Direction:     BUY\n")
                f.write(f"    Entry Price:   ${trade['entry_price']:.2f}\n")
                f.write(f"    Position Size: ${trade['position_size']:.2f}\n")
                f.write(f"    Target:        ${trade['target']:.2f}\n")
                f.write(f"    Stop Loss:     ${trade['stop_loss']:.2f}\n")
                risk_amt = ((trade['entry_price'] - trade['stop_loss']) / trade['entry_price']) * trade['position_size']
                reward_amt = ((trade['target'] - trade['entry_price']) / trade['entry_price']) * trade['position_size']
                rr_ratio = reward_amt / risk_amt if risk_amt > 0 else float('inf')
                f.write(f"    Risk ($):      ${risk_amt:.2f}\n")
                f.write(f"    Reward ($):    ${reward_amt:.2f}\n")
                f.write(f"    R:R Ratio:     {rr_ratio:.2f}\n")
                f.write(f"    Entry Cond:    {trade.get('entry_condition', 'N/A')}\n")
                f.write(f"    Exit Cond:     {trade.get('exit_condition', 'N/A')}\n")
                f.write(f"    Balance After: ${self.balance:.2f}\n")
            elif action == "EXIT":
                f.write(f"<<< TRADE CLOSED [{trade.get('asset_name', 'N/A')}]\n")
                f.write(f"    Entry Time:    {trade['entry_time']}\n")
                f.write(f"    Exit Time:     {trade['exit_time']}\n")
                f.write(f"    Entry Price:   ${trade['entry_price']:.2f}\n")
                f.write(f"    Exit Price:    ${trade['exit_price']:.2f}\n")
                f.write(f"    Position Size: ${trade['position_size']:.2f}\n")
                f.write(f"    Target:        ${trade['target']:.2f}\n")
                f.write(f"    Stop Loss:     ${trade['stop_loss']:.2f}\n")
                pnl = trade['pnl']
                pnl_pct = (pnl / trade['position_size']) * 100 if trade['position_size'] > 0 else 0
                result_emoji = "✅" if pnl > 0 else "❌"
                f.write(f"    PnL:           ${pnl:.2f} ({pnl_pct:+.2f}%) {result_emoji}\n")
                f.write(f"    Exit Reason:   {trade['exit_reason']}\n")
                f.write(f"    Balance After: ${self.balance:.2f}\n")
            f.write("=" * 60 + "\n")



    def _evaluate_condition(self, condition_str: str, locals_dict: dict) -> bool:
        """Evaluates a simple comparison condition string safely using AST parsing.
        
        Supports expressions like 'RSI < 70', 'price > bb_lower * 1.01 and rsi < 30',
        using only variables present in locals_dict.
        """
        if isinstance(condition_str, bool):
            return condition_str
            
        if not condition_str:
            return False
            
        condition_str = str(condition_str)
        condition_str = html.unescape(condition_str)
        
        # Auto-fix common LLM syntax hallucinations
        condition_str = re.sub(r'(?i)(\d+)-day\s*high', r'high_\1', condition_str)
        condition_str = re.sub(r'(?i)(\d+)-day\s*low', r'low_\1', condition_str)
        
        if condition_str.lower() in ("true", "none", "n/a"):
            return True
        if condition_str.lower() in ("false", "hold"):
            return False

        try:
            tree = ast.parse(condition_str, mode='eval')
            result = self._safe_eval_node(tree.body, locals_dict)
            return bool(result)
        except Exception as e:
            print(f"Error evaluating condition '{condition_str}': {e}")
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
            
            # Record equity (including PnL components for metrics calculation)
            realized_pnl = sum(t['pnl'] for t in self.trade_history)
            unrealized_pnl = 0
            for trade in self.portfolio:
                unrealized_pnl += ((price - trade['entry_price']) / trade['entry_price']) * trade['position_size']
            self.equity_curve.append({
                "timestamp": timestamp,
                "balance": self.balance,
                "price": price,
                "realized_pnl": realized_pnl,
                "unrealized_pnl": unrealized_pnl,
            })
            
            print(f"\nStep {i-window+1}/{total_steps} | {timestamp} | Price: ${price:.2f} | Balance: ${self.balance:.2f} | ")
            
            # Build execution namespace for evaluating conditions
            locals_dict = {col: row[col] for col in df.columns}
            locals_dict['price'] = price
            
            # Inject commonly hallucinated prev metrics
            if i > 0:
                prev_row = df.iloc[i-1]
                locals_dict['rsi_prev'] = prev_row.get('rsi', 0)
                locals_dict['rsi_current'] = row.get('rsi', 0)
                locals_dict['rsi5_prev'] = prev_row.get('rsi5', 0)
                locals_dict['rsi5_current'] = row.get('rsi5', 0)
            
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
                
                if row["high"] >= trade['target']:
                    exit_triggered, exit_reason = True, "Target"
                elif row["low"] <= trade['stop_loss']:
                    exit_triggered, exit_reason = True, "Stop Loss"
                elif self._evaluate_condition(trade['exit_condition'], locals_dict):
                    exit_triggered, exit_reason = True, "Dynamic Condition"
                    
                if exit_triggered:
                    if exit_reason == "Stop Loss":
                        exit_price = trade['stop_loss']
                    elif exit_reason == "Target":
                        exit_price = trade['target']
                    elif exit_reason == "Dynamic Condition":
                        exit_price = price
                    pnl = ((exit_price - trade['entry_price']) / trade['entry_price']) * trade['position_size']
                    self.balance += trade['position_size'] + pnl
                    trade['exit_price'] = exit_price
                    trade['exit_time'] = timestamp
                    trade['exit_reason'] = exit_reason
                    trade['pnl'] = pnl
                    self.trade_history.append(trade)
                    self.portfolio.remove(trade)
                    self._log_trade("EXIT", trade)
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
                if len(self.portfolio) >= 0:
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
                        self._log_trade("ENTRY", trade)
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
                self._log_trade("EXIT", trade)
            self.portfolio.clear()

        # Final record
        final_realized_pnl = sum(t['pnl'] for t in self.trade_history)
        self.equity_curve.append({
            "timestamp": df.iloc[window + total_steps - 1]['open_time'].isoformat(),
            "balance": self.balance,
            "price": df.iloc[window + total_steps - 1]['close'],
            "realized_pnl": final_realized_pnl,
            "unrealized_pnl": 0,  # all positions closed
        })

    def compute_advanced_metrics(self) -> dict:
        """Compute all advanced performance metrics from in-memory simulation data.
        
        Calculates: ROI, Sharpe/Sortino/Calmar ratios, max drawdown, drawdown
        duration, alpha vs buy-and-hold, and full per-trade statistics.
        No log parsing — everything comes from self.equity_curve and self.trade_history.
        """
        metrics = {}

        # --- Build equity DataFrame from in-memory curve ---
        eq_df = pd.DataFrame(self.equity_curve)
        eq_df["timestamp"] = pd.to_datetime(eq_df["timestamp"], utc=True)
        eq_df = eq_df.sort_values("timestamp").reset_index(drop=True)

        # Total equity = initial_balance + realised + unrealised
        eq_df["total_equity"] = self.initial_balance + eq_df["realized_pnl"] + eq_df["unrealized_pnl"]

        initial_price = eq_df["price"].iloc[0]
        final_price = eq_df["price"].iloc[-1]
        final_equity = eq_df["total_equity"].iloc[-1]

        # ---- Basic ----
        metrics["Initial Balance"] = self.initial_balance
        metrics["Final Equity"] = final_equity
        metrics["Net Profit"] = final_equity - self.initial_balance
        metrics["Total ROI %"] = ((final_equity - self.initial_balance) / self.initial_balance) * 100

        # ---- Asset Price ----
        metrics["Asset Start Price"] = initial_price
        metrics["Asset End Price"] = final_price
        metrics["Asset Return %"] = ((final_price - initial_price) / initial_price) * 100

        # ---- Returns series ----
        eq_df["returns"] = eq_df["total_equity"].pct_change().fillna(0)

        # ---- Sharpe Ratio (annualized, 4h candles -> 6/day -> 2190/year) ----
        periods_per_year = 6 * 365
        mean_ret = eq_df["returns"].mean()
        std_ret = eq_df["returns"].std()
        metrics["Sharpe Ratio (Annualized)"] = (
            (mean_ret / std_ret) * np.sqrt(periods_per_year) if std_ret > 0 else 0
        )

        # ---- Sortino Ratio (annualized, downside deviation) ----
        downside_returns = eq_df["returns"][eq_df["returns"] < 0]
        downside_std = downside_returns.std()
        metrics["Sortino Ratio (Annualized)"] = (
            (mean_ret / downside_std) * np.sqrt(periods_per_year) if downside_std > 0 else 0
        )

        # ---- Cumulative Returns ----
        eq_df["cumulative_return_pct"] = (
            (eq_df["total_equity"] - self.initial_balance) / self.initial_balance
        ) * 100
        eq_df["asset_cumulative_return_pct"] = (
            (eq_df["price"] - initial_price) / initial_price
        ) * 100

        # ---- Drawdown (on total equity) ----
        eq_df["peak"] = eq_df["total_equity"].cummax()
        eq_df["drawdown_pct"] = ((eq_df["total_equity"] - eq_df["peak"]) / eq_df["peak"]) * 100
        metrics["Max Drawdown %"] = eq_df["drawdown_pct"].min()

        # ---- Asset Drawdown ----
        eq_df["asset_peak"] = eq_df["price"].cummax()
        eq_df["asset_drawdown_pct"] = ((eq_df["price"] - eq_df["asset_peak"]) / eq_df["asset_peak"]) * 100
        metrics["Asset Max Drawdown %"] = eq_df["asset_drawdown_pct"].min()

        # ---- Drawdown Duration ----
        in_drawdown = eq_df["drawdown_pct"] < 0
        if in_drawdown.any():
            drawdown_groups = (~in_drawdown).cumsum()
            drawdown_durations = eq_df[in_drawdown].groupby(
                drawdown_groups[in_drawdown]
            )["timestamp"].agg(lambda x: (x.max() - x.min()))
            if len(drawdown_durations) > 0:
                metrics["Max Drawdown Duration"] = str(drawdown_durations.max())
            else:
                metrics["Max Drawdown Duration"] = "N/A"
        else:
            metrics["Max Drawdown Duration"] = "N/A"

        # ---- Calmar Ratio (annualized return / |max drawdown|) ----
        total_days = (
            (eq_df["timestamp"].iloc[-1] - eq_df["timestamp"].iloc[0]).total_seconds() / 86400
        )
        annualized_return_pct = (
            (metrics["Total ROI %"] / total_days) * 365 if total_days > 0 else 0
        )
        metrics["Annualized Return %"] = annualized_return_pct
        metrics["Calmar Ratio"] = (
            abs(annualized_return_pct / metrics["Max Drawdown %"])
            if metrics["Max Drawdown %"] != 0
            else 0
        )

        # ---- Alpha (strategy vs buy-and-hold) ----
        metrics["Alpha % (vs Buy & Hold)"] = metrics["Total ROI %"] - metrics["Asset Return %"]

        # ---- Per-Trade Metrics (from trade_history, no log parsing) ----
        if self.trade_history:
            trades_df = pd.DataFrame(self.trade_history)
            wins = trades_df[trades_df["pnl"] > 0]
            losses = trades_df[trades_df["pnl"] <= 0]

            metrics["Total Trades"] = len(trades_df)
            metrics["Wins"] = len(wins)
            metrics["Losses"] = len(losses)
            metrics["Win Rate %"] = (len(wins) / len(trades_df)) * 100

            gross_profit = wins["pnl"].sum()
            gross_loss = abs(losses["pnl"].sum())
            metrics["Profit Factor"] = (
                (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
            )

            metrics["Avg Win $"] = wins["pnl"].mean() if len(wins) > 0 else 0
            metrics["Avg Loss $"] = losses["pnl"].mean() if len(losses) > 0 else 0
            metrics["Largest Win $"] = wins["pnl"].max() if len(wins) > 0 else 0
            metrics["Largest Loss $"] = losses["pnl"].min() if len(losses) > 0 else 0

            # Avg trade duration
            trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"], utc=True)
            trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"], utc=True)
            metrics["Avg Trade Duration"] = str(
                (trades_df["exit_time"] - trades_df["entry_time"]).mean()
            )

            # Expectancy: avg $ gained per trade
            metrics["Expectancy $"] = trades_df["pnl"].mean()

            # Consecutive wins / losses
            results = (trades_df["pnl"] > 0).astype(int)
            groups = (results != results.shift()).cumsum()
            streaks = results.groupby(groups).agg(["first", "count"])
            win_streaks = streaks[streaks["first"] == 1]["count"]
            loss_streaks = streaks[streaks["first"] == 0]["count"]
            metrics["Max Consecutive Wins"] = (
                int(win_streaks.max()) if len(win_streaks) > 0 else 0
            )
            metrics["Max Consecutive Losses"] = (
                int(loss_streaks.max()) if len(loss_streaks) > 0 else 0
            )

        return metrics

    def print_metrics(self):
        """Compute and display the full advanced metrics report, then save to CSV."""
        metrics = self.compute_advanced_metrics()

        print("\n" + "=" * 60)
        print("BACKTEST FINAL REPORT — ADVANCED METRICS")
        print("=" * 60)
        for key, val in metrics.items():
            if isinstance(val, float):
                print(f"  {key:.<40} {val:>12.4f}")
            else:
                print(f"  {key:.<40} {str(val):>12}")
        print("=" * 60)

        # Save full metrics to CSV
        metrics_df = pd.DataFrame([metrics])
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

    engine = BacktestEngine(initial_balance=100000)
    
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
