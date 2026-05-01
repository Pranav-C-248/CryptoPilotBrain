import os
import sys
import ast
import time
import traceback
from datetime import datetime, timezone

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from src.dashboard.shared.database import get_db, QueuedTrade, Portfolio, Position, TradeLedger
from src.schema.models import RiskAssessment
from src.tools.binance_client import BinancePublicClient
from src.tools.indicators import add_indicators

class PaperTradingEngine:
    def __init__(self):
        self.client = BinancePublicClient()

    def add_to_queue(self, risk_assessment: RiskAssessment, asset_name: str, signal: str):
        """Adds an approved risk assessment trade into the unentered queue."""
        if signal == "HOLD":
            return
            
        db = next(get_db())
        try:
            # We assume risk_assessment has entry_condition and exit_condition
            # Currently RiskAssessment (from schema) doesn't have these, but the custom one in agents.risk_manager does.
            # We will use getattr to safely get them or default to something basic.
            qt = QueuedTrade(
                asset_name=asset_name,
                signal=signal,
                entry_condition=getattr(risk_assessment, 'entry_condition', 'True'),
                exit_condition=getattr(risk_assessment, 'exit_condition', 'False'),
                target=risk_assessment.target if hasattr(risk_assessment, 'target') else risk_assessment.take_profit_price,
                stop_loss=risk_assessment.stop_loss if hasattr(risk_assessment, 'stop_loss') else risk_assessment.stop_loss_price,
                position_size=risk_assessment.position_size if hasattr(risk_assessment, 'position_size') else risk_assessment.final_position_size,
                timeframe=getattr(risk_assessment, 'timeframe', '4h'),
                status="PENDING_APPROVAL"
            )
            db.add(qt)
            db.commit()
            print(f"[PaperTrader] Queued new {signal} trade for {asset_name}.")
        except Exception as e:
            print(f"[PaperTrader] Error adding to queue: {e}")
            db.rollback()
        finally:
            db.close()

    def process_queue(self):
        """Main loop that iterates over UNENTERED and ENTERED trades."""
        db = next(get_db())
        try:
            # We process UNENTERED first
            unentered = db.query(QueuedTrade).filter(QueuedTrade.status == "UNENTERED").all()
            entered = db.query(QueuedTrade).filter(QueuedTrade.status == "ENTERED").all()
            
            # Optimization: Fetch data for symbols we need
            symbols_needed = set([t.asset_name for t in unentered + entered])
            market_data_cache = {}
            
            for sym in symbols_needed:
                try:
                    df = self.client.get_historical_klines(sym, "1h", limit=50)
                    df = add_indicators(df)
                    
                    # Build locals dict for eval
                    last_row = df.iloc[-1]
                    locals_dict = {col: last_row[col] for col in df.columns}
                    # Add uppercase aliases for common indicators
                    if 'rsi' in locals_dict: locals_dict['RSI'] = locals_dict['rsi']
                    if 'macd' in locals_dict: locals_dict['MACD'] = locals_dict['macd']
                    locals_dict['price'] = last_row['close']
                    
                    market_data_cache[sym] = locals_dict
                except Exception as e:
                    print(f"[PaperTrader] Failed fetching data for {sym}: {e}")
                    
            # 1. Check Unentered
            for trade in unentered:
                if trade.asset_name not in market_data_cache: continue
                self._check_entry(db, trade, market_data_cache[trade.asset_name])
                
            # 2. Check Entered
            for trade in entered:
                if trade.asset_name not in market_data_cache: continue
                self._check_exit(db, trade, market_data_cache[trade.asset_name])
                
        except Exception as e:
            print(f"[PaperTrader] Error processing queue: {e}")
            traceback.print_exc()
        finally:
            db.close()
            
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
        except Exception as e:
            print(f"[PaperTrader] Condition eval failed '{condition_str}': {e}")
            return False

    def _safe_eval_node(self, node: ast.AST, variables: dict):
        """Recursively evaluate an AST node, allowing only safe operations."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, bool, str)):
                return node.value
            raise ValueError(f"Disallowed constant type: {type(node.value)}")

        if isinstance(node, ast.Name):
            if node.id in variables:
                return variables[node.id]
            raise NameError(f"Unknown variable: {node.id}")

        if isinstance(node, ast.UnaryOp):
            operand = self._safe_eval_node(node.operand, variables)
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.Not):
                return not operand
            raise ValueError(f"Disallowed unary op: {type(node.op).__name__}")

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
            for op, comparator in zip(node.ops, node.comparators):
                right = self._safe_eval_node(comparator, variables)
                op_func = cmp_ops.get(type(op))
                if op_func is None:
                    raise ValueError(f"Disallowed comparison: {type(op).__name__}")
                if not op_func(left, right):
                    return False
                left = right
            return True

        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return all(self._safe_eval_node(v, variables) for v in node.values)
            if isinstance(node.op, ast.Or):
                return any(self._safe_eval_node(v, variables) for v in node.values)
            raise ValueError(f"Disallowed bool op: {type(node.op).__name__}")

        raise ValueError(f"Disallowed AST node: {type(node).__name__}")

    def _check_entry(self, db, trade: QueuedTrade, data: dict):
        if self._evaluate_condition(trade.entry_condition, data):
            trade.status = "ENTERED"
            trade.entry_price = data['price']
            trade.entry_time = datetime.now(timezone.utc)
            
            # Update Position
            pos = db.query(Position).filter(Position.symbol == trade.asset_name).first()
            if not pos:
                pos = Position(symbol=trade.asset_name, quantity=0, average_price=0)
                db.add(pos)
            
            # Update Portfolio (deduct position size to simulate margin/usage if BUY, or just adjust cash)
            portfolio = db.query(Portfolio).first()
            if not portfolio:
                portfolio = Portfolio(balance=10000.0)
                db.add(portfolio)
                
            if trade.signal == "BUY":
                quantity = trade.position_size / trade.entry_price
                # Basic averaging
                total_val = (pos.quantity * pos.average_price) + trade.position_size
                pos.quantity += quantity
                pos.average_price = total_val / pos.quantity
                
                # We do NOT deduct balance yet since it's just a position allocation.
                # Actually, some engines deduct balance. Let's just deduct it for paper trading.
                portfolio.balance -= trade.position_size
                
                # Log entry trade
                entry_trade = TradeLedger(
                    symbol=trade.asset_name,
                    action="BUY",
                    quantity=quantity,
                    price=trade.entry_price,
                    fee=trade.position_size * 0.001 # 0.1% fee
                )
                db.add(entry_trade)
            
            # SELL / Shorting could be handled similarly, but we'll focus on BUY for simplicity, 
            # or treat 'SELL' as closing a position. The logic above assumes opening.
            
            db.commit()
            print(f"[PaperTrader] Trade {trade.id} ENTERED at {trade.entry_price}")

    def _check_exit(self, db, trade: QueuedTrade, data: dict):
        price = data['price']
        exit_triggered = False
        
        # Check Stop Loss / Target
        if trade.signal == "BUY":
            if price <= trade.stop_loss:
                exit_triggered = True
                print(f"[PaperTrader] Trade {trade.id} hit Stop Loss {trade.stop_loss}")
            elif trade.target > 0 and price >= trade.target:
                exit_triggered = True
                print(f"[PaperTrader] Trade {trade.id} hit Target {trade.target}")
        
        # Or string condition
        if not exit_triggered and self._evaluate_condition(trade.exit_condition, data):
            exit_triggered = True
            print(f"[PaperTrader] Trade {trade.id} met exit condition: {trade.exit_condition}")
            
        if exit_triggered:
            trade.status = "CLOSED"
            trade.exit_price = price
            trade.exit_time = datetime.now(timezone.utc)
            
            if trade.signal == "BUY":
                trade.realized_pnl = ((trade.exit_price - trade.entry_price) / trade.entry_price) * trade.position_size
            
            # Update Position and Portfolio
            pos = db.query(Position).filter(Position.symbol == trade.asset_name).first()
            portfolio = db.query(Portfolio).first()
            
            if trade.signal == "BUY" and pos:
                quantity = trade.position_size / trade.entry_price
                pos.quantity = max(0, pos.quantity - quantity)
                
                # Restore balance + PnL
                if portfolio:
                    portfolio.balance += trade.position_size + trade.realized_pnl
                
                exit_trade = TradeLedger(
                    symbol=trade.asset_name,
                    action="SELL",
                    quantity=quantity,
                    price=trade.exit_price,
                    fee=trade.position_size * 0.001,
                    realized_pnl=trade.realized_pnl
                )
                db.add(exit_trade)
            
            db.commit()
            print(f"[PaperTrader] Trade {trade.id} CLOSED at {trade.exit_price} | PnL: {trade.realized_pnl:.2f}")

