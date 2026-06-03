"""
CryptoPilot Post-Backtest Trade Analyzer
===================================
Parses existing backtest trade log files to compute trade-specific performance metrics
and generate visual reports.

Usage:
    python analyze_trades.py                                        # auto-picks latest log
    python analyze_trades.py --log-name backtest_20260507_200708    # specific log

Outputs:
    tests/analysis/trade_metrics.csv
    tests/analysis/trade_cumulative_returns.html / .png
    tests/analysis/trade_drawdown.html / .png
    tests/analysis/trade_metrics_summary.html
"""

import os
import re
import sys
import argparse
import pandas as pd
import numpy as np
import plotly.graph_objects as go

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(PROJECT_ROOT, "tests", "logs")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "tests", "analysis")


# ---------------------------------------------
# 1. LOG DISCOVERY
# ---------------------------------------------
def find_latest_log_base() -> str:
    """Find the most recent backtest log base name by looking for _trade.txt files."""
    if not os.path.exists(LOGS_DIR):
        print(f"[ERROR] Logs directory not found: {LOGS_DIR}")
        sys.exit(1)
        
    state_logs = sorted([
        f for f in os.listdir(LOGS_DIR)
        if f.endswith("_trade.txt")
    ])
    if not state_logs:
        print(f"[ERROR] No trade logs found in {LOGS_DIR}")
        sys.exit(1)
    latest = state_logs[-1]
    base = latest.replace("_trade.txt", "")
    return base


# ---------------------------------------------
# 2. TRADE LOG PARSER -> Trade List
# ---------------------------------------------
def parse_trade_log(log_base: str) -> pd.DataFrame:
    """Parse the trade log to extract individual closed trades."""
    path = os.path.join(LOGS_DIR, f"{log_base}_trade.txt")
    if not os.path.exists(path):
        print(f"[ERROR] Trade log not found: {path}")
        sys.exit(1)

    with open(path, "r") as f:
        text = f.read()

    blocks = re.split(r'={50,}', text)

    trades = []
    for block in blocks:
        if "TRADE CLOSED" not in block:
            continue

        def extract(pattern, block, default=None):
            m = re.search(pattern, block)
            return m.group(1).strip() if m else default

        entry_time = extract(r'Entry Time:\s+(.+)', block)
        exit_time = extract(r'Exit Time:\s+(.+)', block)
        entry_price = extract(r'Entry Price:\s+\$([0-9,.]+)', block)
        exit_price = extract(r'Exit Price:\s+\$([0-9,.]+)', block)
        position_size = extract(r'Position Size:\s+\$([0-9,.]+)', block)
        target = extract(r'Target:\s+\$([0-9,.]+)', block)
        stop_loss = extract(r'Stop Loss:\s+\$([0-9,.]+)', block)
        pnl_match = re.search(r'PnL:\s+\$([-0-9,.]+)\s+\(([+-]?[0-9.]+)%\)', block)
        exit_reason = extract(r'Exit Reason:\s+(.+)', block)
        balance_after = extract(r'Balance After:\s+\$([0-9,.]+)', block)

        if not entry_time or not pnl_match:
            continue

        # Guard against missing fields in malformed log blocks
        if not all([exit_time, entry_price, exit_price, position_size, target, stop_loss, balance_after]):
            continue

        pnl_dollar = float(pnl_match.group(1).replace(",", ""))
        pnl_pct = float(pnl_match.group(2))

        trades.append({
            "entry_time": pd.to_datetime(entry_time, utc=True),
            "exit_time": pd.to_datetime(exit_time, utc=True),
            "entry_price": float(entry_price.replace(",", "")),
            "exit_price": float(exit_price.replace(",", "")),
            "position_size": float(position_size.replace(",", "")),
            "target": float(target.replace(",", "")),
            "stop_loss": float(stop_loss.replace(",", "")),
            "pnl": pnl_dollar,
            "pnl_pct": pnl_pct,
            "exit_reason": exit_reason,
            "balance_after": float(balance_after.replace(",", "")),
        })

    return pd.DataFrame(trades)


# ---------------------------------------------
# 3. METRIC CALCULATIONS
# ---------------------------------------------
def compute_metrics(trades_df: pd.DataFrame, initial_balance: float) -> dict:
    """Compute performance metrics strictly from closed trades data."""
    metrics = {}

    if trades_df.empty:
        return metrics, pd.DataFrame()

    trades_df = trades_df.sort_values("exit_time").reset_index(drop=True)
    
    # Calculate equity curve based on realized balance
    # Prepend the initial state
    eq_data = [{"timestamp": trades_df["entry_time"].iloc[0] if len(trades_df) > 0 else pd.Timestamp.utcnow(), 
                "balance": initial_balance}]
    for _, row in trades_df.iterrows():
        eq_data.append({"timestamp": row["exit_time"], "balance": row["balance_after"]})
    
    eq_df = pd.DataFrame(eq_data)
    eq_df["cumulative_return_pct"] = ((eq_df["balance"] - initial_balance) / initial_balance) * 100
    eq_df["peak"] = eq_df["balance"].cummax()
    eq_df["drawdown_pct"] = ((eq_df["balance"] - eq_df["peak"]) / eq_df["peak"]) * 100

    final_equity = eq_df["balance"].iloc[-1]
    
    # -- Backtest Duration --
    total_seconds = (eq_df["timestamp"].iloc[-1] - eq_df["timestamp"].iloc[0]).total_seconds()
    total_days = total_seconds / 86400 if total_seconds > 0 else 1
    metrics["Backtest Duration (Days)"] = round(total_days, 1)

    # -- Basic --
    metrics["Initial Balance"] = initial_balance
    metrics["Final Equity"] = final_equity
    metrics["Net Profit"] = final_equity - initial_balance
    metrics["Total ROI %"] = ((final_equity - initial_balance) / initial_balance) * 100

    # -- Asset Price --
    initial_price = trades_df["entry_price"].iloc[0]
    final_price = trades_df["exit_price"].iloc[-1]
    metrics["Asset Start Price"] = initial_price
    metrics["Asset End Price"] = final_price
    metrics["Asset Return %"] = ((final_price - initial_price) / initial_price) * 100

    # -- Resample for time-based metrics --
    daily_eq = eq_df.groupby("timestamp")["balance"].last().resample("4h").ffill()
    print(daily_eq)
    daily_returns = daily_eq.pct_change().dropna()
    candles_per_day = 6
    periods_per_year = candles_per_day * 365
    metrics["Data Points"] = len(trades_df)
    metrics["Detected Candle Interval (h)"] = 4.0
    metrics["Periods Per Year"] = periods_per_year

    # -- Sharpe Ratio (annualized) --
    mean_ret = daily_returns.mean()
    print(mean_ret)
    std_ret = daily_returns.std()
    print(std_ret)
    metrics["Sharpe Ratio (Annualized)"] = (mean_ret / std_ret) * np.sqrt(periods_per_year) if std_ret > 0 else 0

    # -- Sortino Ratio (annualized, downside deviation only) --
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std()
    metrics["Sortino Ratio (Annualized)"] = (mean_ret / downside_std) * np.sqrt(periods_per_year) if downside_std > 0 else 0

    # -- Drawdowns --
    metrics["Max Drawdown %"] = eq_df["drawdown_pct"].min()
    
    asset_prices = pd.concat([
        pd.DataFrame({"timestamp": trades_df["entry_time"], "price": trades_df["entry_price"]}),
        pd.DataFrame({"timestamp": trades_df["exit_time"], "price": trades_df["exit_price"]})
    ]).sort_values("timestamp").reset_index(drop=True)
    asset_prices["asset_peak"] = asset_prices["price"].cummax()
    asset_prices["asset_drawdown_pct"] = ((asset_prices["price"] - asset_prices["asset_peak"]) / asset_prices["asset_peak"]) * 100
    metrics["Asset Max Drawdown %"] = asset_prices["asset_drawdown_pct"].min()

    in_drawdown = eq_df["drawdown_pct"] < 0
    if in_drawdown.any():
        drawdown_groups = (~in_drawdown).cumsum()
        drawdown_durations = eq_df[in_drawdown].groupby(drawdown_groups[in_drawdown])["timestamp"].agg(
            lambda x: (x.max() - x.min())
        )
        if len(drawdown_durations) > 0:
            metrics["Max Drawdown Duration"] = str(drawdown_durations.max())
        else:
            metrics["Max Drawdown Duration"] = "N/A"
    else:
        metrics["Max Drawdown Duration"] = "N/A"

    # -- Calmar Ratio --
    if total_days > 0 and final_equity > 0:
        annualized_return_pct = ((final_equity / initial_balance) ** (365 / total_days) - 1) * 100
    else:
        annualized_return_pct = 0
    metrics["Annualized Return %"] = annualized_return_pct
    metrics["Calmar Ratio"] = (annualized_return_pct / abs(metrics["Max Drawdown %"])) if metrics["Max Drawdown %"] != 0 else 0

    # -- Alpha --
    metrics["Alpha % (vs Buy & Hold)"] = metrics["Total ROI %"] - metrics["Asset Return %"]

    # -- Per-Trade Metrics (from trade log) --
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]

    metrics["Total Trades"] = len(trades_df)
    metrics["Wins"] = len(wins)
    metrics["Losses"] = len(losses)
    metrics["Win Rate %"] = (len(wins) / len(trades_df)) * 100 if len(trades_df) > 0 else 0

    gross_profit = wins["pnl"].sum()
    gross_loss = abs(losses["pnl"].sum())
    metrics["Profit Factor"] = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    metrics["Avg Win $"] = wins["pnl"].mean() if len(wins) > 0 else 0
    metrics["Avg Loss $"] = losses["pnl"].mean() if len(losses) > 0 else 0
    metrics["Largest Win $"] = wins["pnl"].max() if len(wins) > 0 else 0
    metrics["Largest Loss $"] = losses["pnl"].min() if len(losses) > 0 else 0
    metrics["Avg Trade Duration"] = str((trades_df["exit_time"] - trades_df["entry_time"]).mean())

    # Expectancy: avg $ gained per trade
    metrics["Expectancy $"] = trades_df["pnl"].mean() if len(trades_df) > 0 else 0

    # Consecutive wins/losses
    if len(trades_df) > 0:
        results = (trades_df["pnl"] > 0).astype(int)
        groups = (results != results.shift()).cumsum()
        streaks = results.groupby(groups).agg(["first", "count"])
        win_streaks = streaks[streaks["first"] == 1]["count"]
        loss_streaks = streaks[streaks["first"] == 0]["count"]
        metrics["Max Consecutive Wins"] = int(win_streaks.max()) if len(win_streaks) > 0 else 0
        metrics["Max Consecutive Losses"] = int(loss_streaks.max()) if len(loss_streaks) > 0 else 0
    else:
        metrics["Max Consecutive Wins"] = 0
        metrics["Max Consecutive Losses"] = 0

    return metrics, eq_df


# ---------------------------------------------
# 4. CHART GENERATION
# ---------------------------------------------
COLORS = {
    "bg": "#0d0f1a",
    "paper": "#131627",
    "grid": "#1e2140",
    "accent": "#c2ef4e",
    "red": "#f6465d",
    "text": "#e0e0e0",
    "muted": "#6b7280",
}


def generate_cumulative_returns_chart(eq_df: pd.DataFrame) -> go.Figure:
    """Generate a cumulative returns chart based on realized trades."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=eq_df["timestamp"],
        y=eq_df["cumulative_return_pct"],
        mode="lines+markers",
        name="Realized Return %",
        line=dict(color=COLORS["accent"], width=2),
        fill="tozeroy",
        fillcolor="rgba(194, 239, 78, 0.08)",
    ))

    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["muted"], line_width=1)

    fig.update_layout(
        title=dict(text="Realized Cumulative Returns", font=dict(size=18, color=COLORS["text"])),
        template="plotly_dark",
        plot_bgcolor=COLORS["bg"],
        paper_bgcolor=COLORS["paper"],
        font=dict(color=COLORS["text"]),
        xaxis=dict(gridcolor=COLORS["grid"], title="Date"),
        yaxis=dict(gridcolor=COLORS["grid"], title="Return %", ticksuffix="%"),
        height=450,
        margin=dict(l=60, r=30, t=60, b=40),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)"),
    )
    return fig


def generate_drawdown_chart(eq_df: pd.DataFrame) -> go.Figure:
    """Generate a drawdown chart based on realized equity."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=eq_df["timestamp"],
        y=eq_df["drawdown_pct"],
        mode="lines+markers",
        name="Realized Drawdown %",
        line=dict(color=COLORS["red"], width=2),
        fill="tozeroy",
        fillcolor="rgba(246, 70, 93, 0.15)",
    ))

    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["muted"], line_width=1)

    if not eq_df.empty:
        max_dd_idx = eq_df["drawdown_pct"].idxmin()
        max_dd_val = eq_df.loc[max_dd_idx, "drawdown_pct"]
        max_dd_time = eq_df.loc[max_dd_idx, "timestamp"]
        fig.add_annotation(
            x=max_dd_time, y=max_dd_val,
            text=f"Max DD: {max_dd_val:.2f}%",
            showarrow=True, arrowhead=2,
            font=dict(color=COLORS["red"], size=12),
            arrowcolor=COLORS["red"],
            bgcolor=COLORS["paper"],
            bordercolor=COLORS["red"],
        )

    fig.update_layout(
        title=dict(text="Realized Drawdown", font=dict(size=18, color=COLORS["text"])),
        template="plotly_dark",
        plot_bgcolor=COLORS["bg"],
        paper_bgcolor=COLORS["paper"],
        font=dict(color=COLORS["text"]),
        xaxis=dict(gridcolor=COLORS["grid"], title="Date"),
        yaxis=dict(gridcolor=COLORS["grid"], title="Drawdown %", ticksuffix="%"),
        height=400,
        margin=dict(l=60, r=30, t=60, b=40),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)"),
    )
    return fig


# ---------------------------------------------
# 5. METRICS SUMMARY (standalone HTML report)
# ---------------------------------------------
def generate_metrics_html(metrics: dict, log_base: str) -> str:
    """Generate a standalone HTML metrics summary page."""
    def fmt(val):
        if isinstance(val, float):
            if abs(val) >= 1000:
                return f"{val:,.2f}"
            return f"{val:.4f}"
        return str(val)

    rows_html = ""
    for key, val in metrics.items():
        css_class = ""
        display = fmt(val)
        if isinstance(val, float):
            if "ROI" in key or "Return" in key or "Alpha" in key or "Profit" in key or "Win" in key:
                css_class = "positive" if val > 0 else "negative"
            elif "Drawdown %" in key or "Loss" in key:
                css_class = "negative" if val < 0 else ""
        rows_html += f'<tr><td class="label">{key}</td><td class="value {css_class}">{display}</td></tr>\n'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Trade Metrics - {log_base}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: #0d0f1a;
    color: #e0e0e0;
    font-family: 'Segoe UI', 'Inter', system-ui, sans-serif;
    padding: 40px;
  }}
  h1 {{
    font-size: 24px;
    color: #c2ef4e;
    margin-bottom: 8px;
  }}
  .subtitle {{
    color: #6b7280;
    font-size: 14px;
    margin-bottom: 30px;
  }}
  table {{
    width: 100%;
    max-width: 700px;
    border-collapse: collapse;
    background: #131627;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid #1e2140;
  }}
  tr {{ border-bottom: 1px solid #1e2140; }}
  tr:last-child {{ border-bottom: none; }}
  tr:hover {{ background: #1a1d35; }}
  td {{
    padding: 10px 16px;
    font-size: 14px;
  }}
  .label {{
    color: #9ca3af;
    width: 55%;
  }}
  .value {{
    text-align: right;
    font-family: 'Courier New', monospace;
    font-weight: 600;
    color: #e0e0e0;
  }}
  .positive {{ color: #0ecb81; }}
  .negative {{ color: #f6465d; }}
</style>
</head>
<body>
  <h1>Trade Metrics Summary</h1>
  <div class="subtitle">Source: {log_base}_trade.txt</div>
  <table>
    {rows_html}
  </table>
</body>
</html>"""
    return html


# ---------------------------------------------
# 6. MAIN
# ---------------------------------------------
def main():
    global LOGS_DIR, OUTPUT_DIR
    parser = argparse.ArgumentParser(description="Analyze backtest results from trade log files")
    parser.add_argument("--log-name", type=str, default=None,
                        help="Log base name (e.g. backtest_20260507_200708). Defaults to latest.")
    parser.add_argument("--balance", type=float, default=100000,
                        help="Initial balance used in the backtest (default: 100000)")
    parser.add_argument("--logs-dir", type=str, default=None,
                        help="Directory containing the logs")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save the analysis outputs")
    args, _ = parser.parse_known_args()

    if args.logs_dir:
        LOGS_DIR = os.path.abspath(args.logs_dir)
    if args.output_dir:
        OUTPUT_DIR = os.path.abspath(args.output_dir)

    # Discover log
    log_base = args.log_name or find_latest_log_base()
    print(f"\n{'='*60}")
    print(f"[ANALYZE TRADES] {log_base}")
    print(f"{'='*60}")
    print(f"Logs Dir: {LOGS_DIR}")
    print(f"Output Dir: {OUTPUT_DIR}")

    # Parse logs
    print("\nParsing trade log...")
    trades_df = parse_trade_log(log_base)
    if not trades_df.empty:
        print(f"   [OK] {len(trades_df)} closed trades parsed")
    else:
        print("   [ERROR] No trades parsed. Exiting.")
        sys.exit(1)

    # Compute metrics
    print("\nComputing metrics...")
    metrics, eq_df = compute_metrics(trades_df, args.balance)

    # Print report
    print(f"\n{'='*60}")
    print("TRADE PERFORMANCE METRICS")
    print(f"{'='*60}")
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"  {key:.<35} {val:>12.4f}")
        else:
            print(f"  {key:.<35} {str(val):>12}")
    print(f"{'='*60}")

    # Save metrics CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metrics_df = pd.DataFrame([metrics])
    csv_path = os.path.join(OUTPUT_DIR, "trade_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"\nMetrics saved to {csv_path}")

    # Generate standalone metrics HTML
    metrics_html = generate_metrics_html(metrics, log_base)
    metrics_html_path = os.path.join(OUTPUT_DIR, "trade_metrics_summary.html")
    with open(metrics_html_path, "w") as f:
        f.write(metrics_html)
    print(f"Metrics HTML saved to {metrics_html_path}")

    # Generate charts
    print("\nGenerating charts...")

    if not eq_df.empty:
        fig_returns = generate_cumulative_returns_chart(eq_df)
        fig_returns.write_html(os.path.join(OUTPUT_DIR, "trade_cumulative_returns.html"))
        try:
            fig_returns.write_image(os.path.join(OUTPUT_DIR, "trade_cumulative_returns.png"), scale=2)
        except Exception:
            pass

        fig_dd = generate_drawdown_chart(eq_df)
        fig_dd.write_html(os.path.join(OUTPUT_DIR, "trade_drawdown.html"))
        try:
            fig_dd.write_image(os.path.join(OUTPUT_DIR, "trade_drawdown.png"), scale=2)
        except Exception:
            pass

    print(f"\n[DONE] All outputs saved to {OUTPUT_DIR}/")
    print("   - trade_metrics.csv")
    print("   - trade_metrics_summary.html")
    print("   - trade_cumulative_returns.html")
    print("   - trade_drawdown.html")
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
