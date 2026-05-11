"""
CryptoPilot Post-Backtest Analyzer
===================================
Parses existing backtest log files to compute advanced performance metrics
and generate visual reports WITHOUT re-running the simulation.

Usage:
    python analyze_backtest.py                          # auto-picks latest log
    python analyze_backtest.py --log backtest_20260507_200708   # specific log

Outputs:
    tests/analysis/advanced_metrics.csv
    tests/analysis/cumulative_returns.html / .png
    tests/analysis/drawdown.html / .png
    tests/analysis/combined_dashboard.html
    tests/analysis/metrics_summary.html
"""

import os
import re
import sys
import argparse
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(PROJECT_ROOT, "tests", "logs")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "tests", "analysis")


# ---------------------------------------------
# 1. LOG DISCOVERY
# ---------------------------------------------
def find_latest_log_base() -> str:
    """Find the most recent backtest log base name (without _trade suffix)."""
    state_logs = sorted([
        f for f in os.listdir(LOGS_DIR)
        if f.startswith("backtest_") and not f.endswith("_trade.txt") and f.endswith(".txt")
    ])
    if not state_logs:
        print("[ERROR] No backtest logs found in tests/logs/")
        sys.exit(1)
    latest = state_logs[-1]
    base = latest.replace(".txt", "")
    return base


# ---------------------------------------------
# 2. STATE LOG PARSER -> Equity Curve
# ---------------------------------------------
def parse_state_log(log_base: str, initial_balance: float) -> pd.DataFrame:
    """
    Parse the main state log to extract the equity curve.
    Each step has:
        [timestamp] Price: $X | Balance: $Y
        Realized PnL: $A | Unrealized PnL: $B

    Total equity = initial_balance + realized_pnl + unrealized_pnl
    This accounts for the current value of open positions, not just cash.
    """
    path = os.path.join(LOGS_DIR, f"{log_base}.txt")
    if not os.path.exists(path):
        print(f"[ERROR] State log not found: {path}")
        sys.exit(1)

    with open(path, "r") as f:
        text = f.read()

    # Split into per-step blocks by the separator line
    step_pattern = re.compile(
        r'\[(.+?)\] Price: \$([0-9,.]+) \| Balance: \$([0-9,.]+)\n'
        r'Realized PnL: \$(-?[0-9,.]+) \| Unrealized PnL: \$(-?[0-9,.]+)',
        re.MULTILINE
    )
    matches = step_pattern.findall(text)

    if not matches:
        print(f"[ERROR] Could not parse any equity data from {path}")
        sys.exit(1)

    eq_df = pd.DataFrame(matches, columns=["timestamp", "price", "balance", "realized_pnl", "unrealized_pnl"])
    eq_df["timestamp"] = pd.to_datetime(eq_df["timestamp"], utc=True)
    for col in ["price", "balance", "realized_pnl", "unrealized_pnl"]:
        eq_df[col] = eq_df[col].str.replace(",", "").astype(float)

    # Total equity = initial capital + all gains (realized + unrealized)
    # This captures the true portfolio value including open position market value
    eq_df["total_equity"] = initial_balance + eq_df["realized_pnl"] + eq_df["unrealized_pnl"]

    eq_df = eq_df.sort_values("timestamp").reset_index(drop=True)

    return eq_df


# ---------------------------------------------
# 3. TRADE LOG PARSER -> Trade List
# ---------------------------------------------
def parse_trade_log(log_base: str) -> pd.DataFrame:
    """Parse the trade log to extract individual closed trades."""
    path = os.path.join(LOGS_DIR, f"{log_base}_trade.txt")
    if not os.path.exists(path):
        print(f"[WARN] Trade log not found: {path}. Skipping per-trade metrics.")
        return pd.DataFrame()

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

        if not entry_time or not pnl_match:
            continue

        # Guard against missing fields in malformed log blocks
        if not all([exit_time, entry_price, exit_price, position_size, target, stop_loss]):
            continue

        # The dollar value in the log already includes the sign (e.g. $-360.95)
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
        })

    return pd.DataFrame(trades)


# ---------------------------------------------
# 4. METRIC CALCULATIONS
# ---------------------------------------------
def compute_metrics(eq_df: pd.DataFrame, trades_df: pd.DataFrame, initial_balance: float) -> dict:
    """Compute all advanced performance metrics from parsed data.
    Uses total_equity (balance + open position value) for all calculations."""
    metrics = {}

    final_equity = eq_df["total_equity"].iloc[-1]
    initial_price = eq_df["price"].iloc[0]
    final_price = eq_df["price"].iloc[-1]

    # -- Basic --
    metrics["Initial Balance"] = initial_balance
    metrics["Final Equity"] = final_equity
    metrics["Net Profit"] = final_equity - initial_balance
    metrics["Total ROI %"] = ((final_equity - initial_balance) / initial_balance) * 100

    # -- Asset Price --
    metrics["Asset Start Price"] = initial_price
    metrics["Asset End Price"] = final_price
    metrics["Asset Return %"] = ((final_price - initial_price) / initial_price) * 100

    # -- Returns (based on total equity, not just cash balance) --
    # Use pct_change and drop the first NaN row to avoid diluting mean/std
    eq_df["returns"] = eq_df["total_equity"].pct_change()
    eq_df["asset_returns"] = eq_df["price"].pct_change()

    # -- Sharpe Ratio (annualized) --
    # 4h candles -> 6 per day -> 6 * 365 = 2190 periods/year
    periods_per_year = 6 * 365
    clean_returns = eq_df["returns"].dropna()
    mean_ret = clean_returns.mean()
    std_ret = clean_returns.std()
    metrics["Sharpe Ratio (Annualized)"] = (mean_ret / std_ret) * np.sqrt(periods_per_year) if std_ret > 0 else 0

    # -- Sortino Ratio (annualized, downside deviation only) --
    downside_returns = clean_returns[clean_returns < 0]
    downside_std = downside_returns.std()
    metrics["Sortino Ratio (Annualized)"] = (mean_ret / downside_std) * np.sqrt(periods_per_year) if downside_std > 0 else 0

    # Fill NaN for downstream use (charts etc.) after computing ratios
    eq_df["returns"] = eq_df["returns"].fillna(0)
    eq_df["asset_returns"] = eq_df["asset_returns"].fillna(0)

    # -- Cumulative Returns --
    eq_df["cumulative_return_pct"] = ((eq_df["total_equity"] - initial_balance) / initial_balance) * 100
    eq_df["asset_cumulative_return_pct"] = ((eq_df["price"] - initial_price) / initial_price) * 100

    # -- Drawdown (on total equity) --
    eq_df["peak"] = eq_df["total_equity"].cummax()
    eq_df["drawdown_pct"] = ((eq_df["total_equity"] - eq_df["peak"]) / eq_df["peak"]) * 100
    metrics["Max Drawdown %"] = eq_df["drawdown_pct"].min()

    # -- Asset Drawdown --
    eq_df["asset_peak"] = eq_df["price"].cummax()
    eq_df["asset_drawdown_pct"] = ((eq_df["price"] - eq_df["asset_peak"]) / eq_df["asset_peak"]) * 100
    metrics["Asset Max Drawdown %"] = eq_df["asset_drawdown_pct"].min()

    # -- Drawdown Duration --
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

    # -- Calmar Ratio (annualized return / max drawdown) --
    total_days = (eq_df["timestamp"].iloc[-1] - eq_df["timestamp"].iloc[0]).total_seconds() / 86400
    # Use CAGR (compound annualized growth rate) for proper annualization
    if total_days > 0 and final_equity > 0:
        annualized_return_pct = ((final_equity / initial_balance) ** (365 / total_days) - 1) * 100
    else:
        annualized_return_pct = 0
    metrics["Annualized Return %"] = annualized_return_pct
    # abs() only on denominator so a losing strategy correctly shows a negative Calmar
    metrics["Calmar Ratio"] = (annualized_return_pct / abs(metrics["Max Drawdown %"])) if metrics["Max Drawdown %"] != 0 else 0

    # -- Alpha (strategy return vs buy-and-hold) --
    metrics["Alpha % (vs Buy & Hold)"] = metrics["Total ROI %"] - metrics["Asset Return %"]

    # -- Per-Trade Metrics (from trade log) --
    if not trades_df.empty:
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]

        metrics["Total Trades"] = len(trades_df)
        metrics["Wins"] = len(wins)
        metrics["Losses"] = len(losses)
        metrics["Win Rate %"] = (len(wins) / len(trades_df)) * 100

        gross_profit = wins["pnl"].sum()
        gross_loss = abs(losses["pnl"].sum())
        metrics["Profit Factor"] = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

        metrics["Avg Win $"] = wins["pnl"].mean() if len(wins) > 0 else 0
        metrics["Avg Loss $"] = losses["pnl"].mean() if len(losses) > 0 else 0
        metrics["Largest Win $"] = wins["pnl"].max() if len(wins) > 0 else 0
        metrics["Largest Loss $"] = losses["pnl"].min() if len(losses) > 0 else 0
        metrics["Avg Trade Duration"] = str((trades_df["exit_time"] - trades_df["entry_time"]).mean())

        # Expectancy: avg $ gained per trade
        metrics["Expectancy $"] = trades_df["pnl"].mean()

        # Consecutive wins/losses
        results = (trades_df["pnl"] > 0).astype(int)
        groups = (results != results.shift()).cumsum()
        streaks = results.groupby(groups).agg(["first", "count"])
        win_streaks = streaks[streaks["first"] == 1]["count"]
        loss_streaks = streaks[streaks["first"] == 0]["count"]
        metrics["Max Consecutive Wins"] = int(win_streaks.max()) if len(win_streaks) > 0 else 0
        metrics["Max Consecutive Losses"] = int(loss_streaks.max()) if len(loss_streaks) > 0 else 0

    return metrics, eq_df


# ---------------------------------------------
# 5. CHART GENERATION
# ---------------------------------------------
COLORS = {
    "bg": "#0d0f1a",
    "paper": "#131627",
    "grid": "#1e2140",
    "accent": "#c2ef4e",
    "accent2": "#7b61ff",
    "red": "#f6465d",
    "green": "#0ecb81",
    "text": "#e0e0e0",
    "muted": "#6b7280",
    "orange": "#f0b90b",
}


def generate_cumulative_returns_chart(eq_df: pd.DataFrame) -> go.Figure:
    """Generate a cumulative returns chart with strategy vs asset overlay."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=eq_df["timestamp"],
        y=eq_df["cumulative_return_pct"],
        mode="lines",
        name="Portfolio Return %",
        line=dict(color=COLORS["accent"], width=2),
        fill="tozeroy",
        fillcolor="rgba(194, 239, 78, 0.08)",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=eq_df["timestamp"],
        y=eq_df["asset_cumulative_return_pct"],
        mode="lines",
        name="Asset (Buy & Hold) %",
        line=dict(color=COLORS["orange"], width=1.5, dash="dot"),
    ), secondary_y=False)

    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["muted"], line_width=1)

    fig.update_layout(
        title=dict(text="Cumulative Returns: Portfolio vs Buy & Hold", font=dict(size=18, color=COLORS["text"])),
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
    """Generate a drawdown chart with strategy vs asset."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=eq_df["timestamp"],
        y=eq_df["drawdown_pct"],
        mode="lines",
        name="Portfolio Drawdown %",
        line=dict(color=COLORS["red"], width=2),
        fill="tozeroy",
        fillcolor="rgba(246, 70, 93, 0.15)",
    ))

    fig.add_trace(go.Scatter(
        x=eq_df["timestamp"],
        y=eq_df["asset_drawdown_pct"],
        mode="lines",
        name="Asset Drawdown %",
        line=dict(color=COLORS["orange"], width=1.5, dash="dot"),
    ))

    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["muted"], line_width=1)

    # Annotate max drawdown
    max_dd_idx = eq_df["drawdown_pct"].idxmin()
    max_dd_val = eq_df.loc[max_dd_idx, "drawdown_pct"]
    max_dd_time = eq_df.loc[max_dd_idx, "timestamp"]
    fig.add_annotation(
        x=max_dd_time, y=max_dd_val,
        text=f"Max Portfolio DD: {max_dd_val:.2f}%",
        showarrow=True, arrowhead=2,
        font=dict(color=COLORS["red"], size=12),
        arrowcolor=COLORS["red"],
        bgcolor=COLORS["paper"],
        bordercolor=COLORS["red"],
    )

    fig.update_layout(
        title=dict(text="Drawdown: Portfolio vs Asset", font=dict(size=18, color=COLORS["text"])),
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


def generate_combined_dashboard(eq_df: pd.DataFrame) -> go.Figure:
    """Generate a combined dashboard with equity + price, cumulative returns, and drawdown.
    Metrics are NOT embedded here -- they are a separate output."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("Total Equity ($) vs Asset Price ($)", "Cumulative Returns (%)", "Drawdown (%)"),
        row_heights=[0.4, 0.3, 0.3],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]],
    )

    # Row 1: Total Equity + Asset Price on secondary y
    fig.add_trace(go.Scatter(
        x=eq_df["timestamp"], y=eq_df["total_equity"],
        mode="lines", name="Total Equity ($)",
        line=dict(color=COLORS["accent"], width=2),
    ), row=1, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=eq_df["timestamp"], y=eq_df["price"],
        mode="lines", name="Asset Price ($)",
        line=dict(color=COLORS["orange"], width=1.5, dash="dot"),
        opacity=0.7,
    ), row=1, col=1, secondary_y=True)

    # Row 2: Cumulative Returns (strategy vs buy-and-hold)
    fig.add_trace(go.Scatter(
        x=eq_df["timestamp"], y=eq_df["cumulative_return_pct"],
        mode="lines", name="Portfolio Return %",
        line=dict(color=COLORS["accent2"], width=2),
        fill="tozeroy", fillcolor="rgba(123, 97, 255, 0.08)",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=eq_df["timestamp"], y=eq_df["asset_cumulative_return_pct"],
        mode="lines", name="Buy & Hold %",
        line=dict(color=COLORS["orange"], width=1.5, dash="dot"),
    ), row=2, col=1)

    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["muted"], line_width=1, row=2, col=1)

    # Row 3: Drawdown (portfolio vs asset)
    fig.add_trace(go.Scatter(
        x=eq_df["timestamp"], y=eq_df["drawdown_pct"],
        mode="lines", name="Portfolio DD %",
        line=dict(color=COLORS["red"], width=2),
        fill="tozeroy", fillcolor="rgba(246, 70, 93, 0.12)",
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=eq_df["timestamp"], y=eq_df["asset_drawdown_pct"],
        mode="lines", name="Asset DD %",
        line=dict(color=COLORS["orange"], width=1.5, dash="dot"),
    ), row=3, col=1)

    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["muted"], line_width=1, row=3, col=1)

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor=COLORS["bg"],
        paper_bgcolor=COLORS["paper"],
        font=dict(color=COLORS["text"]),
        height=950,
        showlegend=True,
        legend=dict(x=0.01, y=1.12, orientation="h", bgcolor="rgba(0,0,0,0.5)"),
        margin=dict(l=60, r=60, t=100, b=40),
    )

    # Style each subplot axis
    for i in range(1, 4):
        fig.update_xaxes(gridcolor=COLORS["grid"], row=i, col=1)
        fig.update_yaxes(gridcolor=COLORS["grid"], row=i, col=1)

    fig.update_yaxes(tickprefix="$", title_text="Equity", row=1, col=1, secondary_y=False)
    fig.update_yaxes(tickprefix="$", title_text="Price", row=1, col=1, secondary_y=True)
    fig.update_yaxes(ticksuffix="%", row=2, col=1)
    fig.update_yaxes(ticksuffix="%", row=3, col=1)

    return fig


# ---------------------------------------------
# 6. METRICS SUMMARY (standalone HTML report)
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
            if "ROI" in key or "Return" in key or "Alpha" in key or "Profit" in key:
                css_class = "positive" if val > 0 else "negative"
            elif "Drawdown %" in key:
                css_class = "negative" if val < 0 else ""
        rows_html += f'<tr><td class="label">{key}</td><td class="value {css_class}">{display}</td></tr>\n'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Backtest Metrics - {log_base}</title>
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
  <h1>Backtest Metrics Summary</h1>
  <div class="subtitle">Source: {log_base}</div>
  <table>
    {rows_html}
  </table>
</body>
</html>"""
    return html


# ---------------------------------------------
# 7. MAIN
# ---------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Analyze backtest results from log files")
    parser.add_argument("--log", type=str, default=None,
                        help="Log base name (e.g. backtest_20260507_200708). Defaults to latest.")
    parser.add_argument("--balance", type=float, default=100000,
                        help="Initial balance used in the backtest (default: 100000)")
    args = parser.parse_args()

    # Discover log
    log_base = args.log or find_latest_log_base()
    print(f"\n{'='*60}")
    print(f"[ANALYZE] {log_base}")
    print(f"{'='*60}")

    # Parse logs
    print("\nParsing state log...")
    eq_df = parse_state_log(log_base, args.balance)
    print(f"   [OK] {len(eq_df)} equity snapshots parsed")
    print(f"   Range: {eq_df['timestamp'].iloc[0]} -> {eq_df['timestamp'].iloc[-1]}")

    print("\nParsing trade log...")
    trades_df = parse_trade_log(log_base)
    if not trades_df.empty:
        print(f"   [OK] {len(trades_df)} closed trades parsed")
    else:
        print("   [WARN] No trade log available")

    # Compute metrics
    print("\nComputing metrics...")
    metrics, eq_df = compute_metrics(eq_df, trades_df, args.balance)

    # Print report
    print(f"\n{'='*60}")
    print("ADVANCED BACKTEST METRICS")
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
    csv_path = os.path.join(OUTPUT_DIR, "advanced_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"\nMetrics saved to {csv_path}")

    # Generate standalone metrics HTML
    metrics_html = generate_metrics_html(metrics, log_base)
    metrics_html_path = os.path.join(OUTPUT_DIR, "metrics_summary.html")
    with open(metrics_html_path, "w") as f:
        f.write(metrics_html)
    print(f"Metrics HTML saved to {metrics_html_path}")

    # Generate charts
    print("\nGenerating charts...")

    fig_returns = generate_cumulative_returns_chart(eq_df)
    fig_returns.write_html(os.path.join(OUTPUT_DIR, "cumulative_returns.html"))
    try:
        fig_returns.write_image(os.path.join(OUTPUT_DIR, "cumulative_returns.png"), scale=2)
    except Exception:
        print("   [WARN] PNG export skipped (install kaleido: pip install kaleido)")

    fig_dd = generate_drawdown_chart(eq_df)
    fig_dd.write_html(os.path.join(OUTPUT_DIR, "drawdown.html"))
    try:
        fig_dd.write_image(os.path.join(OUTPUT_DIR, "drawdown.png"), scale=2)
    except Exception:
        pass

    fig_dash = generate_combined_dashboard(eq_df)
    fig_dash.write_html(os.path.join(OUTPUT_DIR, "combined_dashboard.html"))
    try:
        fig_dash.write_image(os.path.join(OUTPUT_DIR, "combined_dashboard.png"), scale=2)
    except Exception:
        pass

    print(f"\n[DONE] All outputs saved to {OUTPUT_DIR}/")
    print("   - advanced_metrics.csv")
    print("   - metrics_summary.html")
    print("   - cumulative_returns.html")
    print("   - drawdown.html")
    print("   - combined_dashboard.html")
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
