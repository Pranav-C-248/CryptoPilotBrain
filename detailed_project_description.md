# CryptoPilot: Detailed Project Description

CryptoPilot is an advanced, AI-powered autonomous cryptocurrency trading agent and backtesting platform. It leverages multiple specialized agents, a semantic knowledge base of trading strategies, and a comprehensive paper-trading engine to analyze market data, manage risk, and execute trades autonomously. The platform also includes a rich Streamlit-based web dashboard for monitoring, configuration, and Explainable AI (XAI).

## 1. System Architecture Overview

The system is built on a modular, agentic architecture written in Python. It separates data ingestion, technical analysis, AI-driven strategy selection, risk management, and trade execution into distinct components.

The major layers are:
- **Core Orchestration**: The `MainDataAgent` orchestrates the analysis cycle, and `main.py` provides the CLI to run either the dashboard or a background worker.
- **Agentic Layer**: Specialized agents (`AnalystAgent`, `RiskManagerAgent`, `SentimentAgent`) handle specific domains of the trading process.
- **Knowledge Base Layer**: A ChromaDB-backed vector database stores and retrieves trading strategies based on current market conditions.
- **Execution Layer**: The `PaperTradingEngine` simulates trade execution using a custom AST-based condition evaluator.
- **Data & Tools Layer**: Utilities for fetching data from Binance (`BinancePublicClient`) and calculating technical indicators (`MarketDataProcessor`).
- **Dashboard (Frontend)**: A multi-page Streamlit application for visualization, portfolio tracking, and manual trade approval.
- **Database**: A local SQLite database using SQLAlchemy to persist state (audit logs, portfolio, positions, trade ledger).
- **Backtesting & Analysis**: Dedicated scripts (`backtesting_lms.py`, `analyze_backtest.py`) for running historical simulations and generating performance reports.

---

## 2. Core Components & Agents (Backend)

### Main Data Agent (`src/agents/data_agent.py`)
This is the primary orchestrator for the live trading loop. It:
1. Iterates through configured trading pairs (e.g., BTC/USDT, ETH/USDT, SOL/USDT).
2. Fetches 4h historical klines via the Binance client.
3. Uses the `MarketDataProcessor` to calculate technical indicators and compile a "market context packet".
4. Invokes the `SentimentAgent` to get a real-time sentiment score from news.
5. Passes the data to the `AnalystAgent` to get a trading signal.
6. Passes the signal to the `RiskManagerAgent` to validate against risk parameters and determine sizing/targets.
7. Logs the outcome to the `AuditLog` in the database.
8. Queues approved trades into the `PaperTradingEngine`.

### Analyst Agent (`src/agents/analyst_lms.py`)
The `AnalystAgent` acts as an "Institutional Grade Quantitative Analyst". It uses an LLM (Large Language Model) to evaluate the market.
- **Regime Detection**: It first detects the market regime (Trending, Volatile, or Sideways) based on market structure, ATR, and volume spikes.
- **Narrative Generation**: It builds human-readable narratives of indicator states (e.g., RSI extremes, Bollinger Band touches).
- **Knowledge Base Querying**: It formulates a highly specific query string describing the current regime, momentum, volatility, and moving average state. It uses this query to retrieve the top 2 matching trading strategies from the `TradingKnowledgeBase`.
- **LLM Evaluation**: It sends a structured prompt to the LLM, containing the market data, technical indicators, sentiment score, and the retrieved candidate strategies. The LLM evaluates the strategies against the current conditions, performs a confluence check, and outputs a JSON response containing a signal (BUY, SELL, or HOLD), confidence score, reasoning, internal monologue, and formal mathematical entry/exit conditions.

### Risk Manager Agent (`src/agents/risk_manager.py`)
The `RiskManagerAgent` is a purely deterministic, rule-based capital preservation layer (no LLM). It evaluates the `AnalystSignal`.
- **Veto Gauntlet**: It runs the signal through a series of checks: portfolio saturation, global confidence floor, RSI extreme vetoes, sentiment conflicts, ATR volatility caps, and strategy-specific guards (e.g., EMA200 slope, MA convergence).
- **Level Calculation**: It calculates Stop Loss and Take Profit levels based on the strategy's risk profile (Aggressive vs. Conservative), ATR (Average True Range) multipliers, and confidence scalars.
- **Position Sizing**: It determines the position size based on equity risk percentages (e.g., 2% of total equity), stop loss distance, and open positions exposure scaling.

### Sentiment Agent (`src/agents/sentiment_agent.py`)
Fetches the latest cryptocurrency news from CoinDesk's RSS feed and analyzes the sentiment of the article titles using a pre-trained FinBERT model (`ProsusAI/finbert`) from the Hugging Face `transformers` library. It aggregates the scores to produce a final sentiment value between 0.0 (Extreme Fear) and 1.0 (Extreme Greed).

---

## 3. Core Engine & Tools

### Paper Trading Engine (`src/core/paper_trader.py`)
Manages the lifecycle of queued trades (UNENTERED -> ENTERED -> CLOSED).
- **AST Evaluator**: It features a custom, safe `ast.parse` evaluator that evaluates the mathematical entry and exit conditions generated by the AnalystAgent's LLM (e.g., `price > ema20 and rsi < 30`) against real-time indicator values.
- **State Management**: It updates the SQLite database, deducting balance, updating open positions, and recording executed trades in the `TradeLedger`. It checks stop losses, take profits, and dynamic exit conditions.

### Knowledge Base (`src/core/knowledge_base_lms.py`)
A semantic search engine using ChromaDB and embedding models (e.g., `nomic-embed-text`). It indexes trading strategies defined in `strategies_processed.json`.
- **Metadata Filtering**: Uses ChromaDB metadata to pre-filter strategies based on the current market regime (`regime_primary` and `regime_also_valid`).
- **Prompt Payload**: Retrieves a lean JSON representation of the strategy mechanics to feed into the Analyst LLM, saving context window tokens.

### Technical Indicators (`src/tools/indicators.py`)
Uses the `ta` (Technical Analysis) library to compute a rich set of indicators on raw OHLCV data:
- RSI (14 & 5)
- EMAs (9, 20, 50, 200)
- Bollinger Bands (20,2σ and 40,2σ)
- ADX (14)
- ATR (Average True Range) and custom expansion flags
- Donchian Channels (10, 20, 50, 55)
- Volume spike detection algorithms
- Custom trend structure detection (Higher Highs / Lower Lows) and Moving Average twist detection.

### LLM Factory (`src/core/llm_factory.py`)
A standardized factory pattern to instantiate LangChain LLM and Embedding clients. It supports multiple providers configurable via `settings.json`:
- LM Studio (Local)
- Ollama (Local)
- OpenAI
- Gemini
- NVIDIA

---

## 4. Dashboard (Frontend)

The frontend is a Streamlit application located in `src/dashboard/`. It provides a sleek, dark-themed UI for interacting with the agent.

### Main App (`app.py`)
- **Sidebar**: Allows users to select the active trading pair (BTC, ETH, SOL), view system status, and configure LLM/Embedding providers and API keys dynamically.
- **Main View**: Displays a live interactive Plotly candlestick chart with Bollinger Bands and RSI overlays. Displays the latest AI analysis, confidence score, and proposed trade parameters.

### Pages
- **Audit Trail (`pages/audit.py`)**: A tabular view of the `AuditLog` database table, showing a history of all AI signals, confidence scores, and reasoning.
- **News (`pages/news.py`)**: Displays the latest articles fetched by the Sentiment Agent, along with the FinBERT AI sentiment score and reasoning for each article.
- **Portfolio (`pages/portfolio.py`)**: Tracks the paper trading account. Shows Available Balance, Total Realized P&L, Active Positions, and a full Trade Ledger.
- **Trading Queue (`pages/trading.py`)**: Manages the `QueuedTrade` table.
  - **Pending Approvals**: If `auto_approve_trades` is disabled, proposed trades sit here. Users can review, manually edit entry/exit conditions, targets, and stop losses, and then Approve or Reject them.
  - **Active Engine Queue**: Shows trades that are approved but waiting for conditions to be met (UNENTERED) and trades currently open (ENTERED).
- **Explainable AI - XAI (`pages/xai.py`)**: Provides deep insights into *why* the AI made a decision. It fetches logs and displays the full "Internal Monologue" of the LLM, complete with syntax highlighting for keywords (BUY, SELL, indicators).

---

## 5. Backtesting Suite

The project includes a robust backtesting environment to simulate the agent's performance on historical data.

### Backtest Engine (`backtesting_lms.py`)
Iterates through a historical CSV dataset (e.g., `BTCUSDT_4h_historical.csv`) candle by candle.
- Maintains a simulated portfolio and equity curve.
- For every step, it calculates indicators, queries the LLM (with caching to save API costs and time), and evaluates the Risk Manager.
- Uses the same AST evaluator as the Paper Trading Engine to simulate entries and exits based on historical prices.
- Generates HTML and PNG charts using Plotly: an Equity Curve chart and a detailed Trade Executions candlestick chart showing entries and exits.

### Backtest Analyzer (`analyze_backtest.py`)
A post-processing script that parses the generated text logs from `backtesting_lms.py` to compute advanced financial metrics without re-running the expensive LLM simulation.
- **Metrics Computed**: Net Profit, ROI, Sharpe Ratio, Sortino Ratio, Max Drawdown, Calmar Ratio, Alpha (vs Buy & Hold), Win Rate, Profit Factor, Expectancy, and consecutive streaks.
- **Outputs**: Generates an `advanced_metrics.csv`, a standalone `metrics_summary.html` report, and interactive Plotly dashboards (`combined_dashboard.html`, `cumulative_returns.html`, `drawdown.html`).

---

## 6. Database Schema (`src/dashboard/shared/database.py`)

The application uses SQLite and SQLAlchemy. Key models:
- `AuditLog`: Records every analysis cycle (symbol, action, confidence, price, JSON factors, text explanation).
- `CachedNews`: Caches recent news to prevent rate-limiting.
- `Portfolio`: Tracks the global account balance.
- `Position`: Tracks currently held assets and their average entry prices.
- `TradeLedger`: Historical record of all executed BUY/SELL actions, prices, fees, and realized P&L.
- `QueuedTrade`: State machine for trades generated by the AI (status: PENDING_APPROVAL, UNENTERED, ENTERED, CLOSED). Stores target, stop loss, and AST string conditions.
