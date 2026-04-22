# CryptoPilot - Project Specification

## Table of Contents
1. [Project Overview](#project-overview)
2. [Tech Stack](#tech-stack)
3. [Project Structure](#project-structure)
4. [Dependencies](#dependencies)
5. [Configuration](#configuration)
6. [Database Schema](#database-schema)
7. [API Integrations](#api-integrations)
8. [Technical Indicators](#technical-indicators)
9. [Dashboard](#dashboard)
10. [Features](#features)
11. [Environment Variables](#environment-variables)

---

## Project Overview

**Project Name**: CryptoPilot  
**Type**: Explainable AI Crypto Trading System  
**Core Functionality**: An AI-powered crypto trading system that analyzes market data, calculates technical indicators, assesses sentiment from news, manages risk, and generates explainable trading signals.  
**Target Users**: Crypto traders seeking AI-assisted trading with full decision transparency.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Web Dashboard | Streamlit (>=1.40.0) |
| Database | SQLite with SQLAlchemy ORM |
| LLM Integration | LangChain with Ollama |
| Language | Python 3.11+ |
| Visualization | Plotly (>=5.24.0) |
| Data Processing | Pandas, NumPy |
| Technical Analysis | ta, talipp |

---

## Project Structure

```
cryptopilot/
├── .env                          # Environment variables
├── .env.example                  # Environment template
├── .streamlit/
│   └── config.toml               # Streamlit configuration
├── data/
│   └── cryptopilot.db            # SQLite database
├── logs/
│   ├── cryptopilot_*.log         # Daily logs
│   └── decisions.jsonl           # Decision audit trail
├── src/
│   ├── main.py                   # CLI entry point
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py             # Configuration settings
│   │   ├── models.py             # Pydantic data models
│   │   ├── explainability.py     # Decision explanation
│   │   └── stubs.py              # Type stubs
│   ├── dashboard/
│   │   ├── __init__.py
│   │   ├── app.py                # Main dashboard
│   │   ├── components/
│   │   │   ├── __init__.py
│   │   │   ├── cards.py          # UI cards
│   │   │   └── charts.py         # Chart components
│   │   ├── pages/
│   │   │   ├── __init__.py
│   │   │   ├── news.py           # News analysis page
│   │   │   ├── audit.py          # Decision audit page
│   │   │   └── xai.py            # Explainability page
│   │   └── shared/
│   │       ├── __init__.py
│   │       ├── database.py       # Database operations
│   │       ├── state.py          # App state management
│   │       └── utils.py          # Utility functions
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── binance_client.py     # Binance API client
│   │   ├── coinbase_client.py    # Coinbase API client
│   │   ├── news_client.py        # News aggregation client
│   │   └── indicators.py         # Technical indicators
├── pyproject.toml                # Project metadata & dependencies
└── README.md                     # Documentation
```

---

## Dependencies

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| python-binance | >=1.0.19 | Binance API client |
| coinbase-advanced-py | >=1.2.0 | Coinbase API client |
| httpx | >=0.27.0 | Async HTTP client |
| pandas | >=2.2.0 | Data manipulation |
| numpy | >=1.26.0 | Numerical computing |
| pydantic | >=2.9.0 | Data validation |
| pydantic-settings | >=2.5.0 | Settings management |
| python-dotenv | >=1.0.0 | Environment variables |
| talipp | >=0.8.0 | Technical analysis |
| ta | >=0.11.0 | Technical indicators |
| feedparser | >=6.0.11 | RSS feed parsing |
| requests | >=2.32.0 | HTTP requests |
| streamlit | >=1.40.0 | Web dashboard |
| plotly | >=5.24.0 | Charts/visualization |
| sqlalchemy | >=2.0.0 | Database ORM |
| aiosqlite | >=0.20.0 | Async SQLite |
| loguru | >=0.7.0 | Logging |
| langchain-ollama | >=0.1.0 | Ollama LLM integration |

---

## Configuration

### pyproject.toml

Project metadata, dependency management, and tool configurations including:
- Ruff linter configuration
- MyPy type checker configuration  
- Pytest configuration

### .streamlit/config.toml

Streamlit dashboard theme configuration:
- Dark mode enabled
- Primary color: #0ecb81 (green)
- Background: #0b0e11
- Secondary: #161b22

### .env.example

Template for environment variables:

```bash
# Exchange APIs (optional)
BINANCE_API_KEY=
BINANCE_API_SECRET=
COINBASE_API_KEY=
COINBASE_API_SECRET=

# News APIs
CRYPTOPANIC_API_KEY=

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
OLLAMA_EMBED_MODEL=nomic-embed-text
```

---

## Database Schema

**Database Type**: SQLite  
**Location**: `data/cryptopilot.db`  
**ORM**: SQLAlchemy 2.0+

### Table: audit_logs

| Column | Type | Constraints |
|--------|------|-------------|
| id | Integer | Primary Key |
| timestamp | DateTime | Indexed |
| symbol | String | NOT NULL |
| action | String | NOT NULL |
| confidence | Float | NOT NULL |
| price | Float | NOT NULL |
| factors | JSON | Nullable |
| explanation | Text | Nullable |
| outcome | String | Nullable |
| profit_loss | Float | Nullable |

### Table: cached_news

Cached news articles with sentiment.

| Column | Type | Constraints |
|--------|------|-------------|
| id | Integer | Primary Key |
| title | String | NOT NULL |
| url | String | NOT NULL |
| source | String | NOT NULL |
| published | DateTime | NOT NULL |
| sentiment_label | String | Nullable |
| sentiment_score | Float | Nullable |
| symbol | String | Nullable |
| cached_at | DateTime | NOT NULL |

---

## API Integrations

### Exchange APIs

#### Binance (`src/tools/binance_client.py`)
- OHLCV candlestick data
- 24-hour ticker statistics
- Order book depth
- Current price

#### Coinbase (`src/tools/coinbase_client.py`)
- Public candles
- OHLCV data

### News APIs

#### CryptoPanic API
- News articles
- User votes
- Currency tagging

#### CoinDesk RSS
- Latest news feed
- News headlines

### LLM

#### Ollama (Local)
- Model: `llama3.2:3b` (default) or `gemma4:e2b`
- Embed model: `nomic-embed-text`
- Base URL: `http://localhost:11434`
- Used for generating human-readable trading explanations

---

## Technical Indicators

| Indicator | Parameters | Description |
|-----------|------------|-------------|
| RSI | Period: 14 | Relative Strength Index |
| MACD | Fast: 12, Slow: 26, Signal: 9 | Moving Average Convergence Divergence |
| Bollinger Bands | Period: 20, Std: 2 | Upper, Middle, Lower bands |
| EMA | Periods: 20, 50, 200 | Exponential Moving Averages |
| ATR | Period: 14 | Average True Range |

Implementation: Uses both `ta` and `talipp` libraries.

---

## Dashboard

### Pages (Streamlit)

1. **Markets** - Live price data, interactive charts, trading view
2. **News** - Sentiment analysis, news headlines
3. **Audit** - Decision logs, reasoning traces
4. **XAI** - Explainability interface for decisions

### Components

- **Cards** (`src/dashboard/components/cards.py`): Reusable UI cards for displaying data
- **Charts** (`src/dashboard/components/charts.py`): Plotly-based visualization components

### Theme

- Full dark mode with custom CSS
- Primary accent: #0ecb81 (green)
- Background: #0b0e11
- Secondary: #161b22
- Configured in `.streamlit/config.toml`

---

## Features

### Core Features

1. **Market Analysis**
   - Data fetching from multiple exchanges (Binance, Coinbase)
   - Technical indicator calculation
   - News sentiment analysis
   - Risk management enforcement

2. **Explainable AI**
   - Full decision logging to JSONL file
   - Factor attribution for each decision
   - LLM-generated natural language explanations

3. **Real-Time Dashboard**
   - Live market prices
   - Interactive charts (Plotly)
   - Sentiment overview

4. **Audit Trail**
   - Complete decision history
   - Reasoning logs
   - Signal export capability

### Supported Cryptocurrencies

- Bitcoin (BTC/USDT)
- Ethereum (ETH/USDT)
- Solana (SOL/USDT)

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BINANCE_API_KEY` | - | Binance API key (optional) |
| `BINANCE_API_SECRET` | - | Binance API secret (optional) |
| `COINBASE_API_KEY` | - | Coinbase API key (optional) |
| `COINBASE_API_SECRET` | - | Coinbase API secret (optional) |
| `CRYPTOPANIC_API_KEY` | - | CryptoPanic API key |
| `OLLAMA_BASE_URL` | http://localhost:11434 | Ollama server URL |
| `OLLAMA_MODEL` | llama3.2:3b | LLM model name |
| `OLLAMA_EMBED_MODEL` | nomic-embed-text | Embedding model name |

---

## Key File Reference

### Entry Points
- `src/main.py` - CLI main entry point

### Core
- `src/core/config.py` - Settings and configuration
- `src/core/models.py` - Pydantic data models

### Dashboard
- `src/dashboard/app.py` - Main Streamlit dashboard
- `src/dashboard/pages/news.py` - News analysis page
- `src/dashboard/pages/audit.py` - Audit trail page
- `src/dashboard/pages/xai.py` - Explainability page
- `src/dashboard/shared/database.py` - Database operations

### Tools
- `src/tools/binance_client.py` - Binance API client
- `src/tools/coinbase_client.py` - Coinbase API client
- `src/tools/news_client.py` - News aggregation client
- `src/tools/indicators.py` - Technical indicators calculation