import streamlit as st
import os
import sys
from datetime import datetime, timezone
import traceback

# Add project root to sys.path so imports work correctly
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from src.dashboard.shared.state import init_session_state
from src.dashboard.shared.database import get_db, AuditLog, QueuedTrade
from src.dashboard.components.cards import render_metric_card, render_signal_card
from src.dashboard.components.charts import render_candlestick_chart
from src.tools.binance_client import BinancePublicClient
from src.tools.indicators import MarketDataProcessor

# Page Configuration
st.set_page_config(
    page_title="CryptoPilot - Markets",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State
init_session_state()

from src.dashboard.shared.theme import inject_theme_css
inject_theme_css()

# Main UI
st.title("CryptoPilot: Markets")

# Sidebar
with st.sidebar:
    st.markdown("## CryptoPilot")
    st.caption("AI-Powered Autonomous Trading Agent")
    st.markdown("---")
    
    st.markdown("### Market Selection")
    symbol = st.selectbox(
        "Trading Pair",
        options=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        index=["BTC/USDT", "ETH/USDT", "SOL/USDT"].index(st.session_state.get('current_symbol', 'BTC/USDT')),
        label_visibility="collapsed"
    )
    if symbol != st.session_state['current_symbol']:
        st.session_state['current_symbol'] = symbol
        st.session_state['last_market_data'] = None
        
    st.markdown("---")
    st.markdown("### System Status")
    st.success("Engine Online")
    st.info("Agent Cycle: 4h")
    
    st.markdown("---")
    
    with st.expander("LLM Configuration"):
        from src.core.config_manager import ConfigManager
        
        # Load current settings
        settings = ConfigManager.load_settings()
        
        # Providers list
        providers = ["LM Studio", "Ollama", "OpenAI", "Gemini", "NVIDIA"]
        
        import requests
        
        @st.cache_data(ttl=10)
        def fetch_lmstudio_models(base_url):
            try:
                url = f"{base_url}/models"
                resp = requests.get(url, timeout=2)
                if resp.status_code == 200:
                    data = resp.json()
                    return [m['id'] for m in data.get('data', [])]
            except Exception:
                pass
            return []

        @st.cache_data(ttl=10)
        def fetch_ollama_models(base_url):
            try:
                # Ollama OpenAI-compat endpoint
                url = f"{base_url}/models"
                resp = requests.get(url, timeout=2)
                if resp.status_code == 200:
                    data = resp.json()
                    return [m['id'] for m in data.get('data', [])]
            except Exception:
                pass
            # Fallback: try the native Ollama API
            try:
                native_url = base_url.replace("/v1", "").rstrip("/") + "/api/tags"
                resp = requests.get(native_url, timeout=2)
                if resp.status_code == 200:
                    data = resp.json()
                    return [m['name'] for m in data.get('models', [])]
            except Exception:
                pass
            return []

        def get_model_options(provider, base_url, ollama_url, is_embedding=False):
            if provider == "LM Studio":
                models = fetch_lmstudio_models(base_url)
                return models if models else (["nomic-embed-text"] if is_embedding else ["gemma4:e2b"])
            elif provider == "Ollama":
                models = fetch_ollama_models(ollama_url)
                return models if models else (["nomic-embed-text"] if is_embedding else ["llama3.1", "gemma3", "qwen3", "deepseek-r1", "mistral", "phi4"])
            elif provider == "OpenAI":
                return ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"] if is_embedding else ["gpt-4.5-turbo", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o1", "o1-mini", "o3-mini", "gpt-3.5-turbo"]
            elif provider == "Gemini":
                return ["text-embedding-004", "models/embedding-001"] if is_embedding else ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-pro", "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"]
            elif provider == "NVIDIA":
                return ["nvidia/nv-embedqa-e5-v5", "nvidia/nv-embedqa-mistral-7b-v2"] if is_embedding else ["meta/llama-3.3-70b-instruct", "meta/llama-3.1-405b-instruct", "meta/llama-3.1-70b-instruct", "deepseek-ai/deepseek-v4-pro", "deepseek-ai/deepseek-coder-33b-instruct", "mistralai/mistral-large", "google/gemma-2-27b-it"]
            return []

        st.markdown("**API Keys & URLs**")
        lm_url = st.text_input("LM Studio Base URL", value=settings.get("lm_studio_base_url", "http://localhost:1234/v1"))
        ollama_url = st.text_input("Ollama Base URL", value=settings.get("ollama_base_url", "http://localhost:11434/v1"))
        
        st.markdown("**LLM Backend**")
        llm_prov = st.selectbox("Provider", providers, index=providers.index(settings.get("llm_provider", "LM Studio")), key="llm_prov")
        
        llm_opts = get_model_options(llm_prov, lm_url, ollama_url, is_embedding=False)
        current_llm = settings.get("llm_model", "gemma4:e2b")
        if current_llm not in llm_opts: llm_opts = [current_llm] + llm_opts
        
        llm_sel = st.selectbox("Model", llm_opts + ["Other (Custom)"], index=llm_opts.index(current_llm) if current_llm in llm_opts else 0, key="llm_sel")
        if llm_sel == "Other (Custom)":
            llm_mod = st.text_input("Enter Custom Model", value=current_llm, key="llm_mod_custom")
        else:
            llm_mod = llm_sel
        
        st.markdown("**Embeddings Backend**")
        embed_prov = st.selectbox("Provider", providers, index=providers.index(settings.get("embed_provider", "LM Studio")), key="embed_prov")
        
        embed_opts = get_model_options(embed_prov, lm_url, ollama_url, is_embedding=True)
        current_embed = settings.get("embed_model", "nomic-embed-text")
        if current_embed not in embed_opts: embed_opts = [current_embed] + embed_opts
        
        embed_sel = st.selectbox("Model", embed_opts + ["Other (Custom)"], index=embed_opts.index(current_embed) if current_embed in embed_opts else 0, key="embed_sel")
        if embed_sel == "Other (Custom)":
            embed_mod = st.text_input("Enter Custom Model", value=current_embed, key="embed_mod_custom")
        else:
            embed_mod = embed_sel
        
        st.markdown("---")
        api_keys = settings.get("api_keys", {})
        
        # Determine which keys to show based on selected providers
        required_keys = set()
        if llm_prov not in ("LM Studio", "Ollama"): required_keys.add(llm_prov.lower())
        if embed_prov not in ("LM Studio", "Ollama"): required_keys.add(embed_prov.lower())
        
        new_keys = api_keys.copy()
        if "openai" in required_keys:
            new_keys["openai"] = st.text_input("OpenAI API Key", value=api_keys.get("openai", ""), type="password")
        if "gemini" in required_keys:
            new_keys["gemini"] = st.text_input("Gemini API Key", value=api_keys.get("gemini", ""), type="password")
        if "nvidia" in required_keys:
            new_keys["nvidia"] = st.text_input("NVIDIA API Key", value=api_keys.get("nvidia", ""), type="password")
            
        if st.button("Save Settings", use_container_width=True):
            old_settings = ConfigManager.load_settings()
            old_embed = old_settings.get("embed_model")
            old_embed_prov = old_settings.get("embed_provider")

            settings["llm_provider"] = llm_prov
            settings["llm_model"] = llm_mod
            settings["embed_provider"] = embed_prov
            settings["embed_model"] = embed_mod
            settings["lm_studio_base_url"] = lm_url
            settings["ollama_base_url"] = ollama_url
            settings["api_keys"] = new_keys
            
            ConfigManager.save_settings(settings)
            
            # Check if embed model changed, rebuild DB if needed
            if old_embed != embed_mod or old_embed_prov != embed_prov:
                from src.core.knowledge_base_lms import TradingKnowledgeBase
                st.info("Embedding model changed. Rebuilding Knowledge Base...")
                kb = TradingKnowledgeBase()
                kb.rebuild()
            
            st.success("Settings Saved!")
            
            # Trigger reload
            st.rerun()

    st.markdown("---")
    st.caption("v1.0.0 | Deepmind")

# Auto-Fetch Market Data (4h timeframe)
if st.session_state['last_market_data'] is None:
    with st.spinner("Fetching latest 4h market data..."):
        try:
            client = BinancePublicClient()
            df = client.get_historical_klines(st.session_state['current_symbol'], "4h", limit=200)
            df = MarketDataProcessor.add_indicators(df)
            st.session_state['last_market_data'] = df
        except Exception as e:
            st.error(f"Failed to fetch market data: {e}")

# Fetch Latest DB Signals
db_gen = get_db()
db = next(db_gen)
try:
    latest_log = db.query(AuditLog).filter(AuditLog.symbol == st.session_state['current_symbol']).order_by(AuditLog.id.desc()).first()
    latest_trade = db.query(QueuedTrade).filter(QueuedTrade.asset_name == st.session_state['current_symbol']).order_by(QueuedTrade.id.desc()).first()
finally:
    db.close()

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Live Chart (4h): {st.session_state['current_symbol']}")
    if st.session_state['last_market_data'] is not None:
        fig = render_candlestick_chart(st.session_state['last_market_data'], title=st.session_state['current_symbol'])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Market data not available.")

with col2:
    st.subheader("Latest AI Analysis")
    
    if st.button("Run Fresh Analysis", use_container_width=True):
        with st.spinner(f"Agent analyzing {st.session_state['current_symbol']}... This may take a minute."):
            try:
                from src.agents.data_agent import MainDataAgent
                agent = MainDataAgent()
                agent.run_for_symbol(st.session_state['current_symbol'])
                st.rerun()
            except Exception as e:
                st.error(f"Analysis failed: {e}")
    if latest_log:
        signal_data = {
            'signal': latest_log.action,
            'confidence': latest_log.confidence
        }
        render_signal_card(signal_data)
        
        st.markdown(f"**Timestamp:** {latest_log.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        st.markdown("### Analyst Reasoning")
        st.info(latest_log.explanation)
        
        if latest_trade and latest_trade.signal != "HOLD":
            st.markdown("### Trade Proposal")
            st.write(f"**Status:** {latest_trade.status}")
            st.write(f"**Size:** ${latest_trade.position_size:.2f}")
            st.write(f"**Target:** ${latest_trade.target:.4f}")
            st.write(f"**Stop Loss:** ${latest_trade.stop_loss:.4f}")
            
    else:
        st.write("No analysis logged yet for this asset. Please wait for the Data Agent's first cycle.")
