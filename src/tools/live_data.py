import asyncio
import threading
import pandas as pd
import ccxt.pro as ccxtpro
from datetime import datetime, timezone
from src.tools.exchange_client import ExchangeClient

class LiveDataCache:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(LiveDataCache, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.cache = {}
        self.cache_lock = threading.Lock()
        self.symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        self.exchanges = ["binance", "kraken", "coinbase", "bybit", "kucoin"]
        
        try:
            from src.core.config_manager import ConfigManager
            settings = ConfigManager.load_settings()
            active_exchange = settings.get("active_exchange", "binance")
            if active_exchange in self.exchanges:
                self.exchanges.remove(active_exchange)
                self.exchanges.insert(0, active_exchange)
        except Exception as e:
            print("Could not load config for active exchange prioritization:", e)
        
        # Start daemon thread
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        
        self._initialized = True

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._main_async())
        
    async def _main_async(self):
        tasks = []
        for ex in self.exchanges:
            self.cache[ex] = {}
            # Instantiate a single REST client per exchange to avoid multiple load_markets() calls
            rest_client = ExchangeClient(ex)
            
            # Pre-load markets sequentially to avoid GIL lock contention and CCXT rate-limiter spam
            try:
                await asyncio.to_thread(rest_client.exchange.load_markets)
            except Exception as e:
                print(f"[{ex}] Failed to pre-load markets: {e}")
            for sym in self.symbols:
                self.cache[ex][sym] = None
                tasks.append(asyncio.create_task(self._watch_symbol(ex, sym, rest_client)))
                await asyncio.sleep(0.1)  # Stagger startup to prevent 429 Rate Limit errors
        
        await asyncio.gather(*tasks)

    def _get_4h_boundary(self):
        now = datetime.now(timezone.utc)
        hour = (now.hour // 4) * 4
        return now.replace(hour=hour, minute=0, second=0, microsecond=0)

    async def _watch_symbol(self, exchange_id: str, symbol: str, rest_client: ExchangeClient):
        # 1. Fetch initial via REST using our robust ExchangeClient pagination
        def fetch_initial():
            return rest_client.get_historical_klines(symbol, "4h", limit=250)
            
        while True:
            try:
                df = await asyncio.to_thread(fetch_initial)
                with self.cache_lock:
                    self.cache[exchange_id][symbol] = df
                break
            except Exception as e:
                print(f"[{exchange_id}] Failed initial fetch for {symbol}: {e}")
                await asyncio.sleep(5) # retry

        # Initialize CCXT pro client inside the event loop
        try:
            ex_class = getattr(ccxtpro, exchange_id)()
        except AttributeError:
            ex_class = ccxtpro.binance()
            
        # Coinbase symbol formatting
        formatted_symbol = symbol
        if exchange_id == 'coinbase':
            if formatted_symbol.endswith('USDT'):
                formatted_symbol = formatted_symbol[:-4] + '/USD'
            elif formatted_symbol.endswith('USD'):
                pass
                
        current_boundary = self._get_4h_boundary()
        
        while True:
            try:
                # Check for boundary rollover (start of a new 4h period)
                if datetime.now(timezone.utc) >= current_boundary + pd.Timedelta(hours=4):
                    df = await asyncio.to_thread(fetch_initial)
                    with self.cache_lock:
                        self.cache[exchange_id][symbol] = df
                    current_boundary = self._get_4h_boundary()
                    continue

                ticker = await ex_class.watch_ticker(formatted_symbol)
                last_price = ticker['last']
                
                with self.cache_lock:
                    df = self.cache[exchange_id][symbol]
                    if df is not None and not df.empty:
                        last_idx = df.index[-1]
                        df.at[last_idx, 'close'] = last_price
                        if last_price > df.at[last_idx, 'high']:
                            df.at[last_idx, 'high'] = last_price
                        if last_price < df.at[last_idx, 'low']:
                            df.at[last_idx, 'low'] = last_price
            except Exception as e:
                # Some exchanges like Coinbase might rarely disconnect or fail to stream
                # print(f"[{exchange_id}] WS Error for {symbol}: {e}")
                await asyncio.sleep(2)

    def get_data(self, exchange_id: str, symbol: str) -> pd.DataFrame:
        with self.cache_lock:
            df = self.cache.get(exchange_id, {}).get(symbol, None)
            if df is not None:
                return df.copy()
            return None
