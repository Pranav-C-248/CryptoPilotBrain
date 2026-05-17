import time
import pandas as pd
from src.tools.exchange_client import ExchangeClient
from src.tools.indicators import MarketDataProcessor

client = ExchangeClient("binance")
df = client.get_historical_klines("BTC/USDT", "4h", 250)

start = time.time()
df = MarketDataProcessor.add_indicators(df)
print(f"Indicators took {time.time() - start:.4f}s")
