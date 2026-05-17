import time
from src.tools.exchange_client import ExchangeClient

start = time.time()
print("Init ExchangeClient...")
client = ExchangeClient("binance")
print(f"Init took {time.time() - start:.2f}s")

start = time.time()
print("Fetching OHLCV...")
df = client.get_historical_klines("BTC/USDT", "4h", 250)
print(f"Fetch took {time.time() - start:.2f}s")
