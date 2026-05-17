import asyncio
import time
from src.tools.live_data import LiveDataCache

async def main():
    print("Starting cache...")
    start = time.time()
    cache = LiveDataCache()
    
    # Wait until Binance BTC/USDT is loaded
    while True:
        df = cache.get_data("binance", "BTC/USDT")
        if df is not None:
            print(f"Data loaded in {time.time() - start:.2f} seconds!")
            print(df.head(2))
            break
        await asyncio.sleep(0.1)

asyncio.run(main())
