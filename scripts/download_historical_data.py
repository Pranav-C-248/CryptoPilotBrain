import os
import sys
import time
import requests
import pandas as pd
from datetime import datetime, timezone

# Ensure imports work from src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.tools.indicators import MarketDataProcessor

def fetch_historical_data(symbol="BTCUSDT", interval="4h", start_year=2026, start_month=1, start_day=1):
    """Fetches historical klines handling Binance pagination limit of 1000."""
    print(f"Fetching {interval} data for {symbol} from {start_year}-{start_month:02d}-{start_day:02d} to today...")
    
    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_time = int(datetime(start_year, start_month, start_day, tzinfo=timezone.utc).timestamp() * 1000)
    
    all_klines = []
    url = "https://api.binance.com/api/v3/klines"
    
    while start_time < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": 1000,
            "startTime": start_time,
            "endTime": end_time
        }
        try:
            res = requests.get(url, params=params)
            res.raise_for_status()
            data = res.json()
        except Exception as e:
            print(f"Error fetching data: {e}")
            time.sleep(5)
            continue
            
        if not data:
            break
            
        all_klines.extend(data)
        start_time = data[-1][0] + 1 # Next candle
        time.sleep(0.5) # Avoid rate limits
        
    df = pd.DataFrame(all_klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)
    
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = df[col].astype(float)
        
    print(f"Fetched {len(df)} candles.")
    df = df.drop_duplicates(subset=['open_time']).sort_values('open_time').reset_index(drop=True)
    
    print("Calculating indicators...")
    df = MarketDataProcessor.add_indicators(df)
    
    return df

if __name__ == "__main__":
    df = fetch_historical_data()
    
    output_dir = os.path.join(project_root, "tests")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "BTCUSDT_4h_historical.csv")
    
    df.to_csv(output_path, index=False)
    print(f"Successfully saved full history to {output_path}")
