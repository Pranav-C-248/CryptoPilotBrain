import pandas as pd
import requests
from datetime import datetime

class BinancePublicClient:
    BASE_URL = "https://api.binance.us/api/v3"

    def __init__(self):
        pass

    def get_historical_klines(self, symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
        """
        Fetches historical klines (OHLCV) from Binance public API.
        interval: e.g., '1h', '4h', '1d'
        """
        url = f"{self.BASE_URL}/klines"
        params = {
            "symbol": symbol.replace("/", ""),
            "interval": interval,
            "limit": limit
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # Binance kline format:
        # [
        #   [
        #     1499040000000,      // Open time
        #     "0.01634790",       // Open
        #     "0.80000000",       // High
        #     "0.01575800",       // Low
        #     "0.01577100",       // Close
        #     "148976.11427815",  // Volume
        #     1499644799999,      // Close time
        #     ...
        #   ]
        # ]

        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ])

        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

        # Convert string to float
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            df[col] = df[col].astype(float)

        return df

    def get_ticker(self, symbol: str) -> dict:
        url = f"{self.BASE_URL}/ticker/24hr"
        params = {"symbol": symbol.replace("/", "")}
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
