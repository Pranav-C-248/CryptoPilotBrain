import pandas as pd
import ccxt

class ExchangeClient:
    def __init__(self, exchange_id: str = "binance"):
        try:
            exchange_class = getattr(ccxt, exchange_id.lower())
            self.exchange = exchange_class({
                'enableRateLimit': True,
            })
        except AttributeError:
            # Fallback to binance if exchange_id is invalid
            self.exchange = ccxt.binance({
                'enableRateLimit': True,
            })
        self.exchange_id = exchange_id.lower()

    def _fetch_paginated_ohlcv(self, symbol: str, timeframe: str, limit: int) -> list:
        all_ohlcv = []
        unit = timeframe[-1]
        try:
            val = int(timeframe[:-1])
        except ValueError:
            val = 1
        multiplier = 1
        if unit == 'm': multiplier = 60
        elif unit == 'h': multiplier = 60 * 60
        elif unit == 'd': multiplier = 24 * 60 * 60
        
        # fetch a bit more than limit to ensure we have enough
        target_fetch = limit + int(limit * 0.1)
        since = self.exchange.milliseconds() - (target_fetch * val * multiplier * 1000)
        
        for i in range(10): # max 10 iterations to be safe
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
                if not ohlcv: break
                
                if all_ohlcv and ohlcv[0][0] <= all_ohlcv[-1][0]:
                    ohlcv = [x for x in ohlcv if x[0] > all_ohlcv[-1][0]]
                    
                if not ohlcv: break
                
                all_ohlcv.extend(ohlcv)
                if len(all_ohlcv) >= limit:
                    break
                since = ohlcv[-1][0] + 1
            except Exception as e:
                # If it fails on the very first try, raise it so fallback can catch it
                if i == 0 and not all_ohlcv:
                    raise e
                # Otherwise, if we fetched some data but then errored (e.g. rate limit), return what we have
                break
                
        return all_ohlcv[-limit:] if len(all_ohlcv) >= limit else all_ohlcv

    def get_historical_klines(self, symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
        """
        Fetches historical klines (OHLCV) using ccxt.
        interval: e.g., '1h', '4h', '1d'
        """
        formatted_symbol = symbol
        if "/" not in formatted_symbol and len(formatted_symbol) > 3:
            if formatted_symbol.endswith("USDT"):
                formatted_symbol = formatted_symbol[:-4] + "/USDT"
            elif formatted_symbol.endswith("USD"):
                formatted_symbol = formatted_symbol[:-3] + "/USD"
        
        try:
            data = self._fetch_paginated_ohlcv(formatted_symbol, interval, limit)
            df = pd.DataFrame(data, columns=["open_time", "open", "high", "low", "close", "volume"])
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        except Exception as e:
            if interval == '4h':
                try:
                    data = self._fetch_paginated_ohlcv(formatted_symbol, '1h', limit * 5)
                    df = pd.DataFrame(data, columns=["open_time", "open", "high", "low", "close", "volume"])
                    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
                    df.set_index('open_time', inplace=True)
                    resampled = df.resample('4h').agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna()
                    resampled.reset_index(inplace=True)
                    df = resampled.tail(limit).reset_index(drop=True)
                except Exception as fallback_e:
                    raise Exception(f"Failed to fetch and resample 4h klines for {formatted_symbol} on {self.exchange_id}. Original error: {e}. Fallback error: {fallback_e}")
            else:
                raise Exception(f"Failed to fetch {interval} klines for {formatted_symbol} on {self.exchange_id}: {e}")

        return df

    def get_ticker(self, symbol: str) -> dict:
        return self.exchange.fetch_ticker(symbol)
