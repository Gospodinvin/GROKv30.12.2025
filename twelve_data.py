import requests
import logging
from typing import List, Dict, Optional
from config import TWELVE_DATA_API_KEY

class TwelveDataClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.twelvedata.com"
        self.session = requests.Session()
        self.session.params = {"apikey": api_key}

    def get_candles(self, symbol: str, interval: str, outputsize: int = 50) -> Optional[List[Dict]]:
        """Получить свечи для символа и таймфрейма"""
        try:
            url = f"{self.base_url}/time_series"
            params = {
                "symbol": symbol,
                "interval": interval,
                "outputsize": outputsize,
                "format": "JSON"
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if "values" not in data:
                logging.warning(f"No data for {symbol} {interval}: {data}")
                return None
                
            # Парсим в формат свечей бота
            candles = []
            for candle in data["values"][:outputsize]:
                candles.append({
                    "open": float(candle["open"]),
                    "high": float(candle["high"]),
                    "low": float(candle["low"]),
                    "close": float(candle["close"]),
                    "volume": float(candle.get("volume", 0))
                })
            
            # Нормализуем цены (делим на max для совместимости с CV)
            max_price = max(c["high"] for c in candles) if candles else 1.0
            for candle in candles:
                candle["open"] /= max_price
                candle["high"] /= max_price
                candle["low"] /= max_price
                candle["close"] /= max_price
            
            return candles[::-1]  # последние в конце
            
        except Exception as e:
            logging.error(f"Twelve Data API error: {e}")
            return None

# Глобальный клиент
client = None

def get_client() -> Optional[TwelveDataClient]:
    global client
    if client is None and TWELVE_DATA_API_KEY is not None:
        try:
            client = TwelveDataClient(TWELVE_DATA_API_KEY)
        except Exception as e:
            logging.error(f"Failed to create TwelveDataClient: {e}")
            client = "error"  # Чтобы не пытаться снова
    return client if client != "error" else None
