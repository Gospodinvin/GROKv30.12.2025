import requests
import logging

BINANCE_ENDPOINTS = [
    "https://api.binance.com",
    "https://data-api.binance.vision"
]

def get_candles(symbol, interval="1m", limit=70):
    symbol = symbol.replace("USD", "USDT")

    for base_url in BINANCE_ENDPOINTS:
        url = f"{base_url}/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        try:
            r = requests.get(url, params=params, timeout=5)
            if r.status_code == 200:
                data = r.json()
                return [
                    {
                        "open": float(c[1]),
                        "high": float(c[2]),
                        "low": float(c[3]),
                        "close": float(c[4]),
                        "volume": float(c[5])
                    }
                    for c in data
                ]
            else:
                logging.error(f"Binance {r.status_code} via {base_url}")
        except Exception as e:
            logging.error(f"Binance error via {base_url}: {e}")

    raise RuntimeError("Binance data unavailable (all endpoints failed)")
