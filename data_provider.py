# data_provider.py
from binance_data import get_candles as get_candles_binance
from twelve_data import get_client
import logging

def get_candles(symbol: str, interval: str = "1m", limit: int = 70):
    """
    Универсальная функция получения свечей.
    Сначала Twelve Data (с правильным форматом интервала), потом Binance.
    """
    original_symbol = symbol.upper()
    binance_symbol = original_symbol.replace("USD", "USDT")
    forex_symbol = original_symbol.replace("USD", "/USD") if "USD" in original_symbol else original_symbol

    # Маппинг интервалов для Twelve Data
    td_interval_map = {
        "1m": "1min",
        "2m": "1min",   # 2min не поддерживается → fallback на 1min
        "5m": "5min",
        "10m": "5min",  # 10min не поддерживается → fallback на 5min
    }
    td_interval = td_interval_map.get(interval, interval)  # для остальных (1h и т.д.) оставляем как есть

    client = get_client()
    if client:
        # Пробуем Forex-формат сначала
        logging.info(f"Пытаемся получить {forex_symbol} {td_interval} через Twelve Data...")
        candles = client.get_candles(symbol=forex_symbol, interval=td_interval, outputsize=limit)
        if candles:
            logging.info(f"Успешно получены данные через Twelve Data ({len(candles)} свечей)")
            return candles

        # Пробуем оригинальный символ
        logging.info(f"Пытаемся получить {original_symbol} {td_interval} через Twelve Data...")
        candles = client.get_candles(symbol=original_symbol, interval=td_interval, outputsize=limit)
        if candles:
            logging.info(f"Успешно получены данные через Twelve Data с оригинальным символом ({len(candles)} свечей)")
            return candles

        logging.warning("Twelve Data не вернула данные по обоим вариантам символа, переходим на Binance...")

    # Fallback на Binance
    logging.info(f"Пытаемся получить {binance_symbol} {interval} через Binance...")
    try:
        candles = get_candles_binance(binance_symbol, interval=interval, limit=limit)
        if candles:
            logging.info(f"Успешно получены данные через Binance ({len(candles)} свечей)")
            return candles
    except Exception as e:
        logging.error(f"Binance не сработал для {binance_symbol}: {e}")

    # Последняя попытка — оригинальный символ в Binance
    logging.info(f"Пытаемся получить {original_symbol} {interval} через Binance (оригинальный символ)...")
    try:
        candles = get_candles_binance(original_symbol, interval=interval, limit=limit)
        if candles:
            logging.info(f"Успешно получены данные через Binance с оригинальным символом ({len(candles)} свечей)")
            return candles
    except Exception as e:
        logging.error(f"Binance не сработал и для оригинального символа {original_symbol}: {e}")

    raise RuntimeError("Не удалось получить данные ни с Twelve Data, ни с Binance")
