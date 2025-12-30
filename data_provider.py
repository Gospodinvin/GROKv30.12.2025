# data_provider.py  (НОВЫЙ ФАЙЛ — единая точка получения свечей)
from binance_data import get_candles as get_candles_binance
from twelve_data import get_client
import logging

def get_candles(symbol: str, interval: str = "1m", limit: int = 70):
    """
    Универсальная функция получения свечей.
    Сначала пытается Twelve Data (для всех рынков), потом Binance (для крипты).
    """
    original_symbol = symbol.upper()
    binance_symbol = original_symbol.replace("USD", "USDT")  # Только для Binance
    forex_symbol = original_symbol.replace("USD", "/USD") if "USD" in original_symbol else original_symbol  # Для Twelve Data Forex

    client = get_client()
    if client:
        logging.info(f"Пытаемся получить {forex_symbol} {interval} через Twelve Data...")
        candles = client.get_candles(symbol=forex_symbol, interval=interval, outputsize=limit)
        if candles:
            logging.info(f"Успешно получены данные через Twelve Data ({len(candles)} свечей)")
            return candles
        logging.warning(f"Twelve Data не вернул данные для {forex_symbol}, пробуем оригинальный {original_symbol}...")
        candles = client.get_candles(symbol=original_symbol, interval=interval, outputsize=limit)
        if candles:
            logging.info(f"Успешно получены данные через Twelve Data с оригинальным символом ({len(candles)} свечей)")
            return candles

        logging.warning(f"Twelve Data не вернул данные, пробуем Binance...")

    # Fallback на Binance
    logging.info(f"Пытаемся получить {binance_symbol} {interval} через Binance...")
    try:
        candles = get_candles_binance(binance_symbol, interval=interval, limit=limit)
        logging.info(f"Успешно получены данные через Binance ({len(candles)} свечей)")
        return candles
    except Exception as e:
        logging.error(f"Binance не сработал для {binance_symbol}: {e}")

    # Попробовать original_symbol в Binance на всякий случай
    try:
        candles = get_candles_binance(original_symbol, interval=interval, limit=limit)
        logging.info(f"Успешно получены данные через Binance с оригинальным символом ({len(candles)} свечей)")
        return candles
    except Exception as e:
        logging.error(f"Оба источника не сработали для {symbol}: {e}")
        raise RuntimeError("Не удалось получить данные ни с Twelve Data, ни с Binance")
