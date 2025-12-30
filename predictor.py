# predictor.py
import httpx
import os
import logging
from features import build_features
from patterns import detect_patterns
from trend import trend_signal, market_regime
from confidence import confidence_from_probs
from model_registry import get_model
from data_provider import get_candles
from cv_extractor import extract_candles
import numpy as np

# Настройки Grok API
XAI_API_KEY = os.getenv("XAI_API_KEY")
GROK_MODEL = "grok-4"  # или "grok-beta" / "grok-4" в зависимости от доступности

async def call_grok(candles: list, patterns: list, regime: str, tf: str, symbol: str) -> float:
    """
    Асинхронный запрос к Grok API для получения вероятности роста.
    """
    if not XAI_API_KEY:
        logging.warning("XAI_API_KEY не задан — Grok отключён, возвращаем 0.5")
        return 0.5

    recent_candles = candles[-10:]
    candle_desc = []
    for i, c in enumerate(recent_candles):
        direction = "🟢" if c["close"] > c["open"] else "🔴"
        body = abs(c["close"] - c["open"])
        candle_desc.append(
            f"{i+1}: {direction} O:{c['open']:.4f} H:{c['high']:.4f} L:{c['low']:.4f} C:{c['close']:.4f} (body {body:.4f})"
        )

    prompt = f"""
Ты — эксперт по техническому анализу финансовых рынков.
Инструмент: {symbol}
Таймфрейм: {tf} минут
Текущий режим рынка: {regime}

Последние 10 свечей (нормализованные цены):
{"\n".join(candle_desc)}

Обнаруженные паттерны: {", ".join(patterns) if patterns else "нет"}

Дай вероятность роста цены на следующие 2–3 свечи.
Ответь ТОЛЬКО одним числом от 0.00 до 1.00 (например: 0.68).
Без текста, пояснений и символов.
"""

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                "https://api.x.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {XAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": GROK_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 10
                }
            )

            if response.status_code == 200:
                text = response.json()["choices"][0]["message"]["content"].strip()
                try:
                    prob = float(text)
                    if 0.0 <= prob <= 1.0:
                        logging.info(f"Grok вернул вероятность: {prob:.3f}")
                        return prob
                except ValueError:
                    pass
                logging.warning(f"Grok вернул некорректный формат: '{text}'")
            else:
                logging.error(f"Grok API ошибка {response.status_code}: {response.text}")

    except Exception as e:
        logging.error(f"Ошибка при запросе к Grok: {e}")

    return 0.5  # fallback


async def analyze(image_bytes=None, tf=None, symbol=None):
    """
    Основная функция анализа — теперь полностью асинхронная.
    """
    logging.debug(f"Запуск анализа: image={bool(image_bytes)}, tf={tf}, symbol={symbol}")

    if image_bytes:
        candles, quality = extract_candles(image_bytes, max_candles=70)
        source = "скриншот графика"
        symbol = symbol or "Неизвестный инструмент"
    else:
        try:
            candles = get_candles(symbol, interval=f"{tf}m", limit=70)
            source = "Twelve Data / Binance API"
            logging.debug(f"Получено {len(candles)} свечей из API")
        except Exception as e:
            logging.error(f"Ошибка получения данных: {e}")
            return None, f"Ошибка получения данных: {str(e)}"
        quality = 1.0

    if len(candles) < 5:
        return None, "Недостаточно свечей для анализа (минимум 5)"

    features = build_features(candles, tf)
    if len(features) == 0:
        features = np.array([[0.1, 0, 0.1]])
        logging.warning("Использованы fallback-признаки")
    X = features[-1].reshape(1, -1)

    model = get_model(tf)
    ml_prob = model.predict_proba(X)[0][1]

    patterns, pattern_score = detect_patterns(candles)
    trend_prob = trend_signal(candles)
    regime = market_regime(candles)

    # Асинхронный вызов Grok
    grok_prob = await call_grok(candles, patterns, regime, tf, symbol)

    # Адаптивные веса
    if regime == "trend":
        weights = [0.35, 0.15, 0.25, 0.25]
    elif regime == "flat":
        weights = [0.15, 0.40, 0.20, 0.25]
    else:
        weights = [0.25, 0.25, 0.25, 0.25]

    final_prob = np.dot(weights, [ml_prob, pattern_score, trend_prob, grok_prob])

    conf_label, conf_score = confidence_from_probs([ml_prob, pattern_score, trend_prob, grok_prob])

    return {
        "prob": round(final_prob, 3),
        "down_prob": round(1 - final_prob, 3),
        "confidence": conf_label,
        "confidence_score": conf_score,
        "regime": regime,
        "patterns": patterns,
        "tf": tf,
        "symbol": symbol,
        "source": source,
        "quality": quality
    }, None
