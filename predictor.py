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

# Новый код: клиент для Grok API
XAI_API_KEY = os.getenv("XAI_API_KEY")
GROK_MODEL = "grok-4"  # или "grok-beta", в зависимости от доступности на момент запуска

async def call_grok(candles: list, patterns: list, regime: str, tf: str, symbol: str) -> float:
    """
    Асинхронный вызов Grok для получения вероятности роста.
    Возвращает вероятность от 0.0 до 1.0.
    """
    if not XAI_API_KEY:
        logging.warning("XAI_API_KEY не задан — пропускаем вызов Grok")
        return 0.5

    # Берём последние 10 свечей для контекста (чтобы не превысить лимит токенов)
    recent_candles = candles[-10:]
    candle_desc = []
    for i, c in enumerate(recent_candles):
        direction = "🟢" if c["close"] > c["open"] else "🔴"
        body = abs(c["close"] - c["open"])
        candle_desc.append(f"{i+1}: {direction} O:{c['open']:.4f} H:{c['high']:.4f} L:{c['low']:.4f} C:{c['close']:.4f} (body {body:.4f})")

    prompt = f"""
Ты — эксперт по техническому анализу финансовых рынков.
Инструмент: {symbol}
Таймфрейм: {tf} минут
Текущий режим рынка: {regime} ({'тренд' if regime == 'trend' else 'флэт' if regime == 'flat' else 'высокая волатильность'})

Последние 10 свечей (нормализованные цены, от старых к новым):
{chr(10).join(candle_desc)}

Обнаруженные паттерны: {', '.join(patterns) if patterns else 'нет значимых'}

На основе этого анализа дай вероятность роста цены на следующие 2–3 свечи (на том же таймфрейме).
Ответь ТОЛЬКО одним числом от 0.00 до 1.00 (например: 0.72).
Не добавляй пояснений, символов или текста.
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
                    "temperature": 0.3,  # низкая для стабильности
                    "max_tokens": 10
                }
            )

            if response.status_code == 200:
                text = response.json()["choices"][0]["message"]["content"].strip()
                # Извлекаем число
                prob = float(text)
                if 0.0 <= prob <= 1.0:
                    logging.info(f"Grok вернул вероятность: {prob:.3f}")
                    return prob
                else:
                    logging.warning(f"Grok вернул некорректное значение: {text}")
                    return 0.5
            else:
                logging.error(f"Grok API error {response.status_code}: {response.text}")
                return 0.5

    except Exception as e:
        logging.error(f"Ошибка вызова Grok: {e}")
        return 0.5


def analyze(image_bytes=None, tf=None, symbol=None):
    logging.debug(f"Starting analyze: image_bytes={bool(image_bytes)}, tf={tf}, symbol={symbol}")
    
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
            logging.error(f"Ошибка получения данных в analyze: {str(e)}")
            return None, f"Ошибка получения данных: {str(e)}"
        quality = 1.0

    if len(candles) < 5:
        return None, "Недостаточно свечей для анализа (минимум 5)"

    features = build_features(candles, tf)
    if len(features) == 0:
        features = np.array([[0.1, 0, 0.1]])
    X = features[-1].reshape(1, -1)

    model = get_model(tf)
    ml_prob = model.predict_proba(X)[0][1]

    patterns, pattern_score = detect_patterns(candles)
    trend_prob = trend_signal(candles)
    regime = market_regime(candles)

    # Новый вызов Grok (асинхронно — но в текущем контексте aiogram использует sync)
    # Поэтому делаем блокирующий вызов через asyncio.run (допустимо в боте)
    import asyncio
    grok_prob = asyncio.run(call_grok(candles, patterns, regime, tf, symbol))

    # Адаптивные веса с учётом Grok
    if regime == "trend":
        weights = [0.35, 0.15, 0.25, 0.25]  # больше веса ML и Grok
    elif regime == "flat":
        weights = [0.15, 0.40, 0.20, 0.25]  # больше паттернам и Grok
    else:
        weights = [0.25, 0.25, 0.25, 0.25]  # равномерно

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
