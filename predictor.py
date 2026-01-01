import httpx
import os
import logging
import numpy as np

from features import build_features
from patterns import detect_patterns
from trend import trend_signal, market_regime
from confidence import confidence_from_probs
from model_registry import get_model
from data_provider import get_candles
from cv_extractor import extract_candles
# Исправленный импорт — добавлены compute_stochastic и compute_adx_strength
from indicators import (
    compute_rsi,
    compute_macd,
    compute_bollinger,
    compute_ema,
    compute_stochastic,
    compute_adx_strength,
    scalping_strategy,
    compute_atr,
    compute_cci,
    compute_parabolic_sar
)

XAI_API_KEY = os.getenv("XAI_API_KEY")
GROK_MODEL = "grok-4"

async def call_grok(candles, patterns, regime, tf, symbol, indicators):
    if not XAI_API_KEY:
        logging.warning("Grok отключён (нет ключа)")
        return 0.5

    recent = candles[-10:]
    desc = []
    for i, c in enumerate(recent):
        dir_ = "🟢" if c["close"] > c["open"] else "🔴"
        body = abs(c["close"] - c["open"])
        desc.append(f"{i+1}: {dir_} O:{c['open']:.4f} H:{c['high']:.4f} L:{c['low']:.4f} C:{c['close']:.4f} (body {body:.4f})")

    ind_desc = (
        f"RSI: {indicators['rsi']:.1f}\n"
        f"Stoch: {indicators.get('stoch', 50):.1f}\n"
        f"ADX: {indicators.get('adx', 20):.1f}\n"
        f"MACD: {indicators.get('macd', 0):.5f}\n"
        f"Bollinger: {indicators['bb']}\n"
    )

    prompt = f"""Ты скальпер на Forex/крипте.

Анализируй {symbol} {tf}мин | Режим: {regime}
Свечи (последние 10): {", ".join([f"{i+1}: O{recent[i]['open']:.2f} C{recent[i]['close']:.2f}" for i in range(len(recent))])}
Паттерны: {", ".join(patterns) or "нет"}
Индикаторы: RSI{indicators['rsi']:.1f}, Stoch{indicators['stoch']:.1f}, ADX{indicators['adx']:.1f}, MACD{indicators['macd']:.2f}, BB{indicators['bb']}, ATR{indicators['atr']:.4f}, CCI{indicators['cci']:.1f}, PSAR{indicators['psar']}

Вероятность роста на 2–3 свечи? Только число 0.00-1.00"""

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://api.x.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": GROK_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 8
                }
            )
            if resp.status_code != 200:
                logging.error(f"Grok error {resp.status_code}: {resp.text}")
                return 0.5
            txt = resp.json()["choices"][0]["message"]["content"].strip()
            prob = float(txt) if txt.replace(".", "").isdigit() else 0.5
            return prob
    except Exception as e:
        logging.error(f"Grok exception: {e}")
        return 0.5

async def analyze(tf: str = "1", symbol: str = None, image_bytes: bytes = None):
    source = "Скриншот"
    quality = 0.0
    candles = []

    if image_bytes:
        candles, quality = extract_candles(image_bytes)
    else:
        interval = tf + "m" if tf != "10" else "1h"  # пример
        candles = get_candles(symbol, interval=interval, limit=70)
        source = "Twelve Data / Binance"

    if len(candles) < 5:
        return None, "Мало свечей"

    closes = np.array([c["close"] for c in candles])
    highs = np.array([c["high"] for c in candles])
    lows = np.array([c["low"] for c in candles])

    indicators = {
        "rsi": compute_rsi(closes),
        "macd": compute_macd(closes),
        "bb": compute_bollinger(closes),
        "ema": compute_ema(closes[-20:] if len(closes) >= 20 else closes),
        "closes": closes,
        "stoch": compute_stochastic(closes, highs, lows),
        "adx": compute_adx_strength(highs, lows, closes),
        "atr": compute_atr(highs, lows, closes),
        "cci": compute_cci(highs, lows, closes),
        "psar": compute_parabolic_sar(highs, lows, closes),
    }

    features = build_features(candles, tf)
    if features is None or features.size == 0 or len(features) == 0:
        features = np.array([[0.1, 0.0, 0.1]])

    X = features[-1].reshape(1, -1)
    ml_probs = get_model(tf).predict_proba(X)[0]  # [prob_down, prob_neutral, prob_up]
    ml_prob_up = ml_probs[2]  # Для старой логики
    ml_prob_down = ml_probs[0]

    patterns, pattern_score = detect_patterns(candles)

    regime = market_regime(candles)
    scalp_adj = scalping_strategy(indicators, patterns, regime)
    pattern_score = np.clip(pattern_score + scalp_adj, 0.0, 1.0)

    trend_prob = trend_signal(candles)

    grok_prob = await call_grok(candles, patterns, regime, tf, symbol or "Неизвестно", indicators)

    # Взвешивание вероятностей
    if int(tf or 0) <= 5:
        weights = [0.20, 0.30, 0.20, 0.30]  # ml, patterns, trend, grok
    elif regime == "trend":
        weights = [0.30, 0.25, 0.20, 0.25]
    elif regime == "flat":
        weights = [0.15, 0.40, 0.20, 0.25]
    else:
        weights = [0.20, 0.30, 0.25, 0.25]

    final_prob_up = np.dot(weights, [ml_prob_up, pattern_score, trend_prob, grok_prob])
    final_prob_down = np.dot(weights, [ml_prob_down, 1 - pattern_score, 1 - trend_prob, 1 - grok_prob])  # Симметрично
    final_prob = final_prob_up - final_prob_down + 0.5  # Нормализуем к 0-1

    conf_label, conf_score = confidence_from_probs([ml_prob_up, pattern_score, trend_prob, grok_prob, ml_prob_down])  # Добавил down

    return {
        "prob": round(final_prob, 3),  # Net prob up
        "down_prob": round(final_prob_down, 3),  # Explicit down
        "up_prob": round(final_prob_up, 3),     # Explicit up
        "neutral_prob": round(1 - final_prob_up - final_prob_down, 3),
        "confidence": conf_label,
        "confidence_score": conf_score,
        "regime": regime,
        "patterns": patterns,
        "tf": tf,
        "symbol": symbol or "Скриншот",
        "source": source,
        "quality": quality,
        "indicators": indicators
    }, None
