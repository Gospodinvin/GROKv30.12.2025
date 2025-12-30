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
from indicators import compute_rsi, compute_macd, compute_bollinger, compute_ema, scalping_strategy
import numpy as np

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
        f"RSI: {indicators['rsi']:.2f}\n"
        f"MACD: {indicators['macd']:.5f}\n"
        f"Bollinger: {indicators['bb']}\n"
        f"EMA9: {indicators['ema']:.5f}"
    )

    prompt = f"""
Эксперт по скальпингу.
{symbol} | {tf} мин | Режим: {regime}
Свечи:
{"\n".join(desc)}
Паттерны: {", ".join(patterns) or "нет"}
Индикаторы:
{ind_desc}
Вероятность роста на 2-3 свечи? Ответь только числом 0.00-1.00
"""

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://api.x.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"},
                json={"model": GROK_MODEL, "messages": [{"role": "user", "content": prompt}],
                      "temperature": 0.3, "max_tokens": 10}
            )
            if resp.status_code == 200:
                text = resp.json()["choices"][0]["message"]["content"].strip()
                try:
                    prob = float(text)
                    if 0 <= prob <= 1:
                        return prob
                except:
                    pass
    except Exception as e:
        logging.error(f"Grok error: {e}")

    return 0.5

async def analyze(image_bytes=None, tf=None, symbol=None):
    if image_bytes:
        candles, quality = extract_candles(image_bytes, max_candles=70)
        source = "скриншот"
        symbol = symbol or "Неизвестно"
    else:
        try:
            candles = get_candles(symbol, interval=f"{tf}m", limit=70)
            source = "Twelve Data / Binance"
        except Exception as e:
            return None, str(e)
        quality = 1.0

    if len(candles) < 5:
        return None, "Мало свечей"

    closes = np.array([c["close"] for c in candles])

    indicators = {
        "rsi": compute_rsi(closes),
        "macd": compute_macd(closes),
        "bb": compute_bollinger(closes),
        "ema": compute_ema(closes[-20:]),
        "closes": closes
    }

    features = build_features(candles, tf) or np.array([[0.1, 0, 0.1]])
    X = features[-1].reshape(1, -1)
    ml_prob = get_model(tf).predict_proba(X)[0][1]

    patterns, pattern_score = detect_patterns(candles)
    trend_prob = trend_signal(candles)
    regime = market_regime(candles)

    scalp_adj = scalping_strategy(indicators, patterns, regime)
    pattern_score = np.clip(pattern_score + scalp_adj, 0.0, 1.0)

    grok_prob = await call_grok(candles, patterns, regime, tf, symbol, indicators)

    if int(tf or 0) <= 5:
        weights = [0.20, 0.30, 0.20, 0.30]
    elif regime == "trend":
        weights = [0.30, 0.25, 0.20, 0.25]
    elif regime == "flat":
        weights = [0.15, 0.40, 0.20, 0.25]
    else:
        weights = [0.20, 0.30, 0.25, 0.25]

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
        "quality": quality,
        "indicators": indicators
    }, None
