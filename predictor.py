# predictor.py  (обновлённый импорт и использование)
from features import build_features
from patterns import detect_patterns
from trend import trend_signal, market_regime
from confidence import confidence_from_probs
from model_registry import get_model
# Заменяем импорт binance_data на новый универсальный
from data_provider import get_candles
from cv_extractor import extract_candles
import numpy as np

def analyze(image_bytes=None, tf=None, symbol=None):
    if image_bytes:
        # Режим по скриншоту
        candles, quality = extract_candles(image_bytes, max_candles=70)
        source = "скриншот графика"
        symbol = symbol or "Неизвестный инструмент"
    else:
        # Режим API — теперь через единый провайдер
        try:
            candles = get_candles(symbol, interval=f"{tf}m", limit=70)
            source = "Twelve Data / Binance API"
        except Exception as e:
            return None, f"Ошибка получения данных: {str(e)}"

        quality = 1.0

    if len(candles) < 5:
        return None, "Недостаточно свечей для анализа (минимум 5)"

    # Признаки строим по всем свечам, но предсказываем по последней завершённой
    features = build_features(candles, tf)
    if len(features) == 0:
        features = np.array([[0.1, 0, 0.1]])  # экстренный fallback
    X = features[-1].reshape(1, -1)

    model = get_model(tf)
    ml_prob = model.predict_proba(X)[0][1]

    patterns, pattern_score = detect_patterns(candles)
    trend_prob = trend_signal(candles)
    regime = market_regime(candles)

    # Адаптивные веса
    if regime == "trend":
        weights = [0.5, 0.2, 0.3]
    elif regime == "flat":
        weights = [0.2, 0.5, 0.3]
    else:
        weights = [0.33, 0.33, 0.34]

    final_prob = np.dot(weights, [ml_prob, pattern_score, trend_prob])

    conf_label, conf_score = confidence_from_probs([ml_prob, pattern_score, trend_prob])

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