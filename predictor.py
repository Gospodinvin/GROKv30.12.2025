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
import logging

def analyze(image_bytes=None, tf=None, symbol=None):
    logging.debug(f"Starting analyze: image_bytes={bool(image_bytes)}, tf={tf}, symbol={symbol}")
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
            logging.debug(f"Получено {len(candles)} свечей из API")
        except Exception as e:
            logging.error(f"Ошибка получения данных в analyze: {str(e)}")
            return None, f"Ошибка получения данных: {str(e)}"

        quality = 1.0

    if len(candles) < 5:
        logging.warning(f"Недостаточно свечей: {len(candles)}")
        return None, "Недостаточно свечей для анализа (минимум 5)"

    # Признаки строим по всем свечам, но предсказываем по последней завершённой
    features = build_features(candles, tf)
    logging.debug(f"Построено {len(features)} признаков")
    if len(features) == 0:
        features = np.array([[0.1, 0, 0.1]])  # экстренный fallback
        logging.warning("Fallback features used")
    if features.shape[0] == 0:  # Дополнительная проверка
        return None, "Не удалось построить признаки (нет свечей)"
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
