# train_models.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.utils import resample
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)

# Исправленный импорт
from binance_data import get_candles as get_candles_binance

# Таймфреймы
TIMEFRAMES = ["1", "2", "5", "10"]
INTERVALS = {"1": "1m", "2": "2m", "5": "5m", "10": "10m"}

# Больше пар для лучшего обучения
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT"]

LIMIT = 5000  # ~3-5 месяцев на 1m

PROFIT_THRESHOLD = 0.18  # чуть повыше, чтобы сигналы были качественнее

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def build_features_single(candle_prev, candle_current, scale):
    body = abs(candle_current["close"] - candle_current["open"]) * scale
    direction = 1 if candle_current["close"] > candle_current["open"] else -1
    vol = (candle_current["high"] - candle_current["low"]) * scale
    return [body, direction, vol]


def prepare_data(candles, tf):
    if len(candles) < 50:
        return None, None

    df = pd.DataFrame(candles)
    scale = {"1": 1.0, "2": 1.2, "5": 1.5, "10": 2.0}[tf]

    X = []
    y = []

    for i in range(len(df) - 4):
        current = df.iloc[i + 1]
        future_close = df.iloc[i + 3]["close"]

        price_change = (future_close - current["close"]) / current["close"] * 100
        label = 1 if price_change > PROFIT_THRESHOLD else 0

        prev = df.iloc[i]
        features = build_features_single(prev, current, scale)

        X.append(features)
        y.append(label)

    return np.array(X), np.array(y)


def train_and_save():
    for tf in TIMEFRAMES:
        print(f"\n=== Обучение модели для {tf}-минутного таймфрейма ===")
        interval = INTERVALS[tf]
        all_X = []
        all_y = []

        for symbol in SYMBOLS:
            try:
                logging.info(f"Загрузка {symbol} {interval}...")
                candles = get_candles_binance(symbol, interval=interval, limit=LIMIT)
                if len(candles) < 100:
                    continue

                X, y = prepare_data(candles, tf)
                if X is not None and len(X) > 0:
                    all_X.extend(X)
                    all_y.extend(y)
                    print(f"  {symbol}: +{len(X)} примеров (всего: {len(all_X)})")
            except Exception as e:
                logging.error(f"Ошибка с {symbol}: {e}")

        if len(all_X) < 500:
            print(f"Недостаточно данных для {tf}m — пропускаем")
            continue

        X = np.array(all_X)
        y = np.array(all_y)

        # Балансировка классов
        pos_idx = y == 1
        neg_idx = y == 0
        if pos_idx.sum() < neg_idx.sum():
            X_pos = resample(X[pos_idx], replace=True, n_samples=neg_idx.sum(), random_state=42)
            X_bal = np.vstack((X_pos, X[neg_idx]))
            y_bal = np.array([1] * neg_idx.sum() + [0] * neg_idx.sum())
        else:
            X_neg = resample(X[neg_idx], replace=True, n_samples=pos_idx.sum(), random_state=42)
            X_bal = np.vstack((X[pos_idx], X_neg))
            y_bal = np.array([1] * pos_idx.sum() + [0] * pos_idx.sum())

        # Обучение
        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_split=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_bal, y_bal)

        preds = model.predict(X_bal)
        print(f"\n{tf}m — Результат на обучающей выборке:")
        print(classification_report(y_bal, preds))

        # Сохранение
        path = os.path.join(MODEL_DIR, f"model_{tf}m.joblib")
        joblib.dump(model, path)
        print(f"Модель сохранена: {path}")

    print("\nОбучение всех моделей завершено!")


if __name__ == "__main__":
    train_and_save()
