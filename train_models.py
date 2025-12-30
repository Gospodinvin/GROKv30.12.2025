# train_models.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os
from binance_data import get_candles_binance
import logging

logging.basicConfig(level=logging.INFO)

# Таймфреймы для обучения
TIMEFRAMES = ["1", "2", "5", "10"]
INTERVALS = {"1": "1m", "2": "2m", "5": "5m", "10": "10m"}

# Символы для обучения (можно добавить больше)
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]

# Сколько свечей брать для обучения
LIMIT = 5000  # ~3–5 месяцев на 1m

# Порог роста для метки (в %)
PROFIT_THRESHOLD = 0.15  # 0.15% за 2–3 свечи — реалистично для скальпинга

# Папка для моделей
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

    # Начинаем с индекса, где можно смотреть вперёд на 3 свечи
    for i in range(len(df) - 4):
        current = df.iloc[i + 1]  # текущая свеча (предсказываем её исход)
        future_close = df.iloc[i + 3]["close"]  # цена через ~2–3 свечи

        # Метка: выросла ли цена достаточно
        price_change = (future_close - current["close"]) / current["close"] * 100
        label = 1 if price_change > PROFIT_THRESHOLD else 0

        # Признаки из предыдущей свечи
        prev = df.iloc[i]
        features = build_features_single(prev, current, scale)

        X.append(features)
        y.append(label)

    return np.array(X), np.array(y)


def train_and_save():
    for tf in TIMEFRAMES:
        print(f"\nОбучение модели для {tf} минута...")
        interval = INTERVALS[tf]
        all_X = []
        all_y = []

        for symbol in SYMBOLS:
            try:
                logging.info(f"Загрузка {symbol} {interval}")
                candles = get_candles_binance(symbol, interval=interval, limit=LIMIT)
                if len(candles) < 100:
                    continue

                X, y = prepare_data(candles, tf)
                if X is not None:
                    all_X.extend(X)
                    all_y.extend(y)
                    logging.info(f"{symbol}: добавлено {len(X)} примеров")
            except Exception as e:
                logging.error(f"Ошибка с {symbol}: {e}")

        if len(all_X) < 100:
            logging.warning(f"Недостаточно данных для {tf}m")
            continue

        X = np.array(all_X)
        y = np.array(all_y)

        # Балансировка классов (опционально)
        from sklearn.utils import resample
        pos = X[y == 1]
        neg = X[y == 0]
        if len(pos) < len(neg):
            pos = resample(pos, replace=True, n_samples=len(neg), random_state=42)
        elif len(neg) < len(pos):
            neg = resample(neg, replace=True, n_samples=len(pos), random_state=42)
        X_bal = np.vstack((pos, neg))
        y_bal = np.array([1] * len(pos) + [0] * len(neg))

        # Обучение
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_bal, y_bal)

        # Оценка
        preds = model.predict(X_bal)
        print(f"\n{tf}m — Отчёт:")
        print(classification_report(y_bal, preds))

        # Сохранение
        path = os.path.join(MODEL_DIR, f"model_{tf}m.joblib")
        joblib.dump(model, path)
        logging.info(f"Модель {tf}m сохранена: {path}")

    print("\nОбучение завершено!")


if __name__ == "__main__":
    train_and_save()
