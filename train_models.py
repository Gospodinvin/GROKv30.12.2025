# train_models.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.utils import resample
import joblib
import os
import logging

from sklearn.model_selection import train_test_split, GridSearchCV  # Новый импорт

logging.basicConfig(level=logging.INFO)

# Теперь импортируем правильно и используем символы с USD (как в боте)
from binance_data import get_candles as get_candles_binance

# Таймфреймы
TIMEFRAMES = ["1", "2", "5", "10"]
INTERVALS = {"1": "1m", "2": "2m", "5": "5m", "10": "10m"}

# Символы в формате, как в твоём боте: с USD (функция в binance_data.py сама заменит на USDT)
SYMBOLS = ["BTCUSD", "ETHUSD", "BNBUSD", "SOLUSD", "XRPUSD", "ADAUSD", "DOGEUSD"]

PROFIT_THRESHOLD = 0.20  # Настроил выше для скальпинга
LIMIT = 10000  # Больше данных

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def prepare_data(candles, tf):
    if len(candles) < 50:
        return None, None

    df = pd.DataFrame(candles)
    X_raw = build_features(candles, tf)  # Теперь с индикаторами!
    y = []

    for i in range(len(df) - 4):
        current = df.iloc[i + 1]
        future_close = df.iloc[i + 3]["close"]
        price_change_pct = (future_close - current["close"]) / current["close"] * 100
        
        if price_change_pct > PROFIT_THRESHOLD:
            label = 1  # Рост
        elif price_change_pct < -PROFIT_THRESHOLD:
            label = -1  # Падение
        else:
            label = 0  # Нейтрал
        y.append(label)

    return X_raw[:len(y)], np.array(y)  # Align lengths

def train_and_save():
    for tf in TIMEFRAMES:
        print(f"\n=== Обучение модели для {tf}-минутного таймфрейма ===")
        interval = INTERVALS[tf]
        all_X = []
        all_y = []

        for symbol in SYMBOLS:
            try:
                logging.info(f"Загрузка {symbol} {interval}...")
                # Здесь binance_data.py сам заменит USD → USDT
                candles = get_candles_binance(symbol, interval=interval, limit=LIMIT)
                if len(candles) < 100:
                    print(f"  {symbol}: мало свечей ({len(candles)})")
                    continue

                X, y = prepare_data(candles, tf)
                if X is not None and len(X) > 0:
                    all_X.extend(X)
                    all_y.extend(y)
                    print(f"  {symbol}: +{len(X)} примеров (всего: {len(all_X)})")
            except Exception as e:
                logging.error(f"Ошибка с {symbol}: {e}")
                print(f"  {symbol}: ошибка — {e}")

        if len(all_X) < 500:
            print(f"Недостаточно данных для {tf}m — пропускаем")
            continue

        X = np.array(all_X)
        y = np.array(all_y)

        # Балансировка (multiclass)
        classes = np.unique(y)
        max_count = max(np.sum(y == c) for c in classes)
        X_bal = []
        y_bal = []
        for c in classes:
            X_c = X[y == c]
            y_c = y[y == c]
            X_bal.extend(resample(X_c, replace=True, n_samples=max_count, random_state=42))
            y_bal.extend([c] * max_count)

        X_bal = np.array(X_bal)
        y_bal = np.array(y_bal)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42)

        # Тюнинг params (lite grid search)
        param_grid = {
            'n_estimators': [400, 600],
            'max_depth': [10, 12],
            'min_samples_split': [8, 10]
        }
        rf = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
        grid = GridSearchCV(rf, param_grid, cv=3, scoring='f1_macro')
        grid.fit(X_train, y_train)

        model = grid.best_estimator_
        print(f"Best params: {grid.best_params_}")

        preds = model.predict(X_test)
        print(f"\n{tf}m — Результат (на тестовой выборке):")
        print(classification_report(y_test, preds))

        # Сохранение
        path = os.path.join(MODEL_DIR, f"model_{tf}m.joblib")
        joblib.dump(model, path)
        print(f"Модель сохранена: {path}\n")

    print("Обучение всех моделей завершено! Модели лежат в папке 'models/'")

if __name__ == "__main__":
    train_and_save()
