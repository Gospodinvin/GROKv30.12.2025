# model.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import logging

class CandleModel:
    def __init__(self, tf: str):
        self.tf = tf
        self.model_path = f"models/model_{tf}m.joblib"
        self.model = None
        self.fallback = True
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                self.fallback = False
                logging.info(f"Загружена обученная модель для {self.tf}m")
            except Exception as e:
                logging.error(f"Ошибка загрузки модели {self.tf}m: {e}")
        else:
            logging.info(f"Обученная модель для {self.tf}m не найдена — используется fallback")

    def predict_proba(self, X):
        if X.shape[0] == 0:
            return np.array([[0.333, 0.333, 0.333]])  # Для 3 классов

        if not self.fallback and self.model is not None:
            probs = self.model.predict_proba(X)
            # Порядок классов: [-1, 0, 1] (падение, нейтрал, рост)
            return probs

        # Fallback multiclass
        momentum = X[:, 0]
        volatility = X[:, 2]
        prob_up = np.clip(0.333 + 0.3 * np.tanh(momentum) - 0.15 * volatility, 0.05, 0.9)
        prob_down = np.clip(0.333 - 0.3 * np.tanh(momentum) + 0.15 * volatility, 0.05, 0.9)
        prob_neutral = 1 - prob_up - prob_down
        return np.vstack([prob_down, prob_neutral, prob_up]).T
