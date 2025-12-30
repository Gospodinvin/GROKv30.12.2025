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
            return np.array([[0.5, 0.5]])

        if not self.fallback and self.model is not None:
            return self.model.predict_proba(X)

        # Fallback (на всякий случай)
        momentum = X[:, 0]
        volatility = X[:, 2]
        prob_up = 0.5 + 0.4 * np.tanh(momentum) - 0.2 * volatility
        prob_up = np.clip(prob_up, 0.05, 0.95)
        return np.vstack([1 - prob_up, prob_up]).T
