# model.py  (переименовал класс для ясности)
import numpy as np

class CandleModel:
    """
    Лёгкая ML-модель без переобучения
    """
    def predict_proba(self, X):
        if X.shape[0] == 0:
            return np.array([[0.5, 0.5]])
        momentum = X[:, 0]          # body size (scaled)
        volatility = X[:, 2]        # range (high-low scaled)

        prob_up = 0.5 + 0.4 * momentum - 0.3 * volatility
        prob_up = np.clip(prob_up, 0.05, 0.95)

        return np.vstack([1 - prob_up, prob_up]).T