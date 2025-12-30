import numpy as np
from sklearn.ensemble import RandomForestClassifier

class CandleModel:
    """
    Улучшенная модель на RandomForest.
    По умолчанию использует простую эвристику, но можно обучить на данных.
    """
    def __init__(self):
        self.fallback = True
        self.model = None

    def fit(self, X, y):
        if len(X) < 10:
            return
        self.model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
        self.model.fit(X, y)
        self.fallback = False

    def predict_proba(self, X):
        if X.shape[0] == 0:
            return np.array([[0.5, 0.5]])
        
        if not self.fallback and self.model is not None:
            return self.model.predict_proba(X)
        
        # Fallback эвристика
        momentum = X[:, 0]
        volatility = X[:, 2]
        prob_up = 0.5 + 0.4 * momentum - 0.25 * volatility
        prob_up = np.clip(prob_up, 0.05, 0.95)
        return np.vstack([1 - prob_up, prob_up]).T
