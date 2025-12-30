import numpy as np

def build_features(candles, tf):
    if len(candles) < 2:
        return np.array([])  # Пустой массив, чтобы выше сработала защита

    X = []
    scale = {"1": 1.0, "2": 1.2, "5": 1.5, "10": 2.0}.get(tf, 1.0)
    
    for i in range(1, len(candles)):
        c = candles[i]
        body = abs(c["close"] - c["open"]) * scale
        direction = np.sign(c["close"] - c["open"])
        vol = (c["high"] - c["low"]) * scale
        X.append([body, direction, vol])
    
    return np.array(X) if X else np.array([])
