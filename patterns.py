import numpy as np

def detect_patterns(candles):
    patterns = []
    if len(candles) < 3:
        return patterns, 0.0

    score = 0.0
    for i in range(-3, 0):  # последние 3 свечи
        prev = candles[i-1] if i > -3 else candles[0]
        c = candles[i]

        o, c_, h, l = c["open"], c["close"], c["high"], c["low"]
        body = abs(c_ - o)
        range_ = h - l
        upper_wick = h - max(o, c_)
        lower_wick = min(o, c_) - l

        direction = 1 if c_ > o else -1

        # Engulfing
        if i > -3:
            prev_body = abs(prev["close"] - prev["open"])
            if body > prev_body * 1.5 and direction == -np.sign(prev["close"] - prev["open"]):
                patterns.append("Engulfing")
                score += 0.25

        # Marubozu / Impulse
        if body > range_ * 0.85:
            patterns.append("Marubozu")
            score += 0.20

        # Hammer / Shooting Star
        if body < range_ * 0.4 and lower_wick > body * 2 and upper_wick < body:
            patterns.append("Hammer" if direction > 0 else "Shooting Star")
            score += 0.22

        # Pinbar
        if (lower_wick > body * 2 and upper_wick < body * 0.5) or (upper_wick > body * 2 and lower_wick < body * 0.5):
            patterns.append("Pinbar")
            score += 0.18

        # Doji
        if body < range_ * 0.1:
            patterns.append("Doji")
            score += 0.10

    patterns = list(set(patterns))
    return patterns, min(score, 1.0)