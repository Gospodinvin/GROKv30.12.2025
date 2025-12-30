import numpy as np

def compute_rsi(closes, period=14):
    deltas = np.diff(closes)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gain[-period:]) if len(gain) >= period else 0.5
    avg_loss = np.mean(loss[-period:]) if len(loss) >= period else 0.5
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def compute_macd(closes, fast=12, slow=26, signal=9):
    if len(closes) < slow:
        return 0.0
    ema_fast = np.mean(closes[-fast:])
    ema_slow = np.mean(closes[-slow:])
    macd_line = ema_fast - ema_slow
    return macd_line

def compute_bollinger(closes, period=20, std_dev=2):
    if len(closes) < period:
        return "neutral"
    sma = np.mean(closes[-period:])
    std = np.std(closes[-period:])
    upper, lower = sma + std_dev * std, sma - std_dev * std
    price = closes[-1]
    if price > upper:
        return "overbought"
    elif price < lower:
        return "oversold"
    return "neutral"

def compute_ema(closes, period=9):
    if len(closes) == 0:
        return 0.0
    alpha = 2 / (period + 1)
    ema = closes[0]
    for p in closes[1:]:
        ema = alpha * p + (1 - alpha) * ema
    return ema

def compute_stochastic(closes, highs, lows, period=14):
    if len(closes) < period:
        return 50.0
    low_min = np.min(lows[-period:])
    high_max = np.max(highs[-period:])
    current_close = closes[-1]
    if high_max == low_min:
        return 50.0
    return 100 * (current_close - low_min) / (high_max - low_min)

def compute_adx_strength(highs, lows, closes, period=14):
    if len(highs) < period + 1:
        return 20.0
    
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(abs(highs[1:] - closes[:-1]), abs(lows[1:] - closes[:-1]))
    )
    atr = np.mean(tr[-period:])
    
    up_move = highs[1:] - highs[:-1]
    down_move = lows[:-1] - lows[1:]
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_di = 100 * np.mean(plus_dm[-period:]) / (atr + 1e-9)
    minus_di = 100 * np.mean(minus_dm[-period:]) / (atr + 1e-9)
    
    dx = abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9) * 100
    return dx

def scalping_strategy(indicators, patterns, regime):
    adj = 0.0
    rsi = indicators['rsi']
    macd = indicators['macd']
    bb = indicators['bb']
    ema = indicators['ema']
    price = indicators['closes'][-1]
    stoch = indicators.get('stoch', 50)
    adx = indicators.get('adx', 20)

    # Buy signals
    if rsi < 30 and any(p in patterns for p in ["Hammer", "Pinbar", "Morning Star", "Bullish Harami"]):
        adj += 0.25
    if macd > 0 and "Engulfing" in patterns:
        adj += 0.20
    if bb == "oversold" and price > ema:
        adj += 0.15
    if stoch < 20 and adx > 25:
        adj += 0.12

    # Sell signals
    if rsi > 70 and any(p in patterns for p in ["Shooting Star", "Evening Star", "Bearish Harami"]):
        adj -= 0.25
    if macd < 0 and "Engulfing" in patterns:
        adj -= 0.20
    if bb == "overbought" and price < ema:
        adj -= 0.15
    if stoch > 80 and adx > 25:
        adj -= 0.12

    if regime == "volatile":
        adj *= 1.2

    return np.clip(adj, -0.3, 0.3)
