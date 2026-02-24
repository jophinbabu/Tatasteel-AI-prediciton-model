"""
Candlestick Pattern Recognition Module

Detects 8 common candlestick patterns used in technical analysis.
Each pattern returns a binary (0/1) column indicating presence.
"""
import pandas as pd
import numpy as np

def _body(df):
    """Real body of candle."""
    return df['Close'] - df['Open']

def _body_abs(df):
    """Absolute body size."""
    return np.abs(_body(df))

def _upper_shadow(df):
    """Upper shadow (wick)."""
    return df['High'] - df[['Open', 'Close']].max(axis=1)

def _lower_shadow(df):
    """Lower shadow (tail)."""
    return df[['Open', 'Close']].min(axis=1) - df['Low']

def _candle_range(df):
    """Full candle range."""
    return df['High'] - df['Low']


def detect_doji(df, threshold=0.05):
    """
    Doji: Body is very small relative to the total range.
    Signals indecision / potential reversal.
    """
    cr = _candle_range(df)
    ba = _body_abs(df)
    # Body is less than threshold of total range
    return ((ba / cr.replace(0, np.nan)) < threshold).fillna(False).astype(int)


def detect_hammer(df, body_ratio=0.3, shadow_ratio=2.0):
    """
    Hammer: Small body at the top, long lower shadow (>= 2x body).
    Bullish reversal signal at bottom of downtrend.
    """
    ba = _body_abs(df)
    ls = _lower_shadow(df)
    us = _upper_shadow(df)
    cr = _candle_range(df)
    
    small_body = (ba / cr.replace(0, np.nan)) < body_ratio
    long_lower = ls >= (ba * shadow_ratio)
    tiny_upper = us <= ba
    
    return (small_body & long_lower & tiny_upper).fillna(False).astype(int)


def detect_inverted_hammer(df, body_ratio=0.3, shadow_ratio=2.0):
    """
    Inverted Hammer: Small body at the bottom, long upper shadow.
    Bullish reversal signal at bottom of downtrend.
    """
    ba = _body_abs(df)
    ls = _lower_shadow(df)
    us = _upper_shadow(df)
    cr = _candle_range(df)
    
    small_body = (ba / cr.replace(0, np.nan)) < body_ratio
    long_upper = us >= (ba * shadow_ratio)
    tiny_lower = ls <= ba
    
    return (small_body & long_upper & tiny_lower).fillna(False).astype(int)


def detect_bullish_engulfing(df):
    """
    Bullish Engulfing: Current green candle body fully engulfs previous red candle body.
    Strong bullish reversal.
    """
    prev_red = df['Close'].shift(1) < df['Open'].shift(1)
    curr_green = df['Close'] > df['Open']
    engulfs = (df['Open'] <= df['Close'].shift(1)) & (df['Close'] >= df['Open'].shift(1))
    
    return (prev_red & curr_green & engulfs).fillna(False).astype(int)


def detect_bearish_engulfing(df):
    """
    Bearish Engulfing: Current red candle body fully engulfs previous green candle body.
    Strong bearish reversal.
    """
    prev_green = df['Close'].shift(1) > df['Open'].shift(1)
    curr_red = df['Close'] < df['Open']
    engulfs = (df['Open'] >= df['Close'].shift(1)) & (df['Close'] <= df['Open'].shift(1))
    
    return (prev_green & curr_red & engulfs).fillna(False).astype(int)


def detect_morning_star(df, doji_threshold=0.1):
    """
    Morning Star (3-candle pattern): 
    Day 1: Long red candle
    Day 2: Small body (doji-ish) that gaps below
    Day 3: Long green candle closing above midpoint of Day 1
    Bullish reversal.
    """
    cr = _candle_range(df)
    ba = _body_abs(df)
    
    # Day 1: Red candle with large body
    day1_red = (df['Close'].shift(2) < df['Open'].shift(2))
    day1_large = (_body_abs(df).shift(2) / cr.shift(2).replace(0, np.nan)) > 0.5
    
    # Day 2: Small body
    day2_small = (ba.shift(1) / cr.shift(1).replace(0, np.nan)) < doji_threshold * 3
    
    # Day 3: Green candle closing above midpoint of Day 1
    day1_mid = (df['Open'].shift(2) + df['Close'].shift(2)) / 2
    day3_green = df['Close'] > df['Open']
    day3_above_mid = df['Close'] > day1_mid
    
    return (day1_red & day1_large & day2_small & day3_green & day3_above_mid).fillna(False).astype(int)


def detect_evening_star(df, doji_threshold=0.1):
    """
    Evening Star (3-candle pattern):
    Day 1: Long green candle
    Day 2: Small body (doji-ish) that gaps above
    Day 3: Long red candle closing below midpoint of Day 1
    Bearish reversal.
    """
    cr = _candle_range(df)
    ba = _body_abs(df)
    
    day1_green = (df['Close'].shift(2) > df['Open'].shift(2))
    day1_large = (_body_abs(df).shift(2) / cr.shift(2).replace(0, np.nan)) > 0.5
    
    day2_small = (ba.shift(1) / cr.shift(1).replace(0, np.nan)) < doji_threshold * 3
    
    day1_mid = (df['Open'].shift(2) + df['Close'].shift(2)) / 2
    day3_red = df['Close'] < df['Open']
    day3_below_mid = df['Close'] < day1_mid
    
    return (day1_green & day1_large & day2_small & day3_red & day3_below_mid).fillna(False).astype(int)


def detect_three_soldiers_crows(df, min_body_ratio=0.5):
    """
    Three White Soldiers: 3 consecutive green candles with large bodies (returns +1)
    Three Black Crows: 3 consecutive red candles with large bodies (returns -1)
    Returns 0 for neither.
    """
    cr = _candle_range(df)
    ba = _body_abs(df)
    body = _body(df)
    
    large_body = (ba / cr.replace(0, np.nan)) > min_body_ratio
    
    green_1 = (body.shift(2) > 0) & large_body.shift(2)
    green_2 = (body.shift(1) > 0) & large_body.shift(1)
    green_3 = (body > 0) & large_body
    three_soldiers = (green_1 & green_2 & green_3).fillna(False)
    
    red_1 = (body.shift(2) < 0) & large_body.shift(2)
    red_2 = (body.shift(1) < 0) & large_body.shift(1)
    red_3 = (body < 0) & large_body
    three_crows = (red_1 & red_2 & red_3).fillna(False)
    
    result = pd.Series(0, index=df.index)
    result[three_soldiers] = 1
    result[three_crows] = -1
    return result


def detect_patterns(df):
    """
    Detects all 8 candlestick patterns and adds them as columns.
    
    Returns: DataFrame with added pattern columns.
    """
    df = df.copy()
    
    df['Pat_Doji'] = detect_doji(df)
    df['Pat_Hammer'] = detect_hammer(df)
    df['Pat_InvHammer'] = detect_inverted_hammer(df)
    df['Pat_BullEngulf'] = detect_bullish_engulfing(df)
    df['Pat_BearEngulf'] = detect_bearish_engulfing(df)
    df['Pat_MorningStar'] = detect_morning_star(df)
    df['Pat_EveningStar'] = detect_evening_star(df)
    df['Pat_3Soldiers_Crows'] = detect_three_soldiers_crows(df)
    
    return df


if __name__ == "__main__":
    # Quick test with random data
    np.random.seed(42)
    n = 100
    close = np.cumsum(np.random.randn(n) * 0.5) + 500
    data = {
        'Open': close + np.random.randn(n) * 1,
        'High': close + np.abs(np.random.randn(n) * 2),
        'Low': close - np.abs(np.random.randn(n) * 2),
        'Close': close,
        'Volume': np.random.uniform(1000, 5000, n)
    }
    df = pd.DataFrame(data)
    result = detect_patterns(df)
    
    pattern_cols = [c for c in result.columns if c.startswith('Pat_')]
    print("Patterns detected:")
    for col in pattern_cols:
        count = (result[col] != 0).sum()
        print(f"  {col}: {count} signals")
