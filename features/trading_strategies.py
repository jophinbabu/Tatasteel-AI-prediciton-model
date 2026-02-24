"""
Trading Strategy Feature Module

Generates 7 strategy-based features derived from chart/graph analysis.
Each strategy produces a signal feature that the model can learn from.
"""
import pandas as pd
import numpy as np


def ema_crossover_signal(df, fast=9, slow=21):
    """
    EMA Crossover Strategy:
    - Returns +1 when fast EMA crosses above slow EMA (bullish)
    - Returns -1 when fast EMA crosses below slow EMA (bearish)
    - Returns 0 otherwise
    Also provides EMA distance ratio as a continuous feature.
    """
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    
    # Crossover detection
    prev_diff = (ema_fast.shift(1) - ema_slow.shift(1))
    curr_diff = (ema_fast - ema_slow)
    
    cross_up = (prev_diff <= 0) & (curr_diff > 0)
    cross_down = (prev_diff >= 0) & (curr_diff < 0)
    
    signal = pd.Series(0, index=df.index)
    signal[cross_up] = 1
    signal[cross_down] = -1
    
    # Continuous feature: normalized distance between EMAs
    ema_dist = (ema_fast - ema_slow) / df['Close']
    
    return signal, ema_dist


def breakout_signal(df, window=20):
    """
    Breakout Detection Strategy:
    - Returns +1 when price breaks above the N-bar high (bullish breakout)
    - Returns -1 when price breaks below the N-bar low (bearish breakout)
    - Returns 0 otherwise
    Also provides the breakout strength (how far beyond the range).
    """
    rolling_high = df['High'].rolling(window=window).max().shift(1)
    rolling_low = df['Low'].rolling(window=window).min().shift(1)
    
    breakout_up = (df['Close'] > rolling_high)
    breakout_down = (df['Close'] < rolling_low)
    
    signal = pd.Series(0, index=df.index)
    signal[breakout_up] = 1
    signal[breakout_down] = -1
    
    # Breakout strength (percentage beyond range)
    range_size = (rolling_high - rolling_low).replace(0, np.nan)
    strength = pd.Series(0.0, index=df.index)
    strength[breakout_up] = (df['Close'][breakout_up] - rolling_high[breakout_up]) / range_size[breakout_up]
    strength[breakout_down] = (rolling_low[breakout_down] - df['Close'][breakout_down]) / range_size[breakout_down]
    
    return signal, strength.fillna(0)


def mean_reversion_signal(df, window=20, z_threshold=2.0):
    """
    Mean Reversion Strategy:
    - Computes z-score of price relative to SMA
    - Returns +1 when z-score < -threshold (oversold, expect reversion up)
    - Returns -1 when z-score > +threshold (overbought, expect reversion down)
    - Returns z-score as continuous feature
    """
    sma = df['Close'].rolling(window=window).mean()
    std = df['Close'].rolling(window=window).std()
    
    z_score = (df['Close'] - sma) / std.replace(0, np.nan)
    z_score = z_score.fillna(0)
    
    signal = pd.Series(0, index=df.index)
    signal[z_score < -z_threshold] = 1   # Oversold -> expect bounce
    signal[z_score > z_threshold] = -1   # Overbought -> expect drop
    
    return signal, z_score


def support_resistance_levels(df, window=20):
    """
    Support/Resistance Strategy:
    - Computes rolling pivot, support, and resistance levels
    - Returns distance from support and resistance as features
    """
    # Classic pivot point
    pivot = (df['High'].rolling(window).max() + 
             df['Low'].rolling(window).min() + 
             df['Close']) / 3
    
    support = 2 * pivot - df['High'].rolling(window).max()
    resistance = 2 * pivot - df['Low'].rolling(window).min()
    
    # Normalized distances
    sr_range = (resistance - support).replace(0, np.nan)
    dist_to_support = (df['Close'] - support) / sr_range
    dist_to_resistance = (resistance - df['Close']) / sr_range
    
    return dist_to_support.fillna(0), dist_to_resistance.fillna(0)


def trend_strength_adx(df, window=14):
    """
    ADX (Average Directional Index) for trend strength.
    - ADX > 25 = trending market
    - ADX < 20 = ranging/choppy market
    Returns: ADX value (0-100), +DI, -DI
    """
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # +DM and -DM
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # True Range
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    
    # Smoothed averages
    atr = tr.rolling(window=window).mean()
    plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(window=window).mean() / atr.replace(0, np.nan))
    
    # DX and ADX
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.rolling(window=window).mean()
    
    return adx.fillna(0), plus_di.fillna(0), minus_di.fillna(0)


def volume_breakout_signal(df, window=20, spike_threshold=2.0):
    """
    Volume Breakout Strategy:
    - Detects volume spikes (volume > threshold × average)
    - Combines with price direction for signal
    - Returns +1 for bullish volume spike, -1 for bearish, 0 otherwise
    Also returns volume spike ratio as continuous feature.
    """
    vol_sma = df['Volume'].rolling(window=window).mean()
    vol_ratio = df['Volume'] / vol_sma.replace(0, np.nan)
    vol_ratio = vol_ratio.fillna(0)
    
    is_spike = vol_ratio > spike_threshold
    price_up = df['Close'] > df['Open']
    price_down = df['Close'] < df['Open']
    
    signal = pd.Series(0, index=df.index)
    signal[is_spike & price_up] = 1
    signal[is_spike & price_down] = -1
    
    return signal, vol_ratio


def momentum_divergence(df, rsi_window=14):
    """
    RSI-Price Momentum Divergence:
    - Bullish divergence: Price makes lower low but RSI makes higher low
    - Bearish divergence: Price makes higher high but RSI makes lower high
    Returns divergence signal (+1 bullish, -1 bearish, 0 none)
    and RSI slope as continuous feature.
    """
    # Compute RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    
    lookback = 10
    
    # Price lows/highs over lookback
    price_lower_low = df['Close'] < df['Close'].rolling(lookback).min().shift(1)
    price_higher_high = df['Close'] > df['Close'].rolling(lookback).max().shift(1)
    
    # RSI lows/highs over lookback
    rsi_higher_low = rsi > rsi.rolling(lookback).min().shift(1)
    rsi_lower_high = rsi < rsi.rolling(lookback).max().shift(1)
    
    signal = pd.Series(0, index=df.index)
    # Bullish divergence: price lower low but RSI higher low
    signal[price_lower_low & rsi_higher_low] = 1
    # Bearish divergence: price higher high but RSI lower high
    signal[price_higher_high & rsi_lower_high] = -1
    
    # RSI slope (rate of change)
    rsi_slope = rsi.diff(5) / 5
    
    return signal, rsi_slope.fillna(0)


def generate_strategy_features(df):
    """
    Generates all 7 strategy-based features.
    Returns DataFrame with strategy signal and continuous feature columns.
    """
    df = df.copy()
    
    # 1. EMA Crossover
    df['Strat_EMA_Cross'], df['Strat_EMA_Dist'] = ema_crossover_signal(df)
    
    # 2. Breakout
    df['Strat_Breakout'], df['Strat_Breakout_Strength'] = breakout_signal(df)
    
    # 3. Mean Reversion
    df['Strat_MeanRev'], df['Strat_ZScore'] = mean_reversion_signal(df)
    
    # 4. Support/Resistance
    df['Strat_Dist_Support'], df['Strat_Dist_Resistance'] = support_resistance_levels(df)
    
    # 5. Trend Strength (ADX)
    df['Strat_ADX'], df['Strat_PlusDI'], df['Strat_MinusDI'] = trend_strength_adx(df)
    
    # 6. Volume Breakout
    df['Strat_VolBreakout'], df['Strat_VolSpike'] = volume_breakout_signal(df)
    
    # 7. Momentum Divergence
    df['Strat_MomDiv'], df['Strat_RSI_Slope'] = momentum_divergence(df)
    
    # Composite: Strategy Agreement Score (how many strategies agree on direction)
    signal_cols = ['Strat_EMA_Cross', 'Strat_Breakout', 'Strat_MeanRev', 
                   'Strat_VolBreakout', 'Strat_MomDiv']
    df['Strat_Agreement'] = df[signal_cols].sum(axis=1)
    
    return df


if __name__ == "__main__":
    np.random.seed(42)
    n = 200
    close = np.cumsum(np.random.randn(n) * 0.5) + 500
    data = {
        'Open': close + np.random.randn(n) * 0.5,
        'High': close + np.abs(np.random.randn(n) * 1.5),
        'Low': close - np.abs(np.random.randn(n) * 1.5),
        'Close': close,
        'Volume': np.random.uniform(1000, 5000, n)
    }
    df = pd.DataFrame(data)
    result = generate_strategy_features(df)
    
    strat_cols = [c for c in result.columns if c.startswith('Strat_')]
    print(f"Strategy features generated: {len(strat_cols)} columns")
    for col in strat_cols:
        print(f"  {col}: non-zero = {(result[col] != 0).sum()}")
