"""
Comprehensive Intraday Feature Engineering Module

Generates 40+ features combining:
- Technical indicators (RSI, MACD, Bollinger, ATR, etc.)
- Time features (hour, minute, day of week, session)
- Candlestick patterns (8 patterns)
- Trading strategy signals (7 strategies + continuous features)
"""
import pandas as pd
import numpy as np

from features.chart_patterns import detect_patterns
from features.trading_strategies import generate_strategy_features


# ─── Core Technical Indicators ───────────────────────────────────────────────

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calculate_stochastic_rsi(data, rsi_window=14, stoch_window=14):
    """Stochastic RSI — RSI of RSI, bounded 0-100."""
    rsi = calculate_rsi(data, rsi_window)
    stoch_rsi = (rsi - rsi.rolling(stoch_window).min()) / \
                (rsi.rolling(stoch_window).max() - rsi.rolling(stoch_window).min()).replace(0, np.nan)
    return (stoch_rsi * 100).fillna(50)

def calculate_macd(data, slow=26, fast=12, smooth=9):
    exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=smooth, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram

def calculate_bollinger_bands(data, window=20, num_std=2):
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

def calculate_atr(data, window=14):
    high_low = data['High'] - data['Low']
    high_cp = np.abs(data['High'] - data['Close'].shift())
    low_cp = np.abs(data['Low'] - data['Close'].shift())
    df = pd.concat([high_low, high_cp, low_cp], axis=1)
    true_range = np.max(df, axis=1)
    return true_range.rolling(window=window).mean()

def calculate_vwap_daily(data):
    """
    VWAP that resets daily (correct implementation).
    Groups by date and computes intraday VWAP for each trading day.
    """
    df = data.copy()
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    
    # Try to group by date
    if hasattr(df.index, 'date'):
        groups = df.index.date
    else:
        # Fallback: treat entire series as one day
        groups = [0] * len(df)
    
    vwap = pd.Series(np.nan, index=df.index)
    for date, group_idx in df.groupby(groups).groups.items():
        mask = df.index.isin(group_idx)
        day_tp = tp[mask]
        day_vol = df['Volume'][mask]
        cum_tp_vol = (day_tp * day_vol).cumsum()
        cum_vol = day_vol.cumsum()
        vwap[mask] = cum_tp_vol / cum_vol.replace(0, np.nan)
    
    return vwap

def calculate_williams_r(data, window=14):
    """Williams %R: Momentum indicator (-100 to 0)."""
    highest_high = data['High'].rolling(window=window).max()
    lowest_low = data['Low'].rolling(window=window).min()
    wr = -100 * (highest_high - data['Close']) / (highest_high - lowest_low).replace(0, np.nan)
    return wr

def calculate_cci(data, window=20):
    """Commodity Channel Index."""
    tp = (data['High'] + data['Low'] + data['Close']) / 3
    sma_tp = tp.rolling(window=window).mean()
    mad = tp.rolling(window=window).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))
    return cci

def calculate_cmf(data, window=20):
    """Chaikin Money Flow (proxy for order flow/buying pressure)"""
    mf_multiplier = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low']).replace(0, np.nan)
    mf_volume = mf_multiplier * data['Volume']
    cmf = mf_volume.rolling(window=window).sum() / data['Volume'].rolling(window=window).sum().replace(0, np.nan)
    return cmf

def calculate_vpt(data):
    """Volume Price Trend (captures cumulative volume multiplied by pct return)"""
    vpt = (data['Volume'] * data['Close'].pct_change()).fillna(0).cumsum()
    return vpt


# ─── Time Features ───────────────────────────────────────────────────────────

def generate_time_features(df):
    """
    Extracts time-based features from datetime index.
    These capture intraday patterns (morning rally, afternoon selling, etc.)
    """
    df = df.copy()
    
    if hasattr(df.index, 'hour'):
        df['Time_Hour'] = df.index.hour
        df['Time_Minute'] = df.index.minute
        df['Time_DayOfWeek'] = df.index.dayofweek  # 0=Monday, 4=Friday
        
        # Session encoding (IST market hours 9:15 - 15:30)
        # Morning: 9:15-11:30, Midday: 11:30-13:30, Afternoon: 13:30-15:30
        hour_min = df['Time_Hour'] * 60 + df['Time_Minute']
        df['Time_Session'] = 0  # pre-market default
        df.loc[(hour_min >= 555) & (hour_min < 690), 'Time_Session'] = 1   # Morning
        df.loc[(hour_min >= 690) & (hour_min < 810), 'Time_Session'] = 2   # Midday
        df.loc[(hour_min >= 810) & (hour_min <= 930), 'Time_Session'] = 3  # Afternoon
        
        # Cyclical encoding for hour (preserves circular nature)
        df['Time_Hour_Sin'] = np.sin(2 * np.pi * df['Time_Hour'] / 24)
        df['Time_Hour_Cos'] = np.cos(2 * np.pi * df['Time_Hour'] / 24)
        df['Time_DOW_Sin'] = np.sin(2 * np.pi * df['Time_DayOfWeek'] / 5)
        df['Time_DOW_Cos'] = np.cos(2 * np.pi * df['Time_DayOfWeek'] / 5)
    else:
        # No datetime index — fill with defaults
        df['Time_Hour'] = 0
        df['Time_Minute'] = 0
        df['Time_DayOfWeek'] = 0
        df['Time_Session'] = 0
        df['Time_Hour_Sin'] = 0.0
        df['Time_Hour_Cos'] = 1.0
        df['Time_DOW_Sin'] = 0.0
        df['Time_DOW_Cos'] = 1.0
    
    return df


# ─── Master Feature Generator ────────────────────────────────────────────────

# Columns to EXCLUDE from model features (raw prices leak absolute levels)
RAW_OHLCV_COLS = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']

# Columns generated during labeling that should not be features
LABEL_COLS = ['Label', 'Forward_Return', 'Future_Price', 'Future_Return']

# Intermediate columns used during calculation
INTERMEDIATE_COLS = ['SMA20', 'SMA50', 'Vol_SMA20', 'OBV', 'VPT', 'BB_Upper', 'BB_Lower', 'VWAP',
                     'MACD_Signal']

# ── Curated feature set (with lag & rolling features) ──
# Core 18 indicators + 12 temporal features = 30 total
CURATED_FEATURES = [
    # Core momentum & trend (8)
    'RSI_9', 'MACD_Hist', 'ROC_10', 'SMA_Slope', 'WILLR_14', 'Volatility_20', 'VWAP_Dist', 'SMA50_Dist',
    # Volatility (3)
    'BB_Width', 'BB_Position', 'ATR_Norm',
    # Volume & Order Flow (4)
    'Vol_Ratio', 'OBV_Norm', 'CMF_20', 'VPT_Norm',
    # Strategy continuous features (4)
    'Strat_EMA_Dist', 'Strat_ZScore', 'Strat_ADX', 'Strat_Agreement',
    # Time (2)
    'Time_Hour_Sin', 'Time_Hour_Cos',
    # ── Lag features (8) — gives temporal context to flat models ──
    'RSI_9_lag3', 'RSI_9_lag6',
    'MACD_Hist_lag3', 'MACD_Hist_lag6',
    'BB_Position_lag3', 'BB_Position_lag6',
    'ATR_Norm_lag3', 'ATR_Norm_lag6',
    # ── Rolling returns (4) — recent price momentum ──
    'Return_3', 'Return_6', 'Return_12', 'Return_Vol_12',
]


def get_feature_columns(df):
    """Returns only the clean feature column names (excludes raw OHLCV, labels, intermediates)."""
    exclude = set(RAW_OHLCV_COLS + LABEL_COLS + INTERMEDIATE_COLS)
    return [c for c in df.columns if c not in exclude]


def get_curated_feature_columns(df):
    """Returns the curated 18-feature subset for better performance with small datasets.
    Falls back to available columns if some are missing."""
    available = [c for c in CURATED_FEATURES if c in df.columns]
    return available


def generate_features(df):
    """
    Generates 40+ features for the trading system.
    
    Feature groups:
    - Core technical indicators (12)
    - Time features (8)
    - Candlestick patterns (8)
    - Trading strategy signals (16)
    
    Returns: DataFrame with all features added (NaN rows dropped).
    """
    df = df.copy()
    
    # ── Core Technical Indicators ──
    
    # 1. RSI (Window 9)
    df['RSI_9'] = calculate_rsi(df, window=9)
    
    # 2. Stochastic RSI
    df['StochRSI_14'] = calculate_stochastic_rsi(df)
    
    # 3-5. MACD, Signal, and Histogram
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df)
    
    # 6-7. Bollinger Bands (width and position)
    df['BB_Upper'], df['BB_Lower'] = calculate_bollinger_bands(df)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['Close']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower']).replace(0, np.nan)
    
    # 8-9. ATR and True Range
    df['TRANGE'] = df['High'] - df['Low']
    df['ATR_5'] = calculate_atr(df, window=5)
    df['ATR_Norm'] = df['ATR_5'] / df['Close']  # Normalized ATR
    
    # 10. VWAP (daily reset)
    df['VWAP'] = calculate_vwap_daily(df)
    df['VWAP_Dist'] = (df['Close'] - df['VWAP']) / df['VWAP'].replace(0, np.nan)
    
    # 11. ROC (Rate of Change)
    df['ROC_10'] = df['Close'].pct_change(periods=10)
    
    # 12. SMA Slope & SMA50 (Macro trend context that fits in 64-bar live fetch)
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA_Slope'] = df['SMA20'].diff() / df['SMA20'].replace(0, np.nan)
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA50_Dist'] = (df['Close'] - df['SMA50']) / df['SMA50'].replace(0, np.nan)
    
    # 13. Volume Ratio
    df['Vol_SMA20'] = df['Volume'].rolling(window=20).mean()
    df['Vol_Ratio'] = df['Volume'] / df['Vol_SMA20'].replace(0, np.nan)
    
    # 14. OBV Normalized
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    obv_range = df['OBV'].rolling(window=20).max() - df['OBV'].rolling(window=20).min()
    df['OBV_Norm'] = (df['OBV'] - df['OBV'].rolling(window=20).min()) / obv_range.replace(0, np.nan)
    
    # Advanced Order Flow & Volume Pricing
    df['CMF_20'] = calculate_cmf(df, window=20)
    df['VPT'] = calculate_vpt(df)
    vpt_range = df['VPT'].rolling(window=20).max() - df['VPT'].rolling(window=20).min()
    df['VPT_Norm'] = (df['VPT'] - df['VPT'].rolling(window=20).min()) / vpt_range.replace(0, np.nan)
    
    # 15. Williams %R
    df['WILLR_14'] = calculate_williams_r(df, window=14)
    
    # 16. CCI
    df['CCI_20'] = calculate_cci(df, window=20)
    
    # 17. Price volatility (rolling std of returns)
    df['Volatility_20'] = df['Close'].pct_change().rolling(20).std()
    
    # ── Time Features ──
    df = generate_time_features(df)
    
    # ── Candlestick Patterns ──
    df = detect_patterns(df)
    
    # ── Trading Strategy Features ──
    df = generate_strategy_features(df)
    
    # ── Lag Features ── (temporal context for flat models like LightGBM)
    for col in ['RSI_9', 'MACD_Hist', 'BB_Position', 'ATR_Norm']:
        if col in df.columns:
            df[f'{col}_lag3'] = df[col].shift(3)
            df[f'{col}_lag6'] = df[col].shift(6)
    
    # ── Rolling Returns ── (recent price momentum)
    df['Return_3'] = df['Close'].pct_change(3)
    df['Return_6'] = df['Close'].pct_change(6)
    df['Return_12'] = df['Close'].pct_change(12)
    df['Return_Vol_12'] = df['Close'].pct_change().rolling(12).std()
    
    # ── Drop NaN rows from indicator warm-up ──
    df.dropna(inplace=True)
    
    # Report
    feature_cols = get_feature_columns(df)
    print(f"Features generated: {len(feature_cols)} features, {len(df)} samples")
    
    return df


if __name__ == "__main__":
    import os
    
    # Load historical data if it exists
    data_file = os.path.join("data", "TATASTEEL_NS_15m.csv")
    if os.path.exists(data_file):
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        featured_df = generate_features(df)
        feature_cols = get_feature_columns(featured_df)
        print(f"\nFeature columns ({len(feature_cols)}):")
        for i, col in enumerate(feature_cols, 1):
            print(f"  {i:2d}. {col}")
    else:
        # Demo with random data
        print("Data file not found, running demo...")
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
        demo_df = pd.DataFrame(data, index=pd.date_range('2025-01-01', periods=n, freq='15min'))
        featured_df = generate_features(demo_df)
        feature_cols = get_feature_columns(featured_df)
        print(f"\nFeature columns ({len(feature_cols)}):")
        for i, col in enumerate(feature_cols, 1):
            print(f"  {i:2d}. {col}")
