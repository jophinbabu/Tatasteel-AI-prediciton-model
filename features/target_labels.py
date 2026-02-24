import pandas as pd
import numpy as np

def generate_labels(df, threshold=0.005, method='dynamic', atr_multiplier=0.8):
    """
    Generates BUY (1), HOLD (0), and SELL (2) labels.
    
    Methods:
        'fixed'   - Fixed threshold (default 0.5%)
        'dynamic' - ATR-based dynamic threshold (adapts to volatility)
    
    Args:
        df: DataFrame with OHLCV data (must have 'Close' column, and 'ATR_5' for dynamic)
        threshold: Price change threshold for fixed method
        method: 'fixed' or 'dynamic'
        atr_multiplier: Multiplier for ATR in dynamic mode (default 1.0x ATR)
    """
    df = df.copy()
    
    # Forward return: current Close to Close 8 bars ahead
    df['Future_Price'] = df['Close'].shift(-8)
    df['Forward_Return'] = (df['Future_Price'] - df['Close']) / df['Close']
    
    # Determine threshold per row
    if method == 'dynamic' and 'ATR_5' in df.columns:
        # Dynamic threshold based on ATR relative to price
        df['_threshold'] = (df['ATR_5'] / df['Close']) * atr_multiplier
        # Floor at minimum 0.1% to ensure profitability
        df['_threshold'] = df['_threshold'].clip(lower=0.001)
    else:
        df['_threshold'] = threshold if threshold else 0.001
    
    # Labels: 0=HOLD, 1=BUY, 2=SELL
    df['Label'] = 0
    df.loc[df['Forward_Return'] > df['_threshold'], 'Label'] = 1
    df.loc[df['Forward_Return'] < -df['_threshold'], 'Label'] = 2
    
    # Clean up helper columns
    df.drop(columns=['_threshold', 'Future_Price'], inplace=True, errors='ignore')
    
    # Drop rows with NaN (end of dataset where forward return can't be computed)
    df.dropna(subset=['Label', 'Forward_Return'], inplace=True)
    
    # Report label distribution
    counts = df['Label'].value_counts().sort_index()
    total = len(df)
    print(f"Label Distribution: HOLD={counts.get(0,0)} ({counts.get(0,0)/total*100:.1f}%), "
          f"BUY={counts.get(1,0)} ({counts.get(1,0)/total*100:.1f}%), "
          f"SELL={counts.get(2,0)} ({counts.get(2,0)/total*100:.1f}%)")
    
    return df

if __name__ == "__main__":
    # Quick test
    np.random.seed(42)
    data = {
        'Close': np.cumsum(np.random.randn(100) * 0.5) + 500,
        'High': np.cumsum(np.random.randn(100) * 0.5) + 502,
        'Low': np.cumsum(np.random.randn(100) * 0.5) + 498,
        'ATR_5': np.abs(np.random.randn(100) * 2) + 3
    }
    df = pd.DataFrame(data)
    
    print("=== Fixed Threshold ===")
    labeled_fixed = generate_labels(df.copy(), threshold=0.005)
    
    print("\n=== Dynamic (ATR-based) Threshold ===")
    labeled_dynamic = generate_labels(df.copy(), method='dynamic')
