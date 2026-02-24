import yfinance as yf
import pandas as pd
import os

def fetch_live_data(ticker="TATASTEEL.NS", interval="15m"):
    """
    Fetches the most recent data for real-time prediction.
    Uses 10d period to ensure enough bars for feature warm-up (indicators need ~30+ bars).
    """
    print(f"Fetching live data for {ticker} (Interval: {interval})...")
    
    # 10d gives ~160 bars of 15m data — enough for feature warm-up + sequence window
    data = yf.download(tickers=ticker, period="10d", interval=interval)
    
    if data.empty:
        print("No live data found.")
        return None
    
    # Flatten multi-index if necessary
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    # Keep last 150 rows (enough for 30-bar feature warm-up + 30-bar sequence + buffer)
    live_data = data.tail(150)
    print(f"Fetched {len(live_data)} recent bars.")
    
    return live_data

if __name__ == "__main__":
    live_df = fetch_live_data()
    if live_df is not None:
        print(live_df.tail())
