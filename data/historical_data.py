import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Mapping of tickers that may have changed on Yahoo Finance
TICKER_ALIASES = {
    "TATAMOTORS.NS": ["TATAMOTORS.NS", "TATAMTRDVR.NS", "TATAMOTORS.BO"],
    "TATASTEEL.NS": ["TATASTEEL.NS", "TATASTEEL.BO"],
}

def _fetch_chunk(ticker, start, end, interval="15m"):
    """Fetch a single chunk of data for a date range."""
    try:
        data = yf.download(tickers=ticker, start=start, end=end, interval=interval)
        if data is not None and not data.empty:
            # Flatten multi-index if necessary
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            return data
    except Exception as e:
        print(f"    Chunk fetch error ({start} to {end}): {e}")
    return None

def fetch_historical_data_chunked(ticker="TATASTEEL.NS", interval="15m", total_days=730):
    """
    Fetches up to ~2 years of 15m data by downloading in 59-day chunks
    (yfinance limits 15m data to 60 days per request).
    
    Args:
        ticker: Stock ticker symbol
        interval: Bar interval (15m, 1h, etc.)
        total_days: Total days of history to attempt fetching
    """
    print(f"Fetching {total_days} days of {interval} data for {ticker} in chunks...")
    
    aliases = TICKER_ALIASES.get(ticker, [ticker])
    all_chunks = []
    working_ticker = None
    
    # Determine chunk size (yfinance limit is 60 days for intraday)
    chunk_days = 59 if interval in ("1m", "2m", "5m", "15m", "30m") else 365
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=total_days)
    
    # Generate date ranges
    current_start = start_date
    chunk_ranges = []
    while current_start < end_date:
        current_end = min(current_start + timedelta(days=chunk_days), end_date)
        chunk_ranges.append((current_start, current_end))
        current_start = current_end
    
    print(f"  Will attempt {len(chunk_ranges)} chunks...")
    
    for alias in aliases:
        print(f"  Trying ticker: {alias}...")
        chunks = []
        success_count = 0
        
        for i, (start, end) in enumerate(chunk_ranges):
            start_str = start.strftime('%Y-%m-%d')
            end_str = end.strftime('%Y-%m-%d')
            print(f"    Chunk {i+1}/{len(chunk_ranges)}: {start_str} to {end_str}", end="")
            
            chunk = _fetch_chunk(alias, start_str, end_str, interval)
            if chunk is not None and len(chunk) > 0:
                chunks.append(chunk)
                success_count += 1
                print(f" -> {len(chunk)} bars")
            else:
                print(f" -> no data")
        
        if chunks:
            all_chunks = chunks
            working_ticker = alias
            print(f"  Success with {alias}! {success_count}/{len(chunk_ranges)} chunks, total bars: {sum(len(c) for c in chunks)}")
            break
        else:
            print(f"  No data from {alias}.")
    
    if not all_chunks:
        print("ERROR: Could not fetch data from any source.")
        return None
    
    # Concatenate, deduplicate, sort
    data = pd.concat(all_chunks)
    data = data[~data.index.duplicated(keep='last')]
    data = data.sort_index()
    
    # Quality report
    completeness = (1 - data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
    date_range = f"{data.index[0]} to {data.index[-1]}"
    print(f"\nData Summary:")
    print(f"  Ticker: {working_ticker}")
    print(f"  Bars: {len(data)}")
    print(f"  Date Range: {date_range}")
    print(f"  Completeness: {completeness:.1f}%")
    
    if completeness < 95:
        print("  WARNING: Data completeness below 95%!")
    
    # Save to CSV
    output_path = os.path.join("data", f"{working_ticker.replace('.', '_')}_{interval}.csv")
    data.to_csv(output_path)
    print(f"  Saved to {output_path}")
    
    return data


def fetch_historical_data(ticker="TATASTEEL.NS", interval="15m", period="2y"):
    """
    Legacy API — fetches historical data with fallback logic.
    For 15m data, now delegates to chunked fetcher for maximum data.
    """
    print(f"Fetching historical data for {ticker} (Interval: {interval}, Period: {period})...")
    
    # For intraday intervals, use chunked approach unless explicit short period
    if interval in ("1m", "2m", "5m", "15m", "30m"):
        # If period is already <= 60d, just let yfinance handle it directly
        if period in ["60d", "30d", "1mo", "5d", "1d"]:
            pass # Fall through to direct download
        else:
            period_days = {"2y": 730, "1y": 365, "6mo": 180, "3mo": 90}
            days = period_days.get(period, 730)
            return fetch_historical_data_chunked(ticker, interval, total_days=days)
    
    # For daily/weekly/short-intraday, use direct download
    aliases = TICKER_ALIASES.get(ticker, [ticker])
    data = None
    
    for alias in aliases:
        try:
            print(f"  Trying ticker: {alias}...")
            # Use period directly for robust relative date handling
            data = yf.download(tickers=alias, period=period, interval=interval)
            if data is not None and not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                print(f"  Success with {alias}! Fetched {len(data)} bars.")
                ticker = alias
                break
            else:
                data = None
        except Exception as e:
            print(f"  Failed for {alias}: {e}")
            data = None
    
    if data is None or data.empty:
        print("ERROR: Could not fetch data from any source.")
        return None
    
    # Quality checks
    completeness = (1 - data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
    print(f"Data Quality: {completeness:.1f}% completeness ({len(data)} bars)")
    if completeness < 95:
        print("WARNING: Data completeness below 95% threshold!")
    
    # Save to CSV
    output_path = os.path.join("data", f"{ticker.replace('.', '_')}_{interval}.csv")
    data.to_csv(output_path)
    print(f"Data saved to {output_path}")
    
    return data

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
    
    # Fetch 1h data (2 years — best for training)
    print("=== Fetching 1-hour data (2 years) ===")
    fetch_historical_data(ticker="TATASTEEL.NS", interval="1h", period="2y")
    
    # Also fetch 15m data (60 days — for live trading)
    print("\n=== Fetching 15-minute data (60 days) ===")
    fetch_historical_data(ticker="TATASTEEL.NS", interval="15m", period="60d")
