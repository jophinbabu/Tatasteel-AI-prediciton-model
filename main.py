import argparse
import sys
import os
import glob

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def find_best_data(ticker, preferred_interval=None):
    """
    Find the best available data file for training.
    Prefers more data: 1h (2y) > 15m (60d) > daily.
    """
    ticker_clean = ticker.replace('.', '_')
    data_dir = "data"
    best = None
    
    # If preferred interval given, try to find that specific file first
    if preferred_interval:
        path = os.path.join(data_dir, f"{ticker_clean}_{preferred_interval}.csv")
        if os.path.exists(path):
            total_lines = sum(1 for _ in open(path)) - 1
            print(f"Using preferred data: {path} ({total_lines} rows, {preferred_interval} interval)")
            return path, preferred_interval

    # If no preferred interval or file not found, proceed with finding the best available
    # Priority order: more data is better for training
    for interval in ["1h", "15m", "1d"]:
        path = os.path.join(data_dir, f"{ticker_clean}_{interval}.csv")
        if os.path.exists(path):
            total_lines = sum(1 for _ in open(path)) - 1  # minus header
            if best is None or total_lines > best[2]:
                best = (path, interval, total_lines)
    
    if best:
        print(f"Best data found: {best[0]} ({best[2]} rows, {best[1]} interval)")
        return best[0], best[1]
    
    return None, None

def main():
    parser = argparse.ArgumentParser(description="Tata Trading System")
    parser.add_argument("--mode", type=str, choices=["fetch", "train", "train-lgbm", "tune-lgbm", "validate-lgbm", "train-hybrid", "trade", "validate", "api", "dashboard"], 
                        default="trade", help="Mode to run the system in")
    parser.add_argument("--ticker", type=str, default="TATASTEEL.NS", help="Stock ticker")
    parser.add_argument("--interval", type=str, default="1h", help="Data interval (15m, 1h, 1d)")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    
    args = parser.parse_args()
    
    if args.mode == "fetch":
        from data.historical_data import fetch_historical_data
        if not os.path.exists("data"):
            os.makedirs("data")
        
        # If interval is specified, use it
        if args.interval:
            print(f"=== Fetching {args.interval} data ===")
            # 60 days for intraday (yfinance limit), 2 years otherwise
            period = "60d" if args.interval in ["1m", "2m", "5m", "15m", "30m"] else "2y"
            if args.interval == "1m": period = "7d" # 1m limit is 7 days
            
            fetch_historical_data(ticker=args.ticker, interval=args.interval, period=period)
        else:
            # Default behavior if no specific interval requested
            print("=== Fetching 1-hour data (2 years) ===")
            fetch_historical_data(ticker=args.ticker, interval="1h", period="2y")
            print("\n=== Fetching 15-minute data (60 days) ===")
            fetch_historical_data(ticker=args.ticker, interval="15m", period="60d")

    elif args.mode == "train":
        from training.trainer import TradingTrainer
        import pandas as pd
        
        data_path, interval = find_best_data(args.ticker, preferred_interval=args.interval)
        if data_path:
            df = pd.read_csv(data_path, index_col=0, parse_dates=True)
            print(f"\nTraining with {len(df)} rows of {interval} data")
            print(f"Date range: {df.index[0]} to {df.index[-1]}")
            
            # Adaptive sequence length based on data size
            if len(df) < 500:
                seq_length = 10
                print(f"Small dataset detected — using seq_length={seq_length}")
            elif len(df) < 2000:
                seq_length = 15
                print(f"Medium dataset — using seq_length={seq_length}")
            elif len(df) < 5000:
                seq_length = 20
                print(f"Large dataset — key improvement: using seq_length={seq_length}")
            else:
                seq_length = 30
                print(f"Massive dataset — using seq_length={seq_length}")
            
            trainer = TradingTrainer(ticker=args.ticker, seq_length=seq_length)
            trainer.train(df, epochs=args.epochs)
        else:
            print(f"Error: No data found. Run: python main.py --mode fetch --ticker {args.ticker}")
    
    elif args.mode == "train-lgbm":
        from training.lgbm_trainer import LGBMTrainer
        import pandas as pd
        
        data_path, interval = find_best_data(args.ticker, preferred_interval=args.interval)
        if data_path:
            df = pd.read_csv(data_path, index_col=0, parse_dates=True)
            print(f"\nLightGBM training with {len(df)} rows of {interval} data")
            print(f"Date range: {df.index[0]} to {df.index[-1]}")
            
            trainer = LGBMTrainer(ticker=args.ticker)
            trainer.train(df)
        else:
            print(f"Error: No data found. Run: python main.py --mode fetch --ticker {args.ticker}")
    
    elif args.mode == "tune-lgbm":
        from training.lgbm_trainer import LGBMTrainer
        import pandas as pd
        
        data_path, interval = find_best_data(args.ticker, preferred_interval=args.interval)
        if data_path:
            df = pd.read_csv(data_path, index_col=0, parse_dates=True)
            print(f"\nOptuna tuning with {len(df)} rows of {interval} data")
            
            trainer = LGBMTrainer(ticker=args.ticker)
            trainer.tune(df, n_trials=50)
        else:
            print(f"Error: No data found. Run: python main.py --mode fetch --ticker {args.ticker}")
    
    elif args.mode == "validate-lgbm":
        from validation.walk_forward_lgbm import walk_forward_lgbm
        import pandas as pd
        
        data_path, interval = find_best_data(args.ticker, preferred_interval=args.interval)
        if data_path:
            df = pd.read_csv(data_path, index_col=0, parse_dates=True)
            print(f"\nWalk-forward validation with {len(df)} rows of {interval} data")
            
            walk_forward_lgbm(df, n_splits=5, ticker=args.ticker)
        else:
            print(f"Error: No data found. Run: python main.py --mode fetch --ticker {args.ticker}")
    
    elif args.mode == "validate":
        from validation.walk_forward import walk_forward_validation
        import pandas as pd
        
        data_path, interval = find_best_data(args.ticker, preferred_interval=args.interval)
        if data_path:
            df = pd.read_csv(data_path, index_col=0, parse_dates=True)
            
            # Adaptive splits based on data size
            if len(df) < 500:
                n_splits = 3
            elif len(df) < 2000:
                n_splits = 5
            else:
                n_splits = 7
            
            print(f"Running walk-forward with {n_splits} splits on {len(df)} rows...")
            walk_forward_validation(df, n_splits=n_splits, epochs_per_fold=20, ticker=args.ticker)
        else:
            print(f"Error: No data found. Run: python main.py --mode fetch --ticker {args.ticker}")
    
    elif args.mode == "train-hybrid":
        from training.hybrid_trainer import HybridTrainer
        import pandas as pd
        
        data_path, interval = find_best_data(args.ticker, preferred_interval=args.interval)
        if data_path:
            df = pd.read_csv(data_path, index_col=0, parse_dates=True)
            print(f"\nHybrid training with {len(df)} rows of {interval} data")
            
            hybrid = HybridTrainer(ticker=args.ticker)
            hybrid.train(df, epochs_lstm=args.epochs, epochs_cnn=args.epochs)
        else:
            print(f"Error: No data found. Run: python main.py --mode fetch --ticker {args.ticker}")
            
    elif args.mode == "trade":
        from execution.trading_system import TradingSystem
        system = TradingSystem(ticker=args.ticker, interval=args.interval)
        system.start()
        
    elif args.mode == "api":
        import uvicorn
        uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
        
    elif args.mode == "dashboard":
        import subprocess
        print("Launching Live Monitoring Dashboard...")
        dashboard_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard", "app.py")
        subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_path])

if __name__ == "__main__":
    main()
