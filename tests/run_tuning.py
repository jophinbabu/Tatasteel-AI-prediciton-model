import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from features.target_labels import generate_labels
from training.lgbm_trainer import LGBMTrainer

def run_tuning():
    data_file = os.path.join("data", "TATASTEEL_NS_15m.csv")
    if not os.path.exists(data_file):
        print(f"Data file {data_file} not found. Start by procuring historical data.")
        return

    print("Loading historical data for tuning...")
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    
    print("\n--- Starting Optuna Hyperparameter Tuning for LightGBM ---")
    trainer = LGBMTrainer(ticker="TATASTEEL.NS")
    
    # Run tuning for 50 trials (takes a few minutes)
    best_f1 = trainer.tune(df, n_trials=50, label_method='dynamic', label_threshold=0.001)
    
    print(f"\nFinal Tuned F1 Macro Score: {best_f1:.4f}")

if __name__ == "__main__":
    run_tuning()
