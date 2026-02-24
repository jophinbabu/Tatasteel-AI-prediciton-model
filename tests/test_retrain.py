import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from features.target_labels import generate_labels
from training.lgbm_trainer import LGBMTrainer

def test_retrain():
    data_file = os.path.join("data", "TATASTEEL_NS_15m.csv")
    if not os.path.exists(data_file):
        print(f"Data file {data_file} not found. Ensure historical data is downloaded.")
        return

    print("Loading data...")
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    
    print("\n--- Training LightGBM Model ---")
    trainer = LGBMTrainer(ticker="TATASTEEL.NS")
    
    # Train the model with the new dynamic threshold
    f1 = trainer.train(df, label_method='dynamic', label_threshold=0.001)
    
    print(f"\nTraining Complete. F1 Macro Score: {f1:.4f}")

if __name__ == "__main__":
    test_retrain()
