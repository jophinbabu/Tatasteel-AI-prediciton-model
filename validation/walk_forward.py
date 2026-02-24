"""
Walk-Forward Cross-Validation for Time Series

Improvements:
- Purge gap between train/test sets (prevents label leakage)
- Fresh model initialization per fold
- Tracks F1, precision, recall per fold
- Proper epoch count per fold
"""
import numpy as np
import pandas as pd
import os
import sys
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.trainer import TradingTrainer


def walk_forward_validation(df, n_splits=10, epochs_per_fold=30, purge_bars=4, 
                            ticker="TATASTEEL.NS", verbose=True):
    """
    Walk-forward cross-validation with purge gap.
    
    Args:
        df: Raw OHLCV DataFrame
        n_splits: Number of forward validation splits
        epochs_per_fold: Training epochs per fold
        purge_bars: Number of bars to skip between train/test (prevents label leakage)
        ticker: Stock ticker
        verbose: Print progress
    """
    # Generate features once (shared across all splits)
    trainer = TradingTrainer(ticker=ticker)
    featured_df = trainer.prepare_features(df)
    feature_cols = trainer.feature_columns if trainer.feature_columns else \
                   [c for c in featured_df.columns if c not in ['Label', 'Forward_Return', 'Future_Price',
                    'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
    
    total_len = len(featured_df)
    
    # Adaptive sequence length based on data size
    if total_len < 500:
        seq_length = 10
    elif total_len < 2000:
        seq_length = 20
    else:
        seq_length = 30
    trainer.seq_length = seq_length
    
    # Auto-reduce splits if data is too small
    min_samples_per_fold = seq_length * 3  # Need at least 3x seq_length per fold
    max_possible_splits = max(1, (total_len // min_samples_per_fold) - 1)
    n_splits = min(n_splits, max_possible_splits)
    
    if verbose:
        print(f"\nData: {total_len} samples, seq_length={seq_length}, n_splits={n_splits}")
    
    split_size = total_len // (n_splits + 1)
    
    results = []
    label_names = ['HOLD', 'BUY', 'SELL']
    
    for i in range(n_splits):
        train_end = split_size * (i + 1)
        # Purge gap: skip 'purge_bars' between train and test
        test_start = train_end + purge_bars
        test_end = min(split_size * (i + 2), total_len)
        
        if test_start >= total_len or test_start >= test_end:
            print(f"Split {i+1}: Not enough data after purge. Skipping.")
            continue
        
        train_df = featured_df.iloc[:train_end]
        test_df = featured_df.iloc[test_start:test_end]
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"Split {i+1}/{n_splits}")
            print(f"  Train: {len(train_df)} samples (rows 0-{train_end})")
            print(f"  Purge: {purge_bars} bars skipped")
            print(f"  Test:  {len(test_df)} samples (rows {test_start}-{test_end})")
        
        # Fresh trainer per fold to avoid weight contamination
        fold_trainer = TradingTrainer(ticker=ticker)
        fold_trainer.seq_length = seq_length
        
        # Scale: fit on train, transform test
        X_train, y_train = fold_trainer.prepare_sequences(train_df, feature_cols, fit_scaler=True)
        X_test, y_test = fold_trainer.prepare_sequences(test_df, feature_cols, fit_scaler=False)
        
        if len(X_train) == 0 or len(X_test) == 0:
            print(f"  Skipping: insufficient data for sequences")
            continue
        
        # Build and train fresh model (auto-sized)
        from models.lstm_cnn_attention import build_model
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        
        fold_trainer.model = build_model(
            input_shape=(seq_length, len(feature_cols)),
            use_focal_loss=True,
            model_size='auto',
            n_samples=len(X_train)
        )
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=0),
        ]
        
        fold_trainer.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs_per_fold,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        
        # Evaluate
        y_pred_probs = fold_trainer.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        loss = fold_trainer.model.evaluate(X_test, y_test, verbose=0)
        if isinstance(loss, list):
            test_loss, test_acc = loss[0], loss[1]
        else:
            test_loss = loss
            test_acc = accuracy_score(y_test, y_pred)
        
        # Detailed metrics
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        
        # Per-class metrics
        buy_mask = y_test == 1
        sell_mask = y_test == 2
        buy_recall = recall_score(y_test == 1, y_pred == 1, zero_division=0) if buy_mask.sum() > 0 else 0
        sell_recall = recall_score(y_test == 2, y_pred == 2, zero_division=0) if sell_mask.sum() > 0 else 0
        
        fold_result = {
            'split': i + 1,
            'test_loss': round(test_loss, 4),
            'accuracy': round(test_acc, 4),
            'f1_macro': round(f1, 4),
            'precision_macro': round(precision, 4),
            'recall_macro': round(recall, 4),
            'buy_recall': round(buy_recall, 4),
            'sell_recall': round(sell_recall, 4),
        }
        results.append(fold_result)
        
        if verbose:
            print(f"  Results: Acc={test_acc:.4f}, F1={f1:.4f}, "
                  f"Prec={precision:.4f}, Rec={recall:.4f}")
            print(f"  BUY Recall={buy_recall:.4f}, SELL Recall={sell_recall:.4f}")
    
    results_df = pd.DataFrame(results)
    
    if verbose and len(results_df) > 0:
        print(f"\n{'='*50}")
        print("WALK-FORWARD SUMMARY")
        print(f"{'='*50}")
        print(results_df.to_string(index=False))
        print(f"\nMean F1 (macro): {results_df['f1_macro'].mean():.4f}")
        print(f"Mean Accuracy:   {results_df['accuracy'].mean():.4f}")
        print(f"Mean BUY Recall: {results_df['buy_recall'].mean():.4f}")
        print(f"Mean SELL Recall:{results_df['sell_recall'].mean():.4f}")
    
    return results_df


if __name__ == "__main__":
    data_file = os.path.join("data", "TATASTEEL_NS_15m.csv")
    if os.path.exists(data_file):
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        results = walk_forward_validation(df, n_splits=10, epochs_per_fold=30)
    else:
        print("Data not found. Run historical_data.py first.")
