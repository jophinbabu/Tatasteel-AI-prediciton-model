"""
Hybrid Trainer

Trains both the LSTM (numerical features) and CNN (chart images) models,
then calibrates ensemble weights on the validation set.

Pipeline:
1. Generate chart images from OHLCV data
2. Train CNN on chart images
3. Train LSTM on numerical features (existing trainer)
4. Align predictions on shared validation set
5. Optimize ensemble weight via grid search on F1
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
from collections import Counter
from sklearn.metrics import f1_score, classification_report, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.chart_image_generator import generate_chart_dataset, render_candlestick_image
from models.cnn_chart_model import build_cnn_model, load_chart_dataset
from training.trainer import TradingTrainer


class HybridTrainer:
    def __init__(self, ticker="TATASTEEL.NS", seq_length=20, chart_window=40):
        self.ticker = ticker
        self.seq_length = seq_length
        self.chart_window = chart_window
        
        self.lstm_trainer = TradingTrainer(ticker=ticker, seq_length=seq_length)
        self.cnn_model = None
        self.ensemble_weight = 0.5  # w * lstm + (1-w) * cnn
        
        # Paths
        self.model_dir = 'models/saved'
        self.cnn_path = os.path.join(self.model_dir, 'chart_cnn.keras')
        self.weight_path = os.path.join(self.model_dir, f'{ticker}_ensemble_weight.joblib')
        self.chart_dir = 'data/chart_images'

    def train(self, df, epochs_lstm=100, epochs_cnn=50, batch_size=64):
        """
        Full hybrid training pipeline.
        """
        print("=" * 60)
        print("HYBRID TRAINING: LSTM + CNN Chart Vision")
        print("=" * 60)
        
        # ── Step 1: Generate chart images ──
        # ── Step 1: Generate chart images ──
        print("\n[1/5] Checking chart images...")
        if os.path.exists(self.chart_dir) and len(os.listdir(self.chart_dir)) > 0:
            # Check if subdirs exist and are populated
            has_images = False
            for subdir in ['HOLD', 'BUY', 'SELL']:
                path = os.path.join(self.chart_dir, subdir)
                if os.path.exists(path) and len(os.listdir(path)) > 0:
                    has_images = True
                    break
            
            if has_images:
                print(f"  Chart images found in {self.chart_dir}. Skipping generation.")
            else:
                print("  Generating chart images...")
                generate_chart_dataset(
                    df, 
                    output_dir=self.chart_dir,
                    window_size=self.chart_window,
                    label_threshold=0.005,
                    forward_bars=4
                )
        else:
            print("  Generating chart images...")
            generate_chart_dataset(
                df, 
                output_dir=self.chart_dir,
                window_size=self.chart_window,
                label_threshold=0.005,
                forward_bars=4
            )
        
        # ── Step 2: Train CNN on chart images ──
        print("\n[2/5] Training CNN chart classifier...")
        self.cnn_model = self._train_cnn(epochs_cnn, batch_size)
        
        # ── Step 3: Train LSTM on numerical features ──
        print("\n[3/5] Training LSTM on numerical features...")
        self.lstm_trainer.train(
            df, epochs=epochs_lstm, batch_size=batch_size,
            use_focal_loss=True, focal_gamma=1.0,
            use_curated_features=True
        )
        
        # ── Step 4: Align predictions on validation data ──
        print("\n[4/5] Aligning predictions for ensemble calibration...")
        lstm_probs, cnn_probs, y_true = self._get_aligned_predictions(df)
        
        if lstm_probs is None or cnn_probs is None:
            print("Could not align predictions. Using equal weights.")
            self.ensemble_weight = 0.5
        else:
            # ── Step 5: Optimize ensemble weight ──
            print("\n[5/5] Optimizing ensemble weights...")
            self.ensemble_weight = self._optimize_weight(lstm_probs, cnn_probs, y_true)
        
        # Save ensemble weight
        os.makedirs(self.model_dir, exist_ok=True)
        joblib.dump(self.ensemble_weight, self.weight_path)
        
        # Final evaluation
        if lstm_probs is not None and cnn_probs is not None:
            self._print_final_report(lstm_probs, cnn_probs, y_true)

    def _train_cnn(self, epochs, batch_size):
        """Train CNN and return the model."""
        from models.cnn_chart_model import train_cnn
        model, _ = train_cnn(
            image_dir=self.chart_dir,
            epochs=epochs,
            batch_size=batch_size,
            model_save_path=self.cnn_path
        )
        return model

    def _get_aligned_predictions(self, df):
        """
        Generate aligned predictions from both models on the validation portion.
        
        The challenge: LSTM and CNN use different input formats, but we need
        predictions for the same time points.
        """
        from features.intraday_features import generate_features, get_feature_columns
        from features.target_labels import generate_labels
        
        # Regenerate features and labels
        featured_df = generate_features(df)
        featured_df = generate_labels(featured_df, threshold=0.005, method='fixed')
        feature_cols = get_feature_columns(featured_df)
        
        # Use last 15% as validation
        split_idx = int(len(featured_df) * 0.85)
        val_df = featured_df.iloc[split_idx:]
        
        if len(val_df) < self.seq_length + self.chart_window:
            return None, None, None
        
        # ── LSTM predictions on validation set ──
        X_val_lstm, y_val = self.lstm_trainer.prepare_sequences(
            val_df, feature_cols, fit_scaler=False
        )
        
        if len(X_val_lstm) == 0:
            return None, None, None
        
        lstm_probs = self.lstm_trainer.model.predict(X_val_lstm, verbose=0)
        
        # ── CNN predictions on validation set ──
        # We need to render chart images for each validation point
        import tensorflow as tf
        
        cnn_probs_list = []
        valid_indices = []
        
        # Each LSTM prediction at index j corresponds to data ending at
        # val_df.iloc[j + seq_length - 1]. For the CNN, we need a window
        # of chart_window bars ending at the same point.
        for j in range(len(X_val_lstm)):
            # The bar index in val_df that this prediction is for
            bar_idx = j + self.seq_length - 1
            
            # We need chart_window bars ending at this bar
            # Get the position in the original featured_df
            abs_idx = split_idx + bar_idx
            start_idx = abs_idx - self.chart_window + 1
            
            if start_idx < 0:
                # Not enough data for this chart window
                cnn_probs_list.append(np.array([1/3, 1/3, 1/3]))
                valid_indices.append(j)
                continue
            
            chart_window_df = df.iloc[start_idx:abs_idx + 1]
            
            if len(chart_window_df) < self.chart_window:
                cnn_probs_list.append(np.array([1/3, 1/3, 1/3]))
                valid_indices.append(j)
                continue
            
            # Render to temp image and predict
            temp_path = os.path.join(self.chart_dir, '_temp_predict.png')
            render_candlestick_image(chart_window_df, temp_path)
            
            # Load and predict
            img = tf.keras.utils.load_img(temp_path, target_size=(224, 224))
            img_array = tf.keras.utils.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            pred = self.cnn_model.predict(img_array, verbose=0)[0]
            cnn_probs_list.append(pred)
            valid_indices.append(j)
        
        # Clean up temp
        temp_path = os.path.join(self.chart_dir, '_temp_predict.png')
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        cnn_probs = np.array(cnn_probs_list)
        y_true = y_val[valid_indices]
        lstm_probs = lstm_probs[valid_indices]
        
        print(f"  Aligned {len(y_true)} predictions from both models")
        
        return lstm_probs, cnn_probs, y_true

    def _optimize_weight(self, lstm_probs, cnn_probs, y_true):
        """
        Grid search for optimal ensemble weight.
        CNN weight is capped at 0.2 — the CNN chart model is unreliable with
        small datasets and should not dominate the ensemble.
        """
        best_f1 = 0
        best_w = 0.8  # Default: LSTM-heavy
        
        # Search w from 0.8 to 1.0 (CNN gets at most 20%)
        for w in np.arange(0.80, 1.01, 0.05):
            ensemble = w * lstm_probs + (1 - w) * cnn_probs
            preds = ensemble.argmax(axis=1)
            f1 = f1_score(y_true, preds, average='macro', zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_w = w
        
        print(f"  Best ensemble weight: LSTM={best_w:.2f}, CNN={1-best_w:.2f}")
        print(f"  Best ensemble F1 (macro): {best_f1:.4f}")
        
        return best_w

    def _print_final_report(self, lstm_probs, cnn_probs, y_true):
        """Print comparison of LSTM-only, CNN-only, and ensemble."""
        label_names = ['HOLD', 'BUY', 'SELL']
        
        # Individual model metrics
        lstm_preds = lstm_probs.argmax(axis=1)
        cnn_preds = cnn_probs.argmax(axis=1)
        
        w = self.ensemble_weight
        ensemble_probs = w * lstm_probs + (1 - w) * cnn_probs
        ensemble_preds = ensemble_probs.argmax(axis=1)
        
        lstm_f1 = f1_score(y_true, lstm_preds, average='macro', zero_division=0)
        cnn_f1 = f1_score(y_true, cnn_preds, average='macro', zero_division=0)
        ensemble_f1 = f1_score(y_true, ensemble_preds, average='macro', zero_division=0)
        
        print("\n" + "=" * 60)
        print("HYBRID MODEL COMPARISON")
        print("=" * 60)
        print(f"\n{'Model':<20} {'F1 (macro)':<12} {'Accuracy':<12}")
        print("-" * 44)
        print(f"{'LSTM only':<20} {lstm_f1:<12.4f} {(lstm_preds == y_true).mean():<12.4f}")
        print(f"{'CNN only':<20} {cnn_f1:<12.4f} {(cnn_preds == y_true).mean():<12.4f}")
        print(f"{'HYBRID (w={w:.2f})':<20} {ensemble_f1:<12.4f} {(ensemble_preds == y_true).mean():<12.4f}")
        
        print(f"\n{'='*60}")
        print("HYBRID Classification Report:")
        print(classification_report(y_true, ensemble_preds, target_names=label_names, zero_division=0))
        
        cm = confusion_matrix(y_true, ensemble_preds)
        print("Confusion Matrix:")
        print(f"         Predicted: HOLD  BUY  SELL")
        for i, name in enumerate(label_names):
            row = "  ".join(f"{cm[i][j]:5d}" for j in range(len(label_names)))
            print(f"  Actual {name:>4}: {row}")

    def predict(self, df):
        """
        Hybrid prediction on live data.
        Returns ensemble probabilities.
        """
        import tensorflow as tf
        from features.intraday_features import generate_features
        
        # LSTM prediction
        featured_df = generate_features(df)
        feature_cols = self.lstm_trainer.feature_columns
        if feature_cols is None:
            from features.intraday_features import get_feature_columns
            feature_cols = get_feature_columns(featured_df)
        
        available_cols = [c for c in feature_cols if c in featured_df.columns]
        X_lstm, _ = self.lstm_trainer.prepare_sequences(featured_df, available_cols, fit_scaler=False)
        
        if len(X_lstm) == 0:
            return None
        
        lstm_probs = self.lstm_trainer.model.predict(X_lstm[-1:], verbose=0)[0]
        
        # CNN prediction (render latest chart window)
        if self.cnn_model is not None and len(df) >= self.chart_window:
            chart_window = df.iloc[-self.chart_window:]
            temp_path = os.path.join(self.chart_dir, '_temp_live.png')
            render_candlestick_image(chart_window, temp_path)
            
            img = tf.keras.utils.load_img(temp_path, target_size=(224, 224))
            img_array = tf.keras.utils.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            cnn_probs = self.cnn_model.predict(img_array, verbose=0)[0]
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            # Ensemble
            w = self.ensemble_weight
            ensemble_probs = w * lstm_probs + (1 - w) * cnn_probs
            return ensemble_probs
        
        return lstm_probs


if __name__ == "__main__":
    data_file = os.path.join("data", "TATASTEEL_NS_1h.csv")
    if os.path.exists(data_file):
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        
        hybrid = HybridTrainer()
        hybrid.train(df, epochs_lstm=50, epochs_cnn=30)
    else:
        print("Data not found. Run: python main.py --mode fetch")
