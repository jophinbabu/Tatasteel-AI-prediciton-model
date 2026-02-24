"""
Enhanced Trading Trainer

Improvements:
- Scaler fit on TRAIN data only (no data leakage)
- Scaler saved/loaded via joblib for live inference
- Raw OHLCV excluded from features
- Class weights computed from label distribution
- Focal Loss support
- F1, Precision, Recall metrics
- Confusion matrix printed after training
"""
import numpy as np
import pandas as pd
import os
import sys
import joblib
from collections import Counter
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.lstm_cnn_attention import build_model
from features.intraday_features import generate_features, get_feature_columns, get_curated_feature_columns
from features.target_labels import generate_labels
from training.callbacks import TradingMetricsCallback


class TradingTrainer:
    def __init__(self, ticker="TATASTEEL.NS", seq_length=30):
        self.ticker = ticker
        self.seq_length = seq_length
        self.scaler = RobustScaler()
        self.model = None
        self.feature_columns = None  # Store feature column names
        
        # Paths
        self.model_dir = 'models/saved'
        self.model_path = os.path.join(self.model_dir, f'{ticker}_model.keras')
        self.scaler_path = os.path.join(self.model_dir, f'{ticker}_scaler.joblib')
        self.features_path = os.path.join(self.model_dir, f'{ticker}_features.joblib')

    def prepare_features(self, df, label_method='fixed', label_threshold=0.005):
        """
        Feature engineering + labeling. Returns DataFrame with all features and labels.
        """
        # Generate features (40+ indicators)
        df = generate_features(df)
        
        # Generate labels
        df = generate_labels(df, threshold=label_threshold, method=label_method)
        
        return df

    def prepare_sequences(self, df, feature_cols, fit_scaler=True):
        """
        Converts DataFrame to scaled sequences for LSTM input.
        
        Args:
            df: DataFrame with features and Label column
            feature_cols: List of feature column names to use
            fit_scaler: If True, fits the scaler (training). If False, only transforms (inference).
        
        Returns: (X_sequences, y_labels) as numpy arrays
        """
        # Scale features
        if fit_scaler:
            scaled = self.scaler.fit_transform(df[feature_cols])
        else:
            scaled = self.scaler.transform(df[feature_cols])
        
        labels = df['Label'].values if 'Label' in df.columns else None
        
        # Generate sequences
        X, y = [], []
        for i in range(len(scaled) - self.seq_length):
            X.append(scaled[i:i + self.seq_length])
            if labels is not None:
                y.append(labels[i + self.seq_length - 1])
        
        X = np.array(X)
        y = np.array(y) if labels is not None else None
        
        return X, y

    def prepare_data(self, df):
        """
        Legacy-compatible method for inference pipeline.
        Uses saved scaler (if available) to transform features.
        """
        df = generate_features(df)
        
        # Load saved feature columns if available
        if self.feature_columns is None and os.path.exists(self.features_path):
            self.feature_columns = joblib.load(self.features_path)
        
        feature_cols = self.feature_columns or get_feature_columns(df)
        
        # Only use columns that exist in the data
        available_cols = [c for c in feature_cols if c in df.columns]
        
        # Load saved scaler if not fitted
        if os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)
            X, _ = self.prepare_sequences(df, available_cols, fit_scaler=False)
        else:
            X, _ = self.prepare_sequences(df, available_cols, fit_scaler=True)
        
        return X, None

    def compute_class_weights(self, y):
        """
        Compute class weights inversely proportional to frequency.
        Returns dict for model.fit() and list for focal loss alpha.
        """
        counter = Counter(y)
        total = len(y)
        n_classes = len(counter)
        
        # Inverse frequency weighting
        weights = {}
        for cls, count in counter.items():
            weights[int(cls)] = total / (n_classes * count)
        
        print(f"Class weights: {weights}")
        
        # Alpha for focal loss: [HOLD, BUY, SELL]
        alpha = [weights.get(i, 1.0) for i in range(n_classes)]
        
        return weights, alpha

    def train(self, df, epochs=100, batch_size=64, label_method='dynamic', 
              label_threshold=0.005, use_focal_loss=False, focal_gamma=1.0,
              use_curated_features=True):
        """
        Full training pipeline with proper data handling.
        """
        print("=" * 60)
        print(f"Training model for {self.ticker}")
        print("=" * 60)
        
        # 1. Feature Engineering + Labeling
        print("\n[1/6] Generating features and labels...")
        df = self.prepare_features(df, label_method, label_threshold)
        
        # 2. Identify feature columns (curated or full set)
        if use_curated_features:
            self.feature_columns = get_curated_feature_columns(df)
            print(f"[2/6] Using CURATED {len(self.feature_columns)} features")
        else:
            self.feature_columns = get_feature_columns(df)
            print(f"[2/6] Using ALL {len(self.feature_columns)} features")
        
        # 3. Time-based train/val split BEFORE scaling (85/15 to maximize training data)
        split_idx = int(len(df) * 0.85)
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]
        
        print(f"[3/6] Train: {len(train_df)} samples, Val: {len(val_df)} samples")
        
        # 4. Scale — fit on TRAIN only, transform both
        print("[4/6] Scaling features (fit on train only)...")
        X_train, y_train = self.prepare_sequences(train_df, self.feature_columns, fit_scaler=True)
        X_val, y_val = self.prepare_sequences(val_df, self.feature_columns, fit_scaler=False)
        
        print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"  X_val:   {X_val.shape}, y_val:   {y_val.shape}")
        
        # 5. Compute class weights
        class_weights, alpha = self.compute_class_weights(y_train)
        
        # 6. Build model (auto-sized based on training data)
        print(f"[5/6] Building model (Focal Loss: {use_focal_loss}, gamma: {focal_gamma})...")
        self.model = build_model(
            input_shape=(self.seq_length, len(self.feature_columns)),
            num_classes=3,
            use_focal_loss=use_focal_loss,
            focal_gamma=focal_gamma,
            class_weights_alpha=alpha if use_focal_loss else None,
            model_size='auto',
            n_samples=len(X_train),
            learning_rate=0.0003
        )
        
        # 7. Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1),
            TradingMetricsCallback(X_val, y_val)
        ]
        
        # 8. Train
        # IMPORTANT: When using focal loss, do NOT also pass class_weight.
        # Focal loss alpha already handles class imbalance — double-applying
        # causes massive over-prediction of minority classes.
        fit_class_weight = None if use_focal_loss else class_weights
        print(f"[6/6] Training for up to {epochs} epochs...")
        print(f"  Class balancing via: {'Focal Loss alpha' if use_focal_loss else 'class_weight'}")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=fit_class_weight,
            verbose=1
        )
        
        # 9. Evaluation
        print("\n" + "=" * 60)
        print("EVALUATION ON VALIDATION SET")
        print("=" * 60)
        
        y_pred_probs = self.model.predict(X_val, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        label_names = ['HOLD', 'BUY', 'SELL']
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred, target_names=label_names, zero_division=0))
        
        print("Confusion Matrix:")
        cm = confusion_matrix(y_val, y_pred)
        print(f"         Predicted: HOLD  BUY  SELL")
        for i, name in enumerate(label_names):
            row = "  ".join(f"{cm[i][j]:5d}" for j in range(len(label_names)))
            print(f"  Actual {name:>4}: {row}")
        
        # 10. Save model, scaler, and feature list
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        self.model.save(self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        joblib.dump(self.feature_columns, self.features_path)
        
        print(f"\nSaved:")
        print(f"  Model:    {self.model_path}")
        print(f"  Scaler:   {self.scaler_path}")
        print(f"  Features: {self.features_path}")
        
        return history


if __name__ == "__main__":
    data_file = os.path.join("data", "TATASTEEL_NS_15m.csv")
    if os.path.exists(data_file):
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        trainer = TradingTrainer()
        trainer.train(df, epochs=5)  # Short run for testing
    else:
        print("Historical data not found. Please run historical_data.py first.")
