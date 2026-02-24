"""
LightGBM Trading Classifier

Gradient Boosted Trees for BUY/HOLD/SELL classification.
Outperforms deep learning on small tabular datasets (<10K samples).
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import RobustScaler
import lightgbm as lgb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.intraday_features import generate_features, get_curated_feature_columns
from features.target_labels import generate_labels


class LGBMTrainer:
    """LightGBM-based trading signal classifier with feature importance analysis."""
    
    def __init__(self, ticker="TATASTEEL.NS"):
        self.ticker = ticker
        self.model = None
        self.scaler = RobustScaler()
        self.feature_columns = None
        self.save_dir = os.path.join("models", "saved")
        os.makedirs(self.save_dir, exist_ok=True)
    
    def train(self, df, label_method='dynamic', label_threshold=0.002, n_folds=5):
        """
        Full training pipeline with stratified k-fold cross-validation.
        """
        print(f"\nTraining LightGBM for {self.ticker}")
        print("=" * 60)
        
        # 1. Generate features and labels
        print("\n[1/5] Generating features and labels...")
        df = generate_features(df)
        df = generate_labels(df, threshold=label_threshold, method=label_method)
        df = df.dropna(subset=['Label'])
        
        # 2. Select curated features
        self.feature_columns = get_curated_feature_columns(df)
        print(f"[2/5] Using {len(self.feature_columns)} curated features")
        
        # 3. Prepare data (no sequences needed — each row is independent)
        X = df[self.feature_columns].values
        y = df['Label'].values.astype(int)
        
        # Handle any remaining NaN
        nan_mask = ~np.isnan(X).any(axis=1)
        X, y = X[nan_mask], y[nan_mask]
        
        print(f"[3/5] Total samples: {len(X)}")
        print(f"  Labels: HOLD={np.sum(y==0)}, BUY={np.sum(y==1)}, SELL={np.sum(y==2)}")
        
        # 4. Time-based train/val split (85/15)
        split_idx = int(len(X) * 0.85)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # 5. Train LightGBM
        print(f"[4/5] Training LightGBM (train={len(X_train)}, val={len(X_val)})...")
        
        # Compute class weights
        classes, counts = np.unique(y_train, return_counts=True)
        total = len(y_train)
        n_classes = len(classes)
        sample_weights = np.ones(len(y_train))
        for cls, cnt in zip(classes, counts):
            weight = total / (n_classes * cnt)
            sample_weights[y_train == cls] = weight
        
        # LightGBM parameters — Optuna-tuned for TATASTEEL 1h data
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 27,
            'max_depth': 7,
            'learning_rate': 0.037,
            'n_estimators': 300,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.6,
            'bagging_freq': 5,
            'min_child_samples': 40,
            'lambda_l1': 0.1,
            'lambda_l2': 1.0,
            'verbose': -1,
        }
        
        train_data = lgb.Dataset(X_train_scaled, label=y_train, weight=sample_weights,
                                 feature_name=self.feature_columns)
        val_data = lgb.Dataset(X_val_scaled, label=y_val, reference=train_data,
                               feature_name=self.feature_columns)
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=50),
        ]
        
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=callbacks,
        )
        
        print(f"  Best iteration: {self.model.best_iteration}")
        
        # 6. Evaluate
        print(f"\n[5/5] Evaluation...")
        y_pred_probs = self.model.predict(X_val_scaled)
        y_pred = y_pred_probs.argmax(axis=1)
        
        print("\n" + "=" * 60)
        print("EVALUATION ON VALIDATION SET")
        print("=" * 60)
        
        print(f"\nClassification Report:")
        print(classification_report(y_val, y_pred, target_names=['HOLD', 'BUY', 'SELL']))
        
        cm = confusion_matrix(y_val, y_pred)
        print(f"Confusion Matrix:")
        print(f"         Predicted: HOLD  BUY  SELL")
        for i, label in enumerate(['HOLD', 'BUY', 'SELL']):
            print(f"  Actual {label:>4s}: {cm[i][0]:>4d} {cm[i][1]:>4d} {cm[i][2]:>4d}")
        
        f1 = f1_score(y_val, y_pred, average='macro')
        print(f"\nF1 Macro: {f1:.4f}")
        
        # 7. Feature importance
        self._print_feature_importance()
        
        # 8. Cross-validation for robust estimate
        self._cross_validate(X, y, params, sample_weights=None)
        
        # 9. Save
        self._save()
        
        return f1
    
    def _print_feature_importance(self):
        """Print feature importance ranking."""
        importance = self.model.feature_importance(importance_type='gain')
        feature_imp = sorted(zip(self.feature_columns, importance), 
                           key=lambda x: x[1], reverse=True)
        
        print(f"\n{'=' * 60}")
        print("FEATURE IMPORTANCE (by gain)")
        print(f"{'=' * 60}")
        
        max_imp = max(importance) if max(importance) > 0 else 1
        for feat, imp in feature_imp:
            bar = '#' * int(30 * imp / max_imp)
            print(f"  {feat:<20s} {imp:>8.1f}  {bar}")
    
    def _cross_validate(self, X, y, params, sample_weights=None):
        """Stratified K-Fold cross-validation for robust performance estimate."""
        print(f"\n{'=' * 60}")
        print("STRATIFIED 5-FOLD CROSS-VALIDATION")
        print(f"{'=' * 60}")
        
        skf = StratifiedKFold(n_splits=5, shuffle=False)  # No shuffle for time-series
        fold_f1s = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_tr, X_vl = X[train_idx], X[val_idx]
            y_tr, y_vl = y[train_idx], y[val_idx]
            
            scaler = RobustScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_vl_s = scaler.transform(X_vl)
            
            # Compute fold-specific weights
            classes, counts = np.unique(y_tr, return_counts=True)
            total = len(y_tr)
            n_classes = len(classes)
            w = np.ones(len(y_tr))
            for cls, cnt in zip(classes, counts):
                w[y_tr == cls] = total / (n_classes * cnt)
            
            train_d = lgb.Dataset(X_tr_s, label=y_tr, weight=w)
            val_d = lgb.Dataset(X_vl_s, label=y_vl, reference=train_d)
            
            model = lgb.train(
                params, train_d, num_boost_round=500,
                valid_sets=[val_d], valid_names=['val'],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
            )
            
            preds = model.predict(X_vl_s).argmax(axis=1)
            f1 = f1_score(y_vl, preds, average='macro')
            fold_f1s.append(f1)
            print(f"  Fold {fold}: F1 macro = {f1:.4f}")
        
        mean_f1 = np.mean(fold_f1s)
        std_f1 = np.std(fold_f1s)
        print(f"\n  Mean F1: {mean_f1:.4f} ± {std_f1:.4f}")
    
    def _save(self):
        """Save model, scaler, and feature list."""
        ticker_clean = self.ticker.replace('.', '_')
        
        model_path = os.path.join(self.save_dir, f"{ticker_clean}_lgbm.txt")
        scaler_path = os.path.join(self.save_dir, f"{ticker_clean}_lgbm_scaler.joblib")
        features_path = os.path.join(self.save_dir, f"{ticker_clean}_lgbm_features.joblib")
        
        self.model.save_model(model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.feature_columns, features_path)
        
        print(f"\nSaved:")
        print(f"  Model:    {model_path}")
        print(f"  Scaler:   {scaler_path}")
        print(f"  Features: {features_path}")
    
    def predict(self, df):
        """Predict on new data."""
        df = generate_features(df)
        X = df[self.feature_columns].values
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict(X_scaled)
        return probs.argmax(axis=1), probs
    
    def tune(self, df, n_trials=50, label_method='dynamic', label_threshold=0.002):
        """
        Auto-tune LightGBM hyperparameters with Optuna.
        After finding the best params, retrains and evaluates.
        """
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        print(f"\nOptuna Hyperparameter Tuning for {self.ticker}")
        print("=" * 60)
        
        # 1. Prepare data
        print("\n[1/4] Preparing data...")
        df = generate_features(df)
        df = generate_labels(df, threshold=label_threshold, method=label_method)
        df = df.dropna(subset=['Label'])
        
        self.feature_columns = get_curated_feature_columns(df)
        X = df[self.feature_columns].values
        y = df['Label'].values.astype(int)
        
        nan_mask = ~np.isnan(X).any(axis=1)
        X, y = X[nan_mask], y[nan_mask]
        
        # Time-based split
        split_idx = int(len(X) * 0.85)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        scaler = RobustScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        
        # Class weights
        classes, counts = np.unique(y_train, return_counts=True)
        total = len(y_train)
        n_classes = len(classes)
        sample_weights = np.ones(len(y_train))
        for cls, cnt in zip(classes, counts):
            sample_weights[y_train == cls] = total / (n_classes * cnt)
        
        train_data = lgb.Dataset(X_train_s, label=y_train, weight=sample_weights)
        val_data = lgb.Dataset(X_val_s, label=y_val, reference=train_data)
        
        # 2. Define objective
        def objective(trial):
            params = {
                'objective': 'multiclass',
                'num_class': 3,
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'verbose': -1,
                'feature_pre_filter': False,
                'num_leaves': trial.suggest_int('num_leaves', 15, 63),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True),
            }
            
            model = lgb.train(
                params, train_data, num_boost_round=500,
                valid_sets=[val_data], valid_names=['val'],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
            )
            
            preds = model.predict(X_val_s).argmax(axis=1)
            return f1_score(y_val, preds, average='macro')
        
        # 3. Run optimization
        print(f"[2/4] Running {n_trials} Optuna trials...")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        best = study.best_params
        print(f"\n[3/4] Best params (F1={study.best_value:.4f}):")
        for k, v in best.items():
            print(f"  {k}: {v}")
        
        # 4. Retrain with best params
        print(f"\n[4/4] Retraining with best params...")
        best_params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'verbose': -1,
            **best,
        }
        
        self.scaler = scaler
        self.model = lgb.train(
            best_params, train_data, num_boost_round=500,
            valid_sets=[train_data, val_data], valid_names=['train', 'val'],
            callbacks=[lgb.early_stopping(50, verbose=True), lgb.log_evaluation(50)],
        )
        
        # Evaluate
        y_pred = self.model.predict(X_val_s).argmax(axis=1)
        
        print("\n" + "=" * 60)
        print("EVALUATION WITH TUNED PARAMS")
        print("=" * 60)
        print(classification_report(y_val, y_pred, target_names=['HOLD', 'BUY', 'SELL']))
        
        cm = confusion_matrix(y_val, y_pred)
        print(f"Confusion Matrix:")
        print(f"         Predicted: HOLD  BUY  SELL")
        for i, label in enumerate(['HOLD', 'BUY', 'SELL']):
            print(f"  Actual {label:>4s}: {cm[i][0]:>4d} {cm[i][1]:>4d} {cm[i][2]:>4d}")
        
        f1 = f1_score(y_val, y_pred, average='macro')
        print(f"\nF1 Macro: {f1:.4f}")
        
        self._print_feature_importance()
        self._save()
        
        return f1
