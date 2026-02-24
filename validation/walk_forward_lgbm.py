"""
Walk-Forward Validation for LightGBM Trading Model

Simulates realistic trading by training on past data, predicting future,
and rolling forward. Includes P&L simulation with transaction costs.
"""
import numpy as np
import pandas as pd
import os
import sys
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, accuracy_score, classification_report
import lightgbm as lgb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.intraday_features import generate_features, get_curated_feature_columns
from features.target_labels import generate_labels


def walk_forward_lgbm(df, n_splits=5, purge_bars=5, ticker="TATASTEEL.NS",
                      confidence_threshold=0.45):
    """
    Walk-forward validation with P&L simulation for LightGBM.
    
    How it works:
    - Split data into expanding training windows and fixed test windows
    - For each fold: train on all past data, predict the next window
    - Only trade when model confidence > threshold
    - Track accuracy AND simulated profit/loss
    """
    print(f"\nWalk-Forward Validation ({ticker})")
    print(f"Confidence threshold: {confidence_threshold:.0%}")
    print("=" * 60)
    
    # 1. Prepare data
    print("[1/4] Preparing features and labels...")
    df = generate_features(df)
    df = generate_labels(df, method='dynamic')
    df = df.dropna(subset=['Label'])
    
    feature_cols = get_curated_feature_columns(df)
    X = df[feature_cols].values
    y = df['Label'].values.astype(int)
    closes = df['Close'].values
    forward_returns = df['Forward_Return'].values if 'Forward_Return' in df.columns else None
    
    # Handle NaN
    nan_mask = ~np.isnan(X).any(axis=1)
    X, y, closes = X[nan_mask], y[nan_mask], closes[nan_mask]
    if forward_returns is not None:
        forward_returns = forward_returns[nan_mask]
    
    total = len(X)
    
    # LightGBM params (Optuna-tuned)
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 25,
        'max_depth': 6,
        'learning_rate': 0.095,
        'feature_fraction': 0.67,
        'bagging_fraction': 0.515,
        'bagging_freq': 5,
        'min_child_samples': 48,
        'lambda_l1': 0.008,
        'lambda_l2': 1.84,
        'verbose': -1,
    }
    
    # 2. Walk-forward splits
    min_train = int(total * 0.5)
    test_total = total - min_train
    fold_size = test_total // n_splits
    
    print(f"[2/4] Running {n_splits} walk-forward splits...")
    print(f"  Total samples: {total}, Initial train: {min_train}, Test/fold: ~{fold_size}")
    
    # Store all fold predictions for threshold optimization
    all_fold_data = []
    
    for fold in range(n_splits):
        train_end = min_train + fold * fold_size
        test_start = train_end + purge_bars
        test_end = min(train_end + fold_size, total) if fold < n_splits - 1 else total
        
        if test_start >= test_end:
            continue
        
        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[test_start:test_end], y[test_start:test_end]
        test_closes = closes[test_start:test_end]
        test_fwd_ret = forward_returns[test_start:test_end] if forward_returns is not None else None
        
        # Scale
        scaler = RobustScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        # Class weights
        classes, counts = np.unique(y_train, return_counts=True)
        n_cls = len(classes)
        w = np.ones(len(y_train))
        for cls, cnt in zip(classes, counts):
            w[y_train == cls] = len(y_train) / (n_cls * cnt)
        
        # Train
        train_data = lgb.Dataset(X_train_s, label=y_train, weight=w)
        val_data = lgb.Dataset(X_test_s, label=y_test, reference=train_data)
        
        model = lgb.train(
            params, train_data, num_boost_round=500,
            valid_sets=[val_data], valid_names=['val'],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
        )
        
        # Predict probabilities
        y_probs = model.predict(X_test_s)
        
        all_fold_data.append({
            'fold': fold + 1,
            'train_size': train_end,
            'y_test': y_test,
            'y_probs': y_probs,
            'closes': test_closes,
            'fwd_returns': test_fwd_ret,
        })
    
    # 3. Evaluate at the given confidence threshold
    print(f"\n[3/4] Results at {confidence_threshold:.0%} confidence threshold:")
    print("-" * 60)
    
    results = []
    all_preds = []
    all_actuals = []
    
    for fd in all_fold_data:
        y_pred, y_conf = apply_confidence_filter(fd['y_probs'], confidence_threshold)
        
        # Only evaluate on samples where we made a prediction
        mask = y_pred >= 0  # -1 means filtered out (becomes HOLD)
        y_pred_final = np.where(y_pred >= 0, y_pred, 0)  # filtered -> HOLD
        
        acc = accuracy_score(fd['y_test'], y_pred_final)
        f1 = f1_score(fd['y_test'], y_pred_final, average='macro', zero_division=0)
        
        pnl = simulate_pnl(y_pred, y_conf, fd['y_test'], fd['closes'], fd['fwd_returns'])
        
        results.append({
            'fold': fd['fold'],
            'train': fd['train_size'],
            'test': len(fd['y_test']),
            'acc': round(acc, 3),
            'f1': round(f1, 3),
            'trades': pnl['n_trades'],
            'filtered': pnl['n_filtered'],
            'wr': round(pnl['win_rate'] * 100, 1),
            'pnl': round(pnl['total_return'] * 100, 2),
        })
        
        all_preds.extend(y_pred_final)
        all_actuals.extend(fd['y_test'])
        
        print(f"  Fold {fd['fold']}: Acc={acc:.3f} F1={f1:.3f} "
              f"Trades={pnl['n_trades']}/{len(fd['y_test'])} "
              f"WR={pnl['win_rate']*100:.0f}% P&L={pnl['total_return']*100:+.2f}%")
    
    results_df = pd.DataFrame(results)
    
    print(f"\n  Mean Accuracy:  {results_df['acc'].mean():.4f}")
    print(f"  Mean F1:        {results_df['f1'].mean():.4f}")
    print(f"  Total P&L:      {results_df['pnl'].sum():+.2f}%")
    print(f"  Total Trades:   {results_df['trades'].sum()} (filtered: {results_df['filtered'].sum()})")
    print(f"  Mean Win Rate:  {results_df['wr'].mean():.1f}%")
    
    # Buy & Hold comparison
    bh_return = (closes[-1] / closes[min_train] - 1) * 100
    print(f"\n  Buy & Hold:     {bh_return:+.2f}%")
    print(f"  Strategy:       {results_df['pnl'].sum():+.2f}%")
    
    # 4. Threshold optimization
    print(f"\n[4/4] Confidence Threshold Optimization:")
    print("-" * 60)
    print(f"  {'Threshold':>10s}  {'Trades':>7s}  {'WinRate':>8s}  {'P&L':>10s}")
    
    best_pnl = -999
    best_thresh = 0.35
    
    for thresh in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        total_pnl = 0
        total_trades = 0
        total_wins = 0
        
        for fd in all_fold_data:
            y_pred, y_conf = apply_confidence_filter(fd['y_probs'], thresh)
            pnl = simulate_pnl(y_pred, y_conf, fd['y_test'], fd['closes'], fd['fwd_returns'])
            total_pnl += pnl['total_return']
            total_trades += pnl['n_trades']
            total_wins += pnl['n_wins']
        
        wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
        marker = " <-- BEST" if total_pnl * 100 > best_pnl else ""
        if total_pnl * 100 > best_pnl:
            best_pnl = total_pnl * 100
            best_thresh = thresh
        
        print(f"  {thresh:>10.0%}  {total_trades:>7d}  {wr:>7.1f}%  {total_pnl*100:>+9.2f}%{marker}")
    
    print(f"\n  Best threshold: {best_thresh:.0%} (P&L: {best_pnl:+.2f}%)")
    
    return results_df


def apply_confidence_filter(y_probs, threshold):
    """
    Apply confidence threshold: only trade when max probability > threshold.
    Returns predictions (-1 for filtered/HOLD) and confidences.
    """
    y_pred = y_probs.argmax(axis=1)
    y_conf = y_probs.max(axis=1)
    
    # Filter: if confidence < threshold OR prediction is HOLD, don't trade
    filtered = y_conf < threshold
    y_pred_filtered = y_pred.copy()
    y_pred_filtered[filtered] = -1  # Mark as "no trade"
    
    return y_pred_filtered, y_conf


def simulate_pnl(predictions, confidences, actuals, closes, forward_returns=None, 
                 tx_cost=0.001):
    """
    Simulate P&L from trading signals with confidence filtering.
    
    Rules:
    - prediction == 1 (BUY): go long, earn forward return
    - prediction == 2 (SELL): go short, earn negative forward return  
    - prediction == 0 or -1 (HOLD/filtered): no trade
    - Transaction cost applied per trade
    """
    trade_returns = []
    
    for i in range(len(predictions) - 1):
        pred = predictions[i]
        
        if pred <= 0:  # HOLD or filtered — skip
            continue
        
        # Calculate actual return for next bar
        if forward_returns is not None and i < len(forward_returns):
            actual_return = forward_returns[i]
        else:
            actual_return = (closes[i+1] - closes[i]) / closes[i]
        
        if pred == 1:  # BUY
            pnl = actual_return - tx_cost
        elif pred == 2:  # SELL
            pnl = -actual_return - tx_cost
        else:
            continue
        
        trade_returns.append(pnl)
    
    n_trades = len(trade_returns)
    n_filtered = sum(1 for p in predictions if p == -1)
    n_wins = sum(1 for r in trade_returns if r > 0)
    
    return {
        'total_return': sum(trade_returns),
        'n_trades': n_trades,
        'n_filtered': n_filtered,
        'n_wins': n_wins,
        'win_rate': n_wins / n_trades if n_trades > 0 else 0,
    }

if __name__ == "__main__":
    data_file = os.path.join("data", "TATASTEEL_NS_15m.csv")
    if os.path.exists(data_file):
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        walk_forward_lgbm(df, n_splits=5, ticker="TATASTEEL.NS", confidence_threshold=0.45)
    else:
        print("Historical data not found. Please run historical_data.py first.")
