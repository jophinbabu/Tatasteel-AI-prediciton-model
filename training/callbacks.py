"""
Training Callbacks

TradingMetricsCallback: Tracks trading-specific metrics per epoch:
- Signal distribution (how many BUY/HOLD/SELL predictions)
- Per-class accuracy
- Win rate estimation on validation set
"""
import numpy as np
from tensorflow.keras.callbacks import Callback


class TradingMetricsCallback(Callback):
    """
    Custom callback to track trading-specific metrics during training.
    Runs on validation data at the end of each epoch.
    """
    def __init__(self, X_val, y_val, log_interval=5):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.log_interval = log_interval
        self.label_names = ['HOLD', 'BUY', 'SELL']
    
    def on_epoch_end(self, epoch, logs=None):
        # Log every N epochs to avoid spam
        if (epoch + 1) % self.log_interval != 0:
            return
        
        # Get predictions
        probs = self.model.predict(self.X_val, verbose=0)
        preds = np.argmax(probs, axis=1)
        
        # Signal distribution
        total = len(preds)
        dist = {name: np.sum(preds == i) for i, name in enumerate(self.label_names)}
        dist_str = ", ".join(f"{name}: {count} ({count/total*100:.1f}%)" for name, count in dist.items())
        
        # Per-class accuracy
        class_acc = []
        for i, name in enumerate(self.label_names):
            mask = self.y_val == i
            if mask.sum() > 0:
                acc = (preds[mask] == i).mean()
                class_acc.append(f"{name}={acc:.2f}")
            else:
                class_acc.append(f"{name}=N/A")
        
        # Average confidence on BUY/SELL signals
        buy_sell_mask = preds != 0
        if buy_sell_mask.sum() > 0:
            avg_conf = probs[buy_sell_mask].max(axis=1).mean()
        else:
            avg_conf = 0.0
        
        # Win rate: correct BUY/SELL predictions
        buy_sell_actual = self.y_val != 0
        buy_sell_pred = preds != 0
        correct_signals = ((preds == self.y_val) & buy_sell_actual).sum()
        total_signals = buy_sell_actual.sum()
        win_rate = correct_signals / total_signals if total_signals > 0 else 0.0
        
        print(f"\n  [TradingMetrics] Epoch {epoch+1}:")
        print(f"    Distribution: {dist_str}")
        print(f"    Class Accuracy: {', '.join(class_acc)}")
        print(f"    Signal Confidence: {avg_conf:.3f}")
        print(f"    BUY/SELL Win Rate: {win_rate:.2%} ({correct_signals}/{total_signals})")
