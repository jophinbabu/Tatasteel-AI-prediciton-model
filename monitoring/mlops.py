import os
import json
from datetime import datetime

class PerformanceMonitor:
    """
    Simple MLOps monitoring for performance tracking.
    """
    def __init__(self, log_dir="monitoring/logs"):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
    def log_prediction(self, ticker, probs, signal, actual=None):
        """
        Logs predictions for later drift analysis.
        """
        log_file = os.path.join(self.log_dir, f"{ticker}_predictions.jsonl")
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "probs": probs.tolist() if hasattr(probs, 'tolist') else probs,
            "signal": signal,
            "actual": actual
        }
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
            
    def check_drift(self, ticker, window=100):
        """
        Heuristic drift detection based on rolling average of max prediction probabilities.
        If confidence drops significantly, it might indicate a regime change (drift).
        """
        log_file = os.path.join(self.log_dir, f"{ticker}_predictions.jsonl")
        if not os.path.exists(log_file):
            return 0.0, False
            
        confidences = []
        try:
            with open(log_file, "r") as f:
                lines = f.readlines()[-window:]
                for line in lines:
                    entry = json.loads(line)
                    confidences.append(max(entry["probs"]))
        except Exception:
            return 0.0, False
            
        if not confidences:
            return 0.0, False
            
        avg_confidence = sum(confidences) / len(confidences)
        # If avg confidence drops below 0.5, flag it (heuristic)
        is_drifting = avg_confidence < 0.5
        
        return avg_confidence, is_drifting

if __name__ == "__main__":
    monitor = PerformanceMonitor()
    monitor.log_prediction("TATASTEEL.NS", [0.1, 0.8, 0.1], "BUY")
