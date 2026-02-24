"""
Probability of Backtest Overfitting (PBO) Calculator

Implements Combinatorial Symmetric Cross-Validation (CSCV) as described
in Bailey et al. (2015). This measures the probability that a backtest-optimized
strategy is overfit to the training data.
"""
import numpy as np
import pandas as pd
from itertools import combinations

def calculate_pbo(validation_results, metric_col='accuracy'):
    """
    Calculates the Probability of Backtest Overfitting (PBO) using CSCV.
    
    Steps:
    1. Take N walk-forward fold results
    2. Split into all possible N/2 in-sample (IS) and out-of-sample (OOS) combinations
    3. For each combo, rank strategies by IS performance
    4. Check if the best IS strategy also performs well OOS
    5. PBO = fraction of combos where best IS strategy underperforms OOS median
    
    Args:
        validation_results: DataFrame with per-fold metrics (from walk-forward)
        metric_col: Column name to use as the performance metric
    
    Returns:
        pbo_score: float [0, 1] — probability of overfitting (lower is better)
        details: dict with breakdown
    """
    if validation_results.empty or len(validation_results) < 4:
        return 1.0, {"error": "Need at least 4 folds for CSCV"}
    
    # Use the metric column, fall back to accuracy
    if metric_col not in validation_results.columns:
        for fallback in ['accuracy', 'test_accuracy', 'f1_macro']:
            if fallback in validation_results.columns:
                metric_col = fallback
                break
        else:
            return 1.0, {"error": f"No metric column found in results"}
    
    performances = validation_results[metric_col].values
    n = len(performances)
    half = n // 2
    
    # Generate all combinations of n choose n/2 for in-sample
    fold_indices = list(range(n))
    is_combos = list(combinations(fold_indices, half))
    
    overfit_count = 0
    total_combos = len(is_combos)
    logit_values = []
    
    for is_indices in is_combos:
        oos_indices = [i for i in fold_indices if i not in is_indices]
        
        is_performances = performances[list(is_indices)]
        oos_performances = performances[list(oos_indices)]
        
        # Best strategy from IS
        best_is_idx = np.argmax(is_performances)
        best_is_perf = is_performances[best_is_idx]
        
        # Corresponding OOS performance of the best IS strategy
        # Map: which OOS fold corresponds to the best IS fold's "strategy"
        best_oos_perf = oos_performances[best_is_idx] if best_is_idx < len(oos_performances) else np.mean(oos_performances)
        
        # Median OOS performance
        oos_median = np.median(oos_performances)
        
        # Is the best IS strategy below OOS median? (overfit signal)
        if best_oos_perf < oos_median:
            overfit_count += 1
        
        # Logit: relative rank of best IS in OOS
        oos_rank = np.sum(oos_performances <= best_oos_perf) / len(oos_performances)
        if 0 < oos_rank < 1:
            logit_values.append(np.log(oos_rank / (1 - oos_rank)))
    
    pbo = overfit_count / total_combos if total_combos > 0 else 1.0
    
    details = {
        "n_folds": n,
        "n_combinations": total_combos,
        "overfit_combos": overfit_count,
        "pbo_score": round(pbo, 4),
        "mean_logit": round(np.mean(logit_values), 4) if logit_values else None,
        "interpretation": _interpret_pbo(pbo),
        "metric_used": metric_col,
        "performances": performances.tolist()
    }
    
    return pbo, details

def _interpret_pbo(pbo):
    """Human-readable interpretation of PBO score."""
    if pbo < 0.15:
        return "LOW risk of overfitting — strategy is robust"
    elif pbo < 0.40:
        return "MODERATE risk — proceed with caution"
    elif pbo < 0.60:
        return "HIGH risk — likely overfit to in-sample data"
    else:
        return "CRITICAL — very likely overfit, do not deploy"

if __name__ == "__main__":
    # Example with realistic walk-forward results
    data = {
        'split': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'accuracy': [0.65, 0.62, 0.64, 0.58, 0.63, 0.66, 0.61, 0.59, 0.64, 0.60]
    }
    df = pd.DataFrame(data)
    
    pbo, details = calculate_pbo(df)
    print(f"PBO Score: {pbo:.4f}")
    print(f"Interpretation: {details['interpretation']}")
    print(f"Combinations tested: {details['n_combinations']}")
    print(f"Overfit combos: {details['overfit_combos']}")
