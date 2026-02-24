import numpy as np

def monte_carlo_simulation(returns, n_simulations=1000):
    """
    Performs Monte Carlo simulation on trade returns to assess risk.
    """
    if len(returns) == 0:
        return None
        
    sim_results = []
    for _ in range(n_simulations):
        # Bootstrap sampling of returns
        sim_path = np.random.choice(returns, size=len(returns), replace=True)
        cumulative_return = np.prod(1 + sim_path)
        sim_results.append(cumulative_return)
        
    return {
        'mean_return': np.mean(sim_results),
        'median_return': np.median(sim_results),
        'percentile_5': np.percentile(sim_results, 5),
        'percentile_95': np.percentile(sim_results, 95)
    }

if __name__ == "__main__":
    # Example returns
    trade_returns = np.random.normal(0.005, 0.02, 100)
    print(monte_carlo_simulation(trade_returns))
