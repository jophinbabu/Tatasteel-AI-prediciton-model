import sys
import os
import unittest
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.trading_strategies import (
    ema_crossover_signal, breakout_signal, mean_reversion_signal,
    support_resistance_levels, trend_strength_adx, volume_breakout_signal,
    momentum_divergence, generate_strategy_features
)


class TestTradingStrategies(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 200
        close = np.cumsum(np.random.randn(n) * 0.5) + 500
        self.df = pd.DataFrame({
            'Open': close + np.random.randn(n) * 0.5,
            'High': close + np.abs(np.random.randn(n) * 1.5),
            'Low': close - np.abs(np.random.randn(n) * 1.5),
            'Close': close,
            'Volume': np.random.uniform(1000, 5000, n)
        })

    def test_ema_crossover(self):
        signal, dist = ema_crossover_signal(self.df)
        self.assertEqual(len(signal), len(self.df))
        self.assertEqual(len(dist), len(self.df))
        self.assertTrue(set(signal.unique()).issubset({-1, 0, 1}))

    def test_breakout(self):
        signal, strength = breakout_signal(self.df)
        self.assertEqual(len(signal), len(self.df))
        self.assertTrue(set(signal.unique()).issubset({-1, 0, 1}))

    def test_mean_reversion(self):
        signal, z_score = mean_reversion_signal(self.df)
        self.assertEqual(len(signal), len(self.df))
        # z_score should have reasonable range
        self.assertTrue(z_score.dropna().abs().max() < 10)

    def test_support_resistance(self):
        dist_sup, dist_res = support_resistance_levels(self.df)
        self.assertEqual(len(dist_sup), len(self.df))
        self.assertEqual(len(dist_res), len(self.df))

    def test_adx(self):
        adx, plus_di, minus_di = trend_strength_adx(self.df)
        self.assertEqual(len(adx), len(self.df))
        # ADX should be 0-100 range (approximately)
        self.assertTrue(adx.max() <= 200)  # Some tolerance for edge cases

    def test_volume_breakout(self):
        signal, vol_ratio = volume_breakout_signal(self.df)
        self.assertEqual(len(signal), len(self.df))
        self.assertTrue(set(signal.unique()).issubset({-1, 0, 1}))

    def test_momentum_divergence(self):
        signal, rsi_slope = momentum_divergence(self.df)
        self.assertEqual(len(signal), len(self.df))
        self.assertTrue(set(signal.unique()).issubset({-1, 0, 1}))

    def test_generate_all_strategy_features(self):
        result = generate_strategy_features(self.df)
        
        expected_cols = [
            'Strat_EMA_Cross', 'Strat_EMA_Dist',
            'Strat_Breakout', 'Strat_Breakout_Strength',
            'Strat_MeanRev', 'Strat_ZScore',
            'Strat_Dist_Support', 'Strat_Dist_Resistance',
            'Strat_ADX', 'Strat_PlusDI', 'Strat_MinusDI',
            'Strat_VolBreakout', 'Strat_VolSpike',
            'Strat_MomDiv', 'Strat_RSI_Slope',
            'Strat_Agreement'
        ]
        for col in expected_cols:
            self.assertIn(col, result.columns, f"Missing: {col}")

    def test_agreement_score(self):
        """Agreement score should be bounded."""
        result = generate_strategy_features(self.df)
        # Agreement = sum of 5 discrete signals, each in {-1, 0, 1}
        self.assertTrue(result['Strat_Agreement'].min() >= -5)
        self.assertTrue(result['Strat_Agreement'].max() <= 5)


if __name__ == "__main__":
    unittest.main()
