import sys
import os
import unittest
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.trainer import TradingTrainer
from features.intraday_features import get_feature_columns


class TestTrainingPipeline(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 300
        close = np.cumsum(np.random.randn(n) * 0.5) + 500
        self.df = pd.DataFrame({
            'Open': close + np.random.randn(n) * 0.5,
            'High': close + np.abs(np.random.randn(n) * 1.5),
            'Low': close - np.abs(np.random.randn(n) * 1.5),
            'Close': close,
            'Volume': np.random.uniform(1000, 5000, n)
        }, index=pd.date_range('2025-01-02 09:15', periods=n, freq='15min'))

    def test_prepare_features(self):
        """Test feature preparation creates labels."""
        trainer = TradingTrainer()
        df = trainer.prepare_features(self.df)
        self.assertIn('Label', df.columns)
        self.assertIn('Forward_Return', df.columns)
        self.assertGreater(len(df), 0)

    def test_feature_columns_exclude_ohlcv(self):
        """Raw OHLCV should NOT be in feature columns."""
        trainer = TradingTrainer()
        df = trainer.prepare_features(self.df)
        feature_cols = get_feature_columns(df)
        
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            self.assertNotIn(col, feature_cols)

    def test_sequence_generation(self):
        """Test that sequences have correct shape."""
        trainer = TradingTrainer(seq_length=30)
        df = trainer.prepare_features(self.df)
        feature_cols = get_feature_columns(df)
        
        X, y = trainer.prepare_sequences(df, feature_cols, fit_scaler=True)
        
        self.assertEqual(X.shape[1], 30)  # seq_length
        self.assertEqual(X.shape[2], len(feature_cols))
        self.assertEqual(len(X), len(y))

    def test_class_weights_computation(self):
        """Test class weights are computed correctly."""
        trainer = TradingTrainer()
        y = np.array([0, 0, 0, 0, 0, 1, 1, 2])  # Imbalanced
        weights, alpha = trainer.compute_class_weights(y)
        
        # HOLD should have lowest weight (most frequent)
        self.assertLess(weights[0], weights[1])
        self.assertLess(weights[0], weights[2])
        
        # Alpha should be a list of 3 values
        self.assertEqual(len(alpha), 3)

    def test_label_distribution(self):
        """Labels should have all 3 classes present in random data."""
        trainer = TradingTrainer()
        df = trainer.prepare_features(self.df)
        
        labels = df['Label'].unique()
        # With random data and low threshold, we should see at least 2 classes
        self.assertGreaterEqual(len(labels), 2)


if __name__ == "__main__":
    unittest.main()
