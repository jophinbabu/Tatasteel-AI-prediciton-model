import sys
import os
import unittest
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.intraday_features import generate_features, get_feature_columns, get_curated_feature_columns


class TestFeatures(unittest.TestCase):
    def setUp(self):
        """Create realistic dummy OHLCV data with datetime index."""
        np.random.seed(42)
        n = 200  # Need enough bars for indicator warm-up
        close = np.cumsum(np.random.randn(n) * 0.5) + 500
        
        data = {
            'Open': close + np.random.randn(n) * 0.5,
            'High': close + np.abs(np.random.randn(n) * 1.5),
            'Low': close - np.abs(np.random.randn(n) * 1.5),
            'Close': close,
            'Volume': np.random.uniform(1000, 5000, n)
        }
        self.df = pd.DataFrame(data, index=pd.date_range('2025-01-02 09:15', periods=n, freq='15min'))

    def test_feature_generation_shape(self):
        """Test that features are generated and NaN rows are dropped."""
        featured_df = generate_features(self.df)
        self.assertGreater(len(featured_df), 0)
        # Should have more columns than original OHLCV (5 cols)
        self.assertGreater(featured_df.shape[1], 30)

    def test_core_indicators_present(self):
        """Test that all core technical indicators exist."""
        featured_df = generate_features(self.df)
        core_cols = [
            'RSI_9', 'StochRSI_14', 'MACD', 'MACD_Hist',
            'BB_Width', 'BB_Position', 'ATR_5', 'ATR_Norm', 'TRANGE',
            'VWAP_Dist', 'ROC_10', 'SMA_Slope', 'Vol_Ratio',
            'OBV_Norm', 'WILLR_14', 'CCI_20', 'Volatility_20'
        ]
        for col in core_cols:
            self.assertIn(col, featured_df.columns, f"Missing indicator: {col}")

    def test_time_features_present(self):
        """Test time features exist."""
        featured_df = generate_features(self.df)
        time_cols = ['Time_Hour', 'Time_Minute', 'Time_DayOfWeek', 'Time_Session',
                     'Time_Hour_Sin', 'Time_Hour_Cos', 'Time_DOW_Sin', 'Time_DOW_Cos']
        for col in time_cols:
            self.assertIn(col, featured_df.columns, f"Missing time feature: {col}")

    def test_candlestick_patterns_present(self):
        """Test pattern columns exist."""
        featured_df = generate_features(self.df)
        pattern_cols = ['Pat_Doji', 'Pat_Hammer', 'Pat_InvHammer',
                       'Pat_BullEngulf', 'Pat_BearEngulf',
                       'Pat_MorningStar', 'Pat_EveningStar',
                       'Pat_3Soldiers_Crows']
        for col in pattern_cols:
            self.assertIn(col, featured_df.columns, f"Missing pattern: {col}")

    def test_strategy_features_present(self):
        """Test strategy feature columns exist."""
        featured_df = generate_features(self.df)
        strat_cols = ['Strat_EMA_Cross', 'Strat_EMA_Dist',
                      'Strat_Breakout', 'Strat_Breakout_Strength',
                      'Strat_MeanRev', 'Strat_ZScore',
                      'Strat_Dist_Support', 'Strat_Dist_Resistance',
                      'Strat_ADX', 'Strat_PlusDI', 'Strat_MinusDI',
                      'Strat_VolBreakout', 'Strat_VolSpike',
                      'Strat_MomDiv', 'Strat_RSI_Slope',
                      'Strat_Agreement']
        for col in strat_cols:
            self.assertIn(col, featured_df.columns, f"Missing strategy: {col}")

    def test_no_nan_values(self):
        """Test no NaN values in final dataframe."""
        featured_df = generate_features(self.df)
        self.assertFalse(featured_df.isna().any().any(), "NaN values found in features")

    def test_get_feature_columns_excludes_ohlcv(self):
        """Test that raw OHLCV columns are excluded from feature list."""
        featured_df = generate_features(self.df)
        feature_cols = get_feature_columns(featured_df)
        
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            self.assertNotIn(col, feature_cols, f"Raw OHLCV {col} should be excluded")

    def test_feature_count_minimum(self):
        """Test we have at least 35 features."""
        featured_df = generate_features(self.df)
        feature_cols = get_feature_columns(featured_df)
        self.assertGreaterEqual(len(feature_cols), 35, 
                               f"Expected 35+ features, got {len(feature_cols)}")

    def test_curated_feature_columns(self):
        """Test curated features are a subset and have expected count."""
        featured_df = generate_features(self.df)
        curated_cols = get_curated_feature_columns(featured_df)
        all_cols = get_feature_columns(featured_df)
        
        # Curated should be a subset of all features
        for col in curated_cols:
            self.assertIn(col, all_cols, f"Curated feature {col} not in full feature set")
        
        # Should have 15-20 curated features
        self.assertGreaterEqual(len(curated_cols), 15, 
                               f"Expected 15+ curated features, got {len(curated_cols)}")
        self.assertLessEqual(len(curated_cols), 22,
                            f"Expected <=22 curated features, got {len(curated_cols)}")


if __name__ == "__main__":
    unittest.main()
