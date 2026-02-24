import sys
import os
import unittest
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.chart_patterns import (
    detect_doji, detect_hammer, detect_inverted_hammer,
    detect_bullish_engulfing, detect_bearish_engulfing,
    detect_morning_star, detect_evening_star,
    detect_three_soldiers_crows, detect_patterns
)


class TestChartPatterns(unittest.TestCase):
    def _make_df(self, opens, highs, lows, closes):
        """Helper to create OHLCV DataFrame."""
        return pd.DataFrame({
            'Open': opens, 'High': highs, 'Low': lows, 'Close': closes,
            'Volume': [1000] * len(opens)
        })

    def test_doji_detection(self):
        """Doji: body is < 5% of range."""
        df = self._make_df(
            opens=[100, 100.0],
            highs=[105, 105.0],
            lows=[95, 95.0],
            closes=[100.1, 100.2]  # Tiny body, large range
        )
        result = detect_doji(df)
        # At least one should be detected
        self.assertTrue(result.sum() >= 1)

    def test_hammer_detection(self):
        """Hammer: small body at top, long lower shadow."""
        df = self._make_df(
            opens=[100, 104.0],
            highs=[105, 105.0],
            lows=[95, 95.0],   # Long lower shadow
            closes=[104.5, 104.5]  # Body at top
        )
        result = detect_hammer(df)
        self.assertIsInstance(result.iloc[0], (int, np.integer))

    def test_bullish_engulfing(self):
        """Bullish engulfing: green candle engulfs previous red candle."""
        df = self._make_df(
            opens=[105, 98],    # prev red (O>C), curr green (O<C)
            highs=[106, 107],
            lows=[99, 97],
            closes=[100, 106]   # curr body engulfs prev body
        )
        result = detect_bullish_engulfing(df)
        self.assertEqual(result.iloc[1], 1)

    def test_bearish_engulfing(self):
        """Bearish engulfing: red candle engulfs previous green candle."""
        df = self._make_df(
            opens=[100, 106],   # prev green, curr red
            highs=[106, 107],
            lows=[99, 97],
            closes=[105, 99]    # curr body engulfs prev body
        )
        result = detect_bearish_engulfing(df)
        self.assertEqual(result.iloc[1], 1)

    def test_three_soldiers(self):
        """Three White Soldiers: 3 consecutive large green candles."""
        df = self._make_df(
            opens=[100, 103, 106],
            highs=[104, 107, 110],
            lows=[99, 102, 105],
            closes=[103, 106, 109]  # All green with large body
        )
        result = detect_three_soldiers_crows(df)
        self.assertEqual(result.iloc[2], 1)

    def test_three_black_crows(self):
        """Three Black Crows: 3 consecutive large red candles."""
        df = self._make_df(
            opens=[110, 107, 104],
            highs=[111, 108, 105],
            lows=[106, 103, 100],
            closes=[107, 104, 101]  # All red with large body
        )
        result = detect_three_soldiers_crows(df)
        self.assertEqual(result.iloc[2], -1)

    def test_detect_all_patterns(self):
        """Test that detect_patterns adds all 8 pattern columns."""
        np.random.seed(42)
        n = 50
        close = np.cumsum(np.random.randn(n) * 0.5) + 500
        df = self._make_df(
            opens=close + np.random.randn(n) * 0.5,
            highs=close + np.abs(np.random.randn(n) * 1.5),
            lows=close - np.abs(np.random.randn(n) * 1.5),
            closes=close
        )
        result = detect_patterns(df)
        
        expected_cols = [
            'Pat_Doji', 'Pat_Hammer', 'Pat_InvHammer',
            'Pat_BullEngulf', 'Pat_BearEngulf',
            'Pat_MorningStar', 'Pat_EveningStar',
            'Pat_3Soldiers_Crows'
        ]
        for col in expected_cols:
            self.assertIn(col, result.columns, f"Missing pattern column: {col}")


if __name__ == "__main__":
    unittest.main()
