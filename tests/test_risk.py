import sys
import os
import unittest

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from risk.manager import RiskManager
from risk.position_sizer import PositionSizer

class TestRisk(unittest.TestCase):
    def test_risk_manager_limits(self):
        rm = RiskManager(daily_loss_limit=0.03)
        # Test basic allowance
        allowed, reason = rm.check_trade_allowed(100000)
        self.assertTrue(allowed)
        
        # Test daily loss limit
        rm.update_metrics(-4000) # Loss of 4k on 100k account (4%)
        allowed, reason = rm.check_trade_allowed(100000)
        self.assertFalse(allowed)
        self.assertIn("Daily loss limit", reason)

    def test_position_sizing(self):
        sizer = PositionSizer(max_risk_percent=0.01)
        # Account 100,000, 1% risk = 1000. Price 500, SL 490 (Risk 10/share). 
        # Size should be 100 shares.
        size = sizer.calculate_position_size(100000, 500, 490)
        self.assertGreater(size, 0)
        self.assertLessEqual(size, 100) # Fixed risk cap

if __name__ == "__main__":
    unittest.main()
