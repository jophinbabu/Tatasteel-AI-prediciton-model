class PositionSizer:
    def __init__(self, max_risk_percent=0.01, kelly_fraction=0.5):
        self.max_risk_percent = max_risk_percent
        self.kelly_fraction = kelly_fraction

    def calculate_position_size(self, account_value, price, stop_loss_price, win_rate=0.55, risk_reward=2.0):
        """
        Calculates position size using Half-Kelly criterion and fixed risk.
        """
        # Fixed Risk Method
        risk_per_share = abs(price - stop_loss_price)
        if risk_per_share == 0:
            return 0
            
        max_loss_amount = account_value * self.max_risk_percent
        shares_fixed_risk = int(max_loss_amount / risk_per_share)
        
        # Kelly Criterion: f* = p - (1-p)/b where b is risk/reward
        # p = win_rate, b = risk_reward
        kelly_f = win_rate - (1 - win_rate) / risk_reward
        shares_kelly = int((account_value * kelly_f * self.kelly_fraction) / price)
        
        # Return the minimum of the two to be conservative
        return max(0, min(shares_fixed_risk, shares_kelly))

if __name__ == "__main__":
    sizer = PositionSizer()
    print(sizer.calculate_position_size(100000, 500, 495)) # 1% risk of 100k is 1000, risk per share 5, so 200 shares
