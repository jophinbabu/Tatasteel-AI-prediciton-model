class RiskManager:
    def __init__(self, max_risk_per_trade=0.01, daily_loss_limit=0.03, max_drawdown_limit=0.1):
        self.max_risk_per_trade = max_risk_per_trade
        self.daily_loss_limit = daily_loss_limit
        self.max_drawdown_limit = max_drawdown_limit
        self.daily_pnl = 0.0
        self.peak_account_value = 0.0
        self.current_drawdown = 0.0

    def check_trade_allowed(self, current_account_value):
        """
        Checks if a trade is allowed based on risk limits.
        """
        # Update peak for drawdown calculation
        if current_account_value > self.peak_account_value:
            self.peak_account_value = current_account_value
        
        # Calculate current drawdown
        if self.peak_account_value > 0:
            self.current_drawdown = (self.peak_account_value - current_account_value) / self.peak_account_value

        # Check Daily Loss
        if self.daily_pnl <= - (current_account_value * self.daily_loss_limit):
            return False, f"Daily loss limit reached ({self.daily_pnl:.2f})"
            
        # Check Max Drawdown
        if self.current_drawdown >= self.max_drawdown_limit:
            return False, f"Max drawdown limit reached ({self.current_drawdown:.2%})"
            
        return True, "Allowed"

    def update_metrics(self, trade_pnl):
        """
        Updates daily PnL and MTM metrics.
        """
        self.daily_pnl += trade_pnl
        print(f"Risk Metrics Updated. Daily PnL: {self.daily_pnl:.2f}")

    def reset_daily(self):
        """
        Resets daily metrics at start of trading day.
        """
        self.daily_pnl = 0.0
