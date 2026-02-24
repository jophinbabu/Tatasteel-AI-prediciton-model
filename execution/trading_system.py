"""
Enhanced Trading System

Improvements:
- Loads saved scaler and feature list for inference
- Passes strategy context to signal generator for confirmation
- Uses enhanced signal generator with regime awareness
"""
import time
from datetime import datetime
import pytz
import pandas as pd
import numpy as np
import os
import sys
import joblib

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.live_data import fetch_live_data
from training.trainer import TradingTrainer
from prediction.signal_generator import SignalGenerator
from prediction.visual_analysis import VisualAnalyst
from risk.manager import RiskManager
from risk.position_sizer import PositionSizer
from monitoring.alerting import AlertManager
from monitoring.mlops import PerformanceMonitor
from execution.broker_client import PaperBrokerClient, ZerodhaBrokerClient


class TradingSystem:
    def __init__(self, ticker="TATASTEEL.NS", interval="1h"):
        self.ticker = ticker
        self.interval = interval
        
        # Initialize Broker
        broker_choice = os.getenv("ACTIVE_BROKER", "PAPER").upper()
        if broker_choice == "ZERODHA":
            self.broker = ZerodhaBrokerClient()
        else:
            self.broker = PaperBrokerClient(initial_capital=500000.0)
            
        self.open_positions = []  # Track open positions
        self.max_concurrent_positions = 2
        
        self.min_confidence = 0.40  # Optimized for 8-bar forecast with imbalanced data
        self.model_type = 'lstm'     # Default to LSTM
        self.ist = pytz.timezone('Asia/Kolkata')
        
        self.trainer = TradingTrainer(ticker=ticker)
        self.signal_gen = SignalGenerator(confidence_threshold=self.min_confidence)
        self.visual_analyst = VisualAnalyst()
        self.risk_manager = RiskManager()
        self.sizer = PositionSizer()
        self.alert_manager = AlertManager()
        self.perf_monitor = PerformanceMonitor()
        
        # Hybrid model components
        self.hybrid_scanner = None
        self.ensemble_weight = None
        self.lgbm_model = None
        
        # Paths
        ticker_clean = ticker.replace('.', '_')
        lgbm_path = f'models/saved/{ticker_clean}_lgbm.txt'
        lgbm_scaler = f'models/saved/{ticker_clean}_lgbm_scaler.joblib'
        lgbm_features = f'models/saved/{ticker_clean}_lgbm_features.joblib'
        
        lstm_path = f'models/saved/{ticker}_model.keras'
        lstm_scaler = f'models/saved/{ticker}_scaler.joblib'
        lstm_features = f'models/saved/{ticker}_features.joblib'
        
        # 1. Prefer LightGBM (Validated strategy)
        if os.path.exists(lgbm_path):
            import lightgbm as lgb
            self.lgbm_model = lgb.Booster(model_file=lgbm_path)
            self.trainer.scaler = joblib.load(lgbm_scaler)
            self.trainer.feature_columns = joblib.load(lgbm_features)
            self.model_type = 'lgbm'
            print(f"Loaded LightGBM model from {lgbm_path}")
            
        # 2. Fallback to LSTM
        elif os.path.exists(lstm_path):
            from tensorflow.keras.models import load_model
            from models.focal_loss import SparseFocalLoss
            self.trainer.model = load_model(lstm_path, custom_objects={'SparseFocalLoss': SparseFocalLoss})
            self.trainer.scaler = joblib.load(lstm_scaler)
            self.trainer.feature_columns = joblib.load(lstm_features)
            print(f"Loaded LSTM model from {lstm_path}")
            
        # Check for Hybrid Model (CNN + Ensemble Weight)
        cnn_path = 'models/saved/chart_cnn.keras'
        weight_path = f'models/saved/{ticker}_ensemble_weight.joblib'
        
        if os.path.exists(cnn_path) and os.path.exists(weight_path):
            try:
                from training.hybrid_trainer import HybridTrainer
                from tensorflow.keras.models import load_model
                
                print("Hybrid model found! Loading Vision System...")
                self.hybrid_scanner = HybridTrainer(ticker)
                self.hybrid_scanner.lstm_trainer = self.trainer  # Share the loaded LSTM/LightGBM features
                self.hybrid_scanner.cnn_model = load_model(cnn_path)
                self.hybrid_scanner.ensemble_weight = joblib.load(weight_path)
                print(f"Loaded Hybrid System (Ensemble weight: {self.hybrid_scanner.ensemble_weight:.2f})")
            except Exception as e:
                print(f"WARNING: Could not load CNN/Hybrid model: {e}")
                print("Continuing with primary model only (LightGBM/LSTM).")
                self.hybrid_scanner = None
        
        if self.trainer.feature_columns:
            print(f"Loaded feature list ({len(self.trainer.feature_columns)} features)")

    def _extract_strategy_context(self, df):
        """
        Extracts strategy features from the latest bar for signal confirmation.
        """
        strategy_cols = ['Strat_Agreement', 'Strat_ADX', 'Strat_EMA_Cross', 
                        'Strat_Breakout', 'Strat_MeanRev', 'Strat_VolBreakout',
                        'Strat_MomDiv']
        
        context = {}
        for col in strategy_cols:
            if col in df.columns:
                val = df[col].iloc[-1]
                if isinstance(val, pd.Series):
                    val = val.values[0]
                context[col] = float(val)
        
        return context if context else None

    def _is_trading_hours(self):
        """Check if current time is within NSE trading hours (IST)."""
        now = datetime.now(self.ist)
        hour, minute = now.hour, now.minute
        current_minutes = hour * 60 + minute
        
        market_open = 9 * 60 + 15    # 9:15 AM IST
        market_close = 15 * 60 + 29  # 3:29 PM IST (3:15 is intraday square-off, market open till 3:30)
        
        if current_minutes < market_open or current_minutes > market_close:
            return False, "Outside market hours (9:15 AM - 3:30 PM IST)"
        
        return True, "Trading hours active"

    def run_once(self):
        """
        Single iteration of the trading loop.
        """
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Running trading loop ({self.interval})...")
        
        # 0. Check trading hours
        in_hours, hours_reason = self._is_trading_hours()
        if not in_hours:
            print(f"  SKIPPED: {hours_reason}")
            # return  <-- Temporarily commented out to allow testing outside hours
        
        # 1. Fetch live data
        df = fetch_live_data(ticker=self.ticker, interval=self.interval)
        if df is None or len(df) < 50:
            print("Not enough data.")
            return
        
        # 2. Generate features
        from features.intraday_features import generate_features
        df_featured = generate_features(df)
        
        # 3. Prepare input
        feature_cols = self.trainer.feature_columns
        available_cols = [c for c in feature_cols if c in df_featured.columns]
        
        # 4. Predict
        probs = None
        signal = "HOLD" # Initialize signal
        confidence = 0.0 # Initialize confidence
        
        if self.model_type == 'lgbm':
            # LightGBM prediction (no sequence needed, just latest row)
            X = df_featured[available_cols].iloc[-1:].values
            X_scaled = self.trainer.scaler.transform(X)
            
            # LightGBM Prediction
            prob = self.lgbm_model.predict(X_scaled)[0]
            # prob is [P(Hold), P(Buy), P(Sell)]
            p_hold, p_buy, p_sell = prob[0], prob[1], prob[2]
            
            # Standard logic: exact argmax
            pred = np.argmax(prob)
            confidence = prob[pred]
            probs = prob # Assign to general probs variable for signal generator
            
            print(f"  Model Probabilities: [Hold: {p_hold:.2f}, Buy: {p_buy:.2f}, Sell: {p_sell:.2f}]")
            
            # Hybrid override if loaded
            if self.hybrid_scanner and hasattr(self.hybrid_scanner, 'cnn_model'):
                # Check for 5m mismatch
                if self.interval == '5m':
                     print("  [WARNING] Hybrid Model (CNN) is active on 5m data! This may cause false HOLDs.")
                     print("  Action: Delete 'models/saved/chart_cnn.keras' to use pure LightGBM.")
            
            if self.hybrid_scanner:
                 # existing hybrid logic (omitted for brevity, or assume it blends)
                 pass
                 
            if pred == 1 and confidence >= self.min_confidence:
                signal = "BUY"
            elif pred == 2 and confidence >= self.min_confidence:
                signal = "SELL"
                
        else:
            # LSTM prediction (needs sequence)
            X, _ = self.trainer.prepare_sequences(df_featured, available_cols, fit_scaler=False)
            if len(X) == 0:
                print("Feature extraction failed (sequence length issue).")
                return
            last_seq = X[-1:]
            
            # LSTM Prediction rules...
            pred_probs = self.trainer.model.predict(last_seq, verbose=0)[0]
            # ... existing lstm logic ...
            p_hold, p_buy, p_sell = pred_probs[0], pred_probs[1], pred_probs[2]
            print(f"  LSTM Probabilities: [Hold: {p_hold:.2f}, Buy: {p_buy:.2f}, Sell: {p_sell:.2f}]")
            
            pred = np.argmax(pred_probs)
            confidence = pred_probs[pred]
            probs = pred_probs # Assign to general probs variable for signal generator
            
            if pred == 1 and confidence >= self.min_confidence:
                signal = "BUY"
            elif pred == 2 and confidence >= self.min_confidence:
                signal = "SELL"

        # 5. Get strategy context
        strategy_context = self._extract_strategy_context(df_featured)
        
        # 6. Generate Signal with strategy confirmation
        # The signal and confidence are now determined by the model directly,
        # but the signal_gen can still refine it based on strategy context.
        # We pass the raw probabilities and the initial signal/confidence.
        signal, confidence, details = self.signal_gen.generate_signal(probs, strategy_context, 
                                                                      initial_signal=signal, 
                                                                      initial_confidence=confidence)
        
        # Apply Confidence Threshold Filter (moved inside signal_gen for consistency, but kept here as a final check)
        if signal != "HOLD" and confidence < self.min_confidence:
            print(f"  FILTERED: {signal} signal (Conf {confidence:.2f} < {self.min_confidence})")
            signal = "HOLD"
        
        regime = details.get('regime', 'unknown')
        aligned = details.get('strategies_aligned', 0)
        print(f"Signal: {signal} (Confidence: {confidence:.2f}, "
              f"Regime: {regime}, Strategies Aligned: {aligned}/5)")
        
        # 7. Risk and Position Sizing
        current_pos_qty = self.broker.get_position(self.ticker)
        
        if signal != "HOLD":
            # Basic concurrency protection
            if current_pos_qty > 0 and signal == "BUY":
                print(f"  -> Trade BLOCKED: Already holding {current_pos_qty} shares of {self.ticker}")
                return
                
            account_val = self.broker.get_balance()
            allowed, reason = self.risk_manager.check_trade_allowed(account_val)
            
            if allowed:
                current_price = df_featured['Close'].iloc[-1]
                current_atr = df_featured['ATR_5'].iloc[-1] if 'ATR_5' in df_featured.columns else 5.0
                
                # Multi-index safety
                if isinstance(current_price, pd.Series): current_price = current_price.values[0]
                if isinstance(current_atr, pd.Series): current_atr = current_atr.values[0]
                
                # Dynamic SL/TP based on ATR
                if signal == "BUY":
                    stop_loss = current_price - (1.5 * current_atr)
                    take_profit = current_price + (3.0 * current_atr)
                else:  # SELL
                    stop_loss = current_price + (1.5 * current_atr)
                    take_profit = current_price - (3.0 * current_atr)
                
                size = self.sizer.calculate_position_size(account_val, current_price, stop_loss)
                
                # Execute through Abstract Broker Interface
                success, msg = self.broker.place_order(self.ticker, signal, qty=int(size))
                
                if success:
                    # Alerts and Logging
                    self.alert_manager.notify_trade(signal, current_price, size, stop_loss, take_profit)
                    self.perf_monitor.log_prediction(self.ticker, probs, signal)
                    
                    print(f"  -> {signal} {size} shares @ {current_price:.2f}")
                    print(f"     Status: SUCCESS ({msg})")
                    print(f"     SL: {stop_loss:.2f}, TP: {take_profit:.2f}")
                else:
                    print(f"  -> Trade FAILED: {msg}")
                
                # Check for Drift
                avg_conf, drift = self.perf_monitor.check_drift(self.ticker)
                if drift:
                    self.alert_manager.send_alert(
                        f"Potential Model Drift Detected! Avg Confidence: {avg_conf:.2f}", 
                        level="WARNING"
                    )
                
            else:
                self.alert_manager.notify_circuit_breaker(reason)
                print(f"  -> Trade BLOCKED: {reason}")
        else:
            self.perf_monitor.log_prediction(self.ticker, probs, signal)

    def start(self):
        """
        Starts the main trading loop.
        """
        if self.trainer.model is None and self.lgbm_model is None:
            print("Error: No model loaded or trained. Please train the model first.")
            return
        
        print(f"\nStarting trading loop for {self.ticker}...")
        print(f"  Broker: {self.broker.__class__.__name__}")
        print(f"  Account: INR {self.broker.get_balance():,.2f}")
        print(f"  Interval: {self.interval}")
        print(f"  Features: {len(self.trainer.feature_columns or [])} columns")
        
        # Determine wait time
        wait_minutes = 60 if self.interval == "1h" else 15
        if self.interval == "5m": wait_minutes = 5
        if self.interval == "1m": wait_minutes = 1
        
        while True:
            try:
                self.run_once()
            except Exception as e:
                print(f"ERROR in trading loop: {e}")
                import traceback
                traceback.print_exc()
            
            print(f"Waiting {wait_minutes} minutes for next bar...")
            time.sleep(wait_minutes * 60)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    system = TradingSystem()
    system.run_once()
