
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from execution.trading_system import TradingSystem

def test_trading_system_init():
    print("Initializing TradingSystem...")
    try:
        system = TradingSystem(ticker="TATASTEEL.NS")
        print(f"Model Type: {system.model_type}")
        print(f"Min Confidence: {system.min_confidence}")
        
        if system.model_type == 'lgbm' and system.lgbm_model is not None:
            print("SUCCESS: LightGBM model loaded correctly.")
        else:
            print("FAILURE: LightGBM model not loaded.")
            
        if system.min_confidence == 0.60:
            print("SUCCESS: Confidence threshold set correctly.")
        else:
            print(f"FAILURE: Confidence threshold incorrect ({system.min_confidence}).")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_trading_system_init()
