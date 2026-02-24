"""
Broker Interface

Provides an abstract base class for executing trades, allowing the AI to
seamlessly switch between simulated paper trading and live broker APIs (Zerodha).
"""
import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

class BaseBrokerClient(ABC):
    """Abstract interface defining required broker interaction methods."""
    
    @abstractmethod
    def get_balance(self) -> float:
        """Fetch the available account margin/balance."""
        pass
        
    @abstractmethod
    def place_order(self, ticker: str, action: str, qty: int) -> Tuple[bool, Optional[str]]:
        """
        Execute an order.
        Returns: (success_boolean, order_id_or_error_msg)
        """
        pass
        
    @abstractmethod
    def get_position(self, ticker: str) -> int:
        """Get the current active quantity held for a ticker. (0 if flat)"""
        pass


class PaperBrokerClient(BaseBrokerClient):
    """
    Simulated broker for backtesting and paper trading.
    Tracks internal variables without risking actual capital.
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        self.capital = initial_capital
        self.positions: Dict[str, int] = {}
        logging.info(f"Initialized PaperBroker with ₹ {initial_capital:,.2f}")
        
    def get_balance(self) -> float:
        return self.capital
        
    def get_position(self, ticker: str) -> int:
        return self.positions.get(ticker, 0)
        
    def place_order(self, ticker: str, action: str, qty: int) -> Tuple[bool, str]:
        if action == "BUY":
            current_qty = self.positions.get(ticker, 0)
            self.positions[ticker] = current_qty + qty
            return True, "Sim_BUY_001"
            
        elif action == "SELL":
            current_qty = self.positions.get(ticker, 0)
            # Simple assumption: full close of position
            self.positions[ticker] = 0
            return True, "Sim_SELL_001"
            
        return False, "Invalid Action"


class ZerodhaBrokerClient(BaseBrokerClient):
    """
    Live implementation for Zerodha Kite Connect API.
    Executes real Intraday (MIS) Market orders on the NSE.
    """
    
    def __init__(self):
        from kiteconnect import KiteConnect
        self.api_key = os.getenv("KITE_API_KEY")
        self.access_token = os.getenv("KITE_ACCESS_TOKEN")
        
        if not self.api_key or not self.access_token:
            raise ValueError("Zerodha credentials missing from .env file.")
            
        self.kite = KiteConnect(api_key=self.api_key)
        self.kite.set_access_token(self.access_token)
        logging.info("Initialized Zerodha Live Broker Client.")
        
    def get_balance(self) -> float:
        try:
            margins = self.kite.margins()
            return margins['equity']['available']['live_balance']
        except Exception as e:
            logging.error(f"Zerodha Balance Error: {e}")
            return 0.0
            
    def get_position(self, ticker: str) -> int:
        try:
            positions = self.kite.positions()
            for pos in positions['net']:
                # Zerodha tickers format: NSE:TATASTEEL
                if ticker.replace(".NS", "") in pos['tradingsymbol']:
                    return pos['quantity']
            return 0
        except Exception as e:
            logging.error(f"Zerodha Position Error: {e}")
            return 0
            
    def place_order(self, ticker: str, action: str, qty: int) -> Tuple[bool, str]:
        from kiteconnect import KiteConnect
        
        symbol = ticker.replace(".NS", "")
        transaction_type = self.kite.TRANSACTION_TYPE_BUY if action == "BUY" else self.kite.TRANSACTION_TYPE_SELL
        
        try:
            order_id = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=self.kite.EXCHANGE_NSE,
                tradingsymbol=symbol,
                transaction_type=transaction_type,
                quantity=qty,
                product=self.kite.PRODUCT_MIS, # Intraday leverage
                order_type=self.kite.ORDER_TYPE_MARKET
            )
            logging.info(f"LIVE {action} Order placed! ID: {order_id}")
            return True, str(order_id)
            
        except Exception as e:
            msg = f"Zerodha Order Failed: {e}"
            logging.error(msg)
            return False, msg
