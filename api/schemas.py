from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

class SignalType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class StatusResponse(BaseModel):
    status: str = "Trading System Active"
    ticker: str
    account_value: float
    daily_pnl: float
    current_drawdown: float
    model_loaded: bool

class TradeSignal(BaseModel):
    signal: SignalType
    confidence: float = Field(..., ge=0, le=1)
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[int] = None

class RunCycleResponse(BaseModel):
    message: str
    signal: Optional[TradeSignal] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool
    data_available: bool
    avg_confidence: Optional[float] = None
    drift_detected: bool = False
