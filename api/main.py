from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import sys

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from execution.trading_system import TradingSystem
from api.schemas import StatusResponse, RunCycleResponse, HealthResponse

app = FastAPI(title="Tata Motors Trading API", version="1.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

system = TradingSystem()

@app.get("/", response_model=StatusResponse)
def read_root():
    return StatusResponse(
        ticker=system.ticker,
        account_value=system.broker.get_balance(),
        daily_pnl=system.risk_manager.daily_pnl,
        current_drawdown=system.risk_manager.current_drawdown,
        model_loaded=(system.trainer.model is not None or getattr(system, 'lgbm_model', None) is not None)
    )

@app.get("/status", response_model=StatusResponse)
def get_status():
    return StatusResponse(
        ticker=system.ticker,
        account_value=system.broker.get_balance(),
        daily_pnl=system.risk_manager.daily_pnl,
        current_drawdown=system.risk_manager.current_drawdown,
        model_loaded=(system.trainer.model is not None or getattr(system, 'lgbm_model', None) is not None)
    )

@app.get("/health", response_model=HealthResponse)
def health_check():
    avg_conf, drift = system.perf_monitor.check_drift(system.ticker)
    return HealthResponse(
        model_loaded=(system.trainer.model is not None or getattr(system, 'lgbm_model', None) is not None),
        data_available=True,
        avg_confidence=avg_conf if avg_conf > 0 else None,
        drift_detected=drift
    )

@app.post("/run", response_model=RunCycleResponse)
def run_cycle():
    """
    Manually trigger a trading cycle.
    """
    try:
        system.run_once()
        return RunCycleResponse(message="Cycle complete. Check logs for details.")
    except Exception as e:
        return RunCycleResponse(message="Cycle failed.", error=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
