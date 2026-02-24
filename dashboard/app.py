import streamlit as st
import pandas as pd
import requests
import json
import os
import sys
import plotly.graph_objects as go
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Tata Trading AI Monitor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
API_URL = "http://127.0.0.1:8000"
TICKER_NAME = "TATASTEEL.NS"

# --- HELPER FUNCTIONS ---

def fetch_api_status():
    """Fetch live data from the FastAPI backend."""
    try:
        status_res = requests.get(f"{API_URL}/status")
        health_res = requests.get(f"{API_URL}/health")
        if status_res.status_code == 200 and health_res.status_code == 200:
            status = status_res.json()
            health = health_res.json()
            return status, health
    except requests.exceptions.ConnectionError:
        pass
    return None, None

def trigger_trading_cycle():
    """Trigger the live trading loop evaluation."""
    try:
        res = requests.post(f"{API_URL}/run")
        if res.status_code == 200:
            st.toast("✅ Trading cycle executed successfully!")
        else:
            st.error(f"Execution failed: {res.text}")
    except Exception as e:
        st.error(f"API Error: Make sure FastAPI is running. {e}")

def load_local_data():
    """Load the historical data for the chart natively if the API doesn't provide it."""
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', f"{TICKER_NAME.replace('.', '_')}_15m.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        return df.tail(100) # last 100 bars for speed
    return None

def create_candlestick_chart(df):
    """Generate a Plotly Candlestick chart."""
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name="Price")])

    # Simple moving averages
    if len(df) >= 20:
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(20).mean(), 
                                 line=dict(color='orange', width=1), name='SMA 20'))
    
    fig.update_layout(
         title='Live AI Price Chart (15m)',
         yaxis_title='Stock Price',
         xaxis_title='Date & Time',
         xaxis_rangeslider_visible=False,
         template='plotly_dark',
         height=500,
         margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig


# --- SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/8/8c/Tata_logo.svg", width=150)
    st.markdown("## System Controls")
    
    if st.button("🚀 Trigger Live Evaluation", type="primary", use_container_width=True):
        trigger_trading_cycle()
        
    st.divider()
    st.markdown("### Agent Settings")
    st.info("The AI operates on a 15-minute timeframe using an Optuna-tuned LightGBM model. Execution confidence is > 55%.")
    st.caption("Auto-refresh is handled passively.")


# --- MAIN DASHBOARD AREA ---
st.title(f"📈 {TICKER_NAME} | Live AI Trading Monitor")

status, health = fetch_api_status()

if status is None:
    st.error("🔴 Connection to the Live Trading System (FastAPI) failed. Please run `python main.py --mode api` in a separate terminal.")
else:
    # Top Row Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    current_val = status.get('account_value', 100000)
    daily_pnl = status.get('daily_pnl', 0.0)
    drawdown = status.get('current_drawdown', 0.0)
    
    col1.metric("Live Portfolio Value", f"₹ {current_val:,.2f}", f"{daily_pnl:+.2f}% Daily PnL", 
                delta_color="normal" if daily_pnl >= 0 else "inverse")
                
    col2.metric("Open Position", "CASH", "No Active Trades" , delta_color="off")
    
    avg_conf = health.get('avg_confidence')
    conf_str = f"{avg_conf*100:.1f}%" if avg_conf else "--"
    col3.metric("AI Confidence (Avg)", conf_str, "Threshold: 55%", delta_color="off")
    
    col4.metric("System Health", "ONLINE", "API Connected", delta_color="normal")
    

    st.divider()

    chart_col, log_col = st.columns([2.5, 1])

    with chart_col:
        st.subheader("Market Action")
        df = load_local_data()
        if df is not None:
            fig = create_candlestick_chart(df)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No live market data found. Ensure the system is actively fetching data.")

    with log_col:
        st.subheader("Trade Alerts")
        st.caption("The last recent system alerts:")
        
        # Load local log file
        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'monitoring', 'alerts.log')
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                logs = f.readlines()
                # reverse list and show top 10
                for line in reversed(logs[-10:]):
                    # Check if it's a trade or drift alert
                    parts = line.split(" - ")
                    if len(parts) > 1:
                        if "WARNING" in parts[1]:
                            st.warning(line.strip())
                        elif "ERROR" in parts[1]:
                            st.error(line.strip())
                        else:
                            st.info(line.strip())
        else:
            st.write("No alerts generated today.")
