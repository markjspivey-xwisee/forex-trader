import os
import sys
from pathlib import Path

# Add the current directory to PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import gc
import time
from dotenv import load_dotenv

# Load environment variables
env_path = Path(current_dir) / '.env'
load_dotenv(dotenv_path=env_path)

# Debug: Print environment variables
st.write("Environment variables loaded from:", env_path)
st.write("OANDA_API_KEY:", "*" * len(os.getenv('OANDA_API_KEY')) if os.getenv('OANDA_API_KEY') else "Not found")
st.write("OANDA_ACCOUNT_ID:", os.getenv('OANDA_ACCOUNT_ID') if os.getenv('OANDA_ACCOUNT_ID') else "Not found")

# Import local modules
from data import DataFetcher, TechnicalIndicators
from models import RandomForestModel, NeuralNetworkModel
from backtesting import Backtester
from trading import OandaClient

# Configure Streamlit for better performance
st.set_page_config(
    page_title='AI-Powered Forex Trading Agent',
    layout='wide',
    initial_sidebar_state='collapsed'
)

# Get API credentials
def get_api_credentials():
    """Get API credentials from secrets or environment variables"""
    try:
        api_key = st.secrets["OANDA_API_KEY"]
        account_id = st.secrets["OANDA_ACCOUNT_ID"]
        st.sidebar.success("OANDA API credentials loaded from secrets")
    except Exception as e:
        api_key = os.getenv('OANDA_API_KEY')
        account_id = os.getenv('OANDA_ACCOUNT_ID')
        if api_key and account_id:
            st.sidebar.info("Using OANDA API credentials from environment variables")
            st.sidebar.write(f"Account ID: {account_id}")
            st.sidebar.write(f"API Key length: {len(api_key)}")
        else:
            st.sidebar.warning("Using simulated data (no API credentials)")
            api_key = None
            account_id = None
    return api_key, account_id

# Initialize session state
if 'api_credentials' not in st.session_state:
    st.session_state.api_credentials = get_api_credentials()

if 'data_fetcher' not in st.session_state:
    api_key, account_id = st.session_state.api_credentials
    st.session_state.data_fetcher = DataFetcher(
        chunk_size=500,
        api_key=api_key,
        account_id=account_id
    )
    
if 'indicators' not in st.session_state:
    st.session_state.indicators = TechnicalIndicators(chunk_size=500)
if 'models' not in st.session_state:
    st.session_state.models = {
        'random_forest': RandomForestModel(),
        'neural_network': NeuralNetworkModel()
    }
if 'backtester' not in st.session_state:
    st.session_state.backtester = Backtester()
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = {
        'random_forest': False,
        'neural_network': False
    }
if 'oanda' not in st.session_state:
    api_key, account_id = st.session_state.api_credentials
    st.session_state.oanda = OandaClient(
        api_key=api_key,
        account_id=account_id
    )
if 'live_trading' not in st.session_state:
    st.session_state.live_trading = {
        'active': False,
        'model': None,
        'start_time': None,
        'last_signal': None,
        'last_signal_time': None
    }

[Previous implementation of helper functions and main()]

if __name__ == '__main__':
    # Get API credentials
    api_key, account_id = get_api_credentials()
    
    # Debug: Print credentials before initializing components
    st.write("Initializing components with:")
    st.write("- API Key length:", len(api_key) if api_key else "Not found")
    st.write("- Account ID:", account_id if account_id else "Not found")
    
    # Initialize components with credentials
    if 'data_fetcher' not in st.session_state:
        st.session_state.data_fetcher = DataFetcher(
            chunk_size=500,
            api_key=api_key,
            account_id=account_id
        )
    if 'oanda' not in st.session_state:
        st.session_state.oanda = OandaClient(
            api_key=api_key,
            account_id=account_id
        )
    
    main()
