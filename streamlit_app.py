import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import os
import sys
import gc
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check OANDA credentials
if not os.getenv('OANDA_API_KEY') or not os.getenv('OANDA_ACCOUNT_ID'):
    st.error("""
    OANDA API credentials not found!
    
    Please set up your OANDA credentials in the .env file:
    1. Go to OANDA's website (https://www.oanda.com/)
    2. Create a practice account if you don't have one
    3. Once logged in, go to "Manage API Access"
    4. Generate an API key for the practice account
    5. Copy your Account ID from the practice account dashboard
    6. Update the .env file with your credentials
    """)
    st.stop()

from data.fetcher import DataFetcher
from data.indicators import TechnicalIndicators
from models.random_forest import RandomForestModel
from models.neural_network import NeuralNetworkModel
from backtesting.backtester import Backtester
from trading.oanda_client import OandaClient

# Rest of the file remains the same...
