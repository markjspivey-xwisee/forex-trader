import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import json
import os
import sys

from data.fetcher import DataFetcher
from data.indicators import TechnicalIndicators
from models.random_forest import RandomForestModel
from models.neural_network import NeuralNetworkModel
from backtesting.backtester import Backtester

st.set_page_config(page_title='AI-Powered Forex Trading Agent', layout='wide')

def main():
    # Title and logo
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Robot/3D/robot_3d.png", width=100)
    with col2:
        st.title('AI-Powered Forex Trading Agent')
    
    # Initialize components
    try:
        data_fetcher = DataFetcher()
        indicators = TechnicalIndicators()
        models = {
            'random_forest': RandomForestModel(),
            'neural_network': NeuralNetworkModel()
        }
        backtester = Backtester()
        
        st.success("Successfully initialized trading components!")
        
        # Get initial data
        data = data_fetcher.fetch_data(days=30)
        data = indicators.add_all_indicators(data)
        
        # Display current price and change
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric('Last Update', datetime.now().strftime('%H:%M:%S'))
        with col2:
            current_price = data['close'].iloc[-1]
            price_change = (current_price - data['close'].iloc[-2]) / data['close'].iloc[-2]
            st.metric('Current Price', f"${current_price:.5f}", f"{price_change:.2%}")
        with col3:
            daily_range = data['high'].iloc[-1] - data['low'].iloc[-1]
            st.metric('Daily Range', f"${daily_range:.5f}")
        
        # Display system status
        st.subheader("System Status")
        st.json({
            "Data Fetcher": "Connected",
            "Technical Indicators": "Ready",
            "Trading Engine": "Active",
            "Models": list(models.keys()),
            "Current Directory": os.getcwd()
        })
        
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        st.code(f"""
        Error Details:
        {type(e).__name__}: {str(e)}
        
        Current Directory Structure:
        {os.listdir(os.getcwd())}
        """)

if __name__ == '__main__':
    main()
