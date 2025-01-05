import streamlit as st
import os
import sys

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from main import ForexTrader
from data.fetcher import DataFetcher
from data.indicators import TechnicalIndicators

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
        trader = ForexTrader()
        
        st.success("Successfully initialized trading components!")
        
        # Display system status
        st.subheader("System Status")
        st.json({
            "Data Fetcher": "Connected",
            "Technical Indicators": "Ready",
            "Trading Engine": "Active",
            "Models Directory": os.path.join(current_dir, "models"),
            "Data Directory": os.path.join(current_dir, "data")
        })
        
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        st.code(f"""
        Error Details:
        {type(e).__name__}: {str(e)}
        
        Current Directory Structure:
        {os.listdir(current_dir)}
        """)

if __name__ == '__main__':
    main()
