import streamlit as st
import os
import sys

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from main import ForexTrader

st.set_page_config(page_title='AI-Powered Forex Trading Agent', layout='wide')

def main():
    # Title and logo
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Robot/3D/robot_3d.png", width=100)
    with col2:
        st.title('AI-Powered Forex Trading Agent')
    
    # Display system status
    st.subheader("System Status")
    st.json({
        "Current Directory": current_dir,
        "Python Path": sys.path,
        "Available Modules": os.listdir(current_dir)
    })
    
    # Initialize trader with default settings
    if st.button("Initialize Trading System"):
        try:
            trader = ForexTrader()
            st.success("Successfully initialized ForexTrader!")
            st.session_state['trader'] = trader
        except Exception as e:
            st.error(f"Error initializing ForexTrader: {str(e)}")
            st.code(f"""
            Error Details:
            {type(e).__name__}: {str(e)}
            """)

if __name__ == '__main__':
    main()
