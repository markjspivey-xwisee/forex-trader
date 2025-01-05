import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import json
import os
import sys

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from src.main import ForexTrader

st.set_page_config(page_title='AI-Powered Forex Trading Agent', layout='wide')

# Rest of your app code...
