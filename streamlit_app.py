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

def plot_price_and_signals(data, signals=None):
    """Create price chart with signals"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03, row_heights=[0.7, 0.3])
    
    # Price candlesticks
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close'],
        name='Price'
    ), row=1, col=1)
    
    # Volume bars
    fig.add_trace(go.Bar(
        x=data.index,
        y=data['volume'],
        name='Volume'
    ), row=2, col=1)
    
    # Add signals if provided
    if signals:
        for signal in signals:
            marker_color = 'green' if signal['signal_type'] == 'BUY' else 'red'
            fig.add_trace(go.Scatter(
                x=[signal['timestamp']],
                y=[data.loc[signal['timestamp'], 'close']],
                mode='markers',
                marker=dict(
                    symbol='triangle-up' if signal['signal_type'] == 'BUY' else 'triangle-down',
                    size=15,
                    color=marker_color
                ),
                name=f"{signal['signal_type']} ({signal['confidence']:.2f})"
            ), row=1, col=1)
    
    fig.update_layout(
        title='EUR/USD Price Chart',
        xaxis_title='Date',
        yaxis_title='Price',
        height=800
    )
    
    return fig

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
        
        # Model selection and training
        st.subheader("Model Training")
        col1, col2 = st.columns(2)
        with col1:
            selected_model = st.selectbox('Select Model', list(models.keys()))
        with col2:
            if st.button('Train Model'):
                with st.spinner('Training model...'):
                    model = models[selected_model]
                    metrics = model.train(data)
                    st.success(f"Training complete! Validation accuracy: {metrics['validation_accuracy']:.2%}")
        
        # Trading signals
        st.subheader("Trading Signals")
        col1, col2 = st.columns(2)
        with col1:
            if st.button('Get Signal'):
                with st.spinner('Analyzing market...'):
                    model = models[selected_model]
                    signal = model.predict(data.iloc[-1])
                    confidence = model.get_prediction_confidence(data.iloc[-1])
                    signal_type = 'BUY' if signal > 0 else 'SELL' if signal < 0 else 'HOLD'
                    st.info(f"Signal: {signal_type} (Confidence: {confidence:.2%})")
        with col2:
            if st.button('Get Ensemble Signal'):
                with st.spinner('Getting ensemble prediction...'):
                    signals = []
                    confidences = []
                    for model in models.values():
                        signal = model.predict(data.iloc[-1])
                        confidence = model.get_prediction_confidence(data.iloc[-1])
                        signals.append(signal)
                        confidences.append(confidence)
                    
                    weighted_signal = sum(s * c for s, c in zip(signals, confidences)) / sum(confidences)
                    avg_confidence = sum(confidences) / len(confidences)
                    signal_type = 'BUY' if weighted_signal > 0.5 else 'SELL' if weighted_signal < -0.5 else 'HOLD'
                    st.info(f"Ensemble Signal: {signal_type} (Confidence: {avg_confidence:.2%})")
        
        # Price chart
        st.subheader("Price Chart")
        fig = plot_price_and_signals(data)
        st.plotly_chart(fig, use_container_width=True)
        
        # System status
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
