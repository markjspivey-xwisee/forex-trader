import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import json
import os
import sys
import gc

from data.fetcher import DataFetcher
from data.indicators import TechnicalIndicators
from models.random_forest import RandomForestModel
from models.neural_network import NeuralNetworkModel
from backtesting.backtester import Backtester

# Configure Streamlit for better performance
st.set_page_config(
    page_title='AI-Powered Forex Trading Agent',
    layout='wide',
    initial_sidebar_state='collapsed'
)

# Initialize session state
if 'data_fetcher' not in st.session_state:
    st.session_state.data_fetcher = DataFetcher()
if 'indicators' not in st.session_state:
    st.session_state.indicators = TechnicalIndicators()
if 'models' not in st.session_state:
    st.session_state.models = {
        'random_forest': RandomForestModel(),
        'neural_network': NeuralNetworkModel()
    }
if 'backtester' not in st.session_state:
    st.session_state.backtester = Backtester()

@st.cache_data(ttl=300, max_entries=10)
def plot_price_and_signals(data, signals=None):
    """Create price chart with signals"""
    try:
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
    except Exception as e:
        st.error(f"Error plotting chart: {str(e)}")
        return None

@st.cache_data(ttl=300, max_entries=10)
def plot_equity_curve(equity_curve):
    """Plot equity curve from backtest results"""
    try:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=[point['timestamp'] for point in equity_curve],
            y=[point['equity'] for point in equity_curve],
            mode='lines',
            name='Portfolio Value'
        ))
        
        fig.update_layout(
            title='Portfolio Value Over Time',
            xaxis_title='Date',
            yaxis_title='Value ($)',
            height=400
        )
        
        return fig
    except Exception as e:
        st.error(f"Error plotting equity curve: {str(e)}")
        return None

def clear_caches():
    """Clear all caches and force garbage collection"""
    try:
        st.cache_data.clear()
        for model in st.session_state.models.values():
            model.clear_cache()
        st.session_state.data_fetcher.clear_cache()
        st.session_state.indicators.clear_cache()
        st.session_state.backtester.clear_cache()
        gc.collect()
    except Exception as e:
        st.error(f"Error clearing caches: {str(e)}")

def main():
    try:
        # Title and logo
        col1, col2, col3 = st.columns([1, 4, 1])
        with col1:
            st.image("https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Robot/3D/robot_3d.png", width=100)
        with col2:
            st.title('AI-Powered Forex Trading Agent')
        with col3:
            if st.button('Clear Cache'):
                clear_caches()
                st.success('Caches cleared!')
        
        # Sidebar controls
        with st.sidebar:
            st.header('Settings')
            days = st.slider('Data Range (Days)', min_value=30, max_value=180, value=60)
            chunk_size = st.slider('Processing Chunk Size', min_value=100, max_value=5000, value=1000, step=100)
            st.session_state.data_fetcher._chunk_size = chunk_size
            st.session_state.indicators._chunk_size = chunk_size
        
        # Get initial data with progress bar
        with st.spinner('Fetching market data...'):
            data = st.session_state.data_fetcher.fetch_data(days=days)
            
            # Validate data before calculating indicators
            if data is None or len(data) == 0:
                st.error("No data available")
                return
            
            if not all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                st.error("Missing required columns in data")
                return
            
            # Calculate indicators with chunking
            data = st.session_state.indicators.add_all_indicators(data)
            
            if data is None or len(data) == 0:
                st.error("Error calculating indicators")
                return
        
        # Display current price and change
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric('Last Update', datetime.now().strftime('%H:%M:%S'))
        with col2:
            current_price = float(data['close'].iloc[-1])
            price_change = float((current_price - data['close'].iloc[-2]) / data['close'].iloc[-2])
            st.metric('Current Price', f"${current_price:.5f}", f"{price_change:.2%}")
        with col3:
            daily_range = float(data['high'].iloc[-1] - data['low'].iloc[-1])
            st.metric('Daily Range', f"${daily_range:.5f}")
        
        # Model selection and training
        st.subheader("Model Training")
        col1, col2 = st.columns(2)
        with col1:
            selected_model = st.selectbox('Select Model', list(st.session_state.models.keys()))
        with col2:
            if st.button('Train Model'):
                with st.spinner('Training model...'):
                    model = st.session_state.models[selected_model]
                    metrics = model.train(data)
                    st.success(f"Training complete! Validation accuracy: {metrics['validation_accuracy']:.2%}")
        
        # Trading signals
        st.subheader("Trading Signals")
        col1, col2 = st.columns(2)
        with col1:
            if st.button('Get Signal'):
                with st.spinner('Analyzing market...'):
                    model = st.session_state.models[selected_model]
                    signal = model.predict(data.iloc[-1])
                    confidence = model.get_prediction_confidence(data.iloc[-1])
                    signal_type = 'BUY' if signal > 0 else 'SELL' if signal < 0 else 'HOLD'
                    st.info(f"Signal: {signal_type} (Confidence: {confidence:.2%})")
        with col2:
            if st.button('Get Ensemble Signal'):
                with st.spinner('Getting ensemble prediction...'):
                    signals = []
                    confidences = []
                    for model in st.session_state.models.values():
                        signal = model.predict(data.iloc[-1])
                        confidence = model.get_prediction_confidence(data.iloc[-1])
                        signals.append(signal)
                        confidences.append(confidence)
                    
                    weighted_signal = sum(s * c for s, c in zip(signals, confidences)) / sum(confidences)
                    avg_confidence = sum(confidences) / len(confidences)
                    signal_type = 'BUY' if weighted_signal > 0.5 else 'SELL' if weighted_signal < -0.5 else 'HOLD'
                    st.info(f"Ensemble Signal: {signal_type} (Confidence: {avg_confidence:.2%})")
        
        # Backtesting
        st.subheader("Backtesting")
        if st.button('Run Backtest'):
            with st.spinner('Running backtest...'):
                model = st.session_state.models[selected_model]
                results = st.session_state.backtester.run(data, model)
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric('Total Return', f"{results['total_return']:.2%}")
                    st.metric('Win Rate', f"{results['win_rate']:.2%}")
                with col2:
                    st.metric('Total Trades', results['total_trades'])
                    st.metric('Average Return', f"{results['avg_return']:.2%}")
                with col3:
                    st.metric('Max Drawdown', f"{results['max_drawdown']:.2%}")
                    st.metric('Sharpe Ratio', f"{results['sharpe_ratio']:.2f}")
                
                # Plot equity curve
                if results['equity_curve']:
                    fig = plot_equity_curve(results['equity_curve'])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
        
        # Price chart
        st.subheader("Price Chart")
        fig = plot_price_and_signals(data)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # System status
        st.subheader("System Status")
        st.json({
            "Data Fetcher": "Connected",
            "Technical Indicators": "Ready",
            "Trading Engine": "Active",
            "Models": list(st.session_state.models.keys()),
            "Memory Usage": f"{gc.get_count()} objects tracked",
            "Current Directory": os.getcwd(),
            "Data Shape": data.shape if data is not None else None,
            "Available Indicators": list(data.columns) if data is not None else None
        })
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.code(f"""
        Error Details:
        {type(e).__name__}: {str(e)}
        
        Current Directory Structure:
        {os.listdir(os.getcwd())}
        """)

if __name__ == '__main__':
    main()
