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
    st.session_state.data_fetcher = DataFetcher(chunk_size=500)  # Reduced chunk size
    
if 'indicators' not in st.session_state:
    st.session_state.indicators = TechnicalIndicators(chunk_size=500)  # Reduced chunk size
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
    st.session_state.oanda = OandaClient()
if 'live_trading' not in st.session_state:
    st.session_state.live_trading = {
        'active': False,
        'model': None,
        'start_time': None,
        'last_signal': None,
        'last_signal_time': None
    }

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
            height=600  # Reduced height
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
            height=300  # Reduced height
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
        st.session_state.models_trained = {
            'random_forest': False,
            'neural_network': False
        }
        gc.collect()
    except Exception as e:
        st.error(f"Error clearing caches: {str(e)}")

def train_all_models(data):
    """Train all models with the given data"""
    try:
        for model_name, model in st.session_state.models.items():
            with st.spinner(f'Training {model_name}...'):
                metrics = model.train(data)
                st.session_state.models_trained[model_name] = True
                st.success(f"{model_name} training complete! Validation accuracy: {metrics['validation_accuracy']:.2%}")
    except Exception as e:
        st.error(f"Error training models: {str(e)}")

def execute_trade(signal, confidence, current_price):
    """Execute a trade based on the signal"""
    try:
        # Get current positions
        positions = st.session_state.oanda.get_open_positions()
        
        # Get account balance
        balance = st.session_state.oanda.get_account_balance()
        if balance is None:
            st.error("Could not get account balance")
            return
        
        position_size = 0.1  # 10% of balance per trade
        units = int((balance * position_size) / current_price)
        
        # Check if we have an open position
        if positions:
            position = positions[0]  # Assume one position at a time
            
            # Check if we should close the position
            if (position['type'] == 'long' and signal < 0) or \
               (position['type'] == 'short' and signal > 0):
                # Close position
                if st.session_state.oanda.close_position(position['id']):
                    st.success(f"Closed {position['type']} position")
        
        # Check if we should open a new position
        elif signal != 0 and confidence >= 0.6:  # Only trade with sufficient confidence
            # Open position
            order_type = 'long' if signal > 0 else 'short'
            if st.session_state.oanda.place_order(order_type, units, current_price):
                st.success(f"Opened {order_type} position with {units} units")
            
            # Update last signal
            st.session_state.live_trading['last_signal'] = signal
            st.session_state.live_trading['last_signal_time'] = datetime.now()
    
    except Exception as e:
        st.error(f"Error executing trade: {str(e)}")

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
            days = st.slider('Data Range (Days)', min_value=7, max_value=60, value=14)  # Reduced range
            chunk_size = st.slider('Processing Chunk Size', min_value=100, max_value=1000, value=500, step=100)  # Reduced max
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
            if st.button('Train Selected Model'):
                with st.spinner('Training model...'):
                    model = st.session_state.models[selected_model]
                    metrics = model.train(data)
                    st.session_state.models_trained[selected_model] = True
                    st.success(f"Training complete! Validation accuracy: {metrics['validation_accuracy']:.2%}")
        
        # Option to train all models
        if st.button('Train All Models'):
            train_all_models(data)
        
        # Live Trading Section
        st.subheader("Live Trading")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            trading_model = st.selectbox(
                'Select Trading Model',
                list(st.session_state.models.keys()),
                key='trading_model'
            )
        
        with col2:
            if not st.session_state.live_trading['active']:
                if st.button('Start Live Trading'):
                    if not st.session_state.models_trained[trading_model]:
                        st.warning(f"Please train the {trading_model} model first!")
                    else:
                        st.session_state.live_trading['active'] = True
                        st.session_state.live_trading['model'] = trading_model
                        st.session_state.live_trading['start_time'] = datetime.now()
                        st.success(f"Started live trading with {trading_model} model")
            else:
                if st.button('Stop Live Trading'):
                    st.session_state.live_trading['active'] = False
                    st.session_state.live_trading['model'] = None
                    st.session_state.oanda.close_all_positions()  # Close all positions when stopping
                    st.warning("Live trading stopped")
        
        with col3:
            balance = st.session_state.oanda.get_account_balance()
            if balance is not None:
                st.metric('Account Balance', f"${balance:.2f}")
        
        # Live trading status
        if st.session_state.live_trading['active']:
            # Get current positions and trade history
            positions = st.session_state.oanda.get_open_positions()
            trades = st.session_state.oanda.get_trade_history()
            
            st.info(f"""
            Live Trading Status:
            - Model: {st.session_state.live_trading['model']}
            - Running since: {st.session_state.live_trading['start_time'].strftime('%Y-%m-%d %H:%M:%S')}
            - Open Positions: {len(positions)}
            - Total Trades: {len(trades)}
            - Last Signal: {st.session_state.live_trading['last_signal']} at {st.session_state.live_trading['last_signal_time'].strftime('%H:%M:%S') if st.session_state.live_trading['last_signal_time'] else 'N/A'}
            """)
            
            # Get latest signal
            model = st.session_state.models[st.session_state.live_trading['model']]
            signal = model.predict(data.iloc[-1])
            confidence = model.get_prediction_confidence(data.iloc[-1])
            
            # Execute trade based on signal
            execute_trade(signal, confidence, current_price)
            
            # Display positions
            if positions:
                st.write("Open Positions:")
                positions_df = pd.DataFrame(positions)
                st.dataframe(positions_df)
            
            # Display trade history
            if trades:
                st.write("Trade History:")
                trades_df = pd.DataFrame(trades)
                st.dataframe(trades_df)
        
        # Trading signals
        st.subheader("Trading Signals")
        col1, col2 = st.columns(2)
        with col1:
            if st.button('Get Signal'):
                if not st.session_state.models_trained[selected_model]:
                    st.warning(f"Please train the {selected_model} first!")
                else:
                    with st.spinner('Analyzing market...'):
                        model = st.session_state.models[selected_model]
                        signal = model.predict(data.iloc[-1])
                        confidence = model.get_prediction_confidence(data.iloc[-1])
                        signal_type = 'BUY' if signal > 0 else 'SELL' if signal < 0 else 'HOLD'
                        st.info(f"Signal: {signal_type} (Confidence: {confidence:.2%})")
        with col2:
            if st.button('Get Ensemble Signal'):
                # Check if all models are trained
                untrained_models = [name for name, trained in st.session_state.models_trained.items() if not trained]
                if untrained_models:
                    st.warning(f"Please train the following models first: {', '.join(untrained_models)}")
                else:
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
            if not st.session_state.models_trained[selected_model]:
                st.warning(f"Please train the {selected_model} first!")
            else:
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
            "Trading Engine": "Active" if st.session_state.live_trading['active'] else "Inactive",
            "OANDA Connection": "Connected" if st.session_state.oanda.api else "Not Connected",
            "Models": {
                name: "Trained" if trained else "Not Trained"
                for name, trained in st.session_state.models_trained.items()
            },
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
    # Get API credentials
    api_key, account_id = get_api_credentials()
    
    # Initialize components with credentials
    if 'data_fetcher' not in st.session_state:
        st.session_state.data_fetcher = DataFetcher(chunk_size=500)
    if 'oanda' not in st.session_state:
        st.session_state.oanda = OandaClient()
    
    main()
