import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import json
import os
import time
import glob

from data_fetcher import OANDADataFetcher
from automated_strategy_generator import StrategyGenerator
from backtester import Backtester

# Page configuration
st.set_page_config(
    page_title="AI Forex Trading Agent",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'strategies' not in st.session_state:
    st.session_state.strategies = []
if 'selected_strategy' not in st.session_state:
    st.session_state.selected_strategy = None
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'strategy_params' not in st.session_state:
    st.session_state.strategy_params = {
        'min_validation_accuracy': 0.55,
        'lookback_periods': 12,
        'confidence_threshold': 0.6
    }

def plot_candlestick_with_indicators(df):
    """Create candlestick chart with technical indicators"""
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03, row_heights=[0.4, 0.2, 0.2, 0.2])

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='OHLC'
        ),
        row=1, col=1
    )

    # Add SMAs and Bollinger Bands
    for period in [20, 50, 200]:
        if f'SMA_{period}' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[f'SMA_{period}'], 
                          name=f'SMA {period}', line=dict(width=1)),
                row=1, col=1
            )
    
    if all(col in df.columns for col in ['BB_upper', 'BB_middle', 'BB_lower']):
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_upper'], name='BB Upper',
                      line=dict(color='gray', dash='dash'), opacity=0.5),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_lower'], name='BB Lower',
                      line=dict(color='gray', dash='dash'), opacity=0.5,
                      fill='tonexty'),
            row=1, col=1
        )

    # Add MACD
    if 'MACD' in df.columns and 'Signal_Line' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Signal_Line'], name='Signal Line', line=dict(color='orange')),
            row=2, col=1
        )

    # Add RSI
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')),
            row=3, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    # Add ATR
    if 'ATR' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['ATR'], name='ATR', line=dict(color='brown')),
            row=4, col=1
        )

    fig.update_layout(
        title='Market Analysis Dashboard',
        yaxis_title='Price',
        yaxis2_title='MACD',
        yaxis3_title='RSI',
        yaxis4_title='ATR',
        xaxis_rangeslider_visible=False,
        height=1000
    )

    return fig

def display_ml_process_info():
    """Display information about the ML process"""
    with st.expander("🧠 Machine Learning Process"):
        st.markdown("""
        ### Model Types
        1. **Random Forest (Conservative)**
           - Balanced approach with limited tree depth
           - Focus on reducing false positives
           - Hyperparameter optimization for stability
        
        2. **Random Forest (Aggressive)**
           - Deeper trees and more estimators
           - Higher sensitivity to market movements
           - Optimized for capturing short-term opportunities
        
        3. **XGBoost**
           - Gradient boosting for improved accuracy
           - Feature importance analysis
           - Adaptive learning rates
        
        4. **Neural Network (MLP)**
           - Multi-layer perceptron architecture
           - Non-linear pattern recognition
           - Adaptive learning with early stopping
        
        5. **LSTM**
           - Long Short-Term Memory network
           - Sequence prediction for time series
           - Memory cells for long-term patterns
        
        ### Training Process
        1. Feature Engineering
           - Technical indicators
           - Moving averages
           - Momentum indicators
           - Volatility measures
        
        2. Data Preparation
           - Time series split
           - Feature scaling
           - Sequence generation for LSTM
        
        3. Model Optimization
           - Hyperparameter tuning
           - Cross-validation
           - Performance metrics
        
        4. Validation
           - Out-of-sample testing
           - Performance thresholds
           - Strategy validation
        """)

def display_strategy_metrics(strategy):
    """Display detailed strategy metrics"""
    if 'metrics' in strategy:
        metrics = strategy['metrics']
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.2%}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.2%}")
        with col4:
            st.metric("F1 Score", f"{metrics['f1']:.2%}")
            
        if 'cross_val' in metrics:
            st.metric("Cross-validation Score", f"{metrics['cross_val']:.2%}")
            
        # Display feature importance if available
        if 'feature_importance' in strategy:
            st.subheader("Feature Importance")
            importance_df = pd.DataFrame(strategy['feature_importance'])
            st.bar_chart(importance_df.set_index('feature')['importance'])
            
        # Display hyperparameters if available
        if 'best_params' in strategy:
            st.subheader("Model Parameters")
            st.json(strategy['best_params'])

def update_data():
    """Fetch and update market data"""
    try:
        fetcher = OANDADataFetcher()
        data = fetcher.fetch_historical_data(days=90)
        st.session_state.data = fetcher.add_features(data)
        st.session_state.last_update = datetime.now()
        
        # Generate strategies if none exist
        if not st.session_state.strategies:
            with st.spinner("Training ML models..."):
                generator = StrategyGenerator()
                strategies = generator.generate_strategies(
                    st.session_state.data,
                    min_validation_accuracy=st.session_state.strategy_params['min_validation_accuracy']
                )
                if strategies:
                    st.session_state.strategies.extend(strategies)
                    st.success(f"Generated {len(strategies)} new trading strategies!")
        
        return True
    except Exception as e:
        st.error(f"Error updating data: {str(e)}")
        return False

def main():
    st.title("🤖 AI-Powered Forex Trading Agent")
    
    # Sidebar controls
    st.sidebar.title("Control Panel")
    
    # ML Parameters
    st.sidebar.subheader("Machine Learning Parameters")
    st.session_state.strategy_params['min_validation_accuracy'] = st.sidebar.slider(
        "Minimum Validation Accuracy",
        min_value=0.5,
        max_value=0.8,
        value=0.55,
        step=0.01,
        help="Minimum required accuracy for strategy validation"
    )
    
    st.session_state.strategy_params['lookback_periods'] = st.sidebar.slider(
        "Lookback Periods",
        min_value=6,
        max_value=24,
        value=12,
        step=1,
        help="Number of periods to look back for predictions"
    )
    
    st.session_state.strategy_params['confidence_threshold'] = st.sidebar.slider(
        "Signal Confidence Threshold",
        min_value=0.5,
        max_value=0.9,
        value=0.6,
        step=0.05,
        help="Minimum probability required for trade signals"
    )
    
    if st.sidebar.button("Update Data & Retrain"):
        with st.spinner("Fetching data and training models..."):
            update_data()
    
    # Auto-refresh every 5 minutes
    if st.sidebar.checkbox("Auto-refresh (5 min)", value=False):
        time.sleep(1)  # Prevent too frequent updates
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Display status
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        st.metric("Last Update", 
                 st.session_state.last_update.strftime('%H:%M:%S') if st.session_state.last_update else "Never")
    with status_col2:
        if st.session_state.data is not None:
            current_price = st.session_state.data['close'].iloc[-1]
            prev_price = st.session_state.data['close'].iloc[-2]
            price_change = ((current_price - prev_price) / prev_price) * 100
            st.metric("Current Price", f"{current_price:.5f}", f"{price_change:+.2f}%")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["Market Analysis", "ML Strategy Performance", "Live Trading"])
    
    with tab1:
        display_ml_process_info()
        if st.session_state.data is not None:
            st.plotly_chart(plot_candlestick_with_indicators(st.session_state.data), use_container_width=True)
            
            with st.expander("Latest Technical Indicators"):
                st.dataframe(st.session_state.data.tail())
        else:
            st.info("Click 'Update Data & Retrain' to start the ML process")
    
    with tab2:
        if st.session_state.strategies:
            st.subheader("Generated Trading Strategies")
            
            # Strategy selection for detailed view
            strategy_names = [f"{s['model_name']} ({s['created_at']})" for s in st.session_state.strategies]
            selected_strategy = st.selectbox(
                "Select Strategy for Detailed Analysis",
                strategy_names,
                key="strategy_analysis"
            )
            
            if selected_strategy:
                strategy_idx = strategy_names.index(selected_strategy)
                strategy = st.session_state.strategies[strategy_idx]
                
                # Display detailed metrics
                st.subheader(f"Strategy Analysis: {strategy['model_name']}")
                display_strategy_metrics(strategy)
                
                # Load and display visualizations
                viz_path = os.path.join('strategies', 'visualizations')
                if os.path.exists(viz_path):
                    plots = glob.glob(os.path.join(viz_path, f"{strategy['model_name']}_{strategy['created_at']}*.png"))
                    if plots:
                        st.subheader("Training Visualizations")
                        for plot in plots:
                            st.image(plot)
        else:
            st.info("No strategies generated yet. Update data to start the ML process.")
    
    with tab3:
        if not st.session_state.strategies:
            st.warning("No strategies available yet. Update data to generate strategies.")
        else:
            st.subheader("Live Trading Dashboard")
            
            # Strategy selection
            strategy_names = [f"{s['model_name']} ({s['created_at']})" for s in st.session_state.strategies]
            selected_strategy = st.selectbox(
                "Select Strategy",
                strategy_names,
                key="live_trading"
            )
            
            if selected_strategy:
                strategy_idx = strategy_names.index(selected_strategy)
                strategy = st.session_state.strategies[strategy_idx]
                
                # Display strategy details
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Model Type", strategy['model_name'])
                with col2:
                    st.metric("Validation Accuracy", f"{strategy['metrics']['accuracy']:.2%}")
                with col3:
                    st.metric("Created", strategy['created_at'])
                
                # Latest signal
                if st.session_state.data is not None:
                    data = st.session_state.data
                    if all(feature in data.columns for feature in strategy['features']):
                        latest_features = data[strategy['features']].iloc[-1]
                        
                        # Load and use strategy
                        generator = StrategyGenerator()
                        model, _ = generator.load_strategy(f"{strategy['model_name']}_{strategy['created_at']}")
                        prediction = model.predict([latest_features])[0]
                        
                        # Get prediction probability
                        prob = model.predict_proba([latest_features])[0]
                        confidence = max(prob)
                        
                        signal_map = {1: "BUY 📈", -1: "SELL 📉", 0: "HOLD ⏸️"}
                        signal = signal_map[prediction]
                        
                        # Display signal and metrics
                        signal_col1, signal_col2, signal_col3 = st.columns(3)
                        with signal_col1:
                            st.metric("Signal", signal)
                        with signal_col2:
                            st.metric("Confidence", f"{confidence:.2%}")
                        with signal_col3:
                            st.metric("Current Price", f"{data['close'].iloc[-1]:.5f}")
                        
                        # Market conditions
                        condition_col1, condition_col2, condition_col3 = st.columns(3)
                        with condition_col1:
                            rsi = data['RSI'].iloc[-1]
                            rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                            st.metric("RSI Status", rsi_status, f"{rsi:.1f}")
                        with condition_col2:
                            trend = "Bullish" if data['SMA_20'].iloc[-1] > data['SMA_50'].iloc[-1] else "Bearish"
                            st.metric("Trend", trend)
                        with condition_col3:
                            volatility = "High" if data['ATR'].iloc[-1] > data['ATR'].mean() else "Low"
                            st.metric("Volatility", volatility)
                        
                        st.markdown("---")
                        st.caption(f"Last updated: {data.index[-1].strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
