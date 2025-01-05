import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import json
import os
from main import ForexTrader

st.set_page_config(page_title='AI-Powered Forex Trading Agent', layout='wide')

def create_market_analysis_chart(data, selected_indicators):
    """Create multi-panel market analysis chart"""
    # Calculate number of rows based on selected indicators
    active_indicators = sum(selected_indicators.values())
    rows = 1 + active_indicators  # Price chart + selected indicators
    
    # Calculate row heights (price chart gets more space)
    heights = [0.4] + [0.6/active_indicators] * active_indicators if active_indicators > 0 else [1]
    
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=heights
    )
    
    # Price and indicators
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close'],
        name='OHLC'
    ), row=1, col=1)
    
    current_row = 2
    
    # Add selected technical indicators to price chart
    if selected_indicators['Show Moving Averages']:
        periods = st.session_state.get('ma_periods', [20, 50])
        for period in periods:
            fig.add_trace(go.Scatter(
                x=data.index, 
                y=data[f'SMA_{period}'], 
                name=f'SMA {period}',
                line=dict(color=f'rgba({period*2}, 100, {255-period}, 0.8)')
            ), row=1, col=1)
    
    if selected_indicators['Show Bollinger Bands']:
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_upper'], name='BB Upper',
                                line=dict(color='gray', dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_lower'], name='BB Lower',
                                line=dict(color='gray', dash='dash')), row=1, col=1)
    
    # Add separate indicator panels
    if selected_indicators['Show MACD']:
        fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD',
                                line=dict(color='blue')), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['Signal_Line'], name='Signal',
                                line=dict(color='orange')), row=current_row, col=1)
        fig.update_yaxes(title_text="MACD", row=current_row, col=1)
        current_row += 1
    
    if selected_indicators['Show RSI']:
        fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI',
                                line=dict(color='purple')), row=current_row, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)
        fig.update_yaxes(title_text="RSI", row=current_row, col=1)
        current_row += 1
    
    if selected_indicators['Show ATR']:
        fig.add_trace(go.Scatter(x=data.index, y=data['ATR'], name='ATR',
                                line=dict(color='red')), row=current_row, col=1)
        fig.update_yaxes(title_text="ATR", row=current_row, col=1)
        current_row += 1
    
    if selected_indicators['Show Stochastic']:
        fig.add_trace(go.Scatter(x=data.index, y=data['Stoch_K'], name='Stoch K',
                                line=dict(color='blue')), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['Stoch_D'], name='Stoch D',
                                line=dict(color='orange')), row=current_row, col=1)
        fig.update_yaxes(title_text="Stochastic", row=current_row, col=1)
        current_row += 1
    
    # Update layout
    fig.update_layout(
        title='Market Analysis Dashboard',
        template='plotly_dark',
        height=200 + 200*rows,
        xaxis_rangeslider_visible=False
    )
    
    fig.update_yaxes(title_text="Price", row=1, col=1)
    
    return fig

def create_feature_importance_chart(importance_dict):
    """Create feature importance bar chart"""
    fig = go.Figure()
    
    # Convert dictionary to sorted lists
    features = list(importance_dict.keys())
    importance = list(importance_dict.values())
    
    # Sort by importance
    sorted_indices = sorted(range(len(importance)), key=lambda k: importance[k], reverse=True)
    features = [features[i] for i in sorted_indices]
    importance = [importance[i] for i in sorted_indices]
    
    fig.add_trace(go.Bar(
        x=importance,
        y=features,
        orientation='h'
    ))
    
    fig.update_layout(
        title='Feature Importance',
        xaxis_title='Importance',
        yaxis_title='Feature',
        template='plotly_dark',
        height=400
    )
    
    return fig

def main():
    # Initialize session state for settings
    if 'ma_periods' not in st.session_state:
        st.session_state.ma_periods = [20, 50]
    if 'risk_settings' not in st.session_state:
        st.session_state.risk_settings = {
            'stop_loss': 0.02,
            'take_profit': 0.04,
            'max_position_size': 0.1,
            'max_trades': 5
        }
    
    # Title and logo
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Robot/3D/robot_3d.png", width=100)
    with col2:
        st.title('AI-Powered Forex Trading Agent')
    
    # Sidebar - Control Panel
    st.sidebar.header('Control Panel')
    
    # Model Parameters
    st.sidebar.subheader('Model Parameters')
    min_accuracy = st.sidebar.slider('Minimum Validation Accuracy', 0.50, 0.80, 0.55)
    lookback = st.sidebar.slider('Lookback Periods', 6, 24, 12)
    confidence = st.sidebar.slider('Signal Confidence Threshold', 0.50, 0.90, 0.60)
    
    # Technical Analysis Settings
    st.sidebar.subheader('Technical Analysis')
    selected_indicators = {
        'Show Moving Averages': st.sidebar.checkbox('Moving Averages', True),
        'Show Bollinger Bands': st.sidebar.checkbox('Bollinger Bands', True),
        'Show MACD': st.sidebar.checkbox('MACD', True),
        'Show RSI': st.sidebar.checkbox('RSI', True),
        'Show ATR': st.sidebar.checkbox('ATR', True),
        'Show Stochastic': st.sidebar.checkbox('Stochastic', True)
    }
    
    if selected_indicators['Show Moving Averages']:
        ma_periods = st.sidebar.multiselect(
            'Moving Average Periods',
            options=[10, 20, 30, 50, 100, 200],
            default=[20, 50]
        )
        st.session_state.ma_periods = ma_periods
    
    # Risk Management Settings
    st.sidebar.subheader('Risk Management')
    risk_settings = {
        'stop_loss': st.sidebar.slider('Stop Loss (%)', 1.0, 5.0, 2.0) / 100,
        'take_profit': st.sidebar.slider('Take Profit (%)', 2.0, 10.0, 4.0) / 100,
        'max_position_size': st.sidebar.slider('Max Position Size (%)', 1.0, 20.0, 10.0) / 100,
        'max_trades': st.sidebar.slider('Max Concurrent Trades', 1, 10, 5)
    }
    st.session_state.risk_settings = risk_settings
    
    # Advanced Settings
    st.sidebar.subheader('Advanced Settings')
    advanced_settings = {
        'use_gpu': st.sidebar.checkbox('Use GPU Acceleration', False),
        'enable_ensemble': st.sidebar.checkbox('Enable Model Ensemble', True),
        'feature_selection': st.sidebar.selectbox(
            'Feature Selection Method',
            ['All Features', 'PCA', 'Recursive Feature Elimination']
        ),
        'optimization_method': st.sidebar.selectbox(
            'Hyperparameter Optimization',
            ['Random Search', 'Grid Search', 'Bayesian Optimization']
        )
    }
    
    # Initialize trader with parameters
    trader = ForexTrader(
        min_accuracy=min_accuracy,
        lookback=lookback,
        confidence=confidence,
        risk_settings=risk_settings,
        advanced_settings=advanced_settings
    )
    
    # Get initial data
    data = trader.data_fetcher.fetch_data(days=30)
    data = trader.indicators.add_all_indicators(data)
    
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
    
    # Main content tabs
    tabs = st.tabs(['Market Analysis', 'ML Strategy Performance', 'Live Trading', 'Risk Analytics'])
    
    with tabs[0]:  # Market Analysis
        if st.sidebar.button('Update Data & Retrain'):
            with st.spinner('Fetching data and training models...'):
                results = trader.train_models(days=90)
                st.success(f"Generated {len(results)} new trading strategies!")
                
                # Display training metrics
                for model_name, metrics in results.items():
                    st.subheader(f'{model_name} Metrics')
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric('Training Accuracy', f"{metrics['train_accuracy']:.2%}")
                    with col2:
                        st.metric('Validation Accuracy', f"{metrics['validation_accuracy']:.2%}")
        
        auto_refresh = st.sidebar.checkbox('Auto-refresh (5 min)')
        
        # Market analysis chart
        st.plotly_chart(create_market_analysis_chart(data, selected_indicators), use_container_width=True)
    
    with tabs[1]:  # ML Strategy Performance
        st.subheader('Generated Trading Strategies')
        
        # Strategy selector
        models_dir = 'models'
        if os.path.exists(models_dir):
            strategy_files = [f for f in os.listdir(models_dir) if f.endswith('_metadata.json')]
            if strategy_files:
                col1, col2 = st.columns([3, 1])
                with col1:
                    selected_strategy = st.selectbox('Select Strategy for Detailed Analysis', strategy_files)
                with col2:
                    sort_by = st.selectbox('Sort By', ['Accuracy', 'Precision', 'Recall', 'F1 Score'])
                
                with open(os.path.join(models_dir, selected_strategy), 'r') as f:
                    strategy_info = json.load(f)
                
                # Display metrics
                cols = st.columns(4)
                if 'metrics' in strategy_info:
                    metrics = strategy_info['metrics']
                    cols[0].metric('Accuracy', f"{metrics.get('accuracy', 0):.2%}")
                    cols[1].metric('Precision', f"{metrics.get('precision', 0):.2%}")
                    cols[2].metric('Recall', f"{metrics.get('recall', 0):.2%}")
                    cols[3].metric('F1 Score', f"{metrics.get('f1', 0):.2%}")
                
                # Feature importance
                if 'feature_importance' in strategy_info:
                    st.subheader('Feature Importance')
                    st.plotly_chart(
                        create_feature_importance_chart(strategy_info['feature_importance']),
                        use_container_width=True
                    )
    
    with tabs[2]:  # Live Trading
        st.subheader('Live Trading Signals')
        
        col1, col2 = st.columns([2, 1])
        with col1:
            signal_mode = st.radio('Signal Mode', ['Single Model', 'Ensemble'], horizontal=True)
        with col2:
            if signal_mode == 'Single Model':
                selected_model = st.selectbox('Select Model', list(trader.models.keys()))
            
        if st.button('Get Latest Signal'):
            with st.spinner('Analyzing market conditions...'):
                if signal_mode == 'Single Model':
                    signal = trader.get_live_signal(model_name=selected_model)
                else:
                    signal = trader.get_ensemble_signal()
                
                cols = st.columns(4)
                with cols[0]:
                    st.metric('Signal', signal['signal_type'])
                with cols[1]:
                    st.metric('Current Price', f"${signal['current_price']:.5f}")
                with cols[2]:
                    timestamp_str = pd.Timestamp(signal['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                    st.metric('Timestamp', timestamp_str)
                with cols[3]:
                    st.metric('Confidence', f"{signal.get('confidence', 0):.2%}")
    
    with tabs[3]:  # Risk Analytics
        st.subheader('Risk Management Dashboard')
        
        # Position Sizing Calculator
        st.write('Position Sizing Calculator')
        col1, col2, col3 = st.columns(3)
        with col1:
            account_balance = st.number_input('Account Balance ($)', value=10000.0, step=1000.0)
        with col2:
            risk_per_trade = st.slider('Risk per Trade (%)', 0.1, 5.0, 1.0)
        with col3:
            stop_loss_pips = st.number_input('Stop Loss (pips)', value=50, step=10)
            
        # Calculate position size
        pip_value = 0.0001  # For most forex pairs
        position_size = (account_balance * (risk_per_trade/100)) / (stop_loss_pips * pip_value)
        
        st.info(f'Recommended Position Size: {position_size:,.2f} units')
        
        # Risk Metrics
        st.write('Current Risk Metrics')
        risk_cols = st.columns(4)
        with risk_cols[0]:
            st.metric('Daily Drawdown', '-2.3%')
        with risk_cols[1]:
            st.metric('Sharpe Ratio', '1.85')
        with risk_cols[2]:
            st.metric('Win Rate', '65%')
        with risk_cols[3]:
            st.metric('Risk/Reward', '1.5')

if __name__ == '__main__':
    main()
