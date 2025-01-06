import pandas as pd
import numpy as np
import streamlit as st
import gc
from io import StringIO

class TechnicalIndicators:
    def __init__(self, chunk_size=500):  # Match DataFetcher's chunk size
        self._chunk_size = chunk_size
    
    def _calculate_sma(self, data, period):
        """Calculate Simple Moving Average"""
        return data['close'].rolling(window=period, min_periods=1).mean()
    
    def _calculate_bollinger_bands(self, data, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = self._calculate_sma(data, period)
        std = data['close'].rolling(window=period, min_periods=1).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band
    
    def _calculate_macd(self, data, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        exp1 = data['close'].ewm(span=fast, adjust=False, min_periods=1).mean()
        exp2 = data['close'].ewm(span=slow, adjust=False, min_periods=1).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False, min_periods=1).mean()
        return macd, signal_line
    
    def _calculate_rsi(self, data, period=14):
        """Calculate RSI"""
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_atr(self, data, period=14):
        """Calculate Average True Range"""
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift())
        low_close = abs(data['low'] - data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(window=period, min_periods=1).mean()
    
    def _calculate_stochastic(self, data, k_period=14, d_period=3):
        """Calculate Stochastic Oscillator"""
        low_min = data['low'].rolling(window=k_period, min_periods=1).min()
        high_max = data['high'].rolling(window=k_period, min_periods=1).max()
        k = 100 * ((data['close'] - low_min) / (high_max - low_min))
        d = k.rolling(window=d_period, min_periods=1).mean()
        return k, d
    
    @staticmethod
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def _process_chunk_cached(chunk_data_json):
        """Process a chunk of data with caching"""
        try:
            # Convert JSON to DataFrame using StringIO
            chunk_data = pd.read_json(StringIO(chunk_data_json))
            
            # Calculate indicators efficiently
            close = chunk_data['close']
            
            # SMA
            sma_20 = close.rolling(window=20, min_periods=1).mean()
            sma_50 = close.rolling(window=50, min_periods=1).mean()
            
            # Bollinger Bands
            std = close.rolling(window=20, min_periods=1).std()
            bb_upper = sma_20 + (std * 2)
            bb_lower = sma_20 - (std * 2)
            
            # MACD
            exp1 = close.ewm(span=12, adjust=False, min_periods=1).mean()
            exp2 = close.ewm(span=26, adjust=False, min_periods=1).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False, min_periods=1).mean()
            
            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # ATR
            high_low = chunk_data['high'] - chunk_data['low']
            high_close = abs(chunk_data['high'] - close.shift())
            low_close = abs(chunk_data['low'] - close.shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=14, min_periods=1).mean()
            
            # Stochastic
            low_min = chunk_data['low'].rolling(window=14, min_periods=1).min()
            high_max = chunk_data['high'].rolling(window=14, min_periods=1).max()
            k = 100 * ((close - low_min) / (high_max - low_min))
            d = k.rolling(window=3, min_periods=1).mean()
            
            # Add indicators to DataFrame
            chunk_data['SMA_20'] = sma_20
            chunk_data['SMA_50'] = sma_50
            chunk_data['BB_upper'] = bb_upper
            chunk_data['BB_lower'] = bb_lower
            chunk_data['MACD'] = macd
            chunk_data['Signal_Line'] = signal
            chunk_data['RSI'] = rsi
            chunk_data['ATR'] = atr
            chunk_data['Stoch_K'] = k
            chunk_data['Stoch_D'] = d
            
            # Convert back to JSON for caching
            return chunk_data.to_json(date_format='iso')
            
        except Exception as e:
            st.error(f"Error processing chunk: {str(e)}")
            return None
    
    def add_all_indicators(self, data):
        """Add all technical indicators to the data"""
        try:
            processed_chunks = []
            
            # Process data in chunks
            for i in range(0, len(data), self._chunk_size):
                chunk = data.iloc[i:i + self._chunk_size].copy()
                
                # Process chunk with caching
                chunk_json = chunk.to_json(date_format='iso')
                processed_chunk_json = self._process_chunk_cached(chunk_json)
                
                if processed_chunk_json is None:
                    return None
                
                processed_chunk = pd.read_json(StringIO(processed_chunk_json))
                processed_chunks.append(processed_chunk)
                
                # Force garbage collection after each chunk
                gc.collect()
            
            # Combine chunks
            result = pd.concat(processed_chunks)
            
            # Fill any NaN values with forward fill then backward fill
            result = result.fillna(method='ffill').fillna(method='bfill')
            
            return result
            
        except Exception as e:
            st.error(f"Error adding indicators: {str(e)}")
            return None
    
    def clear_cache(self):
        """Clear indicator cache"""
        st.cache_data.clear()
