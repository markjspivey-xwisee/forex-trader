import pandas as pd
import numpy as np
import streamlit as st
import gc

class TechnicalIndicators:
    def __init__(self, chunk_size=1000):
        self._chunk_size = chunk_size
        self._cache = {}
    
    def _calculate_sma(self, data, period):
        """Calculate Simple Moving Average"""
        return data['close'].rolling(window=period).mean()
    
    def _calculate_bollinger_bands(self, data, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = self._calculate_sma(data, period)
        std = data['close'].rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band
    
    def _calculate_macd(self, data, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        exp1 = data['close'].ewm(span=fast, adjust=False).mean()
        exp2 = data['close'].ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line
    
    def _calculate_rsi(self, data, period=14):
        """Calculate RSI"""
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_atr(self, data, period=14):
        """Calculate Average True Range"""
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift())
        low_close = abs(data['low'] - data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(window=period).mean()
    
    def _calculate_stochastic(self, data, k_period=14, d_period=3):
        """Calculate Stochastic Oscillator"""
        low_min = data['low'].rolling(window=k_period).min()
        high_max = data['high'].rolling(window=k_period).max()
        k = 100 * ((data['close'] - low_min) / (high_max - low_min))
        d = k.rolling(window=d_period).mean()
        return k, d
    
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def _process_chunk_cached(self, chunk_data_json):
        """Process a chunk of data with caching"""
        try:
            # Convert JSON to DataFrame
            chunk_data = pd.read_json(chunk_data_json)
            
            # Calculate indicators
            chunk_data['SMA_20'] = self._calculate_sma(chunk_data, 20)
            chunk_data['SMA_50'] = self._calculate_sma(chunk_data, 50)
            
            bb_upper, bb_lower = self._calculate_bollinger_bands(chunk_data)
            chunk_data['BB_upper'] = bb_upper
            chunk_data['BB_lower'] = bb_lower
            
            macd, signal = self._calculate_macd(chunk_data)
            chunk_data['MACD'] = macd
            chunk_data['Signal_Line'] = signal
            
            chunk_data['RSI'] = self._calculate_rsi(chunk_data)
            chunk_data['ATR'] = self._calculate_atr(chunk_data)
            
            k, d = self._calculate_stochastic(chunk_data)
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
                
                processed_chunk = pd.read_json(processed_chunk_json)
                processed_chunks.append(processed_chunk)
                
                # Force garbage collection after each chunk
                gc.collect()
            
            # Combine chunks
            result = pd.concat(processed_chunks)
            
            # Drop any NaN values
            result = result.dropna()
            
            return result
            
        except Exception as e:
            st.error(f"Error adding indicators: {str(e)}")
            return None
    
    def clear_cache(self):
        """Clear indicator cache"""
        self._cache.clear()
