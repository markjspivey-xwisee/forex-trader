import pandas as pd
import numpy as np
import streamlit as st
from functools import lru_cache
import gc

class TechnicalIndicators:
    def __init__(self):
        self._cache = {}
        self._chunk_size = 1000  # Number of rows to process at a time
    
    @staticmethod
    @st.cache_data(ttl=300, max_entries=10)  # Cache for 5 minutes, limit entries
    def _process_chunk_cached(chunk_data, chunk_size):
        """Process a single chunk of data with caching"""
        df = chunk_data.copy()
        
        # Moving averages
        for period in [20, 50]:
            df[f'SMA_{period}'] = TechnicalIndicators._calculate_sma(df['close'].values, period)
        
        # Bollinger Bands
        df['BB_middle'] = TechnicalIndicators._calculate_sma(df['close'].values, 20)
        std = TechnicalIndicators._calculate_std(df['close'].values, 20)
        df['BB_upper'] = df['BB_middle'] + 2 * std
        df['BB_lower'] = df['BB_middle'] - 2 * std
        
        # MACD
        df['MACD'], df['Signal_Line'] = TechnicalIndicators._calculate_macd(df['close'].values)
        
        # RSI
        df['RSI'] = TechnicalIndicators._calculate_rsi(df['close'].values)
        
        # ATR
        df['ATR'] = TechnicalIndicators._calculate_atr(
            df['high'].values,
            df['low'].values,
            df['close'].values
        )
        
        # Stochastic Oscillator
        df['Stoch_K'], df['Stoch_D'] = TechnicalIndicators._calculate_stochastic(
            df['high'].values,
            df['low'].values,
            df['close'].values
        )
        
        gc.collect()  # Force garbage collection
        return df
    
    def add_all_indicators(self, data):
        """Add all technical indicators to the dataset with memory management"""
        try:
            # Process data in chunks
            chunks = []
            for i in range(0, len(data), self._chunk_size):
                chunk = data.iloc[i:i + self._chunk_size].copy()
                processed_chunk = self._process_chunk_cached(chunk, self._chunk_size)
                chunks.append(processed_chunk)
                gc.collect()  # Force garbage collection
            
            # Combine chunks
            result = pd.concat(chunks)
            
            # Drop any NaN values that resulted from the calculations
            result = result.dropna()
            
            return result
            
        except Exception as e:
            st.error(f"Error calculating indicators: {str(e)}")
            return data
    
    @staticmethod
    @lru_cache(maxsize=128)
    def _calculate_sma(values, window):
        """Calculate Simple Moving Average with caching"""
        series = pd.Series(values)
        return series.rolling(window=window).mean().values
    
    @staticmethod
    @lru_cache(maxsize=128)
    def _calculate_std(values, window):
        """Calculate Standard Deviation with caching"""
        series = pd.Series(values)
        return series.rolling(window=window).std().values
    
    @staticmethod
    @lru_cache(maxsize=128)
    def _calculate_macd(values):
        """Calculate MACD with caching"""
        series = pd.Series(values)
        exp1 = series.ewm(span=12, adjust=False).mean()
        exp2 = series.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd.values, signal.values
    
    @staticmethod
    @lru_cache(maxsize=128)
    def _calculate_rsi(values):
        """Calculate RSI with caching"""
        series = pd.Series(values)
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        return (100 - (100 / (1 + rs))).values
    
    @staticmethod
    @lru_cache(maxsize=128)
    def _calculate_atr(high_values, low_values, close_values):
        """Calculate ATR with caching"""
        high = pd.Series(high_values)
        low = pd.Series(low_values)
        close = pd.Series(close_values)
        
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(14).mean().values
    
    @staticmethod
    @lru_cache(maxsize=128)
    def _calculate_stochastic(high_values, low_values, close_values):
        """Calculate Stochastic Oscillator with caching"""
        high = pd.Series(high_values)
        low = pd.Series(low_values)
        close = pd.Series(close_values)
        
        low_14 = low.rolling(window=14).min()
        high_14 = high.rolling(window=14).max()
        k = ((close - low_14) / (high_14 - low_14)) * 100
        d = k.rolling(window=3).mean()
        return k.values, d.values
    
    def clear_cache(self):
        """Clear all caches and force garbage collection"""
        self._cache.clear()
        # Clear all lru_cache decorators
        self._calculate_sma.cache_clear()
        self._calculate_std.cache_clear()
        self._calculate_macd.cache_clear()
        self._calculate_rsi.cache_clear()
        self._calculate_atr.cache_clear()
        self._calculate_stochastic.cache_clear()
        gc.collect()
