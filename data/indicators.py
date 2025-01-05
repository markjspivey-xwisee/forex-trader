import pandas as pd
import numpy as np
import streamlit as st
from functools import lru_cache
import gc

class TechnicalIndicators:
    def __init__(self):
        self._cache = {}
        self._chunk_size = 1000  # Number of rows to process at a time
    
    @st.cache_data(ttl=300, max_entries=10)  # Cache for 5 minutes, limit entries
    def add_all_indicators(self, data):
        """Add all technical indicators to the dataset with memory management"""
        try:
            # Process data in chunks
            chunks = []
            for i in range(0, len(data), self._chunk_size):
                chunk = data.iloc[i:i + self._chunk_size].copy()
                chunk = self._process_chunk(chunk)
                chunks.append(chunk)
                gc.collect()  # Force garbage collection
            
            # Combine chunks
            result = pd.concat(chunks)
            
            # Drop any NaN values that resulted from the calculations
            result = result.dropna()
            
            return result
            
        except Exception as e:
            st.error(f"Error calculating indicators: {str(e)}")
            return data
    
    def _process_chunk(self, df):
        """Process a single chunk of data"""
        # Moving averages
        for period in [20, 50]:
            df[f'SMA_{period}'] = self._calculate_sma(df['close'], period)
        
        # Bollinger Bands
        df['BB_middle'] = self._calculate_sma(df['close'], 20)
        std = self._calculate_std(df['close'], 20)
        df['BB_upper'] = df['BB_middle'] + 2 * std
        df['BB_lower'] = df['BB_middle'] - 2 * std
        
        # MACD
        df['MACD'], df['Signal_Line'] = self._calculate_macd(df['close'])
        
        # RSI
        df['RSI'] = self._calculate_rsi(df['close'])
        
        # ATR
        df['ATR'] = self._calculate_atr(df)
        
        # Stochastic Oscillator
        df['Stoch_K'], df['Stoch_D'] = self._calculate_stochastic(df)
        
        return df
    
    @lru_cache(maxsize=128)
    def _calculate_sma(self, series, window):
        """Calculate Simple Moving Average with caching"""
        # Convert series to tuple for caching
        values = tuple(series)
        result = pd.Series(values).rolling(window=window).mean()
        gc.collect()  # Force garbage collection
        return result
    
    @lru_cache(maxsize=128)
    def _calculate_std(self, series, window):
        """Calculate Standard Deviation with caching"""
        values = tuple(series)
        result = pd.Series(values).rolling(window=window).std()
        gc.collect()  # Force garbage collection
        return result
    
    @lru_cache(maxsize=128)
    def _calculate_macd(self, series):
        """Calculate MACD with caching"""
        values = tuple(series)
        series = pd.Series(values)
        exp1 = series.ewm(span=12, adjust=False).mean()
        exp2 = series.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        gc.collect()  # Force garbage collection
        return macd, signal
    
    @lru_cache(maxsize=128)
    def _calculate_rsi(self, series):
        """Calculate RSI with caching"""
        values = tuple(series)
        series = pd.Series(values)
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        result = 100 - (100 / (1 + rs))
        gc.collect()  # Force garbage collection
        return result
    
    @lru_cache(maxsize=128)
    def _calculate_atr(self, df):
        """Calculate ATR with caching"""
        high = tuple(df['high'])
        low = tuple(df['low'])
        close = tuple(df['close'])
        
        df_temp = pd.DataFrame({
            'high': high,
            'low': low,
            'close': close
        })
        
        high_low = df_temp['high'] - df_temp['low']
        high_close = np.abs(df_temp['high'] - df_temp['close'].shift())
        low_close = np.abs(df_temp['low'] - df_temp['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        result = true_range.rolling(14).mean()
        
        del df_temp, ranges  # Explicitly delete temporary objects
        gc.collect()  # Force garbage collection
        
        return result
    
    @lru_cache(maxsize=128)
    def _calculate_stochastic(self, df):
        """Calculate Stochastic Oscillator with caching"""
        high = tuple(df['high'])
        low = tuple(df['low'])
        close = tuple(df['close'])
        
        df_temp = pd.DataFrame({
            'high': high,
            'low': low,
            'close': close
        })
        
        low_14 = df_temp['low'].rolling(window=14).min()
        high_14 = df_temp['high'].rolling(window=14).max()
        k = ((df_temp['close'] - low_14) / (high_14 - low_14)) * 100
        d = k.rolling(window=3).mean()
        
        del df_temp  # Explicitly delete temporary object
        gc.collect()  # Force garbage collection
        
        return k, d
    
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
