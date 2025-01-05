import pandas as pd
import numpy as np
import streamlit as st
from functools import lru_cache

class TechnicalIndicators:
    def __init__(self):
        self._cache = {}
    
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def add_all_indicators(self, data):
        """Add all technical indicators to the dataset"""
        df = data.copy()
        
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
        
        # Drop any NaN values that resulted from the calculations
        return df.dropna()
    
    @lru_cache(maxsize=128)
    def _calculate_sma(self, series, window):
        """Calculate Simple Moving Average with caching"""
        # Convert series to tuple for caching
        values = tuple(series)
        return pd.Series(values).rolling(window=window).mean()
    
    @lru_cache(maxsize=128)
    def _calculate_std(self, series, window):
        """Calculate Standard Deviation with caching"""
        values = tuple(series)
        return pd.Series(values).rolling(window=window).std()
    
    @lru_cache(maxsize=128)
    def _calculate_macd(self, series):
        """Calculate MACD with caching"""
        values = tuple(series)
        series = pd.Series(values)
        exp1 = series.ewm(span=12, adjust=False).mean()
        exp2 = series.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
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
        return 100 - (100 / (1 + rs))
    
    @lru_cache(maxsize=128)
    def _calculate_atr(self, df):
        """Calculate ATR with caching"""
        high = tuple(df['high'])
        low = tuple(df['low'])
        close = tuple(df['close'])
        
        df = pd.DataFrame({
            'high': high,
            'low': low,
            'close': close
        })
        
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(14).mean()
    
    @lru_cache(maxsize=128)
    def _calculate_stochastic(self, df):
        """Calculate Stochastic Oscillator with caching"""
        high = tuple(df['high'])
        low = tuple(df['low'])
        close = tuple(df['close'])
        
        df = pd.DataFrame({
            'high': high,
            'low': low,
            'close': close
        })
        
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        k = ((df['close'] - low_14) / (high_14 - low_14)) * 100
        d = k.rolling(window=3).mean()
        return k, d
