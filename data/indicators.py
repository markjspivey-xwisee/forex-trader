import pandas as pd
import numpy as np
import streamlit as st
from functools import lru_cache
import gc
from io import StringIO

class TechnicalIndicators:
    def __init__(self):
        self._cache = {}
        self._chunk_size = 1000  # Number of rows to process at a time
    
    @staticmethod
    @st.cache_data(ttl=300, max_entries=10)  # Cache for 5 minutes, limit entries
    def _process_chunk_cached(chunk_data_json):
        """Process a single chunk of data with caching"""
        # Convert JSON to DataFrame using StringIO
        chunk_data = pd.read_json(StringIO(chunk_data_json))
        df = chunk_data.copy()
        
        try:
            # Convert values to numpy arrays and ensure same length
            close_values = df['close'].values
            high_values = df['high'].values
            low_values = df['low'].values
            
            # Verify array lengths
            if not (len(close_values) == len(high_values) == len(low_values)):
                raise ValueError("Input arrays must have the same length")
            
            # Moving averages
            for period in [20, 50]:
                df[f'SMA_{period}'] = TechnicalIndicators._calculate_sma(
                    tuple(close_values.tolist()),
                    period
                )
            
            # Bollinger Bands
            df['BB_middle'] = TechnicalIndicators._calculate_sma(
                tuple(close_values.tolist()),
                20
            )
            std = TechnicalIndicators._calculate_std(
                tuple(close_values.tolist()),
                20
            )
            df['BB_upper'] = df['BB_middle'] + 2 * np.array(std)
            df['BB_lower'] = df['BB_middle'] - 2 * np.array(std)
            
            # MACD
            macd, signal = TechnicalIndicators._calculate_macd(
                tuple(close_values.tolist())
            )
            df['MACD'] = macd
            df['Signal_Line'] = signal
            
            # RSI
            df['RSI'] = TechnicalIndicators._calculate_rsi(
                tuple(close_values.tolist())
            )
            
            # ATR
            df['ATR'] = TechnicalIndicators._calculate_atr(
                tuple(high_values.tolist()),
                tuple(low_values.tolist()),
                tuple(close_values.tolist())
            )
            
            # Stochastic Oscillator
            k, d = TechnicalIndicators._calculate_stochastic(
                tuple(high_values.tolist()),
                tuple(low_values.tolist()),
                tuple(close_values.tolist())
            )
            df['Stoch_K'] = k
            df['Stoch_D'] = d
            
            gc.collect()  # Force garbage collection
            return df
            
        except Exception as e:
            st.error(f"Error in _process_chunk_cached: {str(e)}")
            raise
    
    def add_all_indicators(self, data):
        """Add all technical indicators to the dataset with memory management"""
        try:
            # Process data in chunks
            chunks = []
            for i in range(0, len(data), self._chunk_size):
                chunk = data.iloc[i:i + self._chunk_size].copy()
                # Convert chunk to JSON for caching
                chunk_json = chunk.to_json(date_format='iso')
                processed_chunk = self._process_chunk_cached(chunk_json)
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
        try:
            series = pd.Series(values)
            result = series.rolling(window=window).mean()
            return result.values.tolist()
        except Exception as e:
            st.error(f"Error in _calculate_sma: {str(e)}")
            return [0] * len(values)
    
    @staticmethod
    @lru_cache(maxsize=128)
    def _calculate_std(values, window):
        """Calculate Standard Deviation with caching"""
        try:
            series = pd.Series(values)
            result = series.rolling(window=window).std()
            return result.values.tolist()
        except Exception as e:
            st.error(f"Error in _calculate_std: {str(e)}")
            return [0] * len(values)
    
    @staticmethod
    @lru_cache(maxsize=128)
    def _calculate_macd(values):
        """Calculate MACD with caching"""
        try:
            series = pd.Series(values)
            exp1 = series.ewm(span=12, adjust=False).mean()
            exp2 = series.ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            return macd.values.tolist(), signal.values.tolist()
        except Exception as e:
            st.error(f"Error in _calculate_macd: {str(e)}")
            return [0] * len(values), [0] * len(values)
    
    @staticmethod
    @lru_cache(maxsize=128)
    def _calculate_rsi(values):
        """Calculate RSI with caching"""
        try:
            series = pd.Series(values)
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            result = 100 - (100 / (1 + rs))
            return result.values.tolist()
        except Exception as e:
            st.error(f"Error in _calculate_rsi: {str(e)}")
            return [50] * len(values)  # Neutral RSI value
    
    @staticmethod
    @lru_cache(maxsize=128)
    def _calculate_atr(high_values, low_values, close_values):
        """Calculate ATR with caching"""
        try:
            if not (len(high_values) == len(low_values) == len(close_values)):
                raise ValueError("Input arrays must have the same length")
            
            high = pd.Series(high_values)
            low = pd.Series(low_values)
            close = pd.Series(close_values)
            
            high_low = high - low
            high_close = np.abs(high - close.shift())
            low_close = np.abs(low - close.shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            result = true_range.rolling(14).mean()
            return result.values.tolist()
        except Exception as e:
            st.error(f"Error in _calculate_atr: {str(e)}")
            return [0] * len(high_values)
    
    @staticmethod
    @lru_cache(maxsize=128)
    def _calculate_stochastic(high_values, low_values, close_values):
        """Calculate Stochastic Oscillator with caching"""
        try:
            if not (len(high_values) == len(low_values) == len(close_values)):
                raise ValueError("Input arrays must have the same length")
            
            high = pd.Series(high_values)
            low = pd.Series(low_values)
            close = pd.Series(close_values)
            
            low_14 = low.rolling(window=14).min()
            high_14 = high.rolling(window=14).max()
            
            # Handle division by zero
            denominator = high_14 - low_14
            k = np.where(
                denominator != 0,
                ((close - low_14) / denominator) * 100,
                50  # Default to neutral value when denominator is zero
            )
            k_series = pd.Series(k)
            d = k_series.rolling(window=3).mean()
            
            return k_series.values.tolist(), d.values.tolist()
        except Exception as e:
            st.error(f"Error in _calculate_stochastic: {str(e)}")
            return [50] * len(high_values), [50] * len(high_values)  # Neutral values
    
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
