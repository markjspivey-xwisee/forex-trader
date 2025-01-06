import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import json
from io import StringIO

class DataFetcher:
    def __init__(self, chunk_size=1000):
        self._chunk_size = chunk_size
    
    @staticmethod
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def _fetch_data_cached(days):
        """Fetch data with caching"""
        try:
            # Generate sample data for now
            # In production, this would fetch from OANDA API
            dates = pd.date_range(
                end=datetime.now(),
                periods=days * 24 * 60,  # 1-minute intervals
                freq='1min'
            )
            
            data = pd.DataFrame(index=dates)
            
            # Generate realistic price movements
            base_price = 1.10  # EUR/USD typical price
            volatility = 0.0002  # Typical forex volatility
            
            # Random walk for prices
            changes = np.random.normal(0, volatility, len(dates))
            prices = np.exp(np.cumsum(changes)) * base_price
            
            # Generate OHLC data
            data['open'] = prices
            data['close'] = prices * (1 + np.random.normal(0, volatility/2, len(dates)))
            data['high'] = np.maximum(data['open'], data['close']) * (1 + abs(np.random.normal(0, volatility/2, len(dates))))
            data['low'] = np.minimum(data['open'], data['close']) * (1 - abs(np.random.normal(0, volatility/2, len(dates))))
            
            # Generate volume data
            base_volume = 100000  # Base volume
            volume_volatility = 0.3  # Volume volatility
            data['volume'] = np.random.lognormal(np.log(base_volume), volume_volatility, len(dates))
            
            # Convert to JSON for caching
            return data.to_json(date_format='iso')
            
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return None
    
    def fetch_data(self, days=60):
        """Fetch market data with chunking"""
        try:
            # Get data from cache
            data_json = self._fetch_data_cached(days)
            if data_json is None:
                return None
            
            # Process data in chunks
            data = pd.read_json(StringIO(data_json))  # Use StringIO to wrap JSON string
            processed_chunks = []
            
            for i in range(0, len(data), self._chunk_size):
                chunk = data.iloc[i:i + self._chunk_size].copy()
                processed_chunks.append(chunk)
            
            # Combine chunks
            return pd.concat(processed_chunks)
            
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            return None
    
    def clear_cache(self):
        """Clear data cache"""
        st.cache_data.clear()
