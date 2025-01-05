import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import streamlit as st
import gc
from io import StringIO

class DataFetcher:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('OANDA_API_KEY')
        self.account_id = os.getenv('OANDA_ACCOUNT_ID')
        self.instrument = "EUR_USD"
        self.api = API(access_token=self.api_key) if self.api_key else None
        self._cache = {}
        self._cache_duration = timedelta(minutes=5)  # Cache data for 5 minutes
        self._chunk_size = 1000  # Number of rows to process at a time
    
    @staticmethod
    @st.cache_data(ttl=300, max_entries=10)  # Cache for 5 minutes, limit entries
    def _fetch_data_cached(days, chunk_size):
        """Static cached method for fetching data"""
        try:
            # Generate dates
            dates = pd.date_range(end=datetime.now(), periods=days*24, freq='h')
            
            # Generate random data with consistent lengths
            size = len(dates)
            base_price = 1.1
            volatility = 0.02
            
            # Generate price data ensuring high > low
            opens = np.random.normal(base_price, volatility, size)
            closes = np.random.normal(base_price, volatility, size)
            
            # Generate highs and lows ensuring proper order
            price_ranges = np.abs(np.random.normal(0, volatility, size))
            highs = np.maximum(opens, closes) + price_ranges/2
            lows = np.minimum(opens, closes) - price_ranges/2
            
            # Create DataFrame with consistent data
            data = pd.DataFrame({
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'volume': np.random.randint(1000, 5000, size)
            }, index=dates)
            
            # Verify data consistency
            if not (len(data['open']) == len(data['high']) == len(data['low']) == len(data['close']) == len(data['volume'])):
                raise ValueError("Generated data arrays have inconsistent lengths")
            
            if not all(data['high'] >= data['low']):
                raise ValueError("High prices must be greater than or equal to low prices")
            
            # Process in chunks
            chunks = []
            for i in range(0, len(data), chunk_size):
                chunk = data.iloc[i:i + chunk_size].copy()
                chunks.append(chunk)
                gc.collect()  # Force garbage collection
            
            result = pd.concat(chunks)
            
            # Convert to JSON for caching
            return result.to_json(date_format='iso')
            
        except Exception as e:
            st.error(f"Error generating data: {str(e)}")
            raise
    
    def fetch_data(self, days=30):
        """Fetch historical forex data with caching"""
        try:
            cache_key = f"{self.instrument}_{days}"
            current_time = datetime.now()
            
            # Check cache
            if cache_key in self._cache:
                cached_data, cache_time = self._cache[cache_key]
                if current_time - cache_time < self._cache_duration:
                    return cached_data
            
            if self.api:
                # Real API call (commented out for demo)
                # params = {
                #     "count": days * 24,
                #     "granularity": "H1"
                # }
                # r = instruments.InstrumentsCandles(
                #     instrument=self.instrument,
                #     params=params
                # )
                # self.api.request(r)
                # data = self._process_response(r.response)
                # For demo, return simulated data
                data_json = self._fetch_data_cached(days, self._chunk_size)
                data = pd.read_json(StringIO(data_json))
            else:
                # Generate demo data if no API key
                data_json = self._fetch_data_cached(days, self._chunk_size)
                data = pd.read_json(StringIO(data_json))
            
            # Update cache
            self._update_cache(cache_key, data, current_time)
            
            return data
            
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            # Return last cached data if available
            if cache_key in self._cache:
                st.warning("Using cached data due to error")
                return self._cache[cache_key][0]
            # Generate minimal demo data as fallback
            return self._generate_fallback_data(days)
    
    def _generate_fallback_data(self, days):
        """Generate minimal fallback data in case of errors"""
        try:
            dates = pd.date_range(end=datetime.now(), periods=days*24, freq='h')
            base_price = 1.1
            
            # Generate simple, consistent data
            data = pd.DataFrame({
                'open': [base_price] * len(dates),
                'high': [base_price * 1.001] * len(dates),
                'low': [base_price * 0.999] * len(dates),
                'close': [base_price] * len(dates),
                'volume': [1000] * len(dates)
            }, index=dates)
            
            return data
            
        except Exception as e:
            st.error(f"Error generating fallback data: {str(e)}")
            # Return absolute minimal dataset
            return pd.DataFrame({
                'open': [1.1],
                'high': [1.1],
                'low': [1.1],
                'close': [1.1],
                'volume': [1000]
            }, index=[datetime.now()])
    
    def _update_cache(self, key, data, timestamp):
        """Update cache with memory management"""
        try:
            # Remove old entries if cache is too large
            if len(self._cache) >= 10:  # Limit cache size
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]
                gc.collect()  # Force garbage collection
            
            self._cache[key] = (data, timestamp)
            
        except Exception as e:
            st.error(f"Error updating cache: {str(e)}")
            self._cache.clear()  # Clear cache on error
            gc.collect()
    
    def _process_response(self, response):
        """Process API response into DataFrame"""
        try:
            records = []
            for candle in response['candles']:
                records.append({
                    'time': pd.to_datetime(candle['time']),
                    'open': float(candle['mid']['o']),
                    'high': float(candle['mid']['h']),
                    'low': float(candle['mid']['l']),
                    'close': float(candle['mid']['c']),
                    'volume': int(candle['volume'])
                })
            
            df = pd.DataFrame.from_records(records)
            df.set_index('time', inplace=True)
            
            # Verify data consistency
            if not (len(df['open']) == len(df['high']) == len(df['low']) == len(df['close']) == len(df['volume'])):
                raise ValueError("API response data has inconsistent lengths")
            
            if not all(df['high'] >= df['low']):
                raise ValueError("High prices must be greater than or equal to low prices")
            
            return df
            
        except Exception as e:
            st.error(f"Error processing API response: {str(e)}")
            raise
    
    def clear_cache(self):
        """Clear cache and force garbage collection"""
        self._cache.clear()
        gc.collect()
