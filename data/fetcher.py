import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import json
import gc
from io import StringIO
import oandapyV20
import oandapyV20.endpoints.instruments as instruments

class DataFetcher:
    def __init__(self, chunk_size=500):  # Reduced default chunk size
        self._chunk_size = chunk_size
        self._api = None
        self._instrument = "EUR_USD"
        
        # Try to get credentials from Streamlit secrets first
        try:
            self.api_key = st.secrets["OANDA_API_KEY"]
            # Initialize API client
            self._api = oandapyV20.API(
                access_token=self.api_key,
                environment="practice"  # Use 'practice' for demo accounts
            )
        except Exception as e:
            st.warning("Using simulated data (OANDA API credentials not found)")
    
    @staticmethod
    def _generate_sample_data(days):
        """Generate sample data when API is not available"""
        dates = pd.date_range(
            end=datetime.now(),
            periods=days * 24,  # Hourly intervals
            freq='h'
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
        base_volume = 100000
        volume_volatility = 0.3
        data['volume'] = np.random.lognormal(np.log(base_volume), volume_volatility, len(dates))
        
        return data
    
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def _fetch_data_cached(self, days):
        """Fetch data with caching"""
        try:
            if self._api is None:
                # Use simulated data if API is not available
                data = self._generate_sample_data(days)
            else:
                # Fetch real data from OANDA
                params = {
                    "count": days * 24,  # Hourly candles
                    "granularity": "H1",  # 1-hour candles
                    "price": "MBA"  # Mid, Bid, Ask prices
                }
                
                r = instruments.InstrumentsCandles(
                    instrument=self._instrument,
                    params=params
                )
                
                self._api.request(r)
                candles = r.response['candles']
                
                # Convert to DataFrame
                data = []
                for candle in candles:
                    data.append({
                        'time': pd.to_datetime(candle['time']),
                        'open': float(candle['mid']['o']),
                        'high': float(candle['mid']['h']),
                        'low': float(candle['mid']['l']),
                        'close': float(candle['mid']['c']),
                        'volume': float(candle['volume'])
                    })
                
                data = pd.DataFrame(data)
                data.set_index('time', inplace=True)
            
            # Force garbage collection
            gc.collect()
            
            # Convert to JSON for caching
            return data.to_json(date_format='iso')
            
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            # Fall back to simulated data
            data = self._generate_sample_data(days)
            return data.to_json(date_format='iso')
    
    def fetch_data(self, days=60):
        """Fetch market data with chunking"""
        try:
            # Get data from cache
            data_json = self._fetch_data_cached(days)
            if data_json is None:
                return None
            
            # Process data in chunks
            data = pd.read_json(StringIO(data_json))
            processed_chunks = []
            
            for i in range(0, len(data), self._chunk_size):
                chunk = data.iloc[i:i + self._chunk_size].copy()
                processed_chunks.append(chunk)
                
                # Force garbage collection after each chunk
                gc.collect()
            
            # Combine chunks
            result = pd.concat(processed_chunks)
            
            # Force garbage collection
            gc.collect()
            
            return result
            
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            return None
    
    def clear_cache(self):
        """Clear data cache"""
        st.cache_data.clear()
        gc.collect()  # Force garbage collection
