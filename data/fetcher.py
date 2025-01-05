import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import streamlit as st

class DataFetcher:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('OANDA_API_KEY')
        self.account_id = os.getenv('OANDA_ACCOUNT_ID')
        self.instrument = "EUR_USD"
        self.api = API(access_token=self.api_key) if self.api_key else None
        self._cache = {}
        self._cache_duration = timedelta(minutes=5)  # Cache data for 5 minutes
    
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def fetch_data(self, days=30):
        """Fetch historical forex data with caching"""
        cache_key = f"{self.instrument}_{days}"
        current_time = datetime.now()
        
        # Check cache
        if cache_key in self._cache:
            cached_data, cache_time = self._cache[cache_key]
            if current_time - cache_time < self._cache_duration:
                return cached_data
        
        try:
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
                data = self._generate_demo_data(days)
            else:
                # Generate demo data if no API key
                data = self._generate_demo_data(days)
            
            # Update cache
            self._cache[cache_key] = (data, current_time)
            
            return data
        
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            # Return last cached data if available
            if cache_key in self._cache:
                st.warning("Using cached data due to error")
                return self._cache[cache_key][0]
            # Generate demo data as fallback
            return self._generate_demo_data(days)
    
    def _generate_demo_data(self, days):
        """Generate simulated forex data for demo"""
        dates = pd.date_range(end=datetime.now(), periods=days*24, freq='h')
        data = pd.DataFrame({
            'open': np.random.normal(1.1, 0.02, len(dates)),
            'high': np.random.normal(1.1, 0.02, len(dates)),
            'low': np.random.normal(1.1, 0.02, len(dates)),
            'close': np.random.normal(1.1, 0.02, len(dates)),
            'volume': np.random.randint(1000, 5000, len(dates))
        }, index=dates)
        
        # Ensure high/low are actually highest/lowest
        data['high'] = data[['open', 'high', 'low', 'close']].max(axis=1)
        data['low'] = data[['open', 'high', 'low', 'close']].min(axis=1)
        
        return data
    
    def _process_response(self, response):
        """Process API response into DataFrame"""
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
        return df
