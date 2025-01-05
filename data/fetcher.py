import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments

class DataFetcher:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('OANDA_API_KEY')
        self.account_id = os.getenv('OANDA_ACCOUNT_ID')
        self.instrument = "EUR_USD"
        self.api = API(access_token=self.api_key)
    
    def fetch_data(self, days=30):
        """Fetch historical forex data"""
        # For demo, return simulated data
        dates = pd.date_range(end=datetime.now(), periods=days*24, freq='H')
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
