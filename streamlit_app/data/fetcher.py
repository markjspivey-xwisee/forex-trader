from datetime import datetime, timedelta
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
import pandas as pd
from dotenv import load_dotenv
import os

class DataFetcher:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('OANDA_API_KEY')
        self.instrument = os.getenv('INSTRUMENT')
        self.granularity = os.getenv('GRANULARITY')
        self.api = API(access_token=self.api_key)
        
    def fetch_data(self, days=30):
        """Fetch historical candle data"""
        end_dt = datetime.utcnow()
        start_dt = end_dt - timedelta(days=days)
        
        params = {
            "from": start_dt.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "to": end_dt.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "granularity": self.granularity
        }
        
        r = InstrumentsCandles(instrument=self.instrument, params=params)
        self.api.request(r)
        
        return self._process_response(r.response)
    
    def _process_response(self, response):
        """Process API response into DataFrame"""
        data = [{
            'timestamp': pd.to_datetime(candle['time']),
            'open': float(candle['mid']['o']),
            'high': float(candle['mid']['h']),
            'low': float(candle['mid']['l']),
            'close': float(candle['mid']['c']),
            'volume': int(candle['volume'])
        } for candle in response['candles'] if candle['complete']]
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
