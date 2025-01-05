import os
from datetime import datetime, timedelta
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
import pandas as pd
from dotenv import load_dotenv

class OANDADataFetcher:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('OANDA_API_KEY')
        self.instrument = os.getenv('INSTRUMENT')
        self.granularity = os.getenv('GRANULARITY')
        self.api = API(access_token=self.api_key)
        
    def fetch_historical_data(self, days=30):
        """Fetch historical candle data from OANDA"""
        end_dt = datetime.utcnow()
        start_dt = end_dt - timedelta(days=days)
        
        params = {
            "from": start_dt.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "to": end_dt.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "granularity": self.granularity
        }
        
        r = InstrumentsCandles(instrument=self.instrument, params=params)
        self.api.request(r)
        
        # Convert to DataFrame
        data = []
        for candle in r.response['candles']:
            if candle['complete']:
                data.append({
                    'timestamp': pd.to_datetime(candle['time']),
                    'open': float(candle['mid']['o']),
                    'high': float(candle['mid']['h']),
                    'low': float(candle['mid']['l']),
                    'close': float(candle['mid']['c']),
                    'volume': int(candle['volume'])
                })
                
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df

    def add_features(self, df):
        """Add technical indicators"""
        # Simple Moving Averages
        for period in [10, 20, 30, 50, 200]:
            df[f'SMA_{period}'] = df['close'].rolling(window=period).mean()
        
        # Relative Strength Index
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Rate of Change
        for period in [10, 20]:
            df[f'ROC_{period}'] = df['close'].pct_change(periods=period) * 100
        
        # Average True Range (ATR)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        # Bollinger Bands
        df['BB_middle'] = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (std * 2)
        df['BB_lower'] = df['BB_middle'] - (std * 2)
        df['Bollinger_Width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        
        return df.dropna()  # Remove rows with NaN values

if __name__ == '__main__':
    # Test the data fetcher
    fetcher = OANDADataFetcher()
    data = fetcher.fetch_historical_data(days=30)
    data_with_features = fetcher.add_features(data)
    print(data_with_features.tail())
