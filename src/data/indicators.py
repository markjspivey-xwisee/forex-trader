import pandas as pd
import numpy as np

class TechnicalIndicators:
    @staticmethod
    def add_sma(df, periods=[10, 20, 50, 200]):
        """Add Simple Moving Averages"""
        for period in periods:
            df[f'SMA_{period}'] = df['close'].rolling(window=period).mean()
        return df
    
    @staticmethod
    def add_rsi(df, period=14):
        """Add Relative Strength Index"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        return df
    
    @staticmethod
    def add_macd(df, fast=12, slow=26, signal=9):
        """Add MACD and Signal Line"""
        exp1 = df['close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=slow, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        return df
    
    @staticmethod
    def add_roc(df, periods=[10, 20]):
        """Add Rate of Change"""
        for period in periods:
            df[f'ROC_{period}'] = df['close'].pct_change(periods=period) * 100
        return df
    
    @staticmethod
    def add_atr(df, period=14):
        """Add Average True Range"""
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(window=period).mean()
        return df
    
    @staticmethod
    def add_bollinger_bands(df, period=20, std_dev=2):
        """Add Bollinger Bands and Width"""
        df['BB_middle'] = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        df['BB_upper'] = df['BB_middle'] + (std * std_dev)
        df['BB_lower'] = df['BB_middle'] - (std * std_dev)
        df['Bollinger_Width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        return df
    
    @staticmethod
    def add_stochastic(df, k_period=14, d_period=3):
        """Add Stochastic Oscillator"""
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        
        df['Stoch_K'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['Stoch_D'] = df['Stoch_K'].rolling(window=d_period).mean()
        return df
    
    @staticmethod
    def add_obv(df):
        """Add On Balance Volume"""
        df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        return df
    
    @staticmethod
    def add_momentum(df, period=14):
        """Add Momentum Indicator"""
        df['Momentum'] = df['close'].diff(period)
        return df
    
    @staticmethod
    def add_williams_r(df, period=14):
        """Add Williams %R"""
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()
        df['Williams_R'] = -100 * (highest_high - df['close']) / (highest_high - lowest_low)
        return df
    
    @classmethod
    def add_all_indicators(cls, df):
        """Add all technical indicators"""
        df = cls.add_sma(df)
        df = cls.add_rsi(df)
        df = cls.add_macd(df)
        df = cls.add_roc(df)
        df = cls.add_atr(df)
        df = cls.add_bollinger_bands(df)
        df = cls.add_stochastic(df)
        df = cls.add_obv(df)
        df = cls.add_momentum(df)
        df = cls.add_williams_r(df)
        
        # Calculate OHLC
        df['OHLC'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        
        return df.dropna()  # Remove rows with NaN values
