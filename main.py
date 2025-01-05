import os
from data.fetcher import DataFetcher
from data.indicators import TechnicalIndicators
from models.random_forest import RandomForestModel
from models.neural_network import NeuralNetworkModel
from backtesting.backtester import Backtester

class ForexTrader:
    def __init__(self, min_accuracy=0.55, lookback=12, confidence=0.6, risk_settings=None, advanced_settings=None):
        self.min_accuracy = min_accuracy
        self.lookback = lookback
        self.confidence = confidence
        self.risk_settings = risk_settings or {
            'stop_loss': 0.02,
            'take_profit': 0.04,
            'max_position_size': 0.1,
            'max_trades': 5
        }
        self.advanced_settings = advanced_settings or {
            'use_gpu': False,
            'enable_ensemble': True,
            'feature_selection': 'All Features',
            'optimization_method': 'Random Search'
        }
        
        # Initialize components
        self.data_fetcher = DataFetcher()
        self.indicators = TechnicalIndicators()
        self.models = {
            'random_forest': RandomForestModel(),
            'neural_network': NeuralNetworkModel()
        }
        self.backtester = Backtester(
            initial_balance=10000,
            position_size=self.risk_settings['max_position_size']
        )
    
    def train_models(self, days=90):
        """Train all models and return their metrics"""
        data = self.data_fetcher.fetch_data(days=days)
        data = self.indicators.add_all_indicators(data)
        
        results = {}
        for name, model in self.models.items():
            metrics = model.train(data)
            if metrics['validation_accuracy'] >= self.min_accuracy:
                results[name] = metrics
        
        return results
    
    def get_live_signal(self, model_name=None):
        """Get trading signal from a specific model"""
        data = self.data_fetcher.fetch_data(days=self.lookback)
        data = self.indicators.add_all_indicators(data)
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        signal = model.predict(data.iloc[-1])
        confidence = model.get_prediction_confidence(data.iloc[-1])
        
        return {
            'signal_type': 'BUY' if signal > 0 else 'SELL' if signal < 0 else 'HOLD',
            'confidence': confidence,
            'current_price': data['close'].iloc[-1],
            'timestamp': data.index[-1]
        }
    
    def get_ensemble_signal(self):
        """Get trading signal using ensemble of all models"""
        signals = []
        confidences = []
        
        data = self.data_fetcher.fetch_data(days=self.lookback)
        data = self.indicators.add_all_indicators(data)
        
        for model in self.models.values():
            signal = model.predict(data.iloc[-1])
            confidence = model.get_prediction_confidence(data.iloc[-1])
            
            if confidence >= self.confidence:
                signals.append(signal)
                confidences.append(confidence)
        
        if not signals:
            return {
                'signal_type': 'HOLD',
                'confidence': 0.0,
                'current_price': data['close'].iloc[-1],
                'timestamp': data.index[-1]
            }
        
        # Weight signals by confidence
        weighted_signal = sum(s * c for s, c in zip(signals, confidences)) / sum(confidences)
        avg_confidence = sum(confidences) / len(confidences)
        
        return {
            'signal_type': 'BUY' if weighted_signal > 0.5 else 'SELL' if weighted_signal < -0.5 else 'HOLD',
            'confidence': avg_confidence,
            'current_price': data['close'].iloc[-1],
            'timestamp': data.index[-1]
        }
