import argparse
import json
from datetime import datetime
import os
import numpy as np
from data.fetcher import DataFetcher
from data.indicators import TechnicalIndicators
from models.random_forest import AggressiveRandomForest, ConservativeRandomForest
from models.neural_network import MLPModel, LSTMModel
from backtesting.backtester import Backtester
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.ensemble import VotingClassifier

class ForexTrader:
    def __init__(self, min_accuracy=0.55, lookback=12, confidence=0.60,
                 risk_settings=None, advanced_settings=None):
        self.data_fetcher = DataFetcher()
        self.indicators = TechnicalIndicators()
        self.min_accuracy = min_accuracy
        self.lookback = lookback
        self.confidence = confidence
        
        # Default risk settings
        self.risk_settings = risk_settings or {
            'stop_loss': 0.02,
            'take_profit': 0.04,
            'max_position_size': 0.1,
            'max_trades': 5
        }
        
        # Default advanced settings
        self.advanced_settings = advanced_settings or {
            'use_gpu': False,
            'enable_ensemble': True,
            'feature_selection': 'All Features',
            'optimization_method': 'Random Search'
        }
        
        # Initialize models with updated parameters
        self.models = self._initialize_models()
        
    def _initialize_models(self):
        """Initialize models based on settings"""
        models = {
            'random_forest_aggressive': AggressiveRandomForest(),
            'random_forest_conservative': ConservativeRandomForest(),
            'neural_network': MLPModel(),
            'lstm': LSTMModel(sequence_length=self.lookback)
        }
        
        if self.advanced_settings['enable_ensemble']:
            # Create ensemble model
            estimators = [
                ('rf_agg', models['random_forest_aggressive'].model),
                ('rf_con', models['random_forest_conservative'].model),
                ('mlp', models['neural_network'].model)
            ]
            models['ensemble'] = VotingClassifier(
                estimators=estimators,
                voting='soft'
            )
        
        return models
        
    def _apply_feature_selection(self, features):
        """Apply selected feature selection method"""
        if self.advanced_settings['feature_selection'] == 'PCA':
            pca = PCA(n_components=0.95)  # Keep 95% of variance
            features_transformed = pca.fit_transform(features)
            return features_transformed
        elif self.advanced_settings['feature_selection'] == 'Recursive Feature Elimination':
            rfe = RFE(estimator=self.models['random_forest_conservative'].model,
                     n_features_to_select=5)
            features_selected = rfe.fit_transform(features, labels)
            return features_selected
        return features
        
    def _save_results(self, results, prefix='backtest'):
        """Save results to JSON file"""
        os.makedirs('results', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'results/{prefix}_results_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        return filename
        
    def _save_model(self, model, model_info):
        """Save model and its metadata"""
        os.makedirs('models', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_type = model_info['model_type'].lower().replace(' ', '_')
        
        # Save metadata
        metadata_file = f'models/{model_type}_{timestamp}_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(model_info, f, indent=4)
            
        return metadata_file
        
    def train_models(self, days=90):
        """Train all models and save results"""
        print(f"Fetching {days} days of historical data...")
        data = self.data_fetcher.fetch_data(days=days)
        data = self.indicators.add_all_indicators(data)
        
        results = {}
        for name, model in self.models.items():
            if name == 'ensemble':
                continue  # Skip ensemble model here
                
            print(f"\nTraining {name}...")
            try:
                metrics = model.train(data)
                results[name] = metrics
                
                # Save model if accuracy meets threshold
                if metrics['validation_accuracy'] >= self.min_accuracy:
                    metadata_file = self._save_model(model, metrics)
                    print(f"Model saved: {metadata_file}")
                    
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
        
        # Save training results
        results_file = self._save_results(results, prefix='training')
        print(f"\nTraining results saved to: {results_file}")
        return results
        
    def backtest(self, model_name='random_forest_conservative', days=90,
                initial_balance=10000, position_size=0.1):
        """Run backtest for specified model"""
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
            
        print(f"Fetching {days} days of historical data...")
        data = self.data_fetcher.fetch_data(days=days)
        data = self.indicators.add_all_indicators(data)
        
        print("\nTraining model...")
        model = self.models[model_name]
        train_metrics = model.train(data)
        
        print("\nRunning backtest...")
        backtester = Backtester(
            initial_balance=initial_balance,
            position_size=position_size,
            stop_loss=self.risk_settings['stop_loss'],
            take_profit=self.risk_settings['take_profit']
        )
        results = backtester.run(data, model)
        
        # Combine metrics
        results['model_metrics'] = train_metrics
        results['model_name'] = model_name
        
        # Save results
        results_file = self._save_results(results)
        print(f"\nBacktest results saved to: {results_file}")
        
        # Print summary
        print("\nBacktest Summary:")
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']:.2%}")
        print(f"Final Balance: ${results['final_balance']:.2f}")
        
        return results
        
    def get_live_signal(self, model_name='random_forest_conservative'):
        """Get trading signal for current market conditions"""
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
            
        # Get recent data
        data = self.data_fetcher.fetch_data(days=30)  # Enough for indicators
        data = self.indicators.add_all_indicators(data)
        
        # Train on recent data
        model = self.models[model_name]
        model.train(data)
        
        # Get signal for latest data point
        features = data[model.feature_columns].iloc[-1]
        signal = model.predict(features)
        
        # Apply confidence threshold
        if abs(signal) < self.confidence:
            signal = 0
        
        result = {
            'timestamp': data.index[-1],
            'signal': signal,
            'signal_type': 'BUY' if signal == 1 else 'SELL' if signal == -1 else 'HOLD',
            'current_price': data['close'].iloc[-1],
            'model': model_name,
            'confidence': abs(signal)
        }
        
        return result
        
    def get_ensemble_signal(self):
        """Get trading signal using ensemble of models"""
        data = self.data_fetcher.fetch_data(days=30)
        data = self.indicators.add_all_indicators(data)
        
        # Get signals from all models
        signals = []
        confidences = []
        
        for name, model in self.models.items():
            if name != 'ensemble':
                model.train(data)
                features = data[model.feature_columns].iloc[-1]
                signal = model.predict(features)
                signals.append(signal)
                confidences.append(abs(signal))
        
        # Weight signals by model confidence
        weighted_signal = np.average(signals, weights=confidences)
        
        # Apply confidence threshold
        if abs(weighted_signal) < self.confidence:
            final_signal = 0
        else:
            final_signal = np.sign(weighted_signal)
        
        return {
            'timestamp': data.index[-1],
            'signal': final_signal,
            'signal_type': 'BUY' if final_signal > 0 else 'SELL' if final_signal < 0 else 'HOLD',
            'current_price': data['close'].iloc[-1],
            'model': 'ensemble',
            'confidence': abs(weighted_signal)
        }

def main():
    parser = argparse.ArgumentParser(description='Forex Trading System')
    parser.add_argument('--mode', choices=['train', 'backtest', 'trade'],
                      default='backtest', help='Operation mode')
    parser.add_argument('--model', default='random_forest_conservative',
                      help='Model to use')
    parser.add_argument('--days', type=int, default=90,
                      help='Days of historical data')
    parser.add_argument('--balance', type=float, default=10000,
                      help='Initial balance for backtest')
    parser.add_argument('--position-size', type=float, default=0.1,
                      help='Position size as fraction of balance')
    parser.add_argument('--min-accuracy', type=float, default=0.55,
                      help='Minimum validation accuracy')
    parser.add_argument('--lookback', type=int, default=12,
                      help='Lookback periods')
    parser.add_argument('--confidence', type=float, default=0.60,
                      help='Signal confidence threshold')
    
    args = parser.parse_args()
    trader = ForexTrader(
        min_accuracy=args.min_accuracy,
        lookback=args.lookback,
        confidence=args.confidence
    )
    
    if args.mode == 'train':
        trader.train_models(days=args.days)
    elif args.mode == 'backtest':
        trader.backtest(
            model_name=args.model,
            days=args.days,
            initial_balance=args.balance,
            position_size=args.position_size
        )
    else:  # trade
        if args.model == 'ensemble':
            signal = trader.get_ensemble_signal()
        else:
            signal = trader.get_live_signal(model_name=args.model)
        print("\nLive Trading Signal:")
        for key, value in signal.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()
