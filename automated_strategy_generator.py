import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pandas as pd
import json
import os
from datetime import datetime
import joblib
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

class StrategyGenerator:
    def __init__(self, save_dir='strategies'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'visualizations'), exist_ok=True)
        self.feature_combinations = self._generate_feature_combinations()
        self.models = self._initialize_models()
        self.scaler = StandardScaler()
        
    def _generate_feature_combinations(self):
        """Generate different feature combinations for testing"""
        combinations = []
        
        # Technical Indicators
        base_features = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal_Line']
        combinations.append(base_features)
        
        # Moving Averages
        sma_periods = [(10, 30), (20, 50), (50, 200)]
        for short, long in sma_periods:
            features = [f'SMA_{short}', f'SMA_{long}', 'RSI', 'MACD', 'Signal_Line']
            combinations.append(features)
        
        # Momentum
        momentum_features = base_features + ['ROC_10', 'ROC_20']
        combinations.append(momentum_features)
        
        # Volatility
        volatility_features = base_features + ['ATR', 'Bollinger_Width']
        combinations.append(volatility_features)
        
        # Combined
        full_features = list(set(
            base_features + 
            [f'SMA_{p}' for p in [10, 20, 30, 50, 200]] +
            ['ROC_10', 'ROC_20', 'ATR', 'Bollinger_Width']
        ))
        combinations.append(full_features)
        
        return combinations
        
    def _initialize_models(self):
        """Initialize different model architectures"""
        models = {
            'random_forest_conservative': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [10, 20, 30],
                    'class_weight': ['balanced']
                }
            },
            'random_forest_aggressive': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [300, 500, 700],
                    'max_depth': [7, 10, 15],
                    'min_samples_split': [5, 10, 15],
                    'class_weight': ['balanced']
                }
            },
            'xgboost': {
                'model': xgb.XGBClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'neural_network': {
                'model': MLPClassifier(random_state=42),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (100, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['adaptive']
                }
            }
        }
        return models
        
    def _create_lstm_model(self, input_shape):
        """Create LSTM model for sequence prediction"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        return model
        
    def _prepare_sequence_data(self, features, labels, lookback=10):
        """Prepare sequential data for LSTM"""
        X, y = [], []
        for i in range(len(features) - lookback):
            X.append(features[i:(i + lookback)])
            y.append(labels[i + lookback])
        return np.array(X), np.array(y)
        
    def _optimize_hyperparameters(self, model_config, X_train, y_train):
        """Perform hyperparameter optimization"""
        search = RandomizedSearchCV(
            model_config['model'],
            model_config['params'],
            n_iter=10,
            cv=3,
            random_state=42,
            n_jobs=-1
        )
        search.fit(X_train, y_train)
        return search.best_estimator_, search.best_score_, search.best_params_
        
    def _evaluate_model(self, model, X_test, y_test):
        """Calculate comprehensive model metrics"""
        y_pred = model.predict(X_test)
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'cross_val': np.mean(cross_val_score(model, X_test, y_test, cv=5))
        }
        
    def _plot_feature_importance(self, model, features, model_name, timestamp):
        """Plot and save feature importance visualization"""
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(10, 6))
            importances = pd.DataFrame({
                'feature': features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            sns.barplot(x='importance', y='feature', data=importances)
            plt.title(f'Feature Importance - {model_name}')
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.save_dir, 'visualizations', 
                                   f'{model_name}_{timestamp}_importance.png')
            plt.savefig(plot_path)
            plt.close()
            
            return importances.to_dict('records')
        return None
        
    def _plot_training_history(self, history, model_name, timestamp):
        """Plot and save training history for neural networks"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.save_dir, 'visualizations',
                                f'{model_name}_{timestamp}_training.png')
        plt.savefig(plot_path)
        plt.close()
        
    def generate_strategies(self, data, min_validation_accuracy=0.55):
        """Generate and test multiple trading strategies"""
        successful_strategies = []
        
        for feature_set in self.feature_combinations:
            # Filter features that exist in the data
            available_features = [f for f in feature_set if f in data.columns]
            if len(available_features) < 3:
                continue
                
            features = data[available_features]
            
            # Generate labels
            future_returns = data['close'].shift(-12) / data['close'] - 1
            labels = (future_returns > 0).astype(int)
            labels = labels.iloc[:-12]
            
            # Prepare features
            features = features.iloc[:-12]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, shuffle=False
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train and evaluate each model type
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            for model_name, model_config in self.models.items():
                try:
                    # Optimize hyperparameters
                    best_model, best_score, best_params = self._optimize_hyperparameters(
                        model_config, X_train_scaled, y_train
                    )
                    
                    # Evaluate model
                    metrics = self._evaluate_model(best_model, X_test_scaled, y_test)
                    
                    if metrics['accuracy'] >= min_validation_accuracy:
                        # Generate feature importance plot
                        importance = self._plot_feature_importance(
                            best_model, available_features, model_name, timestamp
                        )
                        
                        strategy_info = {
                            'model_name': model_name,
                            'features': available_features,
                            'metrics': metrics,
                            'best_params': best_params,
                            'feature_importance': importance,
                            'created_at': timestamp
                        }
                        
                        # Save strategy
                        strategy_id = f"{model_name}_{timestamp}"
                        self.save_strategy(best_model, strategy_info, strategy_id)
                        successful_strategies.append(strategy_info)
                        
                except Exception as e:
                    print(f"Error training {model_name}: {str(e)}")
                    continue
            
            # Train LSTM model
            try:
                # Prepare sequence data
                X_seq, y_seq = self._prepare_sequence_data(
                    self.scaler.fit_transform(features), labels
                )
                X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(
                    X_seq, y_seq, test_size=0.2, shuffle=False
                )
                
                # Create and train LSTM
                lstm_model = self._create_lstm_model((X_train_seq.shape[1], X_train_seq.shape[2]))
                history = lstm_model.fit(
                    X_train_seq, y_train_seq,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=0
                )
                
                # Evaluate LSTM
                lstm_metrics = {
                    'accuracy': lstm_model.evaluate(X_test_seq, y_test_seq)[1],
                    'training_history': history.history
                }
                
                if lstm_metrics['accuracy'] >= min_validation_accuracy:
                    # Plot training history
                    self._plot_training_history(history, 'lstm', timestamp)
                    
                    strategy_info = {
                        'model_name': 'lstm',
                        'features': available_features,
                        'metrics': lstm_metrics,
                        'created_at': timestamp
                    }
                    
                    # Save strategy
                    strategy_id = f"lstm_{timestamp}"
                    self.save_strategy(lstm_model, strategy_info, strategy_id)
                    successful_strategies.append(strategy_info)
                    
            except Exception as e:
                print(f"Error training LSTM: {str(e)}")
                
        return successful_strategies
        
    def save_strategy(self, model, strategy_info, strategy_id):
        """Save the model and its metadata"""
        # Save model
        model_path = os.path.join(self.save_dir, f'{strategy_id}.joblib')
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata_path = os.path.join(self.save_dir, f'{strategy_id}_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(strategy_info, f, indent=4)
            
    def load_strategy(self, strategy_id):
        """Load a saved strategy"""
        model_path = os.path.join(self.save_dir, f'{strategy_id}.joblib')
        metadata_path = os.path.join(self.save_dir, f'{strategy_id}_metadata.json')
        
        model = joblib.load(model_path)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        return model, metadata
        
    def list_strategies(self):
        """List all saved strategies"""
        strategies = []
        for file in os.listdir(self.save_dir):
            if file.endswith('_metadata.json'):
                with open(os.path.join(self.save_dir, file), 'r') as f:
                    metadata = json.load(f)
                    strategies.append(metadata)
        return strategies

if __name__ == '__main__':
    # Test the strategy generator
    from data_fetcher import OANDADataFetcher
    
    fetcher = OANDADataFetcher()
    data = fetcher.fetch_historical_data(days=90)
    data = fetcher.add_features(data)
    
    generator = StrategyGenerator()
    strategies = generator.generate_strategies(data)
    print(f"Generated {len(strategies)} successful strategies")
