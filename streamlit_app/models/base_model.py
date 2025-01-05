from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BaseModel(ABC):
    def __init__(self, feature_columns=None):
        self.feature_columns = feature_columns or [
            'SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal_Line'
        ]
        self.model = None
        
    def prepare_data(self, df, lookforward_periods=12):
        """Prepare features and labels for training"""
        # Generate labels based on future returns
        future_returns = df['close'].shift(-lookforward_periods) / df['close'] - 1
        labels = (future_returns > 0).astype(int)
        labels = labels.iloc[:-lookforward_periods]
        
        # Prepare features
        features = df[self.feature_columns].iloc[:-lookforward_periods]
        
        return features, labels
    
    @abstractmethod
    def train(self, df):
        """Train the model - must be implemented by subclasses"""
        pass
    
    def predict(self, features):
        """Make trading predictions"""
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame([features], columns=self.feature_columns)
        
        if not hasattr(self.model, 'predict_proba'):
            predictions = self.model.predict(features)
            return predictions[0] if len(predictions) == 1 else predictions
            
        # Get probability of positive return
        prob_positive = self.model.predict_proba(features)[:, 1]
        
        # Convert to trading decisions with confidence thresholds
        decisions = np.where(prob_positive > 0.6, 1, 
                           np.where(prob_positive < 0.4, -1, 0))
        
        return decisions[0] if len(decisions) == 1 else decisions
    
    def get_model_info(self):
        """Get model parameters and information"""
        if hasattr(self.model, 'get_params'):
            return {
                'type': self.model.__class__.__name__,
                'parameters': self.model.get_params(),
                'features': self.feature_columns
            }
        return {
            'type': self.model.__class__.__name__,
            'features': self.feature_columns
        }
