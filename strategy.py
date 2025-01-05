import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

class MLStrategy:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            min_samples_split=20,
            max_depth=5,
            random_state=42,
            class_weight='balanced'
        )
        self.feature_columns = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal_Line']
        
    def generate_labels(self, df, lookforward_periods=12):
        """Generate trading signals based on future price movements"""
        # Calculate future returns
        future_returns = df['close'].shift(-lookforward_periods) / df['close'] - 1
        
        # Create labels: 1 for buy (positive return), 0 for sell (negative return)
        labels = (future_returns > 0).astype(int)
        
        return labels.iloc[:-lookforward_periods]  # Remove last rows where we can't calculate future returns
        
    def prepare_data(self, df):
        """Prepare features and labels for training"""
        # Generate labels
        labels = self.generate_labels(df)
        
        # Prepare features
        features = df[self.feature_columns].iloc[:-12]  # Match the size of labels
        
        return features, labels
        
    def train(self, df):
        """Train the model on historical data"""
        features, labels = self.prepare_data(df)
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels, test_size=0.2, shuffle=False
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Calculate accuracy
        train_accuracy = self.model.score(X_train, y_train)
        val_accuracy = self.model.score(X_val, y_val)
        
        return {
            'train_accuracy': train_accuracy,
            'validation_accuracy': val_accuracy
        }
        
    def predict(self, features):
        """Make trading predictions"""
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame([features], columns=self.feature_columns)
            
        # Get probability of positive return
        prob_positive = self.model.predict_proba(features)[:, 1]
        
        # Convert to trading decisions
        # Only take strong signals (>60% confidence)
        decisions = np.where(prob_positive > 0.6, 1, np.where(prob_positive < 0.4, -1, 0))
        
        return decisions[0] if len(decisions) == 1 else decisions

if __name__ == '__main__':
    # Test strategy with some dummy data
    from data_fetcher import OANDADataFetcher
    
    # Fetch and prepare data
    fetcher = OANDADataFetcher()
    data = fetcher.fetch_historical_data(days=90)  # Get more data for training
    data_with_features = fetcher.add_features(data)
    
    # Create and train strategy
    strategy = MLStrategy()
    metrics = strategy.train(data_with_features)
    print(f"Training metrics: {metrics}")
    
    # Make predictions on latest data
    latest_features = data_with_features[strategy.feature_columns].iloc[-1]
    prediction = strategy.predict(latest_features)
    print(f"Latest prediction: {prediction}")  # 1 for buy, -1 for sell, 0 for hold
