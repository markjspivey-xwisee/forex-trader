import numpy as np
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
from .base_model import BaseModel

class RandomForestModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = None
    
    def _train_model(self, X_train, y_train):
        """Train the random forest model"""
        try:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1  # Use all available cores
            )
            self.model.fit(X_train, y_train)
            
        except Exception as e:
            st.error(f"Error training random forest: {str(e)}")
            raise
    
    def score(self, X, y):
        """Calculate accuracy score"""
        try:
            if self.model is None:
                return 0.0
            return float(self.model.score(X, y))
        except Exception as e:
            st.error(f"Error calculating score: {str(e)}")
            return 0.0
    
    def predict(self, data_point):
        """Make prediction for a single data point"""
        try:
            # Extract features
            features = np.array([data_point[col] for col in self.feature_columns]).reshape(1, -1)
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            
            return int(prediction)  # Convert numpy int to Python int
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return 0
    
    def get_prediction_confidence(self, data_point):
        """Get confidence score for prediction"""
        try:
            # Extract features
            features = np.array([data_point[col] for col in self.feature_columns]).reshape(1, -1)
            
            # Get prediction probabilities
            probs = self.model.predict_proba(features)[0]
            
            # Get confidence (max probability)
            confidence = float(max(probs))  # Convert numpy float to Python float
            
            # Scale confidence to 0.5-1.0 range
            confidence = 0.5 + (confidence * 0.5)
            
            return confidence
            
        except Exception as e:
            st.error(f"Error calculating confidence: {str(e)}")
            return 0.0
