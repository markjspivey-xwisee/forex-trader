import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st
from functools import lru_cache
import gc
import json
import pandas as pd
from io import StringIO

class BaseModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'SMA_20', 'SMA_50',
            'BB_upper', 'BB_lower',
            'MACD', 'Signal_Line',
            'RSI', 'ATR',
            'Stoch_K', 'Stoch_D'
        ]
        self._prediction_cache = {}
        self._confidence_cache = {}
    
    @staticmethod
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def _prepare_data_cached(data_json, feature_columns):
        """Prepare data for training/prediction with caching"""
        # Convert JSON to DataFrame using StringIO
        data = pd.read_json(StringIO(data_json))
        
        # Calculate returns
        data['returns'] = data['close'].pct_change()
        
        # Create labels (1 for positive returns, -1 for negative)
        data['target'] = np.where(data['returns'].shift(-1) > 0, 1, -1)
        
        # Prepare features
        X = data[feature_columns]
        y = data['target']
        
        # Remove any remaining NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        gc.collect()  # Force garbage collection
        
        # Convert to lists for JSON serialization
        X_dict = {col: X[col].values.tolist() for col in X.columns}
        y_list = y.values.tolist()
        
        return json.dumps({'X': X_dict, 'y': y_list})
    
    @staticmethod
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def _split_and_scale_data(data_json):
        """Split and scale data with caching"""
        # Parse JSON data
        data = json.loads(data_json)
        X = pd.DataFrame(data['X'])
        y = np.array(data['y'])
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        gc.collect()  # Force garbage collection
        
        # Convert numpy arrays to lists for JSON serialization
        result = {
            'X_train': X_train_scaled.tolist(),
            'X_val': X_val_scaled.tolist(),
            'y_train': y_train.tolist(),
            'y_val': y_val.tolist(),
            'scaler_mean': scaler.mean_.tolist(),
            'scaler_scale': scaler.scale_.tolist()
        }
        
        return json.dumps(result)
    
    def train(self, data):
        """Train the model and return metrics"""
        try:
            # Prepare data
            data_json = data.to_json(date_format='iso')
            prepared_data_json = self._prepare_data_cached(data_json, self.feature_columns)
            
            # Split and scale data
            split_data_json = self._split_and_scale_data(prepared_data_json)
            split_data = json.loads(split_data_json)
            
            # Convert lists back to numpy arrays
            X_train = np.array(split_data['X_train'])
            X_val = np.array(split_data['X_val'])
            y_train = np.array(split_data['y_train'])
            y_val = np.array(split_data['y_val'])
            
            # Reconstruct scaler
            self.scaler = StandardScaler()
            self.scaler.mean_ = np.array(split_data['scaler_mean'])
            self.scaler.scale_ = np.array(split_data['scaler_scale'])
            
            # Train model (to be implemented by child classes)
            self._train_model(X_train, y_train)
            
            # Calculate metrics
            train_accuracy = float(self.model.score(X_train, y_train))
            val_accuracy = float(self.model.score(X_val, y_val))
            
            # Clear prediction caches after training
            self._prediction_cache.clear()
            self._confidence_cache.clear()
            gc.collect()  # Force garbage collection
            
            return {
                'train_accuracy': train_accuracy,
                'validation_accuracy': val_accuracy
            }
            
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            return {
                'train_accuracy': 0.0,
                'validation_accuracy': 0.0
            }
    
    def predict(self, data_point):
        """Make prediction for a single data point with caching"""
        try:
            # Create cache key from feature values
            cache_key = tuple(float(data_point[col]) for col in self.feature_columns)
            
            # Check cache
            if cache_key in self._prediction_cache:
                return self._prediction_cache[cache_key]
            
            # Make prediction
            features = np.array([data_point[col] for col in self.feature_columns]).reshape(1, -1)
            scaled_features = self.scaler.transform(features)
            prediction = int(self.model.predict(scaled_features)[0])  # Convert to Python int
            
            # Update cache
            self._prediction_cache[cache_key] = prediction
            
            return prediction
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return 0  # Return neutral prediction on error
    
    def get_prediction_confidence(self, data_point):
        """Get confidence score for prediction with caching"""
        try:
            # Create cache key from feature values
            cache_key = tuple(float(data_point[col]) for col in self.feature_columns)
            
            # Check cache
            if cache_key in self._confidence_cache:
                return self._confidence_cache[cache_key]
            
            features = np.array([data_point[col] for col in self.feature_columns]).reshape(1, -1)
            scaled_features = self.scaler.transform(features)
            
            # For models that support predict_proba
            if hasattr(self.model, 'predict_proba'):
                probs = self.model.predict_proba(scaled_features)[0]
                confidence = float(max(probs))  # Convert numpy float to Python float
            else:
                confidence = 0.6  # Default confidence
            
            # Update cache
            self._confidence_cache[cache_key] = confidence
            
            return confidence
            
        except Exception as e:
            st.error(f"Error calculating confidence: {str(e)}")
            return 0.0  # Return zero confidence on error
    
    def _train_model(self, X_train, y_train):
        """To be implemented by child classes"""
        raise NotImplementedError
    
    def clear_cache(self):
        """Clear all caches and force garbage collection"""
        self._prediction_cache.clear()
        self._confidence_cache.clear()
        gc.collect()
