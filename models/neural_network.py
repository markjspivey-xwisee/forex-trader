import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
import streamlit as st
import gc
import json
import pandas as pd
from io import StringIO
from .base_model import BaseModel

class NeuralNetworkModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = None
        self.scaler = StandardScaler()
        self._build_model()
    
    def _build_model(self):
        """Build neural network architecture"""
        try:
            # Clear any existing models
            tf.keras.backend.clear_session()
            gc.collect()
            
            # Build model
            self.model = Sequential([
                Dense(64, activation='relu', input_shape=(len(self.feature_columns),)),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(1, activation='tanh')  # Output between -1 and 1
            ])
            
            # Compile model
            self.model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['accuracy']
            )
            
        except Exception as e:
            st.error(f"Error building neural network: {str(e)}")
            raise
    
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
            
            # Train model
            history = self.model.fit(
                X_train,
                y_train,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                verbose=0,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=5,
                        restore_best_weights=True
                    )
                ]
            )
            
            # Calculate metrics
            train_predictions = self.model.predict(X_train, verbose=0)
            val_predictions = self.model.predict(X_val, verbose=0)
            
            # Convert predictions to binary signals
            train_signals = np.where(train_predictions > 0.5, 1, np.where(train_predictions < -0.5, -1, 0))
            val_signals = np.where(val_predictions > 0.5, 1, np.where(val_predictions < -0.5, -1, 0))
            
            # Calculate accuracies
            train_accuracy = float(np.mean(train_signals.flatten() == y_train))
            val_accuracy = float(np.mean(val_signals.flatten() == y_val))
            
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
        """Make prediction for a single data point"""
        try:
            # Extract features
            features = np.array([data_point[col] for col in self.feature_columns]).reshape(1, -1)
            
            # Scale features
            if not hasattr(self.scaler, 'mean_'):
                st.warning("Model not trained yet. Please train the model first.")
                return 0
            
            scaled_features = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(scaled_features, verbose=0)[0][0]
            
            # Convert to binary signal
            signal = 1 if prediction > 0.5 else -1 if prediction < -0.5 else 0
            
            return signal
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return 0
    
    def get_prediction_confidence(self, data_point):
        """Get confidence score for prediction"""
        try:
            # Extract features
            features = np.array([data_point[col] for col in self.feature_columns]).reshape(1, -1)
            
            # Scale features
            if not hasattr(self.scaler, 'mean_'):
                st.warning("Model not trained yet. Please train the model first.")
                return 0.0
            
            scaled_features = self.scaler.transform(features)
            
            # Get raw prediction
            prediction = abs(self.model.predict(scaled_features, verbose=0)[0][0])
            
            # Convert to confidence score (0.5 to 1.0)
            confidence = 0.5 + (prediction * 0.5)
            
            return float(confidence)
            
        except Exception as e:
            st.error(f"Error calculating confidence: {str(e)}")
            return 0.0
    
    def clear_cache(self):
        """Clear model cache and force garbage collection"""
        super().clear_cache()
        tf.keras.backend.clear_session()
        gc.collect()
