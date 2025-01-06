import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import streamlit as st
import gc
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
            
            # Build model with Input layer
            self.model = Sequential([
                Input(shape=(len(self.feature_columns),)),  # Explicit input shape
                Dense(128, activation='relu'),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')  # Binary classification output
            ])
            
            # Compile model with binary crossentropy
            self.model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
        except Exception as e:
            st.error(f"Error building neural network: {str(e)}")
            raise
    
    def _train_model(self, X_train, y_train):
        """Train the neural network model"""
        try:
            # Convert labels from -1/1 to 0/1 for binary crossentropy
            y_train_binary = (y_train + 1) / 2
            
            # Train model
            history = self.model.fit(
                X_train,
                y_train_binary,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                verbose=1,  # Show progress
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=5,
                        restore_best_weights=True
                    )
                ]
            )
            
            # Force garbage collection
            gc.collect()
            
            return history
            
        except Exception as e:
            st.error(f"Error training neural network: {str(e)}")
            raise
    
    def score(self, X, y):
        """Calculate accuracy score"""
        try:
            if self.model is None:
                return 0.0
            
            # Convert labels from -1/1 to 0/1 for binary crossentropy
            y_binary = (y + 1) / 2
            
            # Get predictions
            predictions = self.model.predict(X, verbose=0)
            
            # Convert predictions to binary signals
            predicted_signals = np.where(predictions > 0.5, 1, 0)
            
            # Calculate accuracy
            accuracy = np.mean(predicted_signals.flatten() == y_binary)
            
            return float(accuracy)
            
        except Exception as e:
            st.error(f"Error calculating score: {str(e)}")
            return 0.0
    
    def predict(self, data_point):
        """Make prediction for a single data point"""
        try:
            # Extract features
            features = np.array([data_point[col] for col in self.feature_columns]).reshape(1, -1)
            
            # Make prediction
            prediction = self.model.predict(features, verbose=0)[0][0]
            
            # Convert to -1/0/1 signal
            signal = 1 if prediction > 0.66 else -1 if prediction < 0.33 else 0
            
            return signal
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return 0
    
    def get_prediction_confidence(self, data_point):
        """Get confidence score for prediction"""
        try:
            # Extract features
            features = np.array([data_point[col] for col in self.feature_columns]).reshape(1, -1)
            
            # Get raw prediction
            prediction = self.model.predict(features, verbose=0)[0][0]
            
            # Convert to confidence score (0.5 to 1.0)
            # Map 0-0.33 and 0.66-1 ranges to 0.5-1.0
            if prediction < 0.33:
                confidence = 0.5 + (0.33 - prediction) * 1.5
            elif prediction > 0.66:
                confidence = 0.5 + (prediction - 0.66) * 1.5
            else:
                confidence = 0.5
            
            return float(confidence)
            
        except Exception as e:
            st.error(f"Error calculating confidence: {str(e)}")
            return 0.0
    
    def clear_cache(self):
        """Clear model cache and force garbage collection"""
        super().clear_cache()
        tf.keras.backend.clear_session()
        gc.collect()
