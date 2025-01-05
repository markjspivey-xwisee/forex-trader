import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
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
    
    def _train_model(self, X_train, y_train):
        """Train the neural network model"""
        try:
            # Ensure data is properly scaled
            if not hasattr(self.scaler, 'mean_'):
                self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            
            # Convert labels to float32
            y_train = y_train.astype(np.float32)
            
            # Train model
            self.model.fit(
                X_train_scaled,
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
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            st.error(f"Error training neural network: {str(e)}")
            raise
    
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
