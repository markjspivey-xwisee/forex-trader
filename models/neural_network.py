import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import numpy as np
from .base_model import BaseModel

class NeuralNetworkModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = None
        self.history = None
    
    def _build_model(self, input_shape):
        """Build and compile the neural network"""
        model = Sequential([
            Input(shape=(input_shape,)),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _train_model(self, X_train, y_train):
        """Train the neural network"""
        # Convert labels from {-1, 1} to {0, 1} for binary classification
        y_train_binary = (y_train + 1) // 2
        
        # Build model with correct input shape
        self.model = self._build_model(X_train.shape[1])
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train_binary,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        return self.model
    
    def predict(self, data_point):
        """Make prediction for a single data point"""
        features = np.array([data_point[col] for col in self.feature_columns]).reshape(1, -1)
        scaled_features = self.scaler.transform(features)
        prob = self.model.predict(scaled_features, verbose=0)[0][0]
        return 1 if prob > 0.5 else -1
    
    def get_prediction_confidence(self, data_point):
        """Get confidence score for prediction"""
        features = np.array([data_point[col] for col in self.feature_columns]).reshape(1, -1)
        scaled_features = self.scaler.transform(features)
        prob = self.model.predict(scaled_features, verbose=0)[0][0]
        return max(prob, 1 - prob)  # Return confidence between 0 and 1
