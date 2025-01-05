from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
from .base_model import BaseModel

class MLPModel(BaseModel):
    def __init__(self, feature_columns=None):
        super().__init__(feature_columns)
        self.scaler = StandardScaler()
        self.model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            random_state=42
        )
    
    def train(self, df):
        """Train the MLP model"""
        features, labels = self.prepare_data(df)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels, test_size=0.2, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate feature importance using permutation importance
        base_score = self.model.score(X_val_scaled, y_val)
        importance_dict = {}
        
        for i, feature in enumerate(self.feature_columns):
            X_val_permuted = X_val_scaled.copy()
            X_val_permuted[:, i] = np.random.permutation(X_val_permuted[:, i])
            permuted_score = self.model.score(X_val_permuted, y_val)
            importance = max(0, base_score - permuted_score)  # Ensure non-negative
            importance_dict[feature] = float(importance)  # Convert numpy float to Python float
        
        # Normalize importance values
        total_importance = sum(importance_dict.values())
        if total_importance > 0:
            importance_dict = {k: v/total_importance for k, v in importance_dict.items()}
        
        return {
            'train_accuracy': self.model.score(X_train_scaled, y_train),
            'validation_accuracy': base_score,
            'feature_importance': importance_dict,
            'model_type': 'Neural Network'
        }
    
    def predict(self, features):
        """Make predictions with scaled features"""
        if not isinstance(features, np.ndarray):
            features = np.array(features).reshape(1, -1)
        scaled_features = self.scaler.transform(features)
        return super().predict(scaled_features)

class LSTMModel(BaseModel):
    def __init__(self, feature_columns=None, sequence_length=10):
        super().__init__(feature_columns)
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.model = self._build_model()
    
    def _build_model(self):
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(50, return_sequences=True, 
                 input_shape=(self.sequence_length, len(self.feature_columns))),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _prepare_sequences(self, features):
        """Convert features into sequences for LSTM"""
        sequences = []
        for i in range(len(features) - self.sequence_length):
            sequences.append(features[i:(i + self.sequence_length)])
        return np.array(sequences)
    
    def train(self, df):
        """Train the LSTM model"""
        features, labels = self.prepare_data(df)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Prepare sequences
        X = self._prepare_sequences(scaled_features)
        y = labels[self.sequence_length:]
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            verbose=0
        )
        
        # Calculate feature importance using integrated gradients
        importance_dict = {}
        baseline = np.zeros_like(X_val[0])
        
        for i, feature in enumerate(self.feature_columns):
            gradients = []
            for seq in X_val[:10]:  # Use first 10 sequences for computational efficiency
                with tf.GradientTape() as tape:
                    inputs = tf.convert_to_tensor(seq[np.newaxis, ...], dtype=tf.float32)
                    tape.watch(inputs)
                    outputs = self.model(inputs)
                    gradients.append(tape.gradient(outputs, inputs))
            
            avg_gradient = np.mean([g.numpy() for g in gradients], axis=0)
            importance = np.mean(np.abs(avg_gradient[:, :, i]))
            importance_dict[feature] = float(importance)
        
        # Normalize importance values
        total_importance = sum(importance_dict.values())
        if total_importance > 0:
            importance_dict = {k: v/total_importance for k, v in importance_dict.items()}
        
        # Get final metrics
        train_metrics = self.model.evaluate(X_train, y_train, verbose=0)
        val_metrics = self.model.evaluate(X_val, y_val, verbose=0)
        
        return {
            'train_accuracy': train_metrics[1],
            'validation_accuracy': val_metrics[1],
            'feature_importance': importance_dict,
            'model_type': 'LSTM Neural Network',
            'training_history': history.history
        }
    
    def predict(self, features):
        """Make predictions using sequences"""
        if not isinstance(features, np.ndarray):
            features = np.array(features).reshape(1, -1)
            
        scaled_features = self.scaler.transform(features)
        
        # For single prediction, we need the last sequence_length samples
        if len(scaled_features) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} samples for prediction")
            
        sequence = scaled_features[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        prob = self.model.predict(sequence, verbose=0)[0][0]
        
        # Convert probability to trading decision
        if prob > 0.6:
            return 1
        elif prob < 0.4:
            return -1
        return 0
