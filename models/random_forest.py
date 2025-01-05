from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseModel

class RandomForestModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1  # Use all available cores
        )
        self.feature_importance = {}
    
    def _train_model(self, X_train, y_train):
        """Train the Random Forest model"""
        self.model.fit(X_train, y_train)
        
        # Store feature importance
        self.feature_importance = dict(zip(
            self.feature_columns,
            self.model.feature_importances_
        ))
        
        return self.model
    
    def get_prediction_confidence(self, data_point):
        """Get confidence score for prediction"""
        features = self._prepare_features(data_point)
        probs = self.model.predict_proba(features)[0]
        return max(probs)  # Return highest probability as confidence
    
    def _prepare_features(self, data_point):
        """Helper method to prepare features for prediction"""
        features = [data_point[col] for col in self.feature_columns]
        return self.scaler.transform([features])
