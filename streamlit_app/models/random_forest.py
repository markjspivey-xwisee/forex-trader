from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from .base_model import BaseModel

class RandomForestModel(BaseModel):
    def __init__(self, feature_columns=None, aggressive=False):
        super().__init__(feature_columns)
        self.aggressive = aggressive
        self.model = self._initialize_model()
    
    def _initialize_model(self):
        """Initialize RandomForest with appropriate parameters"""
        if self.aggressive:
            return RandomForestClassifier(
                n_estimators=500,
                max_depth=10,
                min_samples_split=10,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_split=20,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
    
    def train(self, df):
        """Train the RandomForest model"""
        features, labels = self.prepare_data(df)
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels, test_size=0.2, shuffle=False
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Calculate metrics
        train_accuracy = self.model.score(X_train, y_train)
        val_accuracy = self.model.score(X_val, y_val)
        
        # Get feature importance as dictionary
        importance_dict = {}
        for feature, importance in zip(self.feature_columns, self.model.feature_importances_):
            importance_dict[feature] = float(importance)  # Convert numpy float to Python float
        
        return {
            'train_accuracy': train_accuracy,
            'validation_accuracy': val_accuracy,
            'feature_importance': importance_dict,
            'model_type': 'Aggressive RandomForest' if self.aggressive else 'Conservative RandomForest'
        }

class AggressiveRandomForest(RandomForestModel):
    def __init__(self, feature_columns=None):
        super().__init__(feature_columns, aggressive=True)

class ConservativeRandomForest(RandomForestModel):
    def __init__(self, feature_columns=None):
        super().__init__(feature_columns, aggressive=False)
