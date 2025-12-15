"""
Complexity predictor for determining N allocation
"""

import joblib
import numpy as np
from typing import Dict
from .features import extract_code_metrics, extract_repo_features


class ComplexityPredictor:
    """
    Predicts task complexity and determines N allocation.
    
    Uses Random Forest regression on extracted features.
    """
    
    def __init__(
        self,
        model,
        scaler,
        feature_names,
        transform_config=None
    ):
        """
        Initialize predictor with trained models.
        
        Args:
            model: Trained sklearn model
            scaler: Fitted StandardScaler
            feature_names: List of feature names in order
            transform_config: Dict with transformation settings
        """
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.transform_config = transform_config or {'use_log': False}
        
    @classmethod
    def load(cls, model_path: str):
        """
        Load trained predictor from disk.
        
        Args:
            model_path: Path to saved model (.pkl file)
            
        Returns:
            ComplexityPredictor instance
        """
        import os
        
        model_dir = os.path.dirname(model_path)
        
        model = joblib.load(model_path)
        scaler = joblib.load(os.path.join(model_dir, "feature_scaler.pkl"))
        feature_names = joblib.load(os.path.join(model_dir, "feature_names.pkl"))
        
        try:
            transform_config = joblib.load(os.path.join(model_dir, "transform_config.pkl"))
        except:
            transform_config = {'use_log': False}
        
        return cls(
            model=model,
            scaler=scaler,
            feature_names=feature_names,
            transform_config=transform_config
        )
    
    def predict(self, task: Dict, all_tasks: list = None) -> float:
        """
        Predict token requirements for a task.
        
        Args:
            task: Task dictionary with 'problem_statement' and 'repo'
            all_tasks: Optional list of all tasks for repo features
            
        Returns:
            Predicted token count (clipped to 500-3000 range)
        """
        # Extract features
        text_features = extract_code_metrics(task['problem_statement'])
        
        if all_tasks:
            repo_features = extract_repo_features(task['repo'], all_tasks)
        else:
            # Default repo features if not provided
            repo_features = {
                'repo_task_count': 1,
                'repo_avg_difficulty': 2000
            }
        
        # Combine features
        feature_dict = {**text_features, **repo_features}
        
        # Build feature vector
        X = np.array([[feature_dict.get(f, 0) for f in self.feature_names]])
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        if self.transform_config.get('use_log', False):
            pred_log = self.model.predict(X_scaled)[0]
            pred_tokens = np.expm1(pred_log)  # Inverse of log1p
        else:
            pred_tokens = self.model.predict(X_scaled)[0]
        
        # Clip to reasonable range
        pred_tokens = np.clip(pred_tokens, 500, 3000)
        
        return pred_tokens
    
    def determine_n(self, predicted_tokens: float) -> int:
        """
        Map predicted tokens to N value.
        
        Allocation strategy:
        - < 1000 tokens: N=1 (easy)
        - 1000-1400: N=3 (medium)  
        - 1400-1800: N=5 (hard)
        - > 1800: N=8 (very hard)
        
        Args:
            predicted_tokens: Predicted token requirement
            
        Returns:
            N value (1, 3, 5, or 8)
        """
        if predicted_tokens < 1000:
            return 1
        elif predicted_tokens < 1400:
            return 3
        elif predicted_tokens < 1800:
            return 5
        else:
            return 8