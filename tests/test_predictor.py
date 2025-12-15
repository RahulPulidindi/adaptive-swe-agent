"""
Unit tests for complexity predictor
"""

import pytest
import numpy as np
from unittest.mock import Mock
from complexity_predictor import ComplexityPredictor, extract_code_metrics, extract_repo_features


class TestComplexityPredictor:
    """Tests for ComplexityPredictor"""
    
    def test_determine_n_easy(self):
        """Test N determination for easy tasks"""
        mock_model = Mock()
        mock_scaler = Mock()
        
        predictor = ComplexityPredictor(
            model=mock_model,
            scaler=mock_scaler,
            feature_names=['feat1', 'feat2']
        )
        
        assert predictor.determine_n(500) == 1
        assert predictor.determine_n(999) == 1
    
    def test_determine_n_medium(self):
        """Test N determination for medium tasks"""
        mock_model = Mock()
        mock_scaler = Mock()
        
        predictor = ComplexityPredictor(
            model=mock_model,
            scaler=mock_scaler,
            feature_names=['feat1', 'feat2']
        )
        
        assert predictor.determine_n(1000) == 3
        assert predictor.determine_n(1200) == 3
        assert predictor.determine_n(1399) == 3
    
    def test_determine_n_hard(self):
        """Test N determination for hard tasks"""
        mock_model = Mock()
        mock_scaler = Mock()
        
        predictor = ComplexityPredictor(
            model=mock_model,
            scaler=mock_scaler,
            feature_names=['feat1', 'feat2']
        )
        
        assert predictor.determine_n(1400) == 5
        assert predictor.determine_n(1600) == 5
        assert predictor.determine_n(1799) == 5
    
    def test_determine_n_very_hard(self):
        """Test N determination for very hard tasks"""
        mock_model = Mock()
        mock_scaler = Mock()
        
        predictor = ComplexityPredictor(
            model=mock_model,
            scaler=mock_scaler,
            feature_names=['feat1', 'feat2']
        )
        
        assert predictor.determine_n(1800) == 8
        assert predictor.determine_n(2000) == 8
        assert predictor.determine_n(3000) == 8
    
    def test_predict_clips_values(self):
        """Test prediction clipping to 500-3000 range"""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([5000])  # Too high
        
        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.array([[1, 2]])
        
        predictor = ComplexityPredictor(
            model=mock_model,
            scaler=mock_scaler,
            feature_names=['char_count', 'word_count']
        )
        
        task = {'problem_statement': 'Fix bug', 'repo': 'test/repo'}
        prediction = predictor.predict(task)
        
        assert prediction == 3000  # Clipped
    
    def test_predict_with_log_transform(self):
        """Test prediction with log transform"""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([np.log1p(1500)])
        
        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.array([[1, 2]])
        
        predictor = ComplexityPredictor(
            model=mock_model,
            scaler=mock_scaler,
            feature_names=['char_count', 'word_count'],
            transform_config={'use_log': True}
        )
        
        task = {'problem_statement': 'Fix bug', 'repo': 'test/repo'}
        prediction = predictor.predict(task)
        
        assert 1400 <= prediction <= 1600  # Should be close to 1500


class TestFeatureExtraction:
    """Tests for feature extraction"""
    
    def test_extract_code_metrics_basic(self):
        """Test basic code metrics extraction"""
        problem = "This is a simple bug report with 10 words total here."
        
        features = extract_code_metrics(problem)
        
        assert features['char_count'] == len(problem)
        assert features['word_count'] == 10
        assert features['line_count'] == 1
        assert features['has_code_block'] == 0
        assert features['has_traceback'] == 0
        assert features['has_error'] == 0
    
    def test_extract_code_metrics_with_code_block(self):
        """Test extraction with code blocks"""
        problem = """There's a bug in this code:
```python
def foo():
    pass
```
It doesn't work."""
        
        features = extract_code_metrics(problem)
        
        assert features['has_code_block'] == 1
        assert features['code_block_count'] == 1
        assert features['line_count'] > 1
    
    def test_extract_code_metrics_with_error(self):
        """Test extraction with error mentions"""
        problem = """I got this traceback:
Error: ValueError occurred"""
        
        features = extract_code_metrics(problem)
        
        assert features['has_traceback'] == 1
        assert features['has_error'] == 1
    
    def test_extract_code_metrics_with_files(self):
        """Test extraction with file mentions"""
        problem = "Bug in models.py and views.py files"
        
        features = extract_code_metrics(problem)
        
        assert features['file_mentions'] == 2
    
    def test_extract_repo_features(self):
        """Test repository feature extraction"""
        all_tasks = [
            {'repo': 'django/django', 'instance_id': '1'},
            {'repo': 'django/django', 'instance_id': '2'},
            {'repo': 'sympy/sympy', 'instance_id': '3'}
        ]
        
        features = extract_repo_features('django/django', all_tasks)
        
        assert features['repo_task_count'] == 2
        assert 'repo_avg_difficulty' in features