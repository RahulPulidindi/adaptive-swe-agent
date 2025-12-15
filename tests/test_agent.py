"""
Unit tests for agent implementations
"""

import pytest
from unittest.mock import Mock, patch
from adaptive_agent import BaseAgent, AdaptiveAgent, FixedAgent


class TestBaseAgent:
    """Tests for BaseAgent"""
    
    def test_init(self):
        """Test agent initialization"""
        agent = BaseAgent(api_key="test-key", model="gpt-5.1")
        assert agent.model == "gpt-5.1"
        assert agent.temperature == 0.7
        assert agent.max_completion_tokens == 4096
    
    def test_clean_patch_with_markdown(self):
        """Test patch cleaning with markdown"""
        agent = BaseAgent(api_key="test-key")
        
        patch_with_markdown = """```diff
diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -1,3 +1,3 @@
-old line
+new line
```"""
        
        cleaned = agent._clean_patch(patch_with_markdown)
        assert "```" not in cleaned
        assert "diff --git" in cleaned
    
    def test_clean_patch_without_markdown(self):
        """Test patch cleaning without markdown"""
        agent = BaseAgent(api_key="test-key")
        
        patch = """diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -1,3 +1,3 @@
-old line
+new line"""
        
        cleaned = agent._clean_patch(patch)
        assert cleaned == patch
    
    @patch('adaptive_agent.base_agent.OpenAI')
    def test_solve(self, mock_openai):
        """Test solve method"""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="diff --git a/file.py b/file.py\n"))]
        mock_response.usage = Mock(total_tokens=1000)
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        agent = BaseAgent(api_key="test-key")
        
        result = agent.solve(
            problem_statement="Fix the bug",
            repo="test/repo",
            base_commit="abc123",
            instance_id="test-1"
        )
        
        assert result['instance_id'] == "test-1"
        assert result['success'] is True
        assert result['tokens_used'] == 1000
        assert 'diff --git' in result['model_patch']


class TestAdaptiveAgent:
    """Tests for AdaptiveAgent"""
    
    @patch('adaptive_agent.base_agent.OpenAI')
    def test_solve_adaptive_with_n3(self, mock_openai):
        """Test adaptive solve with N=3"""
        # Mock predictor
        mock_predictor = Mock()
        mock_predictor.predict.return_value = 1200  # Should give N=3
        mock_predictor.determine_n.return_value = 3
        
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="diff --git a/file.py b/file.py\nvalid patch"))]
        mock_response.usage = Mock(total_tokens=500)
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        agent = AdaptiveAgent(api_key="test-key")
        
        task = {
            'problem_statement': 'Fix bug',
            'repo': 'test/repo',
            'base_commit': 'abc123',
            'instance_id': 'test-1'
        }
        
        result = agent.solve_adaptive(task=task, predictor=mock_predictor, early_stop=False)
        
        assert result['n_allocated'] == 3
        assert result['n_used'] <= 3
        assert result['predicted_tokens'] == 1200
    
    def test_select_best_solution_success(self):
        """Test solution selection with successful patches"""
        agent = AdaptiveAgent(api_key="test-key")
        
        solutions = [
            {'success': False, 'model_patch': ''},
            {'success': True, 'model_patch': 'a' * 200},  # Should select this
            {'success': True, 'model_patch': 'b' * 300}
        ]
        
        best = agent._select_best_solution(solutions)
        assert best['model_patch'] == 'a' * 200  # First successful


class TestFixedAgent:
    """Tests for FixedAgent"""
    
    def test_init_with_custom_n(self):
        """Test initialization with custom N"""
        agent = FixedAgent(api_key="test-key", fixed_n=15)
        assert agent.fixed_n == 15
    
    @patch('adaptive_agent.base_agent.OpenAI')
    def test_solve_fixed(self, mock_openai):
        """Test fixed-N solve"""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="patch"))]
        mock_response.usage = Mock(total_tokens=500)
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        agent = FixedAgent(api_key="test-key", fixed_n=5)
        
        task = {
            'problem_statement': 'Fix bug',
            'repo': 'test/repo',
            'base_commit': 'abc123',
            'instance_id': 'test-1'
        }
        
        result = agent.solve_fixed(task=task, early_stop=False)
        
        assert result['n_allocated'] == 5
        assert result['n_used'] == 5  # No early stop