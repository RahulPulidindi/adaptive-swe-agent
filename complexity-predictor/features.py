"""
Feature extraction for complexity prediction
"""

import re
import numpy as np
from typing import Dict, List
from collections import Counter


def extract_code_metrics(problem_statement: str) -> Dict:
    """
    Extract text-based features from problem statement.
    
    Args:
        problem_statement: GitHub issue description
        
    Returns:
        Dictionary of text features
    """
    # Basic text statistics
    char_count = len(problem_statement)
    words = problem_statement.split()
    word_count = len(words)
    line_count = len(problem_statement.split('\n'))
    
    avg_word_length = np.mean([len(w) for w in words]) if words else 0
    
    # Code block detection
    has_code_block = int('```' in problem_statement)
    code_block_count = problem_statement.count('```') // 2
    
    # Error/traceback detection
    has_traceback = int('traceback' in problem_statement.lower())
    has_error = int('error' in problem_statement.lower())
    
    # File mentions
    file_mentions = len(re.findall(r'\.py\b', problem_statement))
    
    # Test mentions
    has_test = int('test' in problem_statement.lower())
    
    return {
        'char_count': char_count,
        'word_count': word_count,
        'line_count': line_count,
        'avg_word_length': avg_word_length,
        'has_code_block': has_code_block,
        'code_block_count': code_block_count,
        'has_traceback': has_traceback,
        'has_error': has_error,
        'file_mentions': file_mentions,
        'has_test': has_test
    }


def extract_repo_features(repo: str, all_tasks: List[Dict]) -> Dict:
    """
    Extract repository-based features.
    
    Args:
        repo: Repository name (e.g., 'django/django')
        all_tasks: List of all tasks for computing repo statistics
        
    Returns:
        Dictionary of repository features
    """
    # Count tasks for this repo
    repo_tasks = [t for t in all_tasks if t.get('repo') == repo]
    repo_task_count = len(repo_tasks)
    
    # Average difficulty 
    repo_avg_difficulty = 2000  # Default
    
    return {
        'repo_task_count': repo_task_count,
        'repo_avg_difficulty': repo_avg_difficulty
    }