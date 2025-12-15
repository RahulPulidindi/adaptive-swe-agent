"""
Adaptive agent with complexity-based N allocation
"""

import time
import numpy as np
from typing import Dict, List, Optional
from .base_agent import BaseAgent


class AdaptiveAgent(BaseAgent):
    """
    Adaptive agent that allocates N based on predicted task complexity.
    
    Uses best-of-N sampling where N is determined by complexity prediction.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize adaptive agent (inherits from BaseAgent)."""
        super().__init__(*args, **kwargs)
        
    def solve_adaptive(
        self,
        task: Dict,
        predictor,
        max_n: int = 8,
        early_stop: bool = True
    ) -> Dict:
        """
        Solve task with adaptive N allocation.
        
        Args:
            task: Task dictionary with problem_statement, repo, base_commit, instance_id
            predictor: Trained ComplexityPredictor instance
            max_n: Maximum N to allocate
            early_stop: Stop early if valid patch found
            
        Returns:
            Solution dictionary with best patch and metadata
        """
        # Predict complexity and determine N
        predicted_tokens = predictor.predict(task)
        n = predictor.determine_n(predicted_tokens)
        n = min(n, max_n)  # Cap at max_n
        
        print(f"  Predicted: {predicted_tokens} tokens → N={n}")
        
        # Generate N solutions
        solutions = []
        start_time = time.time()
        total_tokens = 0
        
        for i in range(n):
            result = self.solve(
                problem_statement=task['problem_statement'],
                repo=task['repo'],
                base_commit=task['base_commit'],
                instance_id=task['instance_id']
            )
            
            solutions.append(result)
            total_tokens += result.get('tokens_used', 0)
            
            # Early stopping if we found a valid patch
            if early_stop and result['success'] and len(result['model_patch']) > 100:
                print(f"    Early stop at attempt {i+1}/{n}")
                break
        
        # Select best solution (first successful one, or longest patch)
        best_solution = self._select_best_solution(solutions)
        
        # Add metadata
        best_solution['n_used'] = len(solutions)
        best_solution['n_allocated'] = n
        best_solution['predicted_tokens'] = predicted_tokens
        best_solution['total_tokens'] = total_tokens
        best_solution['total_duration'] = time.time() - start_time
        
        return best_solution
    
    def _select_best_solution(self, solutions: List[Dict]) -> Dict:
        """
        Select best solution from multiple attempts.
        
        Selection criteria:
        1. First solution with non-empty patch
        2. If all empty, return first one
        
        Args:
            solutions: List of solution dictionaries
            
        Returns:
            Best solution dictionary
        """
        # Find first successful solution
        for sol in solutions:
            if sol['success'] and len(sol.get('model_patch', '')) > 100:
                return sol
        
        # If none successful, return solution with longest patch
        solutions_sorted = sorted(
            solutions,
            key=lambda x: len(x.get('model_patch', '')),
            reverse=True
        )
        
        return solutions_sorted[0] if solutions_sorted else solutions[0]