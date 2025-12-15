"""
Fixed N=10 agent for baseline comparison
"""

import time
from typing import Dict, List
from .base_agent import BaseAgent


class FixedAgent(BaseAgent):
    """
    Fixed N=10 agent that always generates 10 solutions.
    
    Represents the maximum compute allocation strategy.
    """
    
    def __init__(self, *args, fixed_n: int = 10, **kwargs):
        """
        Initialize fixed-N agent.
        
        Args:
            fixed_n: Number of solutions to generate (default: 10)
        """
        super().__init__(*args, **kwargs)
        self.fixed_n = fixed_n
        
    def solve_fixed(
        self,
        task: Dict,
        early_stop: bool = False
    ) -> Dict:
        """
        Solve task with fixed N=10.
        
        Args:
            task: Task dictionary
            early_stop: Whether to stop early (default: False for fair comparison)
            
        Returns:
            Solution dictionary with best patch
        """
        print(f"  Fixed N={self.fixed_n} (no early stop)")
        
        # Generate fixed_n solutions
        solutions = []
        start_time = time.time()
        total_tokens = 0
        
        for i in range(self.fixed_n):
            result = self.solve(
                problem_statement=task['problem_statement'],
                repo=task['repo'],
                base_commit=task['base_commit'],
                instance_id=task['instance_id']
            )
            
            solutions.append(result)
            total_tokens += result.get('tokens_used', 0)
            
            # Optional early stopping
            if early_stop and result['success'] and len(result['model_patch']) > 100:
                print(f"    Early stop at attempt {i+1}/{self.fixed_n}")
                break
        
        # Select best solution
        best_solution = self._select_best_solution(solutions)
        
        # Add metadata
        best_solution['n_used'] = len(solutions)
        best_solution['n_allocated'] = self.fixed_n
        best_solution['total_tokens'] = total_tokens
        best_solution['total_duration'] = time.time() - start_time
        
        return best_solution
    
    def _select_best_solution(self, solutions: List[Dict]) -> Dict:
        """Select best solution (same logic as AdaptiveAgent)."""
        for sol in solutions:
            if sol['success'] and len(sol.get('model_patch', '')) > 100:
                return sol
        
        solutions_sorted = sorted(
            solutions,
            key=lambda x: len(x.get('model_patch', '')),
            reverse=True
        )
        
        return solutions_sorted[0] if solutions_sorted else solutions[0]