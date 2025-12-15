"""
Base agent with N=1 (single attempt) - baseline implementation
"""

import os
import time
import json
from typing import Dict, Optional, List
from openai import OpenAI


class BaseAgent:
    """
    Baseline agent that generates a single solution (N=1).
    
    This serves as the baseline for comparison with adaptive approaches.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5.1",
        temperature: float = 0.7,
        max_completion_tokens: int = 4096
    ):
        """
        Initialize baseline agent.
        
        Args:
            api_key: OpenAI API key
            model: Model name (default: gpt-5.1)
            temperature: Sampling temperature
            max_completion_tokens: Maximum tokens to generate
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        
    def solve(
        self,
        problem_statement: str,
        repo: str,
        base_commit: str,
        instance_id: str
    ) -> Dict:
        """
        Solve a single task with one attempt (N=1).
        
        Args:
            problem_statement: GitHub issue description
            repo: Repository name (e.g., 'django/django')
            base_commit: Git commit hash
            instance_id: Unique task identifier
            
        Returns:
            Dictionary with solution details:
            {
                'instance_id': str,
                'model_patch': str,
                'tokens_used': int,
                'duration': float,
                'success': bool
            }
        """
        start_time = time.time()
        
        # Construct system prompt
        system_prompt = self._get_system_prompt()
        
        # Construct user prompt
        user_prompt = self._get_user_prompt(
            problem_statement=problem_statement,
            repo=repo
        )
        
        try:
            # Generate solution
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_completion_tokens=self.max_completion_tokens
            )
            
            # Extract patch
            patch = response.choices[0].message.content.strip()
            patch = self._clean_patch(patch)
            
            # Calculate metrics
            tokens_used = response.usage.total_tokens
            duration = time.time() - start_time
            
            return {
                'instance_id': instance_id,
                'model_patch': patch,
                'tokens_used': tokens_used,
                'duration': duration,
                'success': len(patch) > 0
            }
            
        except Exception as e:
            return {
                'instance_id': instance_id,
                'model_patch': '',
                'tokens_used': 0,
                'duration': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for the agent."""
        return """You are an expert software engineer tasked with resolving GitHub issues.

Your goal is to:
1. Understand the issue thoroughly
2. Locate the relevant code in the repository
3. Implement a fix that resolves the issue
4. Generate a git diff patch in unified diff format

The patch should:
- Be in unified diff format (diff --git a/... b/...)
- Include proper context lines (usually 3 lines before/after changes)
- Modify only what's necessary to fix the issue
- Be syntactically correct and ready to apply

Respond with ONLY the patch, no explanations or markdown formatting."""
    
    def _get_user_prompt(self, problem_statement: str, repo: str) -> str:
        """Get user prompt with task details."""
        return f"""Repository: {repo}

Issue to fix:
{problem_statement}

Generate a git diff patch that fixes this issue."""
    
    def _clean_patch(self, patch: str) -> str:
        """
        Clean patch by removing markdown formatting.
        
        Args:
            patch: Raw patch from LLM
            
        Returns:
            Cleaned patch string
        """
        # Remove markdown code blocks
        if "```diff" in patch:
            patch = patch.split("```diff")[1].split("```")[0].strip()
        elif "```" in patch:
            patch = patch.split("```")[1].split("```")[0].strip()
        
        return patch