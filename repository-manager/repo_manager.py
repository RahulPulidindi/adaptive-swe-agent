"""
Repository management for SWE-bench tasks
"""

import os
import subprocess
import tempfile
import shutil
from typing import Dict, Optional
from pathlib import Path


class RepositoryManager:
    """
    Manages git repositories for SWE-bench evaluation.
    
    Handles cloning, checkout, and cleanup operations.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize repository manager.
        
        Args:
            cache_dir: Directory to cache cloned repositories
        """
        self.cache_dir = cache_dir or tempfile.mkdtemp(prefix="swe_repos_")
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def get_repo(self, repo: str, base_commit: str) -> str:
        """
        Clone and checkout repository at specific commit.
        
        Args:
            repo: Repository name (e.g., 'django/django')
            base_commit: Git commit hash to checkout
            
        Returns:
            Path to repository directory
        """
        # Sanitize repo name for directory
        repo_dir_name = repo.replace('/', '_')
        repo_path = os.path.join(self.cache_dir, repo_dir_name, base_commit[:8])
        
        # If already exists, return
        if os.path.exists(repo_path):
            return repo_path
        
        # Clone repository
        print(f"  Cloning {repo} ...")
        
        # Create parent directory
        os.makedirs(os.path.dirname(repo_path), exist_ok=True)
        
        # Clone with depth 1 for speed
        repo_url = f"https://github.com/{repo}.git"
        
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, repo_path],
                check=True,
                capture_output=True,
                timeout=300
            )
        except subprocess.TimeoutExpired:
            # Try without depth limit if timeout
            subprocess.run(
                ["git", "clone", repo_url, repo_path],
                check=True,
                capture_output=True,
                timeout=600
            )
        
        # Fetch specific commit if needed
        try:
            subprocess.run(
                ["git", "checkout", base_commit],
                cwd=repo_path,
                check=True,
                capture_output=True
            )
        except subprocess.CalledProcessError:
            # Need to fetch more history
            subprocess.run(
                ["git", "fetch", "--depth", "1000"],
                cwd=repo_path,
                check=True,
                capture_output=True
            )
            subprocess.run(
                ["git", "checkout", base_commit],
                cwd=repo_path,
                check=True,
                capture_output=True
            )
        
        print(f"  ✓ Ready at commit {base_commit[:8]}")
        
        return repo_path
    
    def apply_patch(self, repo_path: str, patch: str) -> Dict:
        """
        Apply patch to repository.
        
        Args:
            repo_path: Path to repository
            patch: Unified diff patch
            
        Returns:
            Dictionary with application result:
            {
                'success': bool,
                'error': Optional[str]
            }
        """
        # Write patch to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
            f.write(patch)
            patch_file = f.name
        
        try:
            # Try to apply patch
            result = subprocess.run(
                ["git", "apply", "--check", patch_file],
                cwd=repo_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # Actually apply the patch
                subprocess.run(
                    ["git", "apply", patch_file],
                    cwd=repo_path,
                    check=True,
                    capture_output=True
                )
                return {'success': True, 'error': None}
            else:
                return {
                    'success': False,
                    'error': result.stderr
                }
                
        except subprocess.CalledProcessError as e:
            return {
                'success': False,
                'error': e.stderr.decode() if hasattr(e.stderr, 'decode') else str(e)
            }
        finally:
            # Clean up patch file
            if os.path.exists(patch_file):
                os.remove(patch_file)
    
    def reset_repo(self, repo_path: str):
        """
        Reset repository to clean state.
        
        Args:
            repo_path: Path to repository
        """
        subprocess.run(
            ["git", "reset", "--hard", "HEAD"],
            cwd=repo_path,
            check=True,
            capture_output=True
        )
        subprocess.run(
            ["git", "clean", "-fdx"],
            cwd=repo_path,
            check=True,
            capture_output=True
        )
    
    def cleanup(self):
        """Remove all cached repositories."""
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)