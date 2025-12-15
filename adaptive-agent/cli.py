"""
Command-line interface for adaptive agent
"""

import argparse
import os
import json
from dotenv import load_dotenv

from adaptive_agent import BaseAgent, AdaptiveAgent, FixedAgent
from complexity_predictor import ComplexityPredictor
from repository_manager import RepositoryManager


def solve_single_task(args):
    """Solve a single task"""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found")
    
    # Load task
    with open(args.task_file, 'r') as f:
        task = json.load(f)
    
    # Initialize agent based on mode
    if args.mode == 'baseline':
        agent = BaseAgent(api_key=api_key, model=args.model)
        result = agent.solve(
            problem_statement=task['problem_statement'],
            repo=task['repo'],
            base_commit=task['base_commit'],
            instance_id=task['instance_id']
        )
    
    elif args.mode == 'adaptive':
        predictor = ComplexityPredictor.load(args.predictor)
        agent = AdaptiveAgent(api_key=api_key, model=args.model)
        result = agent.solve_adaptive(task=task, predictor=predictor)
    
    elif args.mode == 'fixed':
        agent = FixedAgent(api_key=api_key, model=args.model, fixed_n=args.fixed_n)
        result = agent.solve_fixed(task=task)
    
    # Print result
    print(f"\n{'='*60}")
    print(f"RESULT")
    print(f"{'='*60}")
    print(f"Instance: {result['instance_id']}")
    print(f"Success: {result['success']}")
    print(f"Tokens: {result.get('tokens_used', 0)}")
    
    if args.mode in ['adaptive', 'fixed']:
        print(f"N used: {result.get('n_used', 1)}")
    
    print(f"\nPatch length: {len(result.get('model_patch', ''))} chars")
    
    # Save patch
    if args.output:
        with open(args.output, 'w') as f:
            f.write(result['model_patch'])
        print(f"✓ Saved patch to {args.output}")


def validate_task(args):
    """Validate a task's patch"""
    # Load task
    with open(args.task_file, 'r') as f:
        task = json.load(f)
    
    # Load patch
    with open(args.patch_file, 'r') as f:
        patch = f.read()
    
    # Get repository
    repo_manager = RepositoryManager()
    repo_path = repo_manager.get_repo(task['repo'], task['base_commit'])
    
    # Apply patch
    result = repo_manager.apply_patch(repo_path, patch)
    
    print(f"\n{'='*60}")
    print(f"VALIDATION RESULT")
    print(f"{'='*60}")
    print(f"Instance: {task['instance_id']}")
    print(f"Repo: {task['repo']}")
    print(f"Success: {result['success']}")
    
    if not result['success']:
        print(f"\nError:\n{result['error']}")


def main():
    parser = argparse.ArgumentParser(description="Adaptive SWE Agent CLI")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Solve command
    solve_parser = subparsers.add_parser('solve', help='Solve a task')
    solve_parser.add_argument('--task', dest='task_file', required=True, help='Task JSON file')
    solve_parser.add_argument('--mode', choices=['baseline', 'adaptive', 'fixed'], 
                             default='adaptive', help='Agent mode')
    solve_parser.add_argument('--model', default='gpt-5.1', help='Model name')
    solve_parser.add_argument('--predictor', default='models/complexity_predictor.pkl',
                             help='Predictor model (for adaptive mode)')
    solve_parser.add_argument('--fixed-n', type=int, default=10, 
                             help='N value for fixed mode')
    solve_parser.add_argument('--output', help='Output file for patch')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate a patch')
    validate_parser.add_argument('--task', dest='task_file', required=True, help='Task JSON file')
    validate_parser.add_argument('--patch', dest='patch_file', required=True, help='Patch file')
    
    args = parser.parse_args()
    
    if args.command == 'solve':
        solve_single_task(args)
    elif args.command == 'validate':
        validate_task(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()