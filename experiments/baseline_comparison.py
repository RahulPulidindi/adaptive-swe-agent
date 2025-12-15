"""
Baseline vs Adaptive vs Fixed-10 comparison experiment
"""

import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

from adaptive_agent import BaseAgent, AdaptiveAgent, FixedAgent
from complexity_predictor import ComplexityPredictor
from repository_manager import RepositoryManager


def load_tasks(data_file: str, n_tasks: int = 5) -> list:
    """Load tasks from JSONL file."""
    tasks = []
    with open(data_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= n_tasks:
                break
            tasks.append(json.loads(line))
    return tasks


def run_comparison(
    tasks: list,
    api_key: str,
    predictor: ComplexityPredictor,
    output_file: str
):
    """
    Run comparison of Baseline vs Adaptive vs Fixed-10.
    
    Args:
        tasks: List of task dictionaries
        api_key: OpenAI API key
        predictor: Trained complexity predictor
        output_file: Path to save results CSV
    """
    # Initialize agents
    baseline_agent = BaseAgent(api_key=api_key, model="gpt-5.1")
    adaptive_agent = AdaptiveAgent(api_key=api_key, model="gpt-5.1")
    fixed_agent = FixedAgent(api_key=api_key, model="gpt-5.1", fixed_n=10)
    
    # Initialize repository manager
    repo_manager = RepositoryManager()
    
    results = []
    
    print("=" * 60)
    print("BASELINE vs ADAPTIVE vs FIXED-10 EVALUATION")
    print("=" * 60)
    print(f"Running on {len(tasks)} tasks")
    print("=" * 60)
    print()
    
    for i, task in enumerate(tasks, 1):
        print(f"\n[{i}/{len(tasks)}] {task['instance_id']}")
        print(f"Repo: {task['repo']}")
        
        # Get repository
        repo_path = repo_manager.get_repo(task['repo'], task['base_commit'])
        
        result = {
            'instance_id': task['instance_id'],
            'repo': task['repo']
        }
        
        # Run baseline (N=1)
        baseline_result = baseline_agent.solve(
            problem_statement=task['problem_statement'],
            repo=task['repo'],
            base_commit=task['base_commit'],
            instance_id=task['instance_id']
        )
        
        # Check if patch applies
        baseline_applied = False
        if baseline_result['success']:
            apply_result = repo_manager.apply_patch(repo_path, baseline_result['model_patch'])
            baseline_applied = apply_result['success']
            repo_manager.reset_repo(repo_path)
        
        result['baseline_tokens'] = baseline_result['tokens_used']
        result['baseline_time'] = baseline_result['duration']
        result['baseline_has_patch'] = baseline_result['success']
        result['baseline_patch_applied'] = baseline_applied
        result['baseline_patch'] = baseline_result['model_patch']
        
        print(f"  Baseline: tokens={baseline_result['tokens_used']} applied={baseline_applied}")
        
        # Run adaptive
        adaptive_result = adaptive_agent.solve_adaptive(
            task=task,
            predictor=predictor,
            max_n=8
        )
        
        adaptive_applied = False
        if adaptive_result['success']:
            apply_result = repo_manager.apply_patch(repo_path, adaptive_result['model_patch'])
            adaptive_applied = apply_result['success']
            repo_manager.reset_repo(repo_path)
        
        result['adaptive_tokens'] = adaptive_result['total_tokens']
        result['adaptive_time'] = adaptive_result['total_duration']
        result['adaptive_n'] = adaptive_result['n_used']
        result['adaptive_predicted_tokens'] = adaptive_result['predicted_tokens']
        result['adaptive_has_patch'] = adaptive_result['success']
        result['adaptive_patch_applied'] = adaptive_applied
        result['adaptive_patch'] = adaptive_result['model_patch']
        
        print(f"  Adaptive: N={adaptive_result['n_used']} tokens={adaptive_result['total_tokens']} applied={adaptive_applied}")
        
        # Run fixed-10
        fixed_result = fixed_agent.solve_fixed(task=task, early_stop=False)
        
        fixed_applied = False
        if fixed_result['success']:
            apply_result = repo_manager.apply_patch(repo_path, fixed_result['model_patch'])
            fixed_applied = apply_result['success']
            repo_manager.reset_repo(repo_path)
        
        result['fixed10_tokens'] = fixed_result['total_tokens']
        result['fixed10_time'] = fixed_result['total_duration']
        result['fixed10_n'] = fixed_result['n_used']
        result['fixed10_has_patch'] = fixed_result['success']
        result['fixed10_patch_applied'] = fixed_applied
        result['fixed10_patch'] = fixed_result['model_patch']
        
        print(f"  Fixed-10: N={fixed_result['n_used']} tokens={fixed_result['total_tokens']} applied={fixed_applied}")
        
        results.append(result)
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    baseline_apply_rate = df['baseline_patch_applied'].mean() * 100
    adaptive_apply_rate = df['adaptive_patch_applied'].mean() * 100
    fixed10_apply_rate = df['fixed10_patch_applied'].mean() * 100
    
    print(f"Apply rate:")
    print(f"  Baseline: {baseline_apply_rate:.1f}%")
    print(f"  Adaptive: {adaptive_apply_rate:.1f}%")
    print(f"  Fixed-10: {fixed10_apply_rate:.1f}%")
    
    print(f"\nAvg tokens:")
    print(f"  Baseline: {df['baseline_tokens'].mean():.0f}")
    print(f"  Adaptive: {df['adaptive_tokens'].mean():.0f}")
    print(f"  Fixed-10: {df['fixed10_tokens'].mean():.0f}")
    
    print(f"\n✓ Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run baseline comparison experiment")
    parser.add_argument("--data", default="data/swebench_subset_50.jsonl", help="Path to data file")
    parser.add_argument("--n-tasks", type=int, default=5, help="Number of tasks")
    parser.add_argument("--predictor", default="models/complexity_predictor.pkl", help="Predictor model")
    parser.add_argument("--output", default="results/comparison.csv", help="Output file")
    
    args = parser.parse_args()
    
    # Load environment
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment")
    
    # Load tasks
    tasks = load_tasks(args.data, args.n_tasks)
    print(f"Loaded {len(tasks)} tasks")
    
    # Load predictor
    predictor = ComplexityPredictor.load(args.predictor)
    print(f"Loaded predictor from {args.predictor}")
    
    # Run comparison
    run_comparison(
        tasks=tasks,
        api_key=api_key,
        predictor=predictor,
        output_file=args.output
    )


if __name__ == "__main__":
    main()