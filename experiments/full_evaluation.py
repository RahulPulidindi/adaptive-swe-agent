"""
Full evaluation on 50 tasks (or more)
WARNING: This can take 2+ hours and consume significant API credits
"""

import os
import json
import argparse
import time
from dotenv import load_dotenv

from adaptive_agent import AdaptiveAgent
from complexity_predictor import ComplexityPredictor
from repository_manager import RepositoryManager


def load_tasks(data_file: str, n_tasks: int = 50) -> list:
    """Load tasks from JSONL file."""
    tasks = []
    with open(data_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= n_tasks:
                break
            tasks.append(json.loads(line))
    return tasks


def run_full_evaluation(
    tasks: list,
    api_key: str,
    predictor: ComplexityPredictor,
    output_file: str,
    checkpoint_freq: int = 10
):
    """
    Run full evaluation with checkpointing.
    
    Args:
        tasks: List of task dictionaries
        api_key: OpenAI API key
        predictor: Trained complexity predictor
        output_file: Path to save predictions JSONL
        checkpoint_freq: Save checkpoint every N tasks
    """
    # Initialize agent
    agent = AdaptiveAgent(api_key=api_key, model="gpt-5.1")
    
    # Initialize repository manager
    repo_manager = RepositoryManager()
    
    # Load existing results if any
    results = []
    completed_ids = set()
    
    if os.path.exists(output_file):
        print(f"Loading existing results from {output_file}")
        with open(output_file, 'r') as f:
            for line in f:
                pred = json.loads(line)
                results.append(pred)
                completed_ids.add(pred['instance_id'])
        print(f"Resuming from {len(results)} completed tasks")
    
    # Filter tasks
    tasks_to_run = [t for t in tasks if t['instance_id'] not in completed_ids]
    
    print(f"\n{'='*60}")
    print(f"FULL EVALUATION - {len(tasks_to_run)} tasks remaining")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    for i, task in enumerate(tasks_to_run, 1):
        print(f"[{i}/{len(tasks_to_run)}] {task['instance_id']}")
        
        try:
            # Get repository
            repo_path = repo_manager.get_repo(task['repo'], task['base_commit'])
            
            # Solve with adaptive allocation
            result = agent.solve_adaptive(
                task=task,
                predictor=predictor,
                max_n=8
            )
            
            # Check if patch applies
            if result['success']:
                apply_result = repo_manager.apply_patch(repo_path, result['model_patch'])
                if apply_result['success']:
                    print("  ✓ Patch applies cleanly")
                else:
                    print(f"  ✗ Patch failed: {apply_result['error'][:50]}")
                repo_manager.reset_repo(repo_path)
            
            # Format for SWE-bench harness
            prediction = {
                'instance_id': task['instance_id'],
                'model_name_or_path': 'gpt-5.1-adaptive',
                'model_patch': result['model_patch']
            }
            
            results.append(prediction)
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            # Add empty prediction
            results.append({
                'instance_id': task['instance_id'],
                'model_name_or_path': 'gpt-5.1-adaptive',
                'model_patch': ''
            })
        
        # Checkpoint
        if i % checkpoint_freq == 0:
            print(f"\n  Checkpointing at {i}/{len(tasks_to_run)}...")
            with open(output_file, 'w') as f:
                for pred in results:
                    f.write(json.dumps(pred) + '\n')
            
            elapsed = time.time() - start_time
            eta = (elapsed / i) * (len(tasks_to_run) - i)
            print(f"  Elapsed: {elapsed/60:.1f}m, ETA: {eta/60:.1f}m\n")
    
    # Final save
    print(f"\nSaving final results...")
    with open(output_file, 'w') as f:
        for pred in results:
            f.write(json.dumps(pred) + '\n')
    
    elapsed = time.time() - start_time
    non_empty = sum(1 for r in results if len(r['model_patch']) > 0)
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Non-empty patches: {non_empty}/{len(results)} ({non_empty/len(results)*100:.1f}%)")
    print(f"Output: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run full evaluation")
    parser.add_argument("--data", default="data/swebench_subset_50.jsonl", help="Path to data file")
    parser.add_argument("--n-tasks", type=int, default=50, help="Number of tasks")
    parser.add_argument("--predictor", default="models/complexity_predictor.pkl", help="Predictor model")
    parser.add_argument("--output", default="results/full_evaluation.jsonl", help="Output file")
    parser.add_argument("--checkpoint-freq", type=int, default=10, help="Checkpoint frequency")
    
    args = parser.parse_args()
    
    # Confirm before running
    print(f"WARNING: This will evaluate {args.n_tasks} tasks")
    print(f"Estimated time: {args.n_tasks * 0.5:.0f}-{args.n_tasks * 1:.0f} minutes")
    print(f"Estimated cost: ${args.n_tasks * 0.10:.2f}-${args.n_tasks * 0.20:.2f} (approximate)")
    
    response = input("\nContinue? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Load environment
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment")
    
    # Load tasks
    tasks = load_tasks(args.data, args.n_tasks)
    
    # Load predictor
    predictor = ComplexityPredictor.load(args.predictor)
    
    # Run evaluation
    run_full_evaluation(
        tasks=tasks,
        api_key=api_key,
        predictor=predictor,
        output_file=args.output,
        checkpoint_freq=args.checkpoint_freq
    )


if __name__ == "__main__":
    main()