"""
Adaptive agent evaluation on N tasks
"""

import os
import json
import argparse
import time
from tqdm import tqdm
from dotenv import load_dotenv

from adaptive_agent import AdaptiveAgent
from complexity_predictor import ComplexityPredictor
from repository_manager import RepositoryManager


def load_tasks(data_file: str, n_tasks: int = 10) -> list:
    """Load tasks from JSONL file."""
    tasks = []
    with open(data_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= n_tasks:
                break
            tasks.append(json.loads(line))
    return tasks


def run_adaptive_evaluation(
    tasks: list,
    api_key: str,
    predictor: ComplexityPredictor,
    output_file: str
):
    """
    Run adaptive evaluation on tasks.
    
    Args:
        tasks: List of task dictionaries
        api_key: OpenAI API key
        predictor: Trained complexity predictor
        output_file: Path to save predictions JSONL
    """
    # Initialize agent
    agent = AdaptiveAgent(api_key=api_key, model="gpt-5.1")
    
    # Initialize repository manager
    repo_manager = RepositoryManager()
    
    results = []
    start_time = time.time()
    
    print(f"Loaded {len(tasks)} tasks from: {args.data}")
    print()
    
    with tqdm(total=len(tasks), desc=f"Adaptive solve ({len(tasks)})") as pbar:
        for task in tasks:
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
                    print("    ✓ Patch applies cleanly")
                repo_manager.reset_repo(repo_path)
            
            # Format for SWE-bench harness
            prediction = {
                'instance_id': task['instance_id'],
                'model_name_or_path': 'gpt-5.1-adaptive',
                'model_patch': result['model_patch']
            }
            
            results.append(prediction)
            pbar.update(1)
    
    # Save predictions
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for pred in results:
            f.write(json.dumps(pred) + '\n')
    
    elapsed = time.time() - start_time
    non_empty = sum(1 for r in results if len(r['model_patch']) > 0)
    
    print(f"\n✓ Wrote predictions JSONL: {output_file}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Non-empty patches: {non_empty}/{len(results)} ({non_empty/len(results)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Run adaptive evaluation")
    parser.add_argument("--data", default="data/swebench_subset_50.jsonl", help="Path to data file")
    parser.add_argument("--n-tasks", type=int, default=10, help="Number of tasks")
    parser.add_argument("--predictor", default="models/complexity_predictor.pkl", help="Predictor model")
    parser.add_argument("--output", default="results/adaptive_predictions.jsonl", help="Output file")
    
    args = parser.parse_args()
    
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
    run_adaptive_evaluation(
        tasks=tasks,
        api_key=api_key,
        predictor=predictor,
        output_file=args.output
    )


if __name__ == "__main__":
    main()