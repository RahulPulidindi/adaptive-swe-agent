"""
Create a subset of SWE-bench tasks
"""

import json
import argparse
import random
from pathlib import Path


def create_subset(
    input_file: str,
    output_file: str,
    n_tasks: int = 50,
    seed: int = 42
):
    """
    Create a random subset of tasks.
    
    Args:
        input_file: Path to full dataset JSONL
        output_file: Path to save subset JSONL
        n_tasks: Number of tasks to sample
        seed: Random seed for reproducibility
    """
    # Load all tasks
    tasks = []
    with open(input_file, 'r') as f:
        for line in f:
            tasks.append(json.loads(line))
    
    print(f"Loaded {len(tasks)} tasks from {input_file}")
    
    # Sample subset
    random.seed(seed)
    
    if n_tasks >= len(tasks):
        subset = tasks
        print(f"Using all {len(tasks)} tasks (requested {n_tasks})")
    else:
        subset = random.sample(tasks, n_tasks)
        print(f"Sampled {n_tasks} tasks (seed={seed})")
    
    # Save subset
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        for task in subset:
            f.write(json.dumps(task) + '\n')
    
    print(f"✓ Saved subset to {output_file}")
    
    # Print statistics
    repos = {}
    for task in subset:
        repo = task.get('repo', 'unknown')
        repos[repo] = repos.get(repo, 0) + 1
    
    print(f"\nRepository distribution:")
    for repo, count in sorted(repos.items(), key=lambda x: -x[1])[:10]:
        print(f"  {repo}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Create task subset")
    parser.add_argument("--input", default="data/swebench_lite.jsonl", help="Input file")
    parser.add_argument("--output", default="data/swebench_subset_50.jsonl", help="Output file")
    parser.add_argument("--n-tasks", type=int, default=50, help="Number of tasks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    create_subset(
        input_file=args.input,
        output_file=args.output,
        n_tasks=args.n_tasks,
        seed=args.seed
    )


if __name__ == "__main__":
    main()