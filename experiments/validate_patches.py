"""
Validate patches against actual repositories
"""

import os
import json
import argparse
import pandas as pd
from tqdm import tqdm

from repository_manager import RepositoryManager, PatchValidator


def load_predictions(predictions_file: str) -> list:
    """Load predictions from JSONL file."""
    predictions = []
    with open(predictions_file, 'r') as f:
        for line in f:
            predictions.append(json.loads(line))
    return predictions


def load_task_metadata(data_file: str) -> dict:
    """Load task metadata (repo, base_commit) from data file."""
    metadata = {}
    with open(data_file, 'r') as f:
        for line in f:
            task = json.loads(line)
            metadata[task['instance_id']] = {
                'repo': task['repo'],
                'base_commit': task['base_commit']
            }
    return metadata


def validate_patches(
    predictions: list,
    task_metadata: dict,
    output_file: str
):
    """
    Validate patches by attempting to apply them.
    
    Args:
        predictions: List of prediction dictionaries
        task_metadata: Dictionary mapping instance_id to repo/commit
        output_file: Path to save validation results CSV
    """
    repo_manager = RepositoryManager()
    validator = PatchValidator()
    
    results = []
    
    print("=" * 60)
    print("PATCH VALIDATION")
    print("=" * 60)
    print()
    
    for i, pred in enumerate(tqdm(predictions, desc="Validating"), 1):
        instance_id = pred['instance_id']
        patch = pred['model_patch']
        
        # Get task metadata
        if instance_id not in task_metadata:
            print(f"\n[{i}/{len(predictions)}] {instance_id}: No metadata found")
            continue
        
        meta = task_metadata[instance_id]
        
        print(f"\n[{i}/{len(predictions)}] {instance_id}:")
        print(f"  Cloning {meta['repo']} ...")
        
        # Get repository
        try:
            repo_path = repo_manager.get_repo(meta['repo'], meta['base_commit'])
            print(f"  ✓ Ready at commit {meta['base_commit'][:8]}")
        except Exception as e:
            print(f"  ✗ Clone failed: {e}")
            results.append({
                'instance_id': instance_id,
                'repo': meta['repo'],
                'validation_result': 'clone_failed',
                'error': str(e)
            })
            continue
        
        # Validate patch format
        validation = validator.validate_patch(patch)
        
        # Try to apply patch
        if not patch or len(patch) < 10:
            result_type = 'empty_patch'
            error_msg = 'Patch is empty or too short'
        else:
            apply_result = repo_manager.apply_patch(repo_path, patch)
            
            if apply_result['success']:
                result_type = 'success'
                error_msg = None
                print(f"  ✓ Patch applies cleanly")
            else:
                # Classify error
                result_type = validator.classify_patch_error(apply_result['error'])
                error_msg = apply_result['error']
                print(f"  ✗ Patch fails: {error_msg[:100]}")
        
        # Reset repo
        try:
            repo_manager.reset_repo(repo_path)
        except:
            pass
        
        results.append({
            'instance_id': instance_id,
            'repo': meta['repo'],
            'validation_result': result_type,
            'patch_valid_format': validation['valid'],
            'patch_length': len(patch),
            'error': error_msg
        })
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print()
    
    success_count = (df['validation_result'] == 'success').sum()
    total = len(df)
    
    print(f"Patches that apply: {success_count}/{total} ({success_count/total*100:.0f}%)")
    print()
    print("Failure breakdown:")
    print(df['validation_result'].value_counts())
    
    print(f"\n✓ Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Validate patches")
    parser.add_argument("--predictions", required=True, help="Path to predictions JSONL")
    parser.add_argument("--data", default="data/swebench_subset_50.jsonl", help="Path to data file")
    parser.add_argument("--output", default="results/validation.csv", help="Output CSV file")
    
    args = parser.parse_args()
    
    # Load predictions
    predictions = load_predictions(args.predictions)
    print(f"Loaded {len(predictions)} predictions")
    
    # Load task metadata
    task_metadata = load_task_metadata(args.data)
    print(f"Loaded metadata for {len(task_metadata)} tasks")
    
    # Validate
    validate_patches(
        predictions=predictions,
        task_metadata=task_metadata,
        output_file=args.output
    )


if __name__ == "__main__":
    main()