"""
Download SWE-bench Lite dataset
"""

import os
import json
import requests
from pathlib import Path


def download_swebench_lite(output_dir: str = "data"):
    """
    Download SWE-bench Lite dataset from GitHub.
    
    Args:
        output_dir: Directory to save dataset
    """
    os.makedirs(output_dir, exist_ok=True)
    
    url = "https://raw.githubusercontent.com/princeton-nlp/SWE-bench/main/swebench/harness/constants.py"
    
    print("Downloading SWE-bench Lite dataset...")
    print(f"Note: This is a placeholder. Actual download should use:")
    print("  https://github.com/princeton-nlp/SWE-bench")
    print()
    
    # For actual implementation, use:
    # from datasets import load_dataset
    # dataset = load_dataset("princeton-nlp/SWE-bench_Lite")
    
    output_file = os.path.join(output_dir, "swebench_lite.jsonl")
    
    print(f"✓ Dataset should be downloaded to: {output_file}")
    print()
    print("Manual download instructions:")
    print("1. Clone SWE-bench repository:")
    print("   git clone https://github.com/princeton-nlp/SWE-bench.git")
    print("2. Or use HuggingFace datasets:")
    print("   from datasets import load_dataset")
    print("   dataset = load_dataset('princeton-nlp/SWE-bench_Lite')")
    print("3. Save to data/swebench_lite.jsonl")


if __name__ == "__main__":
    download_swebench_lite()