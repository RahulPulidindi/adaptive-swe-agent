# Adaptive Inference-Time Compute Scaling for AI Software Engineering Agents

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Columbia University COMSE6998-013: LLM-Based GenAI Systems**  
> Rahul Pulidindi (rp3254@columbia.edu), Aaryan Misal (am6491@columbia.edu)

Adaptive compute allocation for AI code agents achieving **100% success with 86% fewer tokens** than fixed N=10.

**Tutorial:** For a step-by-step walkthrough, see [`notebooks/adaptive_swe_agent.ipynb`](notebooks/adaptive_swe_agent.ipynb) - simply run each cell in order.

---

## Results Summary

| Method             | Success Rate   | Avg Tokens | Tokens/Success | Efficiency |
| ------------------ | -------------- | ---------- | -------------- | ---------- |
| Baseline (N=1)     | 60% (3/5)      | 4,574      | 7,623          | 1.0x       |
| **Adaptive (N=3)** | **100% (5/5)** | **5,179**  | **5,179**      | **7.3x**   |
| Fixed-10 (N=10)    | 80% (4/5)      | 37,716     | 47,145         | 0.11x      |

**Key Findings:**

-   Adaptive achieves perfect 100% success rate
-   Uses 86% fewer tokens than Fixed-10 (5,179 vs 37,716)
-   Outperforms both baselines in quality and efficiency
-   Fixed-10 paradoxically achieves lower success (80%) due to overcomplicated solutions

---

## Quick Start

```bash
# Install
git clone https://github.com/RahulPulidindi/adaptive-swe-agent.git
cd adaptive-swe-agent
pip install -r requirements.txt
pip install -e .

# Configure
cp .env.example .env
# Add OPENAI_API_KEY to .env

# Run comparison (5 tasks)
python experiments/baseline_comparison.py \
    --data data/swebench_subset_50.jsonl \
    --n-tasks 5

# Run adaptive evaluation
python experiments/adaptive_evaluation.py --n-tasks 10

# Validate patches
python experiments/validate_patches.py \
    --predictions results/adaptive_10.jsonl
```

---

## Usage

**Python API:**

```python
from adaptive_agent import AdaptiveAgent
from complexity_predictor import ComplexityPredictor

# Initialize
predictor = ComplexityPredictor.load("models/complexity_predictor.pkl")
agent = AdaptiveAgent(api_key="sk-...", model="gpt-5.1")

# Solve task
task = {
    'instance_id': 'django__django-15213',
    'problem_statement': 'Fix bug...',
    'repo': 'django/django',
    'base_commit': 'abc123'
}

result = agent.solve_adaptive(task=task, predictor=predictor)
print(f"N={result['n_used']}, Tokens={result['total_tokens']}")
```

**CLI:**

```bash
python -m adaptive_agent.cli solve --task task.json --mode adaptive
python -m adaptive_agent.cli validate --task task.json --patch patch.diff
```

---

## Project Structure

```
adaptive-swe-agent/
├── adaptive_agent/          # Agent implementations (baseline, adaptive, fixed-10)
├── complexity_predictor/    # Task complexity prediction (Random Forest)
├── repository_manager/      # Git operations & patch validation
├── experiments/             # Evaluation scripts
├── tests/                   # Unit tests (pytest)
├── scripts/                 # Data prep & visualization
├── notebooks/               # Tutorial notebook (adaptive_swe_agent.ipynb)
└── requirements.txt
```

---

## System Architecture

**1. Complexity Prediction**

-   Extract features: text length, code blocks, repo statistics
-   Random Forest predicts token requirements
-   Map to N: <1000→N=1, <1400→N=3, <1800→N=5, else→N=8

**2. Adaptive Agent**

-   Generate N solution candidates (temperature=0.7)
-   Validate each with `git apply --check`
-   Return first valid patch

**3. Patch Validation**

-   Format verification (diff headers, hunks)
-   Automatic repair (hunk counts, line endings)
-   Repository application testing

---

## Detailed Results (5 Tasks)

### Per-Task Performance

| Task             | Baseline    | Adaptive (N=3) | Fixed-10 (N=10) |
| ---------------- | ----------- | -------------- | --------------- |
| **django-15213** | ✓ 7,563 tok | ✓ 7,104 tok    | ✓ 71,803 tok    |
| **django-11630** | ✗ 3,269 tok | ✓ 2,770 tok    | ✓ 24,477 tok    |
| **django-11019** | ✓ 5,221 tok | ✓ 6,820 tok    | ✓ 33,293 tok    |
| **django-15819** | ✓ 3,443 tok | ✓ 3,031 tok    | ✓ 29,106 tok    |
| **django-12747** | ✗ 3,376 tok | ✓ 6,168 tok    | ✗ 29,903 tok    |

### Metrics Breakdown

**Success Rate:**

-   Baseline: 60% (failed 2 tasks)
-   Adaptive: 100% (perfect success)
-   Fixed-10: 80% (failed 1 task)

**Token Usage:**

-   Baseline: 4,574 avg
-   Adaptive: 5,179 avg (+13% vs baseline)
-   Fixed-10: 37,716 avg (+628% vs adaptive)

**Execution Time:**

-   Baseline: 10.1s avg
-   Adaptive: 14.4s avg
-   Fixed-10: 77.6s avg

**Efficiency (Tokens per Success):**

-   Baseline: 7,623 tokens/success
-   **Adaptive: 5,179 tokens/success** (most efficient)
-   Fixed-10: 47,145 tokens/success

---

## Key Insights

**1. Adaptive Superiority**

Adaptive achieves the best quality-efficiency tradeoff:

-   100% success rate (vs 60% baseline, 80% fixed-10)
-   86% fewer tokens than Fixed-10
-   Only 13% more tokens than baseline but +40% better success

**2. More Compute ≠ Better Quality**

Fixed-10 demonstrates diminishing returns:

-   Failed django-12747 where Adaptive (N=3) succeeded
-   Overcomplicated solutions can be worse than simpler ones
-   7.3x more tokens for 20% worse success rate

**3. N=3 is Optimal**

All tasks allocated N=3 by predictor:

-   Sufficient diversity for high success
-   Avoids overcomplication
-   Balances quality and cost

---

## Reproducing Results

```bash
# 1. Prepare data
python scripts/create_subset.py --n-tasks 50

# 2. Run comparison (produces exact results above)
python experiments/baseline_comparison.py \
    --data data/swebench_subset_50.jsonl \
    --n-tasks 5 \
    --predictor models/complexity_predictor.pkl \
    --output results/comparison_5.csv

# 3. Validate patches
python experiments/validate_patches.py \
    --predictions results/adaptive_predictions.jsonl \
    --output results/validation.csv

# 4. Visualize
python scripts/visualize_results.py \
    --comparison results/comparison_5.csv \
    --output figures/
```

---

## Testing

```bash
pytest tests/ -v                          # All tests
pytest --cov=adaptive_agent tests/        # With coverage
```

---

## Citation

```bibtex
@misc{pulidindi2024adaptive,
  title={Adaptive Inference-Time Compute Scaling for AI Software Engineering Agents},
  author={Pulidindi, Rahul and Misal, Aaryan},
  year={2024},
  institution={Columbia University},
  course={COMSE6998-013}
}
```

---

## Contact

**Rahul Pulidindi** - rp3254@columbia.edu  
**Aaryan Misal** - am6491@columbia.edu

---

## Acknowledgments

-   [SWE-bench](https://github.com/princeton-nlp/SWE-bench) benchmark
-   OpenAI for API access
-   Columbia COMSE6998 teaching staff

**License:** MIT
