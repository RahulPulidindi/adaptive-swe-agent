# Adaptive Inference-Time Compute Scaling for AI Software Engineering Agents

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Columbia University COMSE6998-013: LLM-Based GenAI Systems**  
> Rahul Pulidindi (rp3254@columbia.edu), Aaryan Misal (am6491@columbia.edu)

Adaptive compute allocation for AI code agents: **80% success with 85% fewer tokens** than fixed N=10.

---

## 📊 Results

| Method             | Success | Tokens    | Efficiency  |
| ------------------ | ------- | --------- | ----------- |
| Baseline (N=1)     | 60%     | 4,472     | 1.0x        |
| **Adaptive (N=3)** | **80%** | **5,579** | **5.3x** ✅ |
| Fixed-10           | 100%    | 37,238    | 0.2x        |

**Key Finding:** N=3 achieves 80% quality at 15% of Fixed-10's cost. Scaling N=3→N=10 costs 567% more for only 20% quality gain (23x worse efficiency).

---

## 🚀 Quick Start

```bash
# Install
git clone https://github.com/yourusername/adaptive-swe-agent.git
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

# Run adaptive (10 tasks)
python experiments/adaptive_evaluation.py \
    --n-tasks 10

# Validate patches
python experiments/validate_patches.py \
    --predictions results/adaptive_10.jsonl
```

---

## 📖 Usage

```python
from adaptive_agent import AdaptiveAgent
from complexity_predictor import ComplexityPredictor

# Load predictor
predictor = ComplexityPredictor.load("models/complexity_predictor.pkl")

# Initialize agent
agent = AdaptiveAgent(api_key="sk-...", model="gpt-5.1")

# Solve task
task = {
    'instance_id': 'django__django-15213',
    'problem_statement': 'Fix bug...',
    'repo': 'django/django',
    'base_commit': 'abc123'
}

result = agent.solve_adaptive(task=task, predictor=predictor)
print(f"N={result['n_used']}, Tokens={result['total_tokens']}, Success={result['success']}")
```

**CLI:**

```bash
# Solve single task
python -m adaptive_agent.cli solve --task task.json --mode adaptive

# Validate patch
python -m adaptive_agent.cli validate --task task.json --patch patch.diff
```

---

## 📁 Structure

```
adaptive-swe-agent/
├── adaptive_agent/          # Agent implementations (baseline, adaptive, fixed-10)
├── complexity_predictor/    # Task complexity prediction (Random Forest)
├── repository_manager/      # Git operations & patch validation
├── experiments/             # Evaluation scripts
├── tests/                   # Unit tests (pytest)
├── scripts/                 # Data prep & visualization
├── requirements.txt
└── README.md
```

---

## 🔬 How It Works

**1. Complexity Prediction**

-   Extract features (text length, code blocks, repo stats)
-   Random Forest predicts token requirements
-   Map to N: <1000→N=1, <1400→N=3, <1800→N=5, else→N=8

**2. Adaptive Agent**

-   Generate N solution candidates (temp=0.7)
-   Validate each with `git apply --check`
-   Return first valid or longest patch

**3. Patch Validation**

-   Format check (diff headers, hunks)
-   Repair hunk counts & line endings
-   Apply to actual repository

---

## 🧪 Reproduce Results

```bash
# 1. Prepare data
python scripts/create_subset.py --n-tasks 50

# 2. Run experiments
python experiments/baseline_comparison.py --n-tasks 5
python experiments/adaptive_evaluation.py --n-tasks 10
python experiments/validate_patches.py --predictions results/adaptive_10.jsonl

# 3. Visualize
python scripts/visualize_results.py --comparison results/comparison_5.csv
```

---

## 📊 Detailed Results

**Per-Task (5 tasks):**
| Task | Baseline | Adaptive (N=3) | Fixed-10 (N=10) |
|------|----------|----------------|-----------------|
| django-15213 | ❌ 7,620 | ✅ 7,071 | ✅ 72,488 |
| django-11630 | ✅ 3,100 | ✅ 2,584 | ✅ 24,169 |
| django-11019 | ✅ 5,183 | ✅ 6,857 | ✅ 32,073 |
| django-15819 | ✅ 2,968 | ✅ 2,804 | ✅ 28,280 |
| django-12747 | ❌ 3,489 | ❌ 8,580 | ✅ 29,182 |

**Efficiency:**

-   Adaptive: 6,974 tokens/success
-   Fixed-10: 37,238 tokens/success
-   **5.3x improvement**

---

## 🧪 Testing

```bash
pytest tests/ -v                          # All tests
pytest --cov=adaptive_agent tests/        # With coverage
```

---

## 📝 Citation

```bibtex
@misc{pulidindi2024adaptive,
  title={Adaptive Inference-Time Compute Scaling for AI Software Engineering Agents},
  author={Pulidindi, Rahul and Misal, Aaryan},
  year={2024},
  institution={Columbia University}
}
```

---

## 📧 Contact

**Rahul Pulidindi** - rp3254@columbia.edu  
**Aaryan Misal** - am6491@columbia.edu

---

## 🙏 Acknowledgments

-   [SWE-bench](https://github.com/princeton-nlp/SWE-bench) benchmark
-   OpenAI API access
-   Columbia COMSE6998 teaching staff

**License:** MIT | **Built at Columbia University**
