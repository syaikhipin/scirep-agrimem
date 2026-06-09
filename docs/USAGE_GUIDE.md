# AgriMemory v2: Usage Guide

Complete guide to running experiments with the enhanced benchmark.

---

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt:**
```
numpy>=1.24.0
scipy>=1.10.0
openai>=1.0.0
tiktoken>=0.5.0
requests>=2.28.0
tqdm>=4.65.0
pyyaml>=6.0
python-dotenv>=1.0.0
```

### 2. API Configuration

Copy and configure environment variables:

```bash
cp .env.example .env
```

**.env:**
```bash
# Chutes.ai API (for DeepSeek-V3 and GLM-4.6V)
CHUTES_API_KEY="your_chutes_api_key"
CHUTES_API_BASE="https://llm.chutes.ai/v1"

# Model configuration
MODEL="deepseek-ai/DeepSeek-V3-0324-TEE"
VISION_MODEL="zai-org/GLM-4.6V"

# Embedding API
EMBEDDING_API_BASE="https://your-embedding-endpoint/v1"
EMBEDDING_MODEL="michaelfeil/bge-small-en-v1.5"

# Evaluation models
EVAL_MODEL="gpt-4o-mini"

# Optional: OpenAI for cross-model judging
OPENAI_API_KEY="your_openai_key"
```

---

## Running Experiments

### Basic Benchmark

Run all memory systems on both datasets:

```bash
python scripts/comprehensive_benchmark_v2.py \
    --dataset both \
    --split test \
    --techniques nms,graphrag,rag,baseline \
    --output results/benchmark_v2.json
```

### NMS Ablation Study

Test individual NMS components:

```bash
# Working memory only
python scripts/comprehensive_benchmark_v2.py \
    --techniques nms_working_only \
    --output results/nms_ablation_working.json

# Episodic memory only
python scripts/comprehensive_benchmark_v2.py \
    --techniques nms_episodic_only \
    --output results/nms_ablation_episodic.json

# Full NMS
python scripts/comprehensive_benchmark_v2.py \
    --techniques nms \
    --output results/nms_full.json
```

### Statistical Analysis

Run with multiple seeds for significance testing:

```bash
python scripts/run_multiple_trials.py \
    --techniques nms,graphrag,rag,baseline \
    --n-runs 3 \
    --seeds 42,43,44 \
    --output results/multi_run_results.json

# Generate statistical report
python scripts/evaluation/analyze_significance.py \
    --input results/multi_run_results.json \
    --output results/statistical_report.txt
```

### Temporal Reasoning Evaluation

Focus on temporal questions:

```bash
python scripts/evaluate_temporal.py \
    --techniques nms,nms_with_temporal,rag \
    --category temporal \
    --output results/temporal_results.json
```

### Multimodal Inference

Evaluate with image inputs:

```bash
python scripts/evaluate_multimodal.py \
    --vision-model zai-org/GLM-4.6V \
    --image-retrieval \
    --output results/multimodal_results.json
```

---

## Dataset Generation

### Generate New Data

Generate expanded AgriConvMem dataset (500 samples):

```bash
python scripts/generation/generate_locomo_v2.py \
    --n-samples 500 \
    --output data/text/agri_locomo_v2/ \
    --image-source data/images/ \
    --model zai-org/GLM-4.6V
```

Generate expanded AgriMultiHop dataset (2000 samples):

```bash
python scripts/generation/generate_hotpotqa_v2.py \
    --n-samples 2000 \
    --output data/text/agri_hotpotqa_v2/ \
    --image-source data/images/ \
    --model zai-org/GLM-4.6V
```

### Quality Control

Run automatic validation:

```bash
python scripts/generation/validate_dataset.py \
    --input data/text/agri_locomo_v2/train.json \
    --output data/text/agri_locomo_v2/validation_report.json
```

---

## Configuration

### Edit Experiment Config

All parameters can be configured in `configs/experiment_config.yaml`:

```yaml
# Example: Change NMS parameters
memory_systems:
  nms:
    working_memory:
      capacity: 7
      decay_rate: 0.1

    episodic_memory:
      temporal_weight: 0.4
      top_k: 5

    coordinator:
      routing_strategy:
        disease_id: ["semantic", "episodic"]
        temporal: ["episodic", "working"]
```

### Custom Prompts

Edit prompts in `configs/prompts/`:

```bash
configs/prompts/
├── answer_prompt.txt          # QA generation prompt
├── judge_prompt.txt           # LLM judge prompt
├── entity_extraction.txt      # Entity extraction for GraphRAG
└── conversation_gen.txt       # Conversation generation
```

---

## Analysis and Visualization

### Generate Performance Report

```bash
python scripts/analysis/generate_report.py \
    --input results/benchmark_v2.json \
    --output report/performance_report.pdf
```

### Error Analysis

```bash
python scripts/analysis/error_analysis.py \
    --predictions results/benchmark_v2.json \
    --output results/error_breakdown.json
```

Generates breakdown by:
- Error category (temporal confusion, entity confusion, etc.)
- Question type
- Memory system

### Visualizations

Generate all figures for paper:

```bash
python scripts/analysis/generate_figures.py \
    --results results/benchmark_v2.json \
    --output figures/
```

Creates:
- Performance comparison (bar charts)
- Category-wise breakdown
- Statistical significance heatmap
- Temporal vs non-temporal performance

---

## Ablation Studies

### NMS Component Ablation

```bash
# Run all ablations
bash scripts/ablation/run_nms_ablations.sh

# Analyze results
python scripts/ablation/analyze_ablations.py \
    --results results/ablations/ \
    --output results/ablation_summary.json
```

Tests configurations:
- Working only
- Episodic only
- Semantic only
- Procedural only
- No coordinator
- All components

### Temporal Module Ablation

```bash
python scripts/evaluate_temporal.py \
    --ablations no_temporal,with_temporal,with_timeline \
    --output results/temporal_ablation.json
```

### GraphRAG Parameters

```bash
python scripts/baselines/tune_graphrag.py \
    --params chunk_size,local_k,global_k \
    --output results/graphrag_tuning.json
```

---

## Advanced Usage

### Custom Memory System

Implement custom system by extending `BaseMemorySystem`:

```python
# scripts/memory_systems/custom_system.py
from scripts.memory_systems.base import BaseMemorySystem

class CustomMemory(BaseMemorySystem):
    def __init__(self, logger):
        super().__init__(logger, name="custom")
        # Your initialization

    def add_conversation(self, conversation, metadata):
        # Your implementation
        pass

    def search(self, query, top_k):
        # Your retrieval logic
        pass
```

Register in benchmark:

```python
# In comprehensive_benchmark_v2.py
def create_memory_system(technique, logger):
    if technique == "custom":
        return CustomMemory(logger)
    # ... existing systems
```

### Batch Evaluation

Evaluate on multiple datasets:

```bash
python scripts/batch_evaluate.py \
    --datasets AgriConvMem,AgriMultiHop,MIRAGE \
    --techniques nms,graphrag,rag \
    --output results/batch_results/
```

### Cross-Domain Evaluation

Test on other domains (if datasets available):

```bash
python scripts/cross_domain_eval.py \
    --source-domain agriculture \
    --target-domains medical,finance \
    --technique nms \
    --output results/cross_domain.json
```

---

## Debugging

### Enable Verbose Logging

```bash
python scripts/comprehensive_benchmark_v2.py \
    --log-level DEBUG \
    --log-dir logs/ \
    --save-predictions
```

### Inspect Retrieved Context

```bash
python scripts/debug/inspect_retrieval.py \
    --technique nms \
    --question "What disease was identified?" \
    --conversation-file data/text/agri_locomo_v2/sample_001.json
```

### Test Individual Components

```bash
# Test temporal extraction
python scripts/temporal/temporal_reasoning.py

# Test NMS
python scripts/memory_systems/nms_openmemory.py

# Test GraphRAG
python scripts/baselines/graph_rag.py

# Test statistical testing
python scripts/evaluation/statistical_tests.py
```

---

## Output Files

After running experiments, you'll have:

```
results/
├── benchmark_v2.json              # Main results
│   {
│     "nms": {
│       "locomo": {
│         "overall": {"f1": 0.623, "em": 0.445, ...},
│         "by_category": {...},
│         "predictions": [...]
│       },
│       "hotpotqa": {...}
│     },
│     "graphrag": {...},
│     ...
│   }
│
├── statistical_report.txt          # Significance testing
├── error_breakdown.json            # Error analysis
├── timing_analysis.json            # Performance metrics
└── ablation_summary.json           # Ablation results
```

---

## Reproducibility Checklist

To reproduce published results:

- [ ] Clone repository
- [ ] Install exact dependencies from `requirements.txt`
- [ ] Set environment variables in `.env`
- [ ] Download datasets (link in README)
- [ ] Run with seed=42:
  ```bash
  python scripts/comprehensive_benchmark_v2.py \
      --config configs/experiment_config.yaml \
      --seed 42 \
      --output results/reproduce.json
  ```
- [ ] Compare results with `results/published_results.json`
- [ ] Generate statistical report
- [ ] Check figures match paper

---

## Troubleshooting

### API Rate Limits

If hitting rate limits:

```bash
# Add delay between requests
python scripts/comprehensive_benchmark_v2.py \
    --request-delay 2.0  # seconds

# Or use batch mode with checkpointing
python scripts/comprehensive_benchmark_v2.py \
    --batch-size 10 \
    --checkpoint-freq 50 \
    --resume-from results/checkpoint.json
```

### Memory Issues

For large datasets:

```bash
# Process in batches
python scripts/comprehensive_benchmark_v2.py \
    --limit 100 \
    --offset 0

# Clear caches
python scripts/utils/clear_caches.py
```

### Embedding API Timeout

```bash
# Increase timeout
export EMBEDDING_TIMEOUT=60

# Use local embeddings
python scripts/utils/setup_local_embeddings.py
```

---

## Citation

If you use this benchmark:

```bibtex
@article{agrimemory2025,
  title={Neuroscientific Memory-Augmented Generation for Multimodal Agricultural Data: A Synthetic Benchmark Evaluation},
  author={...},
  journal={...},
  year={2025}
}
```

---

## Support

- **Issues:** https://github.com/your-repo/issues
- **Discussions:** https://github.com/your-repo/discussions
- **Email:** your-email@domain.com

---

## Acknowledgments

This work addresses reviewer feedback from KDD 2025 submission. We thank the reviewers for their detailed suggestions that led to significant improvements in:

1. Implementation rigor (detailed NMS specifications)
2. Baseline coverage (GraphRAG, RAPTOR, ARM)
3. Statistical validity (significance testing, multiple runs)
4. Multimodal evaluation (true image inference)
5. Temporal reasoning (dedicated module)
6. Documentation completeness (this guide)

---

Last updated: 2025-01-08
