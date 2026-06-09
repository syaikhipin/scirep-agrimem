# AgriMemSynth GitHub Supplementary Release

This folder is a cleaned GitHub-ready supplementary package for the AgriMemSynth benchmark artifacts.

It includes:

- `scripts/`
  Benchmark, generation, memory-system, temporal, evaluation, and validation code.
- `configs/`
  Experiment configuration used by the benchmark code.
- `data/text/`
  Released text benchmark datasets included in this supplementary repository:
  `agri_locomo_v2` and `agri_hotpotqa_v4`.
- `results/full_benchmark_v2/`
  Final benchmark result artifacts reported in the accompanying manuscript tables and figures.
- `docs/`
  Implementation and usage notes copied from the benchmark repository.
- `audits/`
  Experiment-integrity and paper-claim audit reports.

## Recommended repository layout

Upload the contents of this folder as the root of a GitHub repository.

## Quick start

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy the environment template and fill in API credentials:

```bash
cp .env.example .env
```

4. Run the released full benchmark:

```bash
python scripts/comprehensive_benchmark_full.py \
  --dataset both \
  --data-dir data/text \
  --split test \
  --techniques nms,rag,hybrid,bm25,memorygraph \
  --output results/full_benchmark_v2/reproduced_benchmark.json
```

## Main benchmark result files

- `results/full_benchmark_v2/summary_complete.csv`
- `results/full_benchmark_v2/benchmark_results_complete.json`
- `results/full_benchmark_v2/locomo_by_category_complete.csv`
- `results/full_benchmark_v2/hotpotqa_by_type_complete.csv`

## Notes

- This supplementary package intentionally excludes the manuscript files.
- Some optional integrations referenced in the codebase, such as Mem0 or Zep, require extra credentials and packages beyond the core benchmark path.
- The benchmark artifacts included here are the released supplementary package associated with the study, not a full mirror of every intermediate experiment in the workspace.
