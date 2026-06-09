# Experiment Audit Report

**Date**: 2026-06-08
**Auditor**: GPT-5.5 xhigh (cross-model, read-only)
**Project**: `scirep-agrimem`
**Submission package under review**: GitHub supplementary release

## Overall Verdict: WARN

## Integrity Status: warn

The main benchmark tables in the DMKD manuscript are broadly consistent with the released result files, and the audit did **not** find evidence of fake ground truth derived from model predictions or score normalization fraud. The main integrity problem identified during audit was reproducibility drift between the development implementation and earlier copied package artifacts.

## Checks

### A. Ground Truth Provenance: WARN

Status:
- `WARN`

Evidence:
- [comprehensive_benchmark_v2.py](scripts/comprehensive_benchmark_v2.py#L1324)
- [comprehensive_benchmark_v2.py](scripts/comprehensive_benchmark_v2.py#L1446)
- [comprehensive_benchmark_full.py](scripts/comprehensive_benchmark_full.py#L1147)
- [comprehensive_benchmark_full.py](scripts/comprehensive_benchmark_full.py#L1235)
- [generate_locomo_v2.py](scripts/generation/generate_locomo_v2.py#L605)
- [generate_locomo_v2.py](scripts/generation/generate_locomo_v2.py#L621)
- [generate_locomo_v2.py](scripts/generation/generate_locomo_v2.py#L630)
- [generate_hard_multihop.py](scripts/generation/generate_hard_multihop.py#L188)
- [generate_hard_multihop.py](scripts/generation/generate_hard_multihop.py#L293)
- [generate_hard_multihop.py](scripts/generation/generate_hard_multihop.py#L582)
- `sn-article.tex:42`

Details:
- Evaluation references are loaded from dataset fields, not from tested model outputs.
- The benchmark is still a **synthetic proxy** benchmark, not official LoCoMo or HotpotQA ground truth.
- The manuscript mostly says this clearly, but the package should avoid any phrasing that could imply official real-GT evaluation.

### B. Score Normalization: PASS

Status:
- `PASS`

Evidence:
- [comprehensive_benchmark_v2.py](scripts/comprehensive_benchmark_v2.py#L106)
- [comprehensive_benchmark_v2.py](scripts/comprehensive_benchmark_v2.py#L124)
- [comprehensive_benchmark_v2.py](scripts/comprehensive_benchmark_v2.py#L129)
- [comprehensive_benchmark_v2.py](scripts/comprehensive_benchmark_v2.py#L165)
- [comprehensive_benchmark_full.py](scripts/comprehensive_benchmark_full.py#L1181)
- [comprehensive_benchmark_full.py](scripts/comprehensive_benchmark_full.py#L1262)

Details:
- F1, EM, BLEU, and the semantic metrics are computed in standard ways.
- No metric is normalized by prediction statistics from the evaluated model.

### C. Result File Existence: WARN

Status:
- `WARN`

Evidence:
- `sn-article.tex:401`
- [summary_complete.csv](results/full_benchmark_v2/summary_complete.csv#L2)
- [summary_complete.csv](results/full_benchmark_v2/summary_complete.csv#L6)
- [benchmark_results_complete.json](results/full_benchmark_v2/benchmark_results_complete.json#L6)
- [benchmark_results_complete.json](results/full_benchmark_v2/benchmark_results_complete.json#L2070)
- `sn-article.tex:383`
- [comprehensive_benchmark_full.py](scripts/comprehensive_benchmark_full.py#L1371)
- [comprehensive_benchmark_full.py](scripts/comprehensive_benchmark_full.py#L1349)
- [comprehensive_benchmark_full.py](scripts/comprehensive_benchmark_full.py#L1360)
- [README.md](README.md#L141)

Details:
- The main DMKD paper numbers match the released `summary_complete.csv` and `benchmark_results_complete.json`.
- The package is not cleanly reproducible:
  - the copied benchmark script under `scripts/` has stale dataset paths,
  - the manuscript cites the wrong benchmark script,
  - and the package README contains at least one false summary claim.

### D. Dead Code Detection: WARN

Status:
- `WARN`

Evidence:
- [comprehensive_benchmark_full.py](scripts/comprehensive_benchmark_full.py#L1153)
- [comprehensive_benchmark_full.py](scripts/comprehensive_benchmark_full.py#L1240)
- [statistical_tests.py](scripts/evaluation/statistical_tests.py#L315)
- [experiment_config.yaml](configs/experiment_config.yaml#L212)
- [experiment_config.yaml](configs/experiment_config.yaml#L220)
- [experiment_config.yaml](configs/experiment_config.yaml#L222)
- [experiment_config.yaml](configs/experiment_config.yaml#L255)

Details:
- Core metric code is used.
- There is still unused or aspirational evaluation machinery around multi-run stats, judge ensembles, ROUGE-L, and multimodal claims that does not appear in the full result artifacts.

### E. Scope Assessment: WARN

Status:
- `WARN`

Evidence:
- `sn-article.tex:248`
- [benchmark_results_complete.json](results/full_benchmark_v2/benchmark_results_complete.json#L6)
- [benchmark_results_complete.json](results/full_benchmark_v2/benchmark_results_complete.json#L2070)
- `sn-article.tex:562`
- `sn-article.tex:570`

Details:
- The complete scope supported by files in this audit is:
  - 5 systems,
  - 75 AgriConvMem test conversations / 146 questions,
  - 200 AgriMultiHop questions.
- The paper’s caveats are better than the repo’s README/docs, but claims about robustness, repeated runs, human validation, or broader external validity exceed what the result files demonstrate.

### F. Evaluation Type: WARN

Status:
- `WARN`

Classification:
- Full AgriConvMem benchmark: `synthetic_proxy`
- Full AgriMultiHop v4 benchmark: `synthetic_proxy`
- Early/test benchmark JSONs: `synthetic_proxy`
- `validation_report.json`: `simulation_only`
- Human evaluation: `not found`
- Official real GT benchmark: `not found`

## Action Items

- Replace or fix `scripts/comprehensive_benchmark_full.py` so it uses the authoritative dataset paths (`agri_locomo_v2`, `agri_hotpotqa_v4`).
- In the DMKD manuscript, point to the actual benchmark script used for the complete outputs, or rerun with the script currently cited in the text.
- Make `summary_complete.csv` and `benchmark_results_complete.json` the explicit source of paper tables.
- Fix false or unsupported README/package claims, especially the claim that RAG has the highest LLM-judge score on both datasets.
- Remove unsupported claims about human validation, three-run statistics, judge ensembles, ROUGE-L, or multimodal evaluation unless matching result files are added.
- Add run metadata to results: script hash, dataset paths, model, judge model, seed, and environment settings.
- Add answer-correctness validation specifically for `agri_hotpotqa_v4`.
- Rephrase strong comparative language such as “substantially outperforms” unless formal significance support is added.

## Claim Impact

- `C1` Main result tables in the DMKD manuscript: `supported`
- `C2` Benchmark as real-GT official LoCoMo/HotpotQA style evaluation: `needs_qualifier`
- `C3` Supplementary package as a clean reproducible release: `unsupported`
- `C4` Strong superiority language for graph-vs-vector retrieval: `needs_qualifier`

## Package Decision

The earlier submission package snapshot was **not ready to use as-is** if the package itself needed to be fully reproducible and internally consistent. It was numerically close enough for the manuscript tables, but the surrounding package artifacts still contained drift and unsupported claims.
