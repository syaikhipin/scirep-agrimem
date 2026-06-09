# Paper Claim Audit Report

**Date**: 2026-06-09  
**Auditor**: GPT-5.5 xhigh (fresh zero-context thread)  
**Paper**: `sn-article.tex`

## Overall Verdict: WARN

The benchmark result numbers in the manuscript match the supplied raw result files. The remaining warnings are about claims that are outside the audited raw-result scope, not about inflated benchmark numbers.

## Claims Verified

- total claims checked: 24
- exact_match: 13
- rounding_ok: 7
- ambiguous_mapping: 0 after local wording fixes
- missing_evidence: 4
- mismatch: 0 material benchmark mismatches

## Issues Found

### [WARN] External agricultural-impact claim
- **Location**: Introduction
- **Paper says**: crop diseases account for `20--40%` of global yield losses annually
- **Evidence shows**: not present in the supplied raw result files
- **Status**: `missing_evidence`
- **Fix**: acceptable if supported by the cited literature; not verifiable by this raw-result audit

### [WARN] Source-dataset inventory counts
- **Location**: Methods / Source Data
- **Paper says**: PlantVillage `54,306` images, PlantDoc `2,598` images, total `56,904`
- **Evidence shows**: not present in the supplied raw result files
- **Status**: `missing_evidence`
- **Fix**: acceptable if supported by source-dataset references; not verifiable by this raw-result audit

### [WARN] Pipeline-stage description
- **Location**: Methods / AgriSynth Module
- **Paper says**: five generation stages
- **Evidence shows**: no provenance file for stage counting was included in the audited set
- **Status**: `missing_evidence`
- **Fix**: acceptable as workflow description, but not verified against raw-result artifacts

### [WARN] Future public release statement
- **Location**: Data availability
- **Paper says**: framework, datasets, and scripts will be publicly available upon publication
- **Evidence shows**: future availability is outside the audited raw-result scope
- **Status**: `missing_evidence`
- **Fix**: keep only if this release plan is genuine

## Benchmark Claims That Passed

- dataset totals: `500` AgriConvMem conversations, `2,000` AgriMultiHop questions
- test counts: `146` conversational-memory questions, `200` multi-hop questions, `346` total
- architecture count: `5`
- headline semantic results:
  - RAG on AgriConvMem: `85.6%` with `95% CI 79.0--90.4`
  - MemoryGraph on AgriMultiHop: `97.0%` with `95% CI 93.6--98.6`
- overall results table values
- category/type tables and displayed percentages
- main descriptive result: conversational memory is harder than multi-hop reasoning in this benchmark

## Summary

This manuscript is **numerically faithful** to the supplied benchmark outputs. The audit remains `WARN` because some introductory, source-dataset, workflow, and release statements cannot be verified from the audited raw-result set alone.
