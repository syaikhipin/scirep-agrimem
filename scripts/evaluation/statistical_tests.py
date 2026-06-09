#!/usr/bin/env python3
"""
Statistical Significance Testing for AgriMemory Benchmark

Implements:
1. Bootstrap resampling for confidence intervals
2. Paired t-tests for method comparison
3. Effect size calculations (Cohen's d)
4. Multiple comparison correction (Bonferroni)
5. Statistical reporting utilities

Addresses reviewer concern about missing significance testing.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from scipy import stats
from dataclasses import dataclass


@dataclass
class SignificanceResult:
    """Result of statistical significance test."""
    method_a: str
    method_b: str
    metric: str
    mean_diff: float
    p_value: float
    cohen_d: float
    is_significant: bool
    confidence_interval: Tuple[float, float]


class StatisticalTester:
    """Statistical significance testing for benchmark results."""

    def __init__(self, alpha: float = 0.05, n_bootstrap: int = 1000):
        """
        Args:
            alpha: Significance level (default 0.05 for 95% confidence)
            n_bootstrap: Number of bootstrap samples
        """
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap

    def bootstrap_confidence_interval(
        self,
        scores: List[float],
        confidence_level: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Calculate bootstrap confidence interval.

        Args:
            scores: List of metric scores
            confidence_level: Confidence level (default 0.95)

        Returns:
            (mean, lower_bound, upper_bound)
        """
        if not scores:
            return 0.0, 0.0, 0.0

        scores_arr = np.array(scores)
        n = len(scores_arr)

        # Bootstrap resampling
        bootstrap_means = []
        rng = np.random.RandomState(42)

        for _ in range(self.n_bootstrap):
            sample = rng.choice(scores_arr, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))

        bootstrap_means = np.array(bootstrap_means)

        # Calculate percentiles
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        mean = np.mean(scores_arr)
        lower = np.percentile(bootstrap_means, lower_percentile)
        upper = np.percentile(bootstrap_means, upper_percentile)

        return mean, lower, upper

    def paired_t_test(
        self,
        scores_a: List[float],
        scores_b: List[float],
        method_a: str,
        method_b: str,
        metric: str
    ) -> SignificanceResult:
        """
        Perform paired t-test between two methods.

        Args:
            scores_a: Scores for method A
            scores_b: Scores for method B
            method_a: Name of method A
            method_b: Name of method B
            metric: Metric name

        Returns:
            SignificanceResult object
        """
        if len(scores_a) != len(scores_b):
            raise ValueError("Score lists must have same length for paired test")

        scores_a_arr = np.array(scores_a)
        scores_b_arr = np.array(scores_b)

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(scores_a_arr, scores_b_arr)

        # Effect size (Cohen's d for paired samples)
        diff = scores_a_arr - scores_b_arr
        cohen_d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-9)

        # Mean difference
        mean_diff = np.mean(scores_a_arr) - np.mean(scores_b_arr)

        # Confidence interval for mean difference
        ci = stats.t.interval(
            1 - self.alpha,
            len(diff) - 1,
            loc=np.mean(diff),
            scale=stats.sem(diff)
        )

        is_significant = p_value < self.alpha

        return SignificanceResult(
            method_a=method_a,
            method_b=method_b,
            metric=metric,
            mean_diff=mean_diff,
            p_value=p_value,
            cohen_d=cohen_d,
            is_significant=is_significant,
            confidence_interval=ci
        )

    def multiple_comparison_correction(
        self,
        p_values: List[float],
        method: str = "bonferroni"
    ) -> List[float]:
        """
        Apply multiple comparison correction.

        Args:
            p_values: List of p-values
            method: Correction method ('bonferroni' or 'holm')

        Returns:
            Corrected p-values
        """
        n = len(p_values)

        if method == "bonferroni":
            return [min(p * n, 1.0) for p in p_values]

        elif method == "holm":
            # Holm-Bonferroni method
            sorted_idx = np.argsort(p_values)
            sorted_p = np.array(p_values)[sorted_idx]

            corrected = []
            for i, p in enumerate(sorted_p):
                corrected.append(min(p * (n - i), 1.0))

            # Enforce monotonicity
            for i in range(1, len(corrected)):
                corrected[i] = max(corrected[i], corrected[i - 1])

            # Unsort
            result = [0.0] * n
            for i, idx in enumerate(sorted_idx):
                result[idx] = corrected[i]

            return result

        else:
            raise ValueError(f"Unknown correction method: {method}")

    def compare_all_methods(
        self,
        results: Dict[str, List[float]],
        metric: str = "f1"
    ) -> List[SignificanceResult]:
        """
        Perform pairwise comparisons between all methods.

        Args:
            results: Dict mapping method_name -> list of scores
            metric: Metric name

        Returns:
            List of SignificanceResult objects
        """
        methods = list(results.keys())
        comparisons = []

        for i, method_a in enumerate(methods):
            for method_b in methods[i + 1:]:
                try:
                    result = self.paired_t_test(
                        results[method_a],
                        results[method_b],
                        method_a,
                        method_b,
                        metric
                    )
                    comparisons.append(result)
                except Exception as e:
                    print(f"Error comparing {method_a} vs {method_b}: {e}")

        # Apply multiple comparison correction
        if comparisons:
            p_values = [c.p_value for c in comparisons]
            corrected_p = self.multiple_comparison_correction(p_values, method="bonferroni")

            for comp, corrected in zip(comparisons, corrected_p):
                comp.p_value = corrected
                comp.is_significant = corrected < self.alpha

        return comparisons

    def generate_report(
        self,
        results: Dict[str, List[float]],
        metric: str = "f1"
    ) -> str:
        """
        Generate statistical report.

        Args:
            results: Dict mapping method_name -> list of scores
            metric: Metric name

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append(f"Statistical Significance Report: {metric.upper()}")
        report.append("=" * 80)
        report.append("")

        # Summary statistics with confidence intervals
        report.append("Method Performance (mean ± 95% CI):")
        report.append("-" * 80)

        for method, scores in results.items():
            mean, lower, upper = self.bootstrap_confidence_interval(scores)
            std = np.std(scores, ddof=1) if len(scores) > 1 else 0
            report.append(
                f"{method:<20} {mean:.4f} ± {std:.4f}  "
                f"[{lower:.4f}, {upper:.4f}]  (n={len(scores)})"
            )

        report.append("")

        # Pairwise comparisons
        comparisons = self.compare_all_methods(results, metric)

        if comparisons:
            report.append("Pairwise Comparisons (with Bonferroni correction):")
            report.append("-" * 80)
            report.append(
                f"{'Method A':<15} {'Method B':<15} {'Δ mean':<10} "
                f"{'p-value':<12} {'Cohen\'s d':<10} {'Significant':>10}"
            )
            report.append("-" * 80)

            for comp in comparisons:
                sig_marker = "***" if comp.is_significant else ""
                report.append(
                    f"{comp.method_a:<15} {comp.method_b:<15} "
                    f"{comp.mean_diff:>9.4f} {comp.p_value:>11.4f} "
                    f"{comp.cohen_d:>9.3f} {sig_marker:>10}"
                )

            report.append("")
            report.append("*** = significant at α = 0.05 (after Bonferroni correction)")

        report.append("=" * 80)
        return "\n".join(report)

    def effect_size_interpretation(self, cohen_d: float) -> str:
        """
        Interpret Cohen's d effect size.

        Args:
            cohen_d: Cohen's d value

        Returns:
            Interpretation string
        """
        abs_d = abs(cohen_d)

        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"


def aggregate_multiple_runs(
    runs: List[Dict[str, Any]],
    metric: str = "f1"
) -> Dict[str, List[float]]:
    """
    Aggregate scores from multiple experimental runs.

    Args:
        runs: List of result dictionaries from multiple runs
        metric: Metric to extract

    Returns:
        Dict mapping method -> list of scores across runs
    """
    aggregated = {}

    for run in runs:
        for method, method_results in run.items():
            if method not in aggregated:
                aggregated[method] = []

            # Extract metric
            if isinstance(method_results, dict):
                overall = method_results.get("overall", {})
                score = overall.get(metric, 0.0)
                aggregated[method].append(score)

    return aggregated


# Example usage
if __name__ == "__main__":
    # Simulate results from 3 runs
    np.random.seed(42)

    methods = ["rag", "nms", "baseline"]
    n_samples = 100
    n_runs = 3

    # Simulate multiple runs
    runs = []
    for run_id in range(n_runs):
        run_results = {}
        for method in methods:
            # Simulate scores (RAG slightly better)
            if method == "rag":
                scores = np.random.beta(8, 2, n_samples).tolist()
            elif method == "nms":
                scores = np.random.beta(7, 3, n_samples).tolist()
            else:
                scores = np.random.beta(6, 4, n_samples).tolist()

            run_results[method] = scores

        runs.append(run_results)

    # Statistical testing
    tester = StatisticalTester(alpha=0.05, n_bootstrap=1000)

    # Aggregate runs
    print("Aggregating multiple runs...")
    for method in methods:
        all_scores = []
        for run in runs:
            all_scores.extend(run[method])
        mean, lower, upper = tester.bootstrap_confidence_interval(all_scores)
        print(f"{method}: {mean:.4f} [{lower:.4f}, {upper:.4f}]")

    # Test individual run
    print("\n" + "=" * 80)
    print("Testing single run:")
    print("=" * 80)
    report = tester.generate_report(runs[0], metric="f1")
    print(report)
