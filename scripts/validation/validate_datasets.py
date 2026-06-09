#!/usr/bin/env python3
"""
AgriMemory Dataset v2 Validation Script

Validates the generated datasets (AgriConvMem and AgriMultiHop) for:
- Data structure and format correctness
- Image path validity
- QA pair quality
- Statistical distribution checks
- Reviewer requirement compliance

Usage:
    python validate_datasets.py --data-dir data/text --images-dir data/images
"""

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ValidationResult:
    """Holds validation results for a dataset."""
    dataset_name: str
    total_samples: int
    splits: Dict[str, int]
    passed_checks: List[str]
    failed_checks: List[str]
    warnings: List[str]
    statistics: Dict[str, Any]

    @property
    def is_valid(self) -> bool:
        return len(self.failed_checks) == 0


class DatasetValidator:
    """Validates AgriMemory Dataset v2."""

    def __init__(self, data_dir: str, images_dir: str):
        self.data_dir = Path(data_dir)
        self.images_dir = Path(images_dir)
        self.base_dir = self.data_dir.parent.parent  # repository root

    def validate_agriconvmem(self) -> ValidationResult:
        """Validate AgriConvMem (LoCoMo v2) dataset."""
        dataset_path = self.data_dir / "agri_locomo_v2"

        passed = []
        failed = []
        warnings = []
        statistics = {}

        # Load all splits
        all_samples = []
        splits = {}

        for split in ['train', 'val', 'test']:
            split_path = dataset_path / f"{split}.json"
            if not split_path.exists():
                failed.append(f"Missing {split}.json file")
                continue

            with open(split_path) as f:
                data = json.load(f)

            samples = data.get('samples', [])
            splits[split] = len(samples)
            all_samples.extend(samples)

        if not all_samples:
            failed.append("No samples found in dataset")
            return ValidationResult(
                dataset_name="AgriConvMem (LoCoMo v2)",
                total_samples=0,
                splits=splits,
                passed_checks=passed,
                failed_checks=failed,
                warnings=warnings,
                statistics=statistics
            )

        # Check 1: Sample count (Reviewer requirement: 500)
        total = len(all_samples)
        if total >= 500:
            passed.append(f"Sample count: {total} >= 500 (reviewer requirement)")
        else:
            failed.append(f"Sample count: {total} < 500 (reviewer requirement)")

        # Check 2: Dataset info file
        info_path = dataset_path / "dataset_info.json"
        if info_path.exists():
            with open(info_path) as f:
                dataset_info = json.load(f)
            passed.append("dataset_info.json exists")
            statistics['generation_model'] = dataset_info.get('model', 'Unknown')
            statistics['generation_method'] = dataset_info.get('generation_method', 'Unknown')
        else:
            warnings.append("dataset_info.json not found")

        # Check 3: Sample structure validation
        required_keys = ['sample_id', 'farm_metadata', 'conversation']
        structure_valid = True
        for sample in all_samples[:10]:  # Check first 10
            for key in required_keys:
                if key not in sample:
                    structure_valid = False
                    failed.append(f"Missing required key: {key}")
                    break

        if structure_valid:
            passed.append("Sample structure is valid")

        # Check 4: Image path validation
        valid_images = 0
        invalid_images = 0
        image_paths = []

        for sample in all_samples:
            conv = sample.get('conversation', {})
            for key, value in conv.items():
                if key.endswith('_image') and value:
                    image_paths.append(value)
                    full_path = self.base_dir / value
                    if full_path.exists():
                        valid_images += 1
                    else:
                        invalid_images += 1

        statistics['total_images'] = len(image_paths)
        statistics['valid_images'] = valid_images
        statistics['invalid_images'] = invalid_images

        if invalid_images == 0 and valid_images > 0:
            passed.append(f"All {valid_images} image paths are valid")
        elif invalid_images > 0:
            failed.append(f"{invalid_images}/{len(image_paths)} image paths are invalid")
        else:
            warnings.append("No image paths found in dataset")

        # Check 5: QA pairs validation
        qa_count = 0
        qa_categories = Counter()
        samples_with_qa = 0

        for sample in all_samples:
            qa_pairs = sample.get('qa', [])
            if qa_pairs:
                samples_with_qa += 1
                qa_count += len(qa_pairs)
                for qa in qa_pairs:
                    qa_categories[qa.get('category', 'unknown')] += 1

        statistics['total_qa_pairs'] = qa_count
        statistics['samples_with_qa'] = samples_with_qa
        statistics['qa_categories'] = dict(qa_categories)

        if qa_count > 0:
            passed.append(f"QA pairs present: {qa_count} across {samples_with_qa} samples")
        else:
            failed.append("No QA pairs found in dataset")

        # Check 6: Multi-session validation
        session_counts = []
        for sample in all_samples:
            conv = sample.get('conversation', {})
            sessions = [k for k in conv.keys()
                       if k.startswith('session_') and
                       not k.endswith(('_date_time', '_summary', '_image'))]
            session_counts.append(len(sessions))

        statistics['min_sessions'] = min(session_counts) if session_counts else 0
        statistics['max_sessions'] = max(session_counts) if session_counts else 0
        statistics['avg_sessions'] = sum(session_counts) / len(session_counts) if session_counts else 0

        if all(s >= 2 for s in session_counts):
            passed.append(f"Multi-session: All samples have >= 2 sessions")
        else:
            warnings.append(f"Some samples have < 2 sessions")

        # Check 7: Crop diversity
        crops = Counter()
        for sample in all_samples:
            crop = sample.get('farm_metadata', {}).get('primary_crop', 'unknown')
            crops[crop] += 1

        statistics['crops'] = dict(crops)
        statistics['crop_diversity'] = len(crops)

        if len(crops) >= 3:
            passed.append(f"Crop diversity: {len(crops)} different crops")
        else:
            warnings.append(f"Limited crop diversity: only {len(crops)} crops")

        # Check 8: Temporal data
        has_temporal = 0
        for sample in all_samples:
            conv = sample.get('conversation', {})
            if any(k.endswith('_date_time') for k in conv.keys()):
                has_temporal += 1

        statistics['samples_with_temporal'] = has_temporal

        if has_temporal == len(all_samples):
            passed.append("All samples have temporal data")
        elif has_temporal > 0:
            warnings.append(f"Only {has_temporal}/{len(all_samples)} samples have temporal data")
        else:
            failed.append("No temporal data found")

        return ValidationResult(
            dataset_name="AgriConvMem (LoCoMo v2)",
            total_samples=total,
            splits=splits,
            passed_checks=passed,
            failed_checks=failed,
            warnings=warnings,
            statistics=statistics
        )

    def validate_agrimultihop(self) -> ValidationResult:
        """Validate AgriMultiHop (HotpotQA v3) dataset."""
        # Prefer v3 (true multi-hop) if available
        dataset_path = self.data_dir / "agri_hotpotqa_v3"
        if not dataset_path.exists():
            dataset_path = self.data_dir / "agri_hotpotqa_v2"

        passed = []
        failed = []
        warnings = []
        statistics = {}

        # Load all splits
        all_samples = []
        splits = {}

        for split in ['train', 'val', 'test']:
            split_path = dataset_path / f"{split}.json"
            if not split_path.exists():
                failed.append(f"Missing {split}.json file")
                continue

            with open(split_path) as f:
                data = json.load(f)

            # Handle both list and dict formats
            if isinstance(data, list):
                samples = data
            elif isinstance(data, dict):
                samples = data.get('samples', [])
            else:
                samples = []

            splits[split] = len(samples)
            all_samples.extend(samples)

        if not all_samples:
            failed.append("No samples found in dataset")
            return ValidationResult(
                dataset_name="AgriMultiHop (HotpotQA v2)",
                total_samples=0,
                splits=splits,
                passed_checks=passed,
                failed_checks=failed,
                warnings=warnings,
                statistics=statistics
            )

        # Check 1: Sample count (Reviewer requirement: 2000)
        total = len(all_samples)
        if total >= 2000:
            passed.append(f"Sample count: {total} >= 2000 (reviewer requirement)")
        else:
            failed.append(f"Sample count: {total} < 2000 (reviewer requirement)")

        # Check 2: Dataset info file
        info_path = dataset_path / "dataset_info.json"
        if info_path.exists():
            with open(info_path) as f:
                dataset_info = json.load(f)
            passed.append("dataset_info.json exists")
            statistics['generation_model'] = dataset_info.get('model', 'Unknown')
            statistics['generation_method'] = dataset_info.get('generation_method', 'Unknown')
        else:
            warnings.append("dataset_info.json not found")

        # Check 3: Sample structure validation
        required_keys = ['id', 'question', 'answer', 'type', 'context']
        structure_valid = True
        for sample in all_samples[:10]:  # Check first 10
            for key in required_keys:
                if key not in sample:
                    structure_valid = False
                    failed.append(f"Missing required key: {key}")
                    break

        if structure_valid:
            passed.append("Sample structure is valid")

        # Check 4: Question types (bridge and comparison)
        question_types = Counter()
        for sample in all_samples:
            qtype = sample.get('type', 'unknown')
            question_types[qtype] += 1

        statistics['question_types'] = dict(question_types)

        if 'bridge' in question_types and 'comparison' in question_types:
            passed.append(f"Both question types present: bridge ({question_types['bridge']}), comparison ({question_types['comparison']})")
        else:
            failed.append("Missing bridge or comparison question types")

        # Check 5: Context paragraphs
        context_counts = []
        for sample in all_samples:
            context = sample.get('context', [])
            context_counts.append(len(context))

        statistics['min_context'] = min(context_counts) if context_counts else 0
        statistics['max_context'] = max(context_counts) if context_counts else 0
        statistics['avg_context'] = sum(context_counts) / len(context_counts) if context_counts else 0

        if all(c >= 2 for c in context_counts):
            passed.append("All samples have >= 2 context paragraphs (multi-hop)")
        else:
            warnings.append("Some samples have < 2 context paragraphs")

        # Check 6: Supporting facts
        has_supporting = 0
        for sample in all_samples:
            if sample.get('supporting_facts'):
                has_supporting += 1

        statistics['samples_with_supporting_facts'] = has_supporting

        if has_supporting == len(all_samples):
            passed.append("All samples have supporting facts")
        elif has_supporting > 0:
            warnings.append(f"Only {has_supporting}/{len(all_samples)} have supporting facts")
        else:
            failed.append("No supporting facts found")

        # Check 7: Answer quality (non-empty)
        empty_answers = 0
        for sample in all_samples:
            answer = sample.get('answer', '')
            if not answer or answer.strip() == '':
                empty_answers += 1

        statistics['empty_answers'] = empty_answers

        if empty_answers == 0:
            passed.append("All samples have non-empty answers")
        else:
            failed.append(f"{empty_answers} samples have empty answers")

        return ValidationResult(
            dataset_name="AgriMultiHop (HotpotQA v2)",
            total_samples=total,
            splits=splits,
            passed_checks=passed,
            failed_checks=failed,
            warnings=warnings,
            statistics=statistics
        )


def print_validation_report(result: ValidationResult):
    """Print formatted validation report."""
    print("\n" + "=" * 70)
    print(f"VALIDATION REPORT: {result.dataset_name}")
    print("=" * 70)

    print(f"\nTotal Samples: {result.total_samples}")
    print(f"Splits: {result.splits}")
    print(f"Status: {'✅ VALID' if result.is_valid else '❌ INVALID'}")

    # Generation info
    if 'generation_model' in result.statistics:
        print(f"\n--- Generation Info ---")
        print(f"Model: {result.statistics.get('generation_model', 'Unknown')}")
        print(f"Method: {result.statistics.get('generation_method', 'Unknown')}")

    print(f"\n--- Passed Checks ({len(result.passed_checks)}) ---")
    for check in result.passed_checks:
        print(f"  ✅ {check}")

    if result.warnings:
        print(f"\n--- Warnings ({len(result.warnings)}) ---")
        for warning in result.warnings:
            print(f"  ⚠️  {warning}")

    if result.failed_checks:
        print(f"\n--- Failed Checks ({len(result.failed_checks)}) ---")
        for check in result.failed_checks:
            print(f"  ❌ {check}")

    print(f"\n--- Statistics ---")
    for key, value in result.statistics.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    - {k}: {v}")
        else:
            print(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(description="Validate AgriMemory Dataset v2")
    parser.add_argument("--data-dir", type=str, default="data/text",
                        help="Path to data/text directory")
    parser.add_argument("--images-dir", type=str, default="data/images",
                        help="Path to data/images directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for validation results")

    args = parser.parse_args()

    # Handle relative paths
    script_dir = Path(__file__).parent.parent.parent
    data_dir = script_dir / args.data_dir
    images_dir = script_dir / args.images_dir

    print("=" * 70)
    print("AGRIMEMORY DATASET v2 VALIDATION")
    print("=" * 70)
    print(f"\nData directory: {data_dir}")
    print(f"Images directory: {images_dir}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    validator = DatasetValidator(str(data_dir), str(images_dir))

    # Validate both datasets
    results = []

    # AgriConvMem
    print("\nValidating AgriConvMem...")
    agriconvmem_result = validator.validate_agriconvmem()
    print_validation_report(agriconvmem_result)
    results.append(agriconvmem_result)

    # AgriMultiHop
    print("\nValidating AgriMultiHop...")
    agrimultihop_result = validator.validate_agrimultihop()
    print_validation_report(agrimultihop_result)
    results.append(agrimultihop_result)

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    all_valid = all(r.is_valid for r in results)

    print(f"\nAgriConvMem: {'✅ VALID' if agriconvmem_result.is_valid else '❌ INVALID'}")
    print(f"AgriMultiHop: {'✅ VALID' if agrimultihop_result.is_valid else '❌ INVALID'}")
    print(f"\nOverall: {'✅ ALL DATASETS VALID' if all_valid else '❌ SOME DATASETS INVALID'}")

    # Reviewer requirements check
    print("\n--- Reviewer Requirements Compliance ---")
    print(f"  AgriConvMem samples: {agriconvmem_result.total_samples}/500 {'✅' if agriconvmem_result.total_samples >= 500 else '❌'}")
    print(f"  AgriMultiHop samples: {agrimultihop_result.total_samples}/2000 {'✅' if agrimultihop_result.total_samples >= 2000 else '❌'}")

    # Model info
    print("\n--- Models Used for Generation ---")
    for result in results:
        model = result.statistics.get('generation_model', 'Unknown')
        method = result.statistics.get('generation_method', 'Unknown')
        print(f"  {result.dataset_name}:")
        print(f"    Model: {model}")
        print(f"    Method: {method}")

    # Save results if requested
    if args.output:
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'results': [
                {
                    'dataset_name': r.dataset_name,
                    'total_samples': r.total_samples,
                    'splits': r.splits,
                    'is_valid': r.is_valid,
                    'passed_checks': r.passed_checks,
                    'failed_checks': r.failed_checks,
                    'warnings': r.warnings,
                    'statistics': r.statistics
                }
                for r in results
            ]
        }

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    # Exit code based on validation
    sys.exit(0 if all_valid else 1)


if __name__ == "__main__":
    main()
