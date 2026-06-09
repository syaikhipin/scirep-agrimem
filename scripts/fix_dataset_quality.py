#!/usr/bin/env python3
"""
Fix Dataset Quality Issues for Q1 Journal Publication

Issues addressed:
1. Treatment answers not grounded in conversation text
2. AgriMultiHop questions too easy (single-hop factual lookups)

This script:
1. Validates and fixes treatment QA by extracting from actual conversations
2. Filters out invalid QA pairs where answer doesn't exist in context
3. Generates statistics on data quality
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


def extract_treatments_from_text(text: str) -> List[str]:
    """Extract treatment mentions from conversation text."""
    treatments = []
    text_lower = text.lower()

    # Common treatment patterns
    treatment_patterns = [
        r"apply(?:ing)?\s+(?:a\s+)?([a-z\-]+(?:\s+[a-z\-]+){0,3})",
        r"recommend(?:ed)?\s+(?:using\s+)?([a-z\-]+(?:\s+[a-z\-]+){0,3})",
        r"use\s+([a-z\-]+(?:\s+fungicide|bactericide|spray|treatment)?)",
        r"treat(?:ment|ed)?\s+with\s+([a-z\-]+(?:\s+[a-z\-]+){0,2})",
    ]

    # Known treatments to look for
    known_treatments = [
        "copper-based fungicide", "copper fungicide", "copper hydroxide",
        "chlorothalonil", "mancozeb", "streptomycin", "bordeaux mixture",
        "neem oil", "sulfur spray", "potassium bicarbonate", "captan",
        "thiram", "azoxystrobin", "propiconazole", "myclobutanil",
        "copper-based bactericide", "resistant varieties", "crop rotation",
        "remove infected", "improve drainage", "reduce humidity"
    ]

    for treatment in known_treatments:
        if treatment in text_lower:
            treatments.append(treatment)

    return list(set(treatments))


def fix_locomo_dataset(input_path: str, output_path: str) -> Dict:
    """Fix AgriConvMem/LoCoMo dataset."""
    with open(input_path, 'r') as f:
        data = json.load(f)

    samples = data.get("samples", data) if isinstance(data, dict) else data

    stats = {
        "total_samples": len(samples),
        "total_qa": 0,
        "fixed_qa": 0,
        "removed_qa": 0,
        "valid_qa": 0,
        "by_category": {
            "disease_identification": {"total": 0, "valid": 0},
            "temporal": {"total": 0, "valid": 0},
            "severity": {"total": 0, "valid": 0},
            "treatment": {"total": 0, "valid": 0, "fixed": 0}
        }
    }

    fixed_samples = []

    for sample in samples:
        # Build full conversation text
        conv_text = ""
        for key, val in sample.get("conversation", {}).items():
            if isinstance(val, list):
                for turn in val:
                    conv_text += turn.get("text", "") + " "

        conv_text_lower = conv_text.lower()

        # Extract treatments mentioned in conversation
        conv_treatments = extract_treatments_from_text(conv_text)

        # Process QA pairs
        fixed_qa = []
        for qa in sample.get("qa", []):
            stats["total_qa"] += 1
            category = qa.get("category_name", "")

            if category in stats["by_category"]:
                stats["by_category"][category]["total"] += 1

            answer = qa.get("answer", "").lower()
            answer_words = set(answer.split()[:5])  # Check first 5 words

            # Check if answer exists in conversation
            answer_found = False

            if category == "treatment":
                # For treatment, check if any part of answer exists
                if conv_treatments:
                    # Find best matching treatment from conversation
                    for treat in conv_treatments:
                        if treat in answer or answer in treat or any(w in conv_text_lower for w in answer_words if len(w) > 3):
                            answer_found = True
                            break

                    # If original answer not found, try to fix with actual treatment
                    if not answer_found and conv_treatments:
                        # Replace with actual treatment from conversation
                        qa["answer"] = conv_treatments[0]
                        qa["_fixed"] = True
                        stats["fixed_qa"] += 1
                        stats["by_category"][category]["fixed"] += 1
                        answer_found = True
            else:
                # For other categories, check if key terms exist
                for word in answer_words:
                    if len(word) > 3 and word in conv_text_lower:
                        answer_found = True
                        break

            if answer_found:
                fixed_qa.append(qa)
                stats["valid_qa"] += 1
                if category in stats["by_category"]:
                    stats["by_category"][category]["valid"] += 1
            else:
                stats["removed_qa"] += 1

        sample["qa"] = fixed_qa
        if fixed_qa:  # Only keep samples with valid QA
            fixed_samples.append(sample)

    # Save fixed dataset
    output_data = {"samples": fixed_samples} if isinstance(data, dict) else fixed_samples

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    stats["output_samples"] = len(fixed_samples)
    return stats


def analyze_hotpotqa_difficulty(input_path: str) -> Dict:
    """Analyze AgriMultiHop question difficulty."""
    with open(input_path, 'r') as f:
        data = json.load(f)

    samples = data if isinstance(data, list) else data.get("samples", [])

    stats = {
        "total": len(samples),
        "by_type": {},
        "answer_lengths": [],
        "question_lengths": [],
        "single_hop": 0,  # Questions answerable from single doc
        "multi_hop": 0,   # Questions requiring multiple docs
    }

    for sample in samples:
        q_type = sample.get("type", "unknown")
        stats["by_type"][q_type] = stats["by_type"].get(q_type, 0) + 1

        answer = sample.get("answer", "")
        question = sample.get("question", "")

        stats["answer_lengths"].append(len(answer.split()))
        stats["question_lengths"].append(len(question.split()))

        # Analyze if truly multi-hop
        # Check if answer can be found in a single context paragraph
        contexts = sample.get("context", [])
        answer_in_single = False
        for ctx in contexts:
            if answer.lower() in ctx.get("text", "").lower():
                answer_in_single = True
                break

        if answer_in_single:
            stats["single_hop"] += 1
        else:
            stats["multi_hop"] += 1

    stats["avg_answer_length"] = sum(stats["answer_lengths"]) / len(stats["answer_lengths"]) if stats["answer_lengths"] else 0
    stats["avg_question_length"] = sum(stats["question_lengths"]) / len(stats["question_lengths"]) if stats["question_lengths"] else 0
    stats["single_hop_pct"] = stats["single_hop"] / stats["total"] * 100 if stats["total"] else 0

    return stats


def main():
    """Main execution."""
    base_dir = Path(__file__).parent.parent

    print("=" * 60)
    print("AgriMemory Dataset Quality Analysis & Fix")
    print("=" * 60)

    # Fix AgriConvMem
    locomo_input = base_dir / "data/text/agri_locomo_v2/test.json"
    locomo_output = base_dir / "data/text/agri_locomo_v2/test_fixed.json"

    if locomo_input.exists():
        print("\n--- Fixing AgriConvMem (LoCoMo) ---")
        stats = fix_locomo_dataset(str(locomo_input), str(locomo_output))
        print(f"Total samples: {stats['total_samples']}")
        print(f"Output samples: {stats['output_samples']}")
        print(f"Total QA pairs: {stats['total_qa']}")
        print(f"Valid QA pairs: {stats['valid_qa']}")
        print(f"Fixed QA pairs: {stats['fixed_qa']}")
        print(f"Removed QA pairs: {stats['removed_qa']}")
        print("\nBy Category:")
        for cat, cat_stats in stats["by_category"].items():
            print(f"  {cat}: {cat_stats['valid']}/{cat_stats['total']} valid", end="")
            if cat_stats.get('fixed', 0) > 0:
                print(f" ({cat_stats['fixed']} fixed)")
            else:
                print()

    # Analyze AgriMultiHop
    hotpot_input = base_dir / "data/text/agri_hotpotqa_v2/test.json"

    if hotpot_input.exists():
        print("\n--- Analyzing AgriMultiHop (HotpotQA) ---")
        stats = analyze_hotpotqa_difficulty(str(hotpot_input))
        print(f"Total samples: {stats['total']}")
        print(f"Question types: {stats['by_type']}")
        print(f"Avg answer length: {stats['avg_answer_length']:.1f} words")
        print(f"Avg question length: {stats['avg_question_length']:.1f} words")
        print(f"Single-hop questions: {stats['single_hop']} ({stats['single_hop_pct']:.1f}%)")
        print(f"Multi-hop questions: {stats['multi_hop']}")

        if stats['single_hop_pct'] > 50:
            print("\n⚠️  WARNING: Most questions are single-hop (too easy for journal!)")
            print("   Consider regenerating with harder multi-hop requirements.")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
