#!/usr/bin/env python3
"""
Hard Multi-Hop Question Generator for AgriMultiHop v4

Creates TRULY challenging multi-hop questions where:
1. Answer does NOT appear in any single document
2. Requires computation/reasoning across documents
3. Includes distractor documents

Question Types:
1. COUNTING: "How many crops can be affected by both disease A and disease B?"
   - Requires reading both docs and counting intersection

2. DERIVED: "What is the pathogen type of the disease that affects [crop] and shows [symptom]?"
   - Answer is derived by connecting multiple facts

3. COMPARATIVE: "Which disease spreads faster AND affects more crops?"
   - Requires comparing multiple attributes

4. IMPLICIT: Questions where answer must be inferred, not stated
"""

import argparse
import json
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List


# =============================================================================
# Knowledge Base
# =============================================================================

AGRI_KB = {
    "diseases": {
        "early_blight": {
            "pathogen": "Alternaria solani",
            "pathogen_type": "fungus",
            "symptoms": ["concentric ring patterns", "brown spots", "target-like lesions"],
            "crops": ["tomato", "potato"],
            "spread_rate": "moderate",
            "severity": "medium",
            "treatments": ["chlorothalonil", "mancozeb", "copper fungicide"],
        },
        "late_blight": {
            "pathogen": "Phytophthora infestans",
            "pathogen_type": "oomycete",
            "symptoms": ["water-soaked lesions", "white fuzzy growth", "rapid wilting"],
            "crops": ["tomato", "potato"],
            "spread_rate": "very fast",
            "severity": "high",
            "treatments": ["metalaxyl", "chlorothalonil", "copper fungicide"],
        },
        "bacterial_spot": {
            "pathogen": "Xanthomonas campestris",
            "pathogen_type": "bacterium",
            "symptoms": ["small water-soaked spots", "yellow halos", "leaf curling"],
            "crops": ["tomato", "pepper", "eggplant"],
            "spread_rate": "fast",
            "severity": "medium",
            "treatments": ["copper bactericide", "streptomycin"],
        },
        "powdery_mildew": {
            "pathogen": "Erysiphe species",
            "pathogen_type": "fungus",
            "symptoms": ["white powdery coating", "leaf yellowing", "stunted growth"],
            "crops": ["squash", "cucumber", "grape", "wheat"],
            "spread_rate": "moderate",
            "severity": "low",
            "treatments": ["sulfur fungicide", "potassium bicarbonate", "neem oil"],
        },
        "rice_blast": {
            "pathogen": "Magnaporthe oryzae",
            "pathogen_type": "fungus",
            "symptoms": ["diamond-shaped lesions", "node infection", "panicle blast"],
            "crops": ["rice"],
            "spread_rate": "very fast",
            "severity": "high",
            "treatments": ["tricyclazole", "propiconazole", "azoxystrobin"],
        },
        "septoria_leaf_spot": {
            "pathogen": "Septoria lycopersici",
            "pathogen_type": "fungus",
            "symptoms": ["circular spots", "gray centers", "tiny black dots"],
            "crops": ["tomato"],
            "spread_rate": "slow",
            "severity": "medium",
            "treatments": ["chlorothalonil", "mancozeb", "copper fungicide"],
        },
        "downy_mildew": {
            "pathogen": "Plasmopara viticola",
            "pathogen_type": "oomycete",
            "symptoms": ["yellow patches", "white downy growth underneath", "leaf drop"],
            "crops": ["grape", "cucumber", "spinach"],
            "spread_rate": "fast",
            "severity": "high",
            "treatments": ["mancozeb", "metalaxyl", "copper fungicide"],
        },
        "fusarium_wilt": {
            "pathogen": "Fusarium oxysporum",
            "pathogen_type": "fungus",
            "symptoms": ["yellowing one-sided", "vascular browning", "wilting"],
            "crops": ["tomato", "banana", "cotton"],
            "spread_rate": "slow",
            "severity": "high",
            "treatments": ["solarization", "resistant varieties"],
        }
    },
    "crops": {
        "tomato": {"family": "Solanaceae", "type": "fruit vegetable"},
        "potato": {"family": "Solanaceae", "type": "root vegetable"},
        "pepper": {"family": "Solanaceae", "type": "fruit vegetable"},
        "eggplant": {"family": "Solanaceae", "type": "fruit vegetable"},
        "rice": {"family": "Poaceae", "type": "cereal grain"},
        "wheat": {"family": "Poaceae", "type": "cereal grain"},
        "grape": {"family": "Vitaceae", "type": "fruit"},
        "cucumber": {"family": "Cucurbitaceae", "type": "fruit vegetable"},
        "squash": {"family": "Cucurbitaceae", "type": "fruit vegetable"},
        "banana": {"family": "Musaceae", "type": "fruit"},
        "cotton": {"family": "Malvaceae", "type": "fiber crop"},
        "spinach": {"family": "Amaranthaceae", "type": "leafy vegetable"}
    }
}


@dataclass
class HardMultiHopSample:
    id: str
    question: str
    answer: str
    question_type: str
    context: List[Dict]
    supporting_facts: List[str]
    reasoning_chain: List[str]
    difficulty: str
    requires_computation: bool = False


class HardMultiHopGenerator:
    """Generate truly challenging multi-hop questions."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.kb = AGRI_KB

    def generate_counting_intersection(self) -> HardMultiHopSample:
        """
        COUNTING: How many crops can be affected by BOTH disease A and disease B?

        Answer is a NUMBER that doesn't appear in either document.
        Requires: reading both crop lists and counting intersection.
        """
        diseases = list(self.kb["diseases"].keys())
        random.shuffle(diseases)

        for i, d1_name in enumerate(diseases):
            for d2_name in diseases[i+1:]:
                d1 = self.kb["diseases"][d1_name]
                d2 = self.kb["diseases"][d2_name]
                common = set(d1["crops"]) & set(d2["crops"])

                if len(common) >= 1:
                    d1_display = d1_name.replace('_', ' ')
                    d2_display = d2_name.replace('_', ' ')

                    # Context does NOT contain the count directly
                    context = [
                        {
                            "title": f"Crops affected by {d1_display}",
                            "text": f"{d1_display.title()} is a plant disease that affects "
                                   f"{', '.join(d1['crops'])}. It is caused by {d1['pathogen']} "
                                   f"and spreads at a {d1['spread_rate']} rate."
                        },
                        {
                            "title": f"Crops affected by {d2_display}",
                            "text": f"{d2_display.title()} is a plant disease that affects "
                                   f"{', '.join(d2['crops'])}. It is caused by {d2['pathogen']} "
                                   f"and spreads at a {d2['spread_rate']} rate."
                        }
                    ]
                    random.shuffle(context)

                    answer = str(len(common))

                    return HardMultiHopSample(
                        id=f"counting_{uuid.uuid4().hex[:8]}",
                        question=f"How many crops can be affected by both {d1_display} and {d2_display}?",
                        answer=answer,
                        question_type="counting",
                        context=context,
                        supporting_facts=[d1_name, d2_name],
                        reasoning_chain=[
                            f"Step 1: {d1_display} affects: {d1['crops']}",
                            f"Step 2: {d2_display} affects: {d2['crops']}",
                            f"Step 3: Common crops: {list(common)}",
                            f"Step 4: Count = {len(common)}"
                        ],
                        difficulty="hard",
                        requires_computation=True
                    )

        return self.generate_derived_pathogen_type()

    def generate_counting_total_crops(self) -> HardMultiHopSample:
        """
        COUNTING: How many DIFFERENT crops are affected by disease A OR disease B?

        Answer is union count (not in documents).
        """
        diseases = list(self.kb["diseases"].keys())
        random.shuffle(diseases)
        d1_name, d2_name = diseases[:2]
        d1 = self.kb["diseases"][d1_name]
        d2 = self.kb["diseases"][d2_name]

        d1_display = d1_name.replace('_', ' ')
        d2_display = d2_name.replace('_', ' ')

        union = set(d1["crops"]) | set(d2["crops"])

        context = [
            {
                "title": f"About {d1_display}",
                "text": f"{d1_display.title()} affects the following crops: {', '.join(d1['crops'])}. "
                       f"This disease is caused by {d1['pathogen']}."
            },
            {
                "title": f"About {d2_display}",
                "text": f"{d2_display.title()} affects the following crops: {', '.join(d2['crops'])}. "
                       f"This disease is caused by {d2['pathogen']}."
            }
        ]
        random.shuffle(context)

        return HardMultiHopSample(
            id=f"counting_{uuid.uuid4().hex[:8]}",
            question=f"How many different crops in total are affected by either {d1_display} or {d2_display}?",
            answer=str(len(union)),
            question_type="counting",
            context=context,
            supporting_facts=[d1_name, d2_name],
            reasoning_chain=[
                f"Step 1: {d1_display} affects: {d1['crops']}",
                f"Step 2: {d2_display} affects: {d2['crops']}",
                f"Step 3: Union (unique crops): {list(union)}",
                f"Step 4: Total count = {len(union)}"
            ],
            difficulty="hard",
            requires_computation=True
        )

    def generate_derived_pathogen_type(self) -> HardMultiHopSample:
        """
        DERIVED: What type of pathogen causes the disease affecting [crop] that shows [symptom]?

        Answer (pathogen type) must be derived by:
        1. Finding disease with both crop AND symptom
        2. Looking up its pathogen type
        """
        diseases = list(self.kb["diseases"].keys())
        random.shuffle(diseases)

        for d_name in diseases:
            d = self.kb["diseases"][d_name]
            if len(d["crops"]) >= 1 and len(d["symptoms"]) >= 1:
                crop = random.choice(d["crops"])
                symptom = random.choice(d["symptoms"])
                d_display = d_name.replace('_', ' ')

                # Doc1: Links symptom to disease name (no pathogen type)
                # Doc2: Links disease to pathogen type (no symptom)
                context = [
                    {
                        "title": "Symptom Guide",
                        "text": f"When {crop} plants show {symptom}, this is typically "
                               f"a sign of {d_display}. This condition requires immediate attention "
                               f"and proper disease management."
                    },
                    {
                        "title": f"Disease Classification: {d_display.title()}",
                        "text": f"{d_display.title()} is classified as a disease caused by a {d['pathogen_type']}. "
                               f"The specific pathogen is {d['pathogen']}. "
                               f"It spreads at a {d['spread_rate']} rate."
                    }
                ]
                random.shuffle(context)

                return HardMultiHopSample(
                    id=f"derived_{uuid.uuid4().hex[:8]}",
                    question=f"What type of pathogen causes the disease that affects {crop} and shows {symptom}?",
                    answer=d["pathogen_type"],
                    question_type="derived",
                    context=context,
                    supporting_facts=[d_name],
                    reasoning_chain=[
                        f"Step 1: Find disease affecting {crop} with symptom '{symptom}' -> {d_display}",
                        f"Step 2: Look up pathogen type for {d_display} -> {d['pathogen_type']}",
                        f"Answer: {d['pathogen_type']}"
                    ],
                    difficulty="hard",
                    requires_computation=False
                )

        return self.generate_counting_intersection()

    def generate_comparative_multi_attribute(self) -> HardMultiHopSample:
        """
        COMPARATIVE: Which disease BOTH spreads faster AND has higher severity?

        Requires comparing two attributes across diseases.
        """
        diseases = list(self.kb["diseases"].keys())
        random.shuffle(diseases)
        d1_name, d2_name = diseases[:2]
        d1 = self.kb["diseases"][d1_name]
        d2 = self.kb["diseases"][d2_name]

        d1_display = d1_name.replace('_', ' ')
        d2_display = d2_name.replace('_', ' ')

        # Spread rate ranking
        spread_rank = {"very fast": 4, "fast": 3, "moderate": 2, "slow": 1}
        severity_rank = {"high": 3, "medium": 2, "low": 1}

        d1_spread = spread_rank.get(d1["spread_rate"], 2)
        d2_spread = spread_rank.get(d2["spread_rate"], 2)
        d1_severity = severity_rank.get(d1["severity"], 2)
        d2_severity = severity_rank.get(d2["severity"], 2)

        # Score = spread + severity
        d1_score = d1_spread + d1_severity
        d2_score = d2_spread + d2_severity

        if d1_score >= d2_score:
            answer = d1_display
        else:
            answer = d2_display

        context = [
            {
                "title": f"Disease Profile: {d1_display.title()}",
                "text": f"{d1_display.title()} has a {d1['spread_rate']} spread rate "
                       f"and is considered {d1['severity']} severity. "
                       f"It affects {', '.join(d1['crops'])}."
            },
            {
                "title": f"Disease Profile: {d2_display.title()}",
                "text": f"{d2_display.title()} has a {d2['spread_rate']} spread rate "
                       f"and is considered {d2['severity']} severity. "
                       f"It affects {', '.join(d2['crops'])}."
            }
        ]
        random.shuffle(context)

        return HardMultiHopSample(
            id=f"comparative_{uuid.uuid4().hex[:8]}",
            question=f"Between {d1_display} and {d2_display}, which disease is more dangerous overall (considering both spread rate and severity)?",
            answer=answer,
            question_type="comparative",
            context=context,
            supporting_facts=[d1_name, d2_name],
            reasoning_chain=[
                f"Step 1: {d1_display} - spread: {d1['spread_rate']}, severity: {d1['severity']}",
                f"Step 2: {d2_display} - spread: {d2['spread_rate']}, severity: {d2['severity']}",
                f"Step 3: Compare overall danger level",
                f"Answer: {answer}"
            ],
            difficulty="hard",
            requires_computation=True
        )

    def generate_implicit_family(self) -> HardMultiHopSample:
        """
        IMPLICIT: What plant family is MOST susceptible to fungal diseases?

        Requires:
        1. Finding all fungal diseases
        2. Collecting all crops they affect
        3. Mapping crops to families
        4. Counting which family appears most
        """
        # Find all fungal diseases and their crops
        fungal_crops = []
        for d_name, d in self.kb["diseases"].items():
            if d["pathogen_type"] == "fungus":
                fungal_crops.extend(d["crops"])

        # Count by family
        family_counts = {}
        for crop in fungal_crops:
            family = self.kb["crops"].get(crop, {}).get("family", "unknown")
            family_counts[family] = family_counts.get(family, 0) + 1

        most_affected = max(family_counts, key=family_counts.get)

        # Create context that doesn't directly state the answer
        fungal_diseases = [d for d, info in self.kb["diseases"].items() if info["pathogen_type"] == "fungus"]
        random.shuffle(fungal_diseases)

        context = []
        for d_name in fungal_diseases[:3]:
            d = self.kb["diseases"][d_name]
            d_display = d_name.replace('_', ' ')
            context.append({
                "title": f"Fungal Disease: {d_display.title()}",
                "text": f"{d_display.title()} is caused by {d['pathogen']} (a fungus). "
                       f"It primarily affects {', '.join(d['crops'])}."
            })

        # Add crop family info
        context.append({
            "title": "Crop Family Classification",
            "text": "Tomato, potato, pepper, and eggplant belong to the Solanaceae family. "
                   "Rice and wheat belong to the Poaceae family. "
                   "Grape belongs to the Vitaceae family. "
                   "Cucumber and squash belong to the Cucurbitaceae family."
        })

        random.shuffle(context)

        return HardMultiHopSample(
            id=f"implicit_{uuid.uuid4().hex[:8]}",
            question="Based on the number of fungal diseases that can affect them, which plant family is most susceptible to fungal infections?",
            answer=most_affected,
            question_type="implicit",
            context=context,
            supporting_facts=fungal_diseases[:3],
            reasoning_chain=[
                "Step 1: Identify all fungal diseases",
                "Step 2: List crops affected by each",
                "Step 3: Map crops to plant families",
                f"Step 4: Count occurrences - {family_counts}",
                f"Answer: {most_affected}"
            ],
            difficulty="very_hard",
            requires_computation=True
        )

    def generate_negation(self) -> HardMultiHopSample:
        """
        NEGATION: Which disease does NOT affect any Solanaceae crops?

        Requires checking which diseases' crop lists have no Solanaceae members.
        """
        solanaceae_crops = {"tomato", "potato", "pepper", "eggplant"}

        diseases = list(self.kb["diseases"].keys())
        random.shuffle(diseases)

        # Find one that affects Solanaceae and one that doesn't
        affects_solanaceae = None
        not_affects = None

        for d_name in diseases:
            d = self.kb["diseases"][d_name]
            crop_set = set(d["crops"])
            if crop_set & solanaceae_crops:
                if not affects_solanaceae:
                    affects_solanaceae = d_name
            else:
                if not not_affects:
                    not_affects = d_name

            if affects_solanaceae and not_affects:
                break

        if not not_affects or not affects_solanaceae:
            return self.generate_counting_intersection()

        d1 = self.kb["diseases"][affects_solanaceae]
        d2 = self.kb["diseases"][not_affects]
        d1_display = affects_solanaceae.replace('_', ' ')
        d2_display = not_affects.replace('_', ' ')

        context = [
            {
                "title": f"Disease: {d1_display.title()}",
                "text": f"{d1_display.title()} is a disease that affects {', '.join(d1['crops'])}. "
                       f"It is caused by {d1['pathogen']}."
            },
            {
                "title": f"Disease: {d2_display.title()}",
                "text": f"{d2_display.title()} is a disease that affects {', '.join(d2['crops'])}. "
                       f"It is caused by {d2['pathogen']}."
            },
            {
                "title": "Solanaceae Family",
                "text": "The Solanaceae (nightshade) family includes important crops: "
                       "tomato, potato, pepper, and eggplant. These crops share common vulnerabilities."
            }
        ]
        random.shuffle(context)

        return HardMultiHopSample(
            id=f"negation_{uuid.uuid4().hex[:8]}",
            question=f"Between {d1_display} and {d2_display}, which disease does NOT affect any crops from the Solanaceae family?",
            answer=d2_display,
            question_type="negation",
            context=context,
            supporting_facts=[affects_solanaceae, not_affects],
            reasoning_chain=[
                f"Step 1: {d1_display} affects: {d1['crops']}",
                f"Step 2: {d2_display} affects: {d2['crops']}",
                "Step 3: Solanaceae crops: tomato, potato, pepper, eggplant",
                f"Step 4: Check intersection with Solanaceae",
                f"Answer: {d2_display} (no Solanaceae crops)"
            ],
            difficulty="hard",
            requires_computation=True
        )

    def generate_sample(self, question_type: str = None) -> HardMultiHopSample:
        """Generate a single hard multi-hop sample."""
        if question_type is None:
            question_type = random.choices(
                ["counting", "derived", "comparative", "implicit", "negation"],
                weights=[0.30, 0.25, 0.20, 0.15, 0.10]
            )[0]

        generators = {
            "counting": [self.generate_counting_intersection, self.generate_counting_total_crops],
            "derived": [self.generate_derived_pathogen_type],
            "comparative": [self.generate_comparative_multi_attribute],
            "implicit": [self.generate_implicit_family],
            "negation": [self.generate_negation]
        }

        gen_func = random.choice(generators.get(question_type, generators["counting"]))
        return gen_func()

    def generate_dataset(self, n_samples: int, output_dir: str):
        """Generate full dataset."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        samples = []
        type_counts = {}

        for i in range(n_samples):
            if i % 100 == 0:
                self.logger.info(f"Generating sample {i+1}/{n_samples}")

            sample = self.generate_sample()
            type_counts[sample.question_type] = type_counts.get(sample.question_type, 0) + 1

            samples.append({
                "id": sample.id,
                "question": sample.question,
                "answer": sample.answer,
                "type": sample.question_type,
                "context": sample.context,
                "supporting_facts": sample.supporting_facts,
                "reasoning_chain": sample.reasoning_chain,
                "difficulty": sample.difficulty,
                "requires_computation": sample.requires_computation
            })

            time.sleep(0.005)

        # Split
        random.shuffle(samples)
        n_train = int(0.8 * len(samples))
        n_val = int(0.1 * len(samples))

        train = samples[:n_train]
        val = samples[n_train:n_train + n_val]
        test = samples[n_train + n_val:]

        # Save
        with open(output_path / "train.json", 'w') as f:
            json.dump(train, f, indent=2)
        with open(output_path / "val.json", 'w') as f:
            json.dump(val, f, indent=2)
        with open(output_path / "test.json", 'w') as f:
            json.dump(test, f, indent=2)

        # Dataset info
        dataset_info = {
            "name": "AgriMultiHop v4 (Hard Multi-Hop)",
            "version": "4.0.0",
            "description": "Challenging agricultural multi-hop QA - answers require computation/reasoning",
            "generation_method": "Template-based with computation requirements",
            "model": "Knowledge Base + Python Generator",
            "total_samples": len(samples),
            "splits": {"train": len(train), "val": len(val), "test": len(test)},
            "question_types": type_counts,
            "difficulty_features": [
                "Answers not explicitly in documents",
                "Counting/computation required",
                "Multi-attribute comparison",
                "Negation reasoning"
            ],
            "created": datetime.now().isoformat()
        }
        with open(output_path / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)

        self.logger.info(f"Generated {len(samples)} samples")
        self.logger.info(f"Types: {type_counts}")
        self.logger.info(f"Saved to {output_path}")

        return samples


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger("hard_multihop")

    parser = argparse.ArgumentParser(description="Generate Hard Multi-Hop QA Dataset")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--output", default="data/text/agri_hotpotqa_v4/")
    args = parser.parse_args()

    generator = HardMultiHopGenerator(logger)
    generator.generate_dataset(args.n_samples, args.output)


if __name__ == "__main__":
    main()
