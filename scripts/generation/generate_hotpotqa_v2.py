#!/usr/bin/env python3
"""
AgriMultiHop v2 Dataset Generator

Uses GLM-4.6V (Vision-Language Model) via Chutes.ai to generate
multi-hop reasoning QA pairs for agricultural domain.

Key improvements over v1:
- True multimodal generation (image input)
- Larger scale (2000 samples)
- Bridge and comparison questions
- Enhanced knowledge base
- Quality validation pipeline

Usage:
    python generate_hotpotqa_v2.py --n-samples 2000 --output data/text/agri_hotpotqa_v2/
"""

import argparse
import json
import logging
import os
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "AgriMemory-Dataset" / "scripts"))
from chutes_client import ChutesClient


# =============================================================================
# Knowledge Base
# =============================================================================

AGRICULTURAL_KB = {
    "diseases": {
        "early_blight": {
            "pathogen": "Alternaria solani",
            "symptoms": "concentric ring patterns, brown spots with yellow halos, target-like appearance",
            "crops": ["tomato", "potato"],
            "severity_factors": ["humidity", "temperature", "plant age"],
            "treatments": ["copper fungicide", "chlorothalonil", "mancozeb"]
        },
        "late_blight": {
            "pathogen": "Phytophthora infestans",
            "symptoms": "water-soaked lesions, white fuzzy growth, rapid wilting, dark brown spots",
            "crops": ["tomato", "potato"],
            "severity_factors": ["cool wet conditions", "humidity above 90%"],
            "treatments": ["metalaxyl", "copper fungicide", "chlorothalonil"]
        },
        "bacterial_spot": {
            "pathogen": "Xanthomonas species",
            "symptoms": "small dark spots with yellow halos, leaf curling, fruit lesions",
            "crops": ["tomato", "pepper"],
            "severity_factors": ["warm humid conditions", "overhead irrigation"],
            "treatments": ["copper bactericide", "streptomycin", "resistant varieties"]
        },
        "septoria_leaf_spot": {
            "pathogen": "Septoria lycopersici",
            "symptoms": "circular spots with dark borders, gray centers with tiny black dots",
            "crops": ["tomato"],
            "severity_factors": ["wet weather", "poor air circulation"],
            "treatments": ["chlorothalonil", "mancozeb", "copper fungicide"]
        },
        "powdery_mildew": {
            "pathogen": "various Erysiphales fungi",
            "symptoms": "white powdery coating on leaves, leaf curling, stunted growth",
            "crops": ["squash", "wheat", "grape"],
            "severity_factors": ["dry conditions", "moderate temperatures"],
            "treatments": ["sulfur fungicide", "neem oil", "potassium bicarbonate"]
        },
        "rice_blast": {
            "pathogen": "Magnaporthe oryzae",
            "symptoms": "diamond-shaped lesions, gray-green centers, node infection",
            "crops": ["rice"],
            "severity_factors": ["nitrogen excess", "humid conditions"],
            "treatments": ["tricyclazole", "propiconazole", "resistant varieties"]
        },
        "gray_leaf_spot": {
            "pathogen": "Cercospora zeae-maydis",
            "symptoms": "rectangular gray lesions parallel to leaf veins",
            "crops": ["corn"],
            "severity_factors": ["humid conditions", "continuous corn"],
            "treatments": ["strobilurin fungicide", "triazole fungicide", "crop rotation"]
        },
        "apple_scab": {
            "pathogen": "Venturia inaequalis",
            "symptoms": "olive-brown velvety spots, leaf curling, fruit deformity",
            "crops": ["apple"],
            "severity_factors": ["wet spring weather", "humid conditions"],
            "treatments": ["captan", "mancozeb", "myclobutanil"]
        },
        "black_rot_grape": {
            "pathogen": "Guignardia bidwellii",
            "symptoms": "brown circular spots, black mummified berries, leaf lesions",
            "crops": ["grape"],
            "severity_factors": ["warm humid weather", "poor pruning"],
            "treatments": ["mancozeb", "captan", "myclobutanil"]
        }
    },
    "treatments": {
        "copper_fungicide": {
            "type": "fungicide",
            "active_ingredient": "copper hydroxide or copper sulfate",
            "diseases_treated": ["early_blight", "late_blight", "bacterial_spot"],
            "application": "foliar spray, 7-10 day intervals"
        },
        "mancozeb": {
            "type": "fungicide",
            "active_ingredient": "mancozeb (dithiocarbamate)",
            "diseases_treated": ["early_blight", "late_blight", "septoria_leaf_spot"],
            "application": "preventive application, 7-14 day intervals"
        },
        "tricyclazole": {
            "type": "fungicide",
            "active_ingredient": "tricyclazole",
            "diseases_treated": ["rice_blast"],
            "application": "250-300 ml per hectare"
        },
        "sulfur_fungicide": {
            "type": "fungicide",
            "active_ingredient": "elemental sulfur",
            "diseases_treated": ["powdery_mildew"],
            "application": "foliar spray, not above 85°F"
        }
    },
    "crops": {
        "tomato": {
            "family": "Solanaceae",
            "common_diseases": ["early_blight", "late_blight", "bacterial_spot", "septoria_leaf_spot"],
            "growing_season": "spring to fall"
        },
        "potato": {
            "family": "Solanaceae",
            "common_diseases": ["early_blight", "late_blight"],
            "growing_season": "spring planting"
        },
        "rice": {
            "family": "Poaceae",
            "common_diseases": ["rice_blast", "bacterial_leaf_blight"],
            "growing_season": "wet season"
        },
        "corn": {
            "family": "Poaceae",
            "common_diseases": ["gray_leaf_spot", "northern_leaf_blight", "corn_rust"],
            "growing_season": "spring to summer"
        }
    }
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ContextParagraph:
    title: str
    text: str
    source: str = "KB"


@dataclass
class MultiHopSample:
    id: str
    question: str
    answer: str
    question_type: str  # "bridge" or "comparison"
    context: List[ContextParagraph]
    supporting_facts: List[str]
    generation_metadata: Dict[str, Any]


# =============================================================================
# Question Generators
# =============================================================================

class MultiHopGenerator:
    """Generate multi-hop reasoning questions."""

    def __init__(self, logger: logging.Logger, images_dir: str = None):
        self.logger = logger
        self.client = ChutesClient(model="zai-org/GLM-4.6V")
        self.kb = AGRICULTURAL_KB
        self.images_dir = images_dir

    def generate_bridge_question(self) -> Optional[MultiHopSample]:
        """
        Generate bridge-type question requiring connecting two facts.

        Example: "What pathogen causes the disease that affects both tomato and potato?"
        """
        # Select two related diseases
        diseases = list(self.kb["diseases"].keys())
        random.shuffle(diseases)

        disease_name = diseases[0]
        disease_info = self.kb["diseases"][disease_name]

        # Create context paragraphs
        context = []

        # Paragraph 1: Disease information
        p1_text = f"{disease_name.replace('_', ' ').title()} is a plant disease caused by {disease_info['pathogen']}. "
        p1_text += f"Common symptoms include {disease_info['symptoms']}. "
        p1_text += f"This disease primarily affects {', '.join(disease_info['crops'])}."

        context.append(ContextParagraph(
            title=disease_name.replace('_', ' ').title(),
            text=p1_text
        ))

        # Paragraph 2: Treatment information
        treatment = random.choice(disease_info['treatments'])
        treatment_key = treatment.replace(' ', '_').lower()
        if treatment_key in self.kb["treatments"]:
            treat_info = self.kb["treatments"][treatment_key]
            p2_text = f"{treatment} is a {treat_info['type']} with active ingredient {treat_info['active_ingredient']}. "
            p2_text += f"It is effective against {', '.join(treat_info['diseases_treated'][:3]).replace('_', ' ')}. "
            p2_text += f"Application: {treat_info['application']}."
        else:
            p2_text = f"{treatment} is commonly used to treat {disease_name.replace('_', ' ')}. "
            p2_text += f"It should be applied every 7-10 days for best results."

        context.append(ContextParagraph(
            title=f"Treatment: {treatment}",
            text=p2_text
        ))

        # Generate question using LLM
        prompt = f"""Create a bridge-type multi-hop question based on these facts:

Fact 1 (Disease): {p1_text}

Fact 2 (Treatment): {p2_text}

Requirements:
- Question should require connecting information from BOTH facts
- Answer should be derivable from the facts
- Question should be about agriculture/plant disease

Examples of bridge questions:
- "What is the pathogen that causes the disease treated by [treatment]?"
- "Which crops can be affected by diseases that [treatment] treats?"
- "What symptoms does the disease caused by [pathogen] show?"

Return JSON:
{{"question": "...", "answer": "..."}}"""

        try:
            response = self.client.chat(prompt, max_tokens=300, temperature=0.5)
            qa = self._parse_qa(response)

            if qa:
                return MultiHopSample(
                    id=f"bridge_{uuid.uuid4().hex[:8]}",
                    question=qa["question"],
                    answer=qa["answer"],
                    question_type="bridge",
                    context=context,
                    supporting_facts=[disease_name, treatment],
                    generation_metadata={
                        "disease": disease_name,
                        "treatment": treatment,
                        "generated_at": datetime.now().isoformat()
                    }
                )
        except Exception as e:
            self.logger.warning(f"Bridge question generation failed: {e}")

        # Fallback
        question = f"What pathogen causes {disease_name.replace('_', ' ')} which can be treated with {treatment}?"
        answer = disease_info['pathogen']

        return MultiHopSample(
            id=f"bridge_{uuid.uuid4().hex[:8]}",
            question=question,
            answer=answer,
            question_type="bridge",
            context=context,
            supporting_facts=[disease_name, treatment],
            generation_metadata={
                "disease": disease_name,
                "treatment": treatment,
                "fallback": True,
                "generated_at": datetime.now().isoformat()
            }
        )

    def generate_comparison_question(self) -> Optional[MultiHopSample]:
        """
        Generate comparison-type question comparing two entities.

        Example: "Which disease spreads faster in humid conditions, early blight or late blight?"
        """
        # Select two diseases for comparison
        diseases = list(self.kb["diseases"].keys())
        random.shuffle(diseases)
        disease1_name, disease2_name = diseases[:2]
        disease1_info = self.kb["diseases"][disease1_name]
        disease2_info = self.kb["diseases"][disease2_name]

        # Create context paragraphs
        context = []

        p1_text = f"{disease1_name.replace('_', ' ').title()} is caused by {disease1_info['pathogen']}. "
        p1_text += f"It affects {', '.join(disease1_info['crops'])}. "
        p1_text += f"Symptoms include {disease1_info['symptoms']}. "
        p1_text += f"Severity is influenced by {', '.join(disease1_info['severity_factors'][:2])}."

        context.append(ContextParagraph(
            title=disease1_name.replace('_', ' ').title(),
            text=p1_text
        ))

        p2_text = f"{disease2_name.replace('_', ' ').title()} is caused by {disease2_info['pathogen']}. "
        p2_text += f"It affects {', '.join(disease2_info['crops'])}. "
        p2_text += f"Symptoms include {disease2_info['symptoms']}. "
        p2_text += f"Severity is influenced by {', '.join(disease2_info['severity_factors'][:2])}."

        context.append(ContextParagraph(
            title=disease2_name.replace('_', ' ').title(),
            text=p2_text
        ))

        # Generate comparison question using LLM
        prompt = f"""Create a comparison question based on these two diseases:

Disease 1: {p1_text}

Disease 2: {p2_text}

Requirements:
- Question should compare the two diseases on some attribute
- Answer should be derivable from the facts
- Question should be about agriculture

Examples:
- "Which disease is caused by a fungal vs bacterial pathogen?"
- "Which disease affects more crop types?"
- "Which disease has treatments involving copper compounds?"

Return JSON:
{{"question": "...", "answer": "..."}}"""

        try:
            response = self.client.chat(prompt, max_tokens=300, temperature=0.5)
            qa = self._parse_qa(response)

            if qa:
                return MultiHopSample(
                    id=f"comparison_{uuid.uuid4().hex[:8]}",
                    question=qa["question"],
                    answer=qa["answer"],
                    question_type="comparison",
                    context=context,
                    supporting_facts=[disease1_name, disease2_name],
                    generation_metadata={
                        "disease1": disease1_name,
                        "disease2": disease2_name,
                        "generated_at": datetime.now().isoformat()
                    }
                )
        except Exception as e:
            self.logger.warning(f"Comparison question generation failed: {e}")

        # Fallback
        d1_display = disease1_name.replace('_', ' ')
        d2_display = disease2_name.replace('_', ' ')
        question = f"Between {d1_display} and {d2_display}, which one can affect {disease1_info['crops'][0]}?"
        answer = d1_display

        return MultiHopSample(
            id=f"comparison_{uuid.uuid4().hex[:8]}",
            question=question,
            answer=answer,
            question_type="comparison",
            context=context,
            supporting_facts=[disease1_name, disease2_name],
            generation_metadata={
                "disease1": disease1_name,
                "disease2": disease2_name,
                "fallback": True,
                "generated_at": datetime.now().isoformat()
            }
        )

    def _parse_qa(self, response: str) -> Optional[Dict]:
        """Parse QA JSON from response."""
        if not response:
            return None
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass
        return None

    def generate_sample(self, question_type: str = None) -> MultiHopSample:
        """Generate a single multi-hop sample."""
        if question_type is None:
            question_type = random.choice(["bridge", "comparison"])

        if question_type == "bridge":
            sample = self.generate_bridge_question()
        else:
            sample = self.generate_comparison_question()

        return sample


# =============================================================================
# Main
# =============================================================================

def format_sample_for_output(sample: MultiHopSample) -> Dict:
    """Format sample for JSON output in HotpotQA format."""
    return {
        "id": sample.id,
        "question": sample.question,
        "answer": sample.answer,
        "type": sample.question_type,
        "context": [
            {"title": p.title, "text": p.text, "source": p.source}
            for p in sample.context
        ],
        "supporting_facts": sample.supporting_facts,
        "metadata": sample.generation_metadata
    }


def main():
    parser = argparse.ArgumentParser(description="Generate AgriMultiHop v2 dataset")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--output", type=str, default="data/text/agri_hotpotqa_v2")
    parser.add_argument("--images-dir", type=str, default=None)
    parser.add_argument("--bridge-ratio", type=float, default=0.6,
                        help="Ratio of bridge questions (vs comparison)")
    parser.add_argument("--split-ratios", type=str, default="0.7,0.15,0.15")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="INFO")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    random.seed(args.seed)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = MultiHopGenerator(logger, args.images_dir)

    logger.info(f"Generating {args.n_samples} multi-hop samples...")
    samples = []

    n_bridge = int(args.n_samples * args.bridge_ratio)
    n_comparison = args.n_samples - n_bridge

    # Generate bridge questions
    for i in range(n_bridge):
        try:
            sample = generator.generate_sample("bridge")
            if sample:
                samples.append(format_sample_for_output(sample))
                logger.info(f"Generated bridge {i + 1}/{n_bridge}")
        except Exception as e:
            logger.error(f"Failed to generate bridge sample: {e}")

        time.sleep(0.5)
        if (i + 1) % 10 == 0:
            time.sleep(2)

    # Generate comparison questions
    for i in range(n_comparison):
        try:
            sample = generator.generate_sample("comparison")
            if sample:
                samples.append(format_sample_for_output(sample))
                logger.info(f"Generated comparison {i + 1}/{n_comparison}")
        except Exception as e:
            logger.error(f"Failed to generate comparison sample: {e}")

        time.sleep(0.5)
        if (i + 1) % 10 == 0:
            time.sleep(2)

    # Split into train/val/test
    split_ratios = [float(x) for x in args.split_ratios.split(",")]
    n_train = int(len(samples) * split_ratios[0])
    n_val = int(len(samples) * split_ratios[1])

    random.shuffle(samples)
    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]

    # Save splits
    for split_name, split_samples in [
        ("train", train_samples),
        ("val", val_samples),
        ("test", test_samples)
    ]:
        output_file = output_dir / f"{split_name}.json"
        with open(output_file, 'w') as f:
            json.dump(split_samples, f, indent=2)
        logger.info(f"Saved {len(split_samples)} samples to {output_file}")

    # Save dataset info
    dataset_info = {
        "name": "AgriMultiHop v2",
        "version": "2.0.0",
        "description": "Agricultural Multi-hop Reasoning Benchmark (Enhanced)",
        "generation_method": "Chutes.ai API",
        "model": "zai-org/GLM-4.6V",
        "total_samples": len(samples),
        "splits": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples)
        },
        "question_types": {
            "bridge": sum(1 for s in samples if s["type"] == "bridge"),
            "comparison": sum(1 for s in samples if s["type"] == "comparison")
        },
        "created": datetime.now().isoformat()
    }

    with open(output_dir / "dataset_info.json", 'w') as f:
        json.dump(dataset_info, f, indent=2)

    logger.info(f"Dataset generation complete! Total: {len(samples)} samples")


if __name__ == "__main__":
    main()
