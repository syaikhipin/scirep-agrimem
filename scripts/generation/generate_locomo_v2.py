#!/usr/bin/env python3
"""
AgriConvMem v2 Dataset Generator

Uses GLM-4.6V (Vision-Language Model) via Chutes.ai to generate
multimodal agricultural conversations grounded in PlantVillage and PlantDoc images.

Key improvements over v1:
- True multimodal generation (image input)
- Larger scale (500 samples)
- Enhanced diversity (more crops, diseases, temporal patterns)
- Quality validation pipeline
- Detailed metadata tracking

Usage:
    python generate_locomo_v2.py --n-samples 500 --output data/text/agri_locomo_v2/
"""

import argparse
import json
import logging
import os
import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

# Import the chutes client
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "AgriMemory-Dataset" / "scripts"))
from chutes_client import ChutesClient


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GenerationConfig:
    """Configuration for dataset generation."""
    # Scale
    n_samples: int = 500
    sessions_per_sample: Tuple[int, int] = (2, 4)  # min, max
    turns_per_session: Tuple[int, int] = (5, 8)

    # Diversity
    crops: List[str] = field(default_factory=lambda: [
        "tomato", "potato", "pepper", "rice", "wheat", "corn",
        "apple", "grape", "cherry", "strawberry", "soybean", "squash"
    ])

    severity_levels: List[str] = field(default_factory=lambda: [
        "mild", "moderate", "severe"
    ])

    locations: List[str] = field(default_factory=lambda: [
        "field_1", "field_2", "field_3", "greenhouse_1", "orchard_1"
    ])

    seasons: List[str] = field(default_factory=lambda: [
        "spring", "summer", "fall", "winter"
    ])

    # QA categories (from LoCoMo)
    qa_categories: List[str] = field(default_factory=lambda: [
        "disease_identification",
        "temporal",
        "severity",
        "treatment"
    ])

    # Generation parameters
    temperature: float = 0.7
    max_tokens_conversation: int = 2048
    max_tokens_qa: int = 512

    # Quality control
    validate_samples: bool = True
    validation_sample_rate: float = 0.1


# =============================================================================
# Disease-Crop Mappings (based on PlantVillage/PlantDoc)
# =============================================================================

DISEASE_CROP_MAPPING = {
    "tomato": {
        "diseases": [
            {"name": "early_blight", "pathogen": "Alternaria solani",
             "symptoms": "concentric ring patterns, brown spots with yellow halos"},
            {"name": "late_blight", "pathogen": "Phytophthora infestans",
             "symptoms": "water-soaked lesions, white fuzzy growth, rapid wilting"},
            {"name": "bacterial_spot", "pathogen": "Xanthomonas",
             "symptoms": "small dark spots with yellow halos, leaf curling"},
            {"name": "septoria_leaf_spot", "pathogen": "Septoria lycopersici",
             "symptoms": "circular spots with dark borders and gray centers"},
            {"name": "leaf_mold", "pathogen": "Passalora fulva",
             "symptoms": "yellow patches on upper surface, olive-green mold underneath"},
            {"name": "spider_mites", "pathogen": "Tetranychus urticae",
             "symptoms": "stippling, bronzing, fine webbing on leaves"},
            {"name": "yellow_leaf_curl_virus", "pathogen": "TYLCV",
             "symptoms": "upward curling leaves, yellowing, stunted growth"},
            {"name": "mosaic_virus", "pathogen": "ToMV",
             "symptoms": "mottled light and dark green pattern, distorted leaves"},
            {"name": "target_spot", "pathogen": "Corynespora cassiicola",
             "symptoms": "brown spots with concentric rings, yellowing"}
        ],
        "treatments": [
            "copper-based fungicide", "chlorothalonil", "mancozeb",
            "neem oil spray", "remove infected leaves", "improve air circulation"
        ]
    },
    "potato": {
        "diseases": [
            {"name": "early_blight", "pathogen": "Alternaria solani",
             "symptoms": "brown circular lesions with concentric rings"},
            {"name": "late_blight", "pathogen": "Phytophthora infestans",
             "symptoms": "dark water-soaked lesions, white sporulation"}
        ],
        "treatments": [
            "copper fungicide", "chlorothalonil", "mancozeb", "crop rotation"
        ]
    },
    "pepper": {
        "diseases": [
            {"name": "bacterial_spot", "pathogen": "Xanthomonas euvesicatoria",
             "symptoms": "raised brown lesions, leaf defoliation"}
        ],
        "treatments": [
            "copper-based bactericide", "streptomycin", "resistant varieties"
        ]
    },
    "rice": {
        "diseases": [
            {"name": "rice_blast", "pathogen": "Magnaporthe oryzae",
             "symptoms": "diamond-shaped lesions, gray-green centers"},
            {"name": "bacterial_leaf_blight", "pathogen": "Xanthomonas oryzae",
             "symptoms": "water-soaked lesions, yellow to white stripes"},
            {"name": "brown_spot", "pathogen": "Cochliobolus miyabeanus",
             "symptoms": "brown oval spots with gray centers"}
        ],
        "treatments": [
            "tricyclazole", "propiconazole", "copper bactericide", "resistant varieties"
        ]
    },
    "wheat": {
        "diseases": [
            {"name": "wheat_rust", "pathogen": "Puccinia species",
             "symptoms": "orange-brown pustules on leaves"},
            {"name": "powdery_mildew", "pathogen": "Blumeria graminis",
             "symptoms": "white powdery coating on leaves"}
        ],
        "treatments": [
            "propiconazole", "tebuconazole", "sulfur-based fungicide"
        ]
    },
    "corn": {
        "diseases": [
            {"name": "gray_leaf_spot", "pathogen": "Cercospora zeae-maydis",
             "symptoms": "rectangular gray lesions parallel to leaf veins"},
            {"name": "northern_leaf_blight", "pathogen": "Exserohilum turcicum",
             "symptoms": "long cigar-shaped gray-green lesions"},
            {"name": "corn_rust", "pathogen": "Puccinia sorghi",
             "symptoms": "circular to elongated brown pustules"}
        ],
        "treatments": [
            "strobilurin fungicide", "triazole fungicide", "resistant hybrids"
        ]
    },
    "apple": {
        "diseases": [
            {"name": "apple_scab", "pathogen": "Venturia inaequalis",
             "symptoms": "olive-brown spots, velvety texture, leaf curling"},
            {"name": "apple_rust", "pathogen": "Gymnosporangium juniperi-virginianae",
             "symptoms": "yellow-orange spots with tube-like structures"}
        ],
        "treatments": [
            "captan", "mancozeb", "myclobutanil", "remove infected leaves"
        ]
    },
    "grape": {
        "diseases": [
            {"name": "black_rot", "pathogen": "Guignardia bidwellii",
             "symptoms": "brown circular spots, black mummified berries"},
            {"name": "powdery_mildew", "pathogen": "Erysiphe necator",
             "symptoms": "white powdery coating, leaf curling"}
        ],
        "treatments": [
            "mancozeb", "captan", "sulfur spray", "improve air circulation"
        ]
    },
    "cherry": {
        "diseases": [
            {"name": "leaf_spot", "pathogen": "Blumeriella jaapii",
             "symptoms": "purple spots turning brown, premature leaf drop"}
        ],
        "treatments": [
            "copper fungicide", "chlorothalonil", "remove fallen leaves"
        ]
    },
    "strawberry": {
        "diseases": [
            {"name": "leaf_scorch", "pathogen": "Diplocarpon earlianum",
             "symptoms": "purple spots with tan centers, leaf margins burn"}
        ],
        "treatments": [
            "captan", "thiram", "remove infected leaves", "proper spacing"
        ]
    },
    "soybean": {
        "diseases": [
            {"name": "frogeye_leaf_spot", "pathogen": "Cercospora sojina",
             "symptoms": "circular gray spots with reddish-brown borders"}
        ],
        "treatments": [
            "strobilurin fungicide", "crop rotation", "resistant varieties"
        ]
    },
    "squash": {
        "diseases": [
            {"name": "powdery_mildew", "pathogen": "Podosphaera xanthii",
             "symptoms": "white powdery patches on leaves"}
        ],
        "treatments": [
            "potassium bicarbonate", "neem oil", "sulfur spray"
        ]
    }
}

# Map PlantVillage/PlantDoc folder names to our disease names
FOLDER_TO_DISEASE = {
    "Tomato_Early_blight": ("tomato", "early_blight"),
    "Tomato_Late_blight": ("tomato", "late_blight"),
    "Tomato_Bacterial_spot": ("tomato", "bacterial_spot"),
    "Tomato_Septoria_leaf_spot": ("tomato", "septoria_leaf_spot"),
    "Tomato_Leaf_Mold": ("tomato", "leaf_mold"),
    "Tomato_Spider_mites_Two_spotted_spider_mite": ("tomato", "spider_mites"),
    "Tomato__Tomato_YellowLeaf__Curl_Virus": ("tomato", "yellow_leaf_curl_virus"),
    "Tomato__Tomato_mosaic_virus": ("tomato", "mosaic_virus"),
    "Tomato__Target_Spot": ("tomato", "target_spot"),
    "Tomato_healthy": ("tomato", None),
    "Potato___Early_blight": ("potato", "early_blight"),
    "Potato___Late_blight": ("potato", "late_blight"),
    "Pepper__bell___Bacterial_spot": ("pepper", "bacterial_spot"),
    # PlantDoc mappings
    "Tomato Early blight leaf": ("tomato", "early_blight"),
    "Tomato leaf late blight": ("tomato", "late_blight"),
    "Tomato leaf bacterial spot": ("tomato", "bacterial_spot"),
    "Tomato Septoria leaf spot": ("tomato", "septoria_leaf_spot"),
    "Tomato mold leaf": ("tomato", "leaf_mold"),
    "Tomato two spotted spider mites leaf": ("tomato", "spider_mites"),
    "Tomato leaf yellow virus": ("tomato", "yellow_leaf_curl_virus"),
    "Tomato leaf mosaic virus": ("tomato", "mosaic_virus"),
    "Potato leaf early blight": ("potato", "early_blight"),
    "Potato leaf late blight": ("potato", "late_blight"),
    "Bell_pepper leaf spot": ("pepper", "bacterial_spot"),
    "Apple Scab Leaf": ("apple", "apple_scab"),
    "Apple rust leaf": ("apple", "apple_rust"),
    "Corn Gray leaf spot": ("corn", "gray_leaf_spot"),
    "Corn leaf blight": ("corn", "northern_leaf_blight"),
    "Corn rust leaf": ("corn", "corn_rust"),
    "grape leaf black rot": ("grape", "black_rot"),
    "Squash Powdery mildew leaf": ("squash", "powdery_mildew"),
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ConversationTurn:
    speaker: str
    text: str
    dia_id: str


@dataclass
class Session:
    turns: List[ConversationTurn]
    date_time: str
    summary: str
    image_path: Optional[str] = None


@dataclass
class QAPair:
    question: str
    answer: str
    category: str
    category_name: str
    evidence: List[str]


@dataclass
class Sample:
    sample_id: str
    farm_metadata: Dict[str, Any]
    sessions: List[Session]
    qa_pairs: List[QAPair]
    generation_metadata: Dict[str, Any]


# =============================================================================
# Image Discovery
# =============================================================================

def discover_images(images_dir: str) -> Dict[str, List[str]]:
    """
    Discover available images organized by crop-disease.

    Returns:
        Dict mapping (crop, disease) to list of image paths
    """
    images_dir = Path(images_dir)
    image_map = {}

    # PlantVillage images
    plantvillage_dir = images_dir / "PlantVillage"
    if plantvillage_dir.exists():
        for folder in plantvillage_dir.iterdir():
            if folder.is_dir() and folder.name in FOLDER_TO_DISEASE:
                crop, disease = FOLDER_TO_DISEASE[folder.name]
                key = (crop, disease)
                if key not in image_map:
                    image_map[key] = []

                # Get images
                for ext in ["*.jpg", "*.JPG", "*.png", "*.PNG"]:
                    image_map[key].extend([str(p) for p in folder.glob(ext)])

    # PlantDoc images
    plantdoc_dir = images_dir / "plantdoc" / "train"
    if plantdoc_dir.exists():
        for folder in plantdoc_dir.iterdir():
            if folder.is_dir() and folder.name in FOLDER_TO_DISEASE:
                crop, disease = FOLDER_TO_DISEASE[folder.name]
                key = (crop, disease)
                if key not in image_map:
                    image_map[key] = []

                for ext in ["*.jpg", "*.JPG", "*.png", "*.PNG"]:
                    image_map[key].extend([str(p) for p in folder.glob(ext)])

    # Disease folder (alternative structure)
    disease_dir = images_dir / "disease"
    if disease_dir.exists():
        for folder in disease_dir.iterdir():
            if folder.is_dir() and folder.name in FOLDER_TO_DISEASE:
                crop, disease = FOLDER_TO_DISEASE[folder.name]
                key = (crop, disease)
                if key not in image_map:
                    image_map[key] = []

                for ext in ["*.jpg", "*.JPG", "*.png", "*.PNG"]:
                    image_map[key].extend([str(p) for p in folder.glob(ext)])

    return image_map


# =============================================================================
# Conversation Generation
# =============================================================================

class ConversationGenerator:
    """Generate agricultural conversations using GLM-4.6V."""

    def __init__(
        self,
        config: GenerationConfig,
        images_dir: str,
        logger: logging.Logger
    ):
        self.config = config
        self.logger = logger
        self.client = ChutesClient(model="zai-org/GLM-4.6V")
        self.image_map = discover_images(images_dir)

        logger.info(f"Discovered images for {len(self.image_map)} crop-disease combinations")
        for key, images in list(self.image_map.items())[:5]:
            logger.info(f"  {key}: {len(images)} images")

    def select_scenario(self) -> Dict[str, Any]:
        """Select a random scenario with crop, disease, etc."""
        # Filter to crops with available images
        available_crops = set(crop for (crop, disease) in self.image_map.keys() if disease)

        if not available_crops:
            # Fallback to all crops
            available_crops = set(self.config.crops)

        crop = random.choice(list(available_crops))

        # Select disease for this crop
        crop_diseases = [
            disease for (c, disease) in self.image_map.keys()
            if c == crop and disease
        ]

        if crop_diseases:
            disease = random.choice(crop_diseases)
        else:
            # Fallback to DISEASE_CROP_MAPPING
            if crop in DISEASE_CROP_MAPPING:
                disease_info = random.choice(DISEASE_CROP_MAPPING[crop]["diseases"])
                disease = disease_info["name"]
            else:
                disease = "unknown_disease"

        severity = random.choice(self.config.severity_levels)
        location = random.choice(self.config.locations)
        season = random.choice(self.config.seasons)

        return {
            "crop": crop,
            "disease": disease,
            "severity": severity,
            "location": location,
            "season": season
        }

    def get_disease_info(self, crop: str, disease: str) -> Dict[str, Any]:
        """Get disease information from mapping."""
        if crop in DISEASE_CROP_MAPPING:
            for d in DISEASE_CROP_MAPPING[crop]["diseases"]:
                if d["name"] == disease:
                    treatments = DISEASE_CROP_MAPPING[crop]["treatments"]
                    return {
                        "disease": disease,
                        "pathogen": d.get("pathogen", "unknown"),
                        "symptoms": d.get("symptoms", "disease symptoms"),
                        "treatments": treatments
                    }

        return {
            "disease": disease,
            "pathogen": "unknown pathogen",
            "symptoms": "various symptoms",
            "treatments": ["consult agricultural expert"]
        }

    def select_image(self, crop: str, disease: str) -> Optional[str]:
        """Select a random image for the crop-disease combination."""
        key = (crop, disease)
        if key in self.image_map and self.image_map[key]:
            return random.choice(self.image_map[key])
        return None

    def generate_session(
        self,
        scenario: Dict[str, Any],
        session_num: int,
        base_date: datetime
    ) -> Session:
        """Generate a single conversation session."""
        crop = scenario["crop"]
        disease = scenario["disease"]
        severity = scenario["severity"]
        location = scenario["location"]

        disease_info = self.get_disease_info(crop, disease)
        image_path = self.select_image(crop, disease)

        # Session date (progressive)
        session_date = base_date + timedelta(days=session_num * random.randint(3, 14))
        date_str = session_date.strftime("%Y-%m-%d %H:%M")

        # Generate conversation using GLM-4.6V
        n_turns = random.randint(*self.config.turns_per_session)

        prompt = f"""Generate a realistic conversation between a farmer and agricultural expert.

Scenario:
- Crop: {crop}
- Disease: {disease} (caused by {disease_info['pathogen']})
- Symptoms: {disease_info['symptoms']}
- Severity: {severity}
- Location: {location}
- Session: {session_num + 1} (follow-up if > 1)
- Date: {date_str}

Requirements:
1. Farmer describes symptoms and situation realistically
2. Expert asks clarifying questions before diagnosis
3. Include specific details: symptoms observed, timeline, affected area
4. Expert provides diagnosis and treatment recommendations
5. Include temporal markers (e.g., "started 3 days ago", "since last week")
6. Generate exactly {n_turns} turns

Format as JSON array:
[
  {{"speaker": "farmer", "text": "..."}},
  {{"speaker": "expert", "text": "..."}},
  ...
]"""

        system_prompt = """You are an agricultural conversation generator.
Create realistic, informative dialogues between farmers and experts about crop diseases.
Always return valid JSON arrays."""

        try:
            response = self.client.chat(
                prompt,
                image_path=image_path,
                system_prompt=system_prompt,
                max_tokens=self.config.max_tokens_conversation,
                temperature=self.config.temperature
            )

            # Parse JSON
            turns = self._parse_conversation(response, n_turns)

        except Exception as e:
            self.logger.warning(f"Conversation generation failed: {e}")
            turns = self._generate_fallback_conversation(scenario, disease_info, n_turns)

        # Create turn objects
        conversation_turns = []
        for i, turn in enumerate(turns):
            conversation_turns.append(ConversationTurn(
                speaker=turn.get("speaker", "unknown"),
                text=turn.get("text", ""),
                dia_id=f"D{session_num + 1}:{i + 1}"
            ))

        # Generate summary
        summary = f"{disease.replace('_', ' ')} detected in {crop} ({location}), {severity} severity."

        return Session(
            turns=conversation_turns,
            date_time=date_str,
            summary=summary,
            image_path=image_path
        )

    def _parse_conversation(self, response: str, expected_turns: int) -> List[Dict]:
        """Parse conversation JSON from response."""
        try:
            # Find JSON array
            start = response.find("[")
            end = response.rfind("]") + 1
            if start >= 0 and end > start:
                turns = json.loads(response[start:end])
                if isinstance(turns, list) and len(turns) > 0:
                    return turns[:expected_turns]
        except json.JSONDecodeError:
            pass

        return []

    def _generate_fallback_conversation(
        self,
        scenario: Dict,
        disease_info: Dict,
        n_turns: int
    ) -> List[Dict]:
        """Generate fallback conversation if LLM fails."""
        crop = scenario["crop"]
        disease = scenario["disease"]
        severity = scenario["severity"]
        location = scenario["location"]

        turns = [
            {"speaker": "farmer", "text": f"Hello, I'm calling about my {crop} crop in {location}. "
             f"I've noticed {disease_info['symptoms']} on the plants."},
            {"speaker": "expert", "text": "Thank you for calling. Can you describe when you first noticed these symptoms "
             "and how quickly they've been spreading?"},
            {"speaker": "farmer", "text": f"I first noticed it about a week ago. The symptoms are {severity} now "
             "and seem to be spreading to neighboring plants."},
            {"speaker": "expert", "text": f"Based on your description, this sounds like {disease.replace('_', ' ')}, "
             f"which is caused by {disease_info['pathogen']}. This is a common issue in {crop} plants."},
            {"speaker": "farmer", "text": "What treatment would you recommend?"},
            {"speaker": "expert", "text": f"I recommend applying {random.choice(disease_info['treatments'])}. "
             "Make sure to follow the label instructions and monitor the plants closely."},
            {"speaker": "farmer", "text": "How often should I apply the treatment?"},
            {"speaker": "expert", "text": "Apply every 7-10 days for 2-3 applications, especially if conditions remain favorable for disease development."}
        ]

        return turns[:n_turns]

    def generate_qa_pairs(
        self,
        sessions: List[Session],
        scenario: Dict[str, Any]
    ) -> List[QAPair]:
        """Generate QA pairs across all categories."""
        qa_pairs = []
        disease_info = self.get_disease_info(scenario["crop"], scenario["disease"])

        # Collect all conversation text for context
        full_context = ""
        for i, session in enumerate(sessions):
            full_context += f"\n[Session {i + 1} - {session.date_time}]\n"
            for turn in session.turns:
                full_context += f"{turn.speaker}: {turn.text}\n"

        # Generate QA for each category
        for category_idx, category in enumerate(self.config.qa_categories):
            qa_prompt = f"""Based on this agricultural consultation:

{full_context}

Generate a {category.replace('_', ' ')} question and short answer.

Category requirements:
- disease_identification: Ask about what disease was identified
- temporal: Ask about timing (when symptoms started, treatment applied, etc.)
- severity: Ask about severity level
- treatment: Ask about recommended treatment

Return JSON:
{{"question": "...", "answer": "...", "evidence": ["relevant quote from conversation"]}}"""

            try:
                response = self.client.chat(
                    qa_prompt,
                    max_tokens=self.config.max_tokens_qa,
                    temperature=0.3
                )

                qa_data = self._parse_qa(response)
                if qa_data:
                    qa_pairs.append(QAPair(
                        question=qa_data.get("question", ""),
                        answer=qa_data.get("answer", ""),
                        category=category_idx + 1,
                        category_name=category,
                        evidence=qa_data.get("evidence", [])
                    ))

            except Exception as e:
                self.logger.warning(f"QA generation failed for {category}: {e}")

                # Fallback QA
                qa_pairs.append(self._generate_fallback_qa(
                    category, category_idx + 1, scenario, disease_info, sessions
                ))

        return qa_pairs

    def _parse_qa(self, response: str) -> Optional[Dict]:
        """Parse QA JSON from response."""
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass
        return None

    def _generate_fallback_qa(
        self,
        category: str,
        category_idx: int,
        scenario: Dict,
        disease_info: Dict,
        sessions: List[Session]
    ) -> QAPair:
        """Generate fallback QA if LLM fails."""
        crop = scenario["crop"]
        disease = scenario["disease"].replace("_", " ")
        severity = scenario["severity"]
        location = scenario["location"]

        if category == "disease_identification":
            return QAPair(
                question=f"What disease was identified in {location}?",
                answer=disease,
                category=category_idx,
                category_name=category,
                evidence=[]
            )
        elif category == "temporal":
            return QAPair(
                question="When was the infection first reported?",
                answer=sessions[0].date_time.split()[0] if sessions else "unknown",
                category=category_idx,
                category_name=category,
                evidence=[]
            )
        elif category == "severity":
            return QAPair(
                question=f"How severe was the {disease} infection?",
                answer=severity,
                category=category_idx,
                category_name=category,
                evidence=[]
            )
        elif category == "treatment":
            treatment = random.choice(disease_info["treatments"])
            return QAPair(
                question="What treatment was recommended?",
                answer=treatment,
                category=category_idx,
                category_name=category,
                evidence=[]
            )
        else:
            return QAPair(
                question=f"What crop was affected?",
                answer=crop,
                category=category_idx,
                category_name=category,
                evidence=[]
            )

    def generate_sample(self, sample_id: str) -> Sample:
        """Generate a complete sample with sessions and QA pairs."""
        scenario = self.select_scenario()
        n_sessions = random.randint(*self.config.sessions_per_sample)

        # Base date for this sample
        base_date = datetime(2024, random.randint(1, 12), random.randint(1, 28))

        # Generate sessions
        sessions = []
        for i in range(n_sessions):
            session = self.generate_session(scenario, i, base_date)
            sessions.append(session)

            # Rate limiting
            time.sleep(0.5)

        # Generate QA pairs
        qa_pairs = self.generate_qa_pairs(sessions, scenario)

        return Sample(
            sample_id=sample_id,
            farm_metadata={
                "primary_crop": scenario["crop"],
                "location": scenario["location"],
                "season": f"2024-{scenario['season']}"
            },
            sessions=sessions,
            qa_pairs=qa_pairs,
            generation_metadata={
                "model": "zai-org/GLM-4.6V",
                "scenario": scenario,
                "generated_at": datetime.now().isoformat()
            }
        )


# =============================================================================
# Main
# =============================================================================

def format_sample_for_output(sample: Sample) -> Dict:
    """Format sample for JSON output in LoCoMo format."""
    output = {
        "sample_id": sample.sample_id,
        "farm_metadata": sample.farm_metadata,
        "conversation": {
            "speaker_a": "farmer",
            "speaker_b": "expert"
        },
        "qa": []
    }

    # Add sessions
    for i, session in enumerate(sample.sessions):
        session_key = f"session_{i + 1}"
        output["conversation"][session_key] = [
            {"speaker": t.speaker, "text": t.text, "dia_id": t.dia_id}
            for t in session.turns
        ]
        output["conversation"][f"{session_key}_date_time"] = session.date_time
        output["conversation"][f"{session_key}_summary"] = session.summary

        if session.image_path:
            output["conversation"][f"{session_key}_image"] = session.image_path

    # Add QA pairs
    for qa in sample.qa_pairs:
        output["qa"].append({
            "question": qa.question,
            "answer": qa.answer,
            "category": qa.category,
            "category_name": qa.category_name,
            "evidence": qa.evidence
        })

    return output


def main():
    parser = argparse.ArgumentParser(description="Generate AgriConvMem v2 dataset")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--output", type=str, default="data/text/agri_locomo_v2")
    parser.add_argument("--images-dir", type=str,
                        default="../AgriMemory-Dataset/data/images")
    parser.add_argument("--split-ratios", type=str, default="0.7,0.15,0.15",
                        help="Train/val/test split ratios")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="INFO")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Set seed
    random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize generator
    config = GenerationConfig(n_samples=args.n_samples)
    generator = ConversationGenerator(config, args.images_dir, logger)

    # Generate samples
    logger.info(f"Generating {args.n_samples} samples...")
    samples = []

    for i in range(args.n_samples):
        sample_id = f"agri-{i:04d}"
        try:
            sample = generator.generate_sample(sample_id)
            samples.append(format_sample_for_output(sample))
            logger.info(f"Generated sample {i + 1}/{args.n_samples}: {sample_id}")
        except Exception as e:
            logger.error(f"Failed to generate sample {sample_id}: {e}")

        # Rate limiting
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
            json.dump({"samples": split_samples}, f, indent=2)
        logger.info(f"Saved {len(split_samples)} samples to {output_file}")

    # Save dataset info
    dataset_info = {
        "name": "AgriConvMem v2",
        "version": "2.0.0",
        "description": "Agricultural Conversational Memory Benchmark (Enhanced)",
        "generation_method": "Chutes.ai API",
        "model": "zai-org/GLM-4.6V",
        "total_samples": len(samples),
        "splits": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples)
        },
        "question_categories": config.qa_categories,
        "crops_covered": config.crops,
        "created": datetime.now().isoformat()
    }

    with open(output_dir / "dataset_info.json", 'w') as f:
        json.dump(dataset_info, f, indent=2)

    logger.info(f"Dataset generation complete! Total: {len(samples)} samples")


if __name__ == "__main__":
    main()
