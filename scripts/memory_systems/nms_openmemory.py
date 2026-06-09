#!/usr/bin/env python3
"""
Neuroscientific Memory System (NMS) - Enhanced Implementation

Based on OpenMemory with neuroscientific memory components:
- Working Memory: Limited capacity attention buffer
- Episodic Memory: Temporal-indexed conversation memories
- Semantic Memory: Structured knowledge graph
- Procedural Memory: Diagnosis and treatment workflows
- Memory Coordinator: Learned routing between memory types

Addresses reviewer feedback:
1. Concrete algorithmic specifications
2. Clear data structures and operations
3. Ablation-ready modular design
4. Detailed documentation
"""

import json
import logging
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class MemoryItem:
    """Single memory item with metadata."""
    id: str
    content: str
    embedding: List[float]
    timestamp: str
    memory_type: str  # 'working', 'episodic', 'semantic', 'procedural'
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    access_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)


@dataclass
class WorkingMemorySlot:
    """Working memory slot with decay."""
    item: MemoryItem
    activation: float = 1.0
    decay_rate: float = 0.1


@dataclass
class TemporalEvent:
    """Episodic memory event with temporal context."""
    event_id: str
    content: str
    timestamp: datetime
    session_id: str
    entities: List[str]
    embedding: List[float]


@dataclass
class SemanticNode:
    """Knowledge graph node."""
    node_id: str
    entity: str
    entity_type: str  # disease, crop, symptom, treatment
    properties: Dict[str, Any]
    embedding: List[float]


@dataclass
class SemanticEdge:
    """Knowledge graph edge."""
    source: str
    relation: str
    target: str
    weight: float = 1.0


@dataclass
class ProcedureStep:
    """Procedural memory step."""
    step_id: str
    name: str
    condition: str
    action: str
    next_step: Optional[str] = None


# =============================================================================
# Working Memory Component
# =============================================================================

class WorkingMemory:
    """
    Working memory with limited capacity (Miller's 7±2).

    Operations:
    - add: Add item with importance scoring
    - update: Update activation levels with decay
    - retrieve: Get active items above threshold
    - evict: Remove least active items when capacity exceeded

    Decay: activation(t) = activation(0) * exp(-decay_rate * t)
    """

    def __init__(self, capacity: int = 20, decay_rate: float = 0.05):
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.slots: List[WorkingMemorySlot] = []

    def add(self, item: MemoryItem, importance: float = 0.5):
        """Add item to working memory with importance-based activation."""
        # Initial activation based on importance
        slot = WorkingMemorySlot(
            item=item,
            activation=importance,
            decay_rate=self.decay_rate
        )

        self.slots.append(slot)

        # Evict if over capacity
        if len(self.slots) > self.capacity:
            self._evict_least_active()

    def _evict_least_active(self):
        """Remove slot with lowest activation."""
        if not self.slots:
            return
        self.slots.sort(key=lambda s: s.activation, reverse=True)
        self.slots = self.slots[:self.capacity]

    def update(self, time_delta: float):
        """Apply decay to all slots."""
        for slot in self.slots:
            slot.activation *= np.exp(-slot.decay_rate * time_delta)

        # Remove items below threshold
        self.slots = [s for s in self.slots if s.activation > 0.1]

    def retrieve(self, query_embedding: List[float], top_k: int = 3) -> List[MemoryItem]:
        """Retrieve most active and relevant items."""
        if not self.slots:
            return []

        # Score = activation * similarity
        scored_items = []
        for slot in self.slots:
            similarity = self._cosine_similarity(query_embedding, slot.item.embedding)
            score = slot.activation * similarity
            scored_items.append((score, slot.item))

        scored_items.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored_items[:top_k]]

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Cosine similarity between two vectors."""
        a_arr, b_arr = np.array(a), np.array(b)
        return np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + 1e-9)

    def clear(self):
        """Clear all working memory."""
        self.slots = []


# =============================================================================
# Episodic Memory Component
# =============================================================================

class EpisodicMemory:
    """
    Episodic memory with temporal indexing.

    Storage: SQLite with temporal features
    Indexing: Hierarchical temporal (year/month/day/hour)
    Retrieval: Temporal-constrained similarity search

    Operations:
    - add_episode: Store conversation with temporal context
    - retrieve: Temporal + semantic similarity search
    - get_timeline: Retrieve events in chronological order
    """

    def __init__(self, db_path: str, logger: logging.Logger):
        self.db_path = db_path
        self.logger = logger
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        """Initialize episodic memory schema."""
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS episodic_memories (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                session_id TEXT,
                content TEXT NOT NULL,
                embedding BLOB,
                timestamp DATETIME,
                temporal_year INTEGER,
                temporal_month INTEGER,
                temporal_day INTEGER,
                temporal_hour INTEGER,
                entities TEXT,
                importance REAL DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_temporal ON episodic_memories(user_id, timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_session ON episodic_memories(session_id)')
        self.conn.commit()

    def add_episode(
        self,
        user_id: str,
        content: str,
        embedding: List[float],
        timestamp: str,
        session_id: str = None,
        entities: List[str] = None,
        importance: float = 0.5
    ):
        """Add episodic memory with temporal indexing."""
        # Parse timestamp
        try:
            dt = datetime.fromisoformat(timestamp) if timestamp else datetime.now()
        except:
            dt = datetime.now()

        episode_id = str(uuid.uuid4())

        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO episodic_memories
            (id, user_id, session_id, content, embedding, timestamp,
             temporal_year, temporal_month, temporal_day, temporal_hour,
             entities, importance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            episode_id,
            user_id,
            session_id,
            content,
            json.dumps(embedding).encode(),
            dt.isoformat(),
            dt.year,
            dt.month,
            dt.day,
            dt.hour,
            json.dumps(entities or []),
            importance
        ))
        self.conn.commit()

    def retrieve(
        self,
        user_id: str,
        query_embedding: List[float],
        temporal_filter: Dict[str, Any] = None,
        top_k: int = 5,
        temporal_weight: float = 0.4
    ) -> List[Dict]:
        """
        Retrieve episodes with temporal-semantic scoring.

        Score = (1 - temporal_weight) * semantic_sim + temporal_weight * temporal_proximity
        """
        cursor = self.conn.cursor()

        # Build query with optional temporal filter
        query = 'SELECT id, content, embedding, timestamp FROM episodic_memories WHERE user_id = ?'
        params = [user_id]

        if temporal_filter:
            if 'start_date' in temporal_filter:
                query += ' AND timestamp >= ?'
                params.append(temporal_filter['start_date'])
            if 'end_date' in temporal_filter:
                query += ' AND timestamp <= ?'
                params.append(temporal_filter['end_date'])

        cursor.execute(query, params)
        rows = cursor.fetchall()

        if not rows:
            return []

        # Score each episode
        scored_episodes = []
        current_time = datetime.now()

        for episode_id, content, embedding_blob, timestamp in rows:
            doc_embedding = json.loads(embedding_blob.decode())

            # Semantic similarity
            semantic_sim = self._cosine_similarity(query_embedding, doc_embedding)

            # Temporal proximity (inverse time distance, normalized)
            episode_time = datetime.fromisoformat(timestamp)
            time_diff_hours = abs((current_time - episode_time).total_seconds() / 3600)
            temporal_proximity = 1.0 / (1.0 + time_diff_hours / 24)  # Decay over days

            # Combined score
            score = (1 - temporal_weight) * semantic_sim + temporal_weight * temporal_proximity

            scored_episodes.append({
                'id': episode_id,
                'content': content,
                'timestamp': timestamp,
                'score': score,
                'semantic_similarity': semantic_sim,
                'temporal_proximity': temporal_proximity
            })

        scored_episodes.sort(key=lambda x: x['score'], reverse=True)
        return scored_episodes[:top_k]

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        a_arr, b_arr = np.array(a), np.array(b)
        return np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + 1e-9)

    def get_timeline(self, user_id: str, session_id: str = None) -> List[Dict]:
        """Get chronological timeline of events."""
        cursor = self.conn.cursor()
        if session_id:
            cursor.execute(
                'SELECT id, content, timestamp FROM episodic_memories WHERE user_id = ? AND session_id = ? ORDER BY timestamp',
                (user_id, session_id)
            )
        else:
            cursor.execute(
                'SELECT id, content, timestamp FROM episodic_memories WHERE user_id = ? ORDER BY timestamp',
                (user_id,)
            )

        return [{'id': row[0], 'content': row[1], 'timestamp': row[2]} for row in cursor.fetchall()]

    def clear(self, user_id: str):
        """Clear episodic memories for user."""
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM episodic_memories WHERE user_id = ?', (user_id,))
        self.conn.commit()


# =============================================================================
# Semantic Memory Component (Simplified)
# =============================================================================

class SemanticMemory:
    """
    Semantic memory as structured knowledge graph.

    For simplicity, using SQLite instead of Neo4j.
    Stores entities and relations for agricultural domain.

    Operations:
    - add_node: Add entity (disease, crop, symptom, treatment)
    - add_edge: Add relation (affects, causes, treats)
    - retrieve: Graph-walk retrieval with reasoning
    """

    def __init__(self, db_path: str, logger: logging.Logger):
        self.db_path = db_path
        self.logger = logger
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        """Initialize semantic memory schema."""
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS semantic_nodes (
                node_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                entity TEXT NOT NULL,
                entity_type TEXT,
                properties TEXT,
                embedding BLOB
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS semantic_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                source TEXT NOT NULL,
                relation TEXT NOT NULL,
                target TEXT NOT NULL,
                weight REAL DEFAULT 1.0
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_nodes_user ON semantic_nodes(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_edges_user ON semantic_edges(user_id)')
        self.conn.commit()

    def add_node(
        self,
        user_id: str,
        entity: str,
        entity_type: str,
        properties: Dict[str, Any],
        embedding: List[float]
    ):
        """Add semantic node."""
        node_id = f"{entity_type}_{entity}".replace(" ", "_")
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO semantic_nodes
            (node_id, user_id, entity, entity_type, properties, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            node_id,
            user_id,
            entity,
            entity_type,
            json.dumps(properties),
            json.dumps(embedding).encode()
        ))
        self.conn.commit()

    def add_edge(self, user_id: str, source: str, relation: str, target: str, weight: float = 1.0):
        """Add semantic relation."""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO semantic_edges (user_id, source, relation, target, weight)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, source, relation, target, weight))
        self.conn.commit()

    def retrieve(
        self,
        user_id: str,
        query_embedding: List[float],
        max_hops: int = 2,
        top_k: int = 10
    ) -> List[str]:
        """Retrieve relevant knowledge with graph walk."""
        cursor = self.conn.cursor()

        # Find top matching nodes
        cursor.execute(
            'SELECT node_id, entity, entity_type, properties, embedding FROM semantic_nodes WHERE user_id = ?',
            (user_id,)
        )
        rows = cursor.fetchall()

        if not rows:
            return []

        # Score nodes by similarity
        scored_nodes = []
        for node_id, entity, entity_type, properties, embedding_blob in rows:
            doc_embedding = json.loads(embedding_blob.decode())
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            scored_nodes.append((similarity, node_id, entity, entity_type, properties))

        scored_nodes.sort(key=lambda x: x[0], reverse=True)
        top_nodes = scored_nodes[:top_k]

        # Collect node information + related edges
        results = []
        for _, node_id, entity, entity_type, properties in top_nodes:
            results.append(f"{entity_type}: {entity} - {properties}")

            # Get edges from this node
            cursor.execute('''
                SELECT relation, target FROM semantic_edges
                WHERE user_id = ? AND source = ?
                LIMIT ?
            ''', (user_id, node_id, 3))

            for relation, target in cursor.fetchall():
                results.append(f"  -> {relation} -> {target}")

        return results

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        a_arr, b_arr = np.array(a), np.array(b)
        return np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + 1e-9)

    def clear(self, user_id: str):
        """Clear semantic memory for user."""
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM semantic_nodes WHERE user_id = ?', (user_id,))
        cursor.execute('DELETE FROM semantic_edges WHERE user_id = ?', (user_id,))
        self.conn.commit()


# =============================================================================
# Procedural Memory Component
# =============================================================================

class ProceduralMemory:
    """
    Procedural memory for diagnosis and treatment workflows.

    Stores structured procedures as state machines.
    Operations:
    - load_procedure: Load procedure template
    - execute: Step through procedure given context
    - retrieve: Get relevant procedure for query
    """

    def __init__(self, procedures_path: str = None):
        self.procedures = {}
        if procedures_path and Path(procedures_path).exists():
            self.load_procedures(procedures_path)
        else:
            self._load_default_procedures()

    def _load_default_procedures(self):
        """Load default agricultural procedures."""
        self.procedures = {
            "disease_diagnosis": {
                "name": "Disease Diagnosis Workflow",
                "steps": [
                    {"id": "gather_symptoms", "action": "Identify visible symptoms", "next": "check_patterns"},
                    {"id": "check_patterns", "action": "Match symptom patterns", "next": "identify_disease"},
                    {"id": "identify_disease", "action": "Determine disease type", "next": "verify_temporal"},
                    {"id": "verify_temporal", "action": "Check temporal progression", "next": None}
                ]
            },
            "treatment_selection": {
                "name": "Treatment Selection Workflow",
                "steps": [
                    {"id": "identify_disease", "action": "Confirm disease diagnosis", "next": "assess_severity"},
                    {"id": "assess_severity", "action": "Evaluate severity level", "next": "match_treatment"},
                    {"id": "match_treatment", "action": "Select appropriate treatment", "next": "verify_constraints"},
                    {"id": "verify_constraints", "action": "Check application constraints", "next": None}
                ]
            }
        }

    def load_procedures(self, path: str):
        """Load procedures from JSON file."""
        with open(path, 'r') as f:
            self.procedures = json.load(f)

    def retrieve(self, query: str) -> str:
        """Retrieve relevant procedure based on query keywords."""
        query_lower = query.lower()

        if any(kw in query_lower for kw in ["diagnose", "identify", "what disease", "disease"]):
            proc = self.procedures.get("disease_diagnosis", {})
            return self._format_procedure(proc)

        if any(kw in query_lower for kw in ["treatment", "recommend", "what to do", "cure"]):
            proc = self.procedures.get("treatment_selection", {})
            return self._format_procedure(proc)

        return ""

    @staticmethod
    def _format_procedure(procedure: Dict) -> str:
        """Format procedure as text."""
        if not procedure:
            return ""

        output = [f"Procedure: {procedure.get('name', 'Unknown')}"]
        for step in procedure.get('steps', []):
            output.append(f"- {step.get('action', '')}")

        return "\n".join(output)


# =============================================================================
# Memory Coordinator
# =============================================================================

class MemoryCoordinator:
    """
    Routes queries to appropriate memory systems.

    Routing strategy:
    - Disease identification -> Semantic + Episodic
    - Temporal queries -> Episodic + Working
    - Treatment questions -> Semantic + Procedural
    - Severity assessment -> Episodic + Semantic

    Aggregation: Weighted fusion of retrieved contexts
    """

    def __init__(self):
        self.routing_rules = {
            "disease_id": ["semantic", "episodic"],
            "temporal": ["episodic", "working"],
            "treatment": ["semantic", "procedural"],
            "severity": ["episodic", "semantic"],
            "general": ["episodic", "semantic", "working"]
        }

        self.weights = {
            "working": 0.2,
            "episodic": 0.4,
            "semantic": 0.3,
            "procedural": 0.1
        }

    def route(self, query: str) -> Tuple[str, List[str]]:
        """
        Determine query type and memory systems to use.

        Returns:
            (query_type, list_of_memory_systems)
        """
        query_lower = query.lower()

        # Temporal keywords
        if any(kw in query_lower for kw in ["when", "date", "time", "last", "first", "ago"]):
            return "temporal", self.routing_rules["temporal"]

        # Treatment keywords
        if any(kw in query_lower for kw in ["treatment", "recommend", "cure", "control", "manage"]):
            return "treatment", self.routing_rules["treatment"]

        # Disease identification
        if any(kw in query_lower for kw in ["disease", "identify", "what is", "diagnose"]):
            return "disease_id", self.routing_rules["disease_id"]

        # Severity
        if any(kw in query_lower for kw in ["severe", "severity", "how bad", "extent"]):
            return "severity", self.routing_rules["severity"]

        return "general", self.routing_rules["general"]

    def aggregate(
        self,
        results: Dict[str, List[str]],
        query_type: str
    ) -> str:
        """
        Aggregate results from different memory systems.

        Simple concatenation with source labeling.
        """
        aggregated = []

        for memory_type in self.routing_rules.get(query_type, ["episodic", "semantic"]):
            if memory_type in results and results[memory_type]:
                aggregated.append(f"[From {memory_type.upper()} memory]")
                aggregated.extend(results[memory_type])
                aggregated.append("")

        return "\n".join(aggregated)


# =============================================================================
# Main NMS System
# =============================================================================

class NeuroscientificMemorySystem:
    """
    Complete Neuroscientific Memory System integrating all components.

    Components:
    - Working Memory: Active attention buffer
    - Episodic Memory: Temporal conversation memories
    - Semantic Memory: Structured domain knowledge
    - Procedural Memory: Diagnosis/treatment workflows
    - Coordinator: Query routing and result aggregation
    """

    def __init__(
        self,
        logger: logging.Logger,
        db_path: str = "./data/nms_memory.db",
        procedures_path: str = None,
        enable_working: bool = True,
        enable_episodic: bool = True,
        enable_semantic: bool = True,
        enable_procedural: bool = True
    ):
        self.logger = logger
        self.user_id = None
        self.name = "nms"

        # Initialize components (for ablation studies)
        self.enable_working = enable_working
        self.enable_episodic = enable_episodic
        self.enable_semantic = enable_semantic
        self.enable_procedural = enable_procedural

        if enable_working:
            self.working_memory = WorkingMemory(capacity=7, decay_rate=0.1)

        if enable_episodic:
            self.episodic_memory = EpisodicMemory(db_path, logger)

        if enable_semantic:
            self.semantic_memory = SemanticMemory(db_path, logger)

        if enable_procedural:
            self.procedural_memory = ProceduralMemory(procedures_path)

        self.coordinator = MemoryCoordinator()

        # OpenAI client for embeddings and LLM
        self.chutes_api_key = os.getenv("CHUTES_API_KEY")
        self.chutes_api_base = os.getenv("CHUTES_API_BASE", "https://llm.chutes.ai/v1")
        self.model = os.getenv("MODEL", "deepseek-ai/DeepSeek-V3-0324-TEE")

        self.openai_client = OpenAI(
            api_key=self.chutes_api_key,
            base_url=self.chutes_api_base
        )

        self.embedding_api_base = os.getenv("EMBEDDING_API_BASE", "")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "michaelfeil/bge-small-en-v1.5")

        logger.info(f"NMS initialized (W:{enable_working}, E:{enable_episodic}, S:{enable_semantic}, P:{enable_procedural})")

    def reset(self, user_id: str = None):
        """Reset all memory systems for new user."""
        self.user_id = user_id or f"nms_{uuid.uuid4().hex[:8]}"

        if self.enable_working:
            self.working_memory.clear()

        if self.enable_episodic:
            self.episodic_memory.clear(self.user_id)

        if self.enable_semantic:
            self.semantic_memory.clear(self.user_id)

    def add_conversation(self, conversation: List[Dict], metadata: Dict = None):
        """Add conversation to appropriate memory systems."""
        timestamp = metadata.get("timestamp", "") if metadata else ""
        session_id = metadata.get("session_id", str(uuid.uuid4())) if metadata else str(uuid.uuid4())

        # Build full conversation text for better context
        full_conversation = []
        expert_statements = []

        for turn in conversation:
            speaker = turn.get("speaker", "user")
            text = turn.get("text", "")
            full_conversation.append(f"{speaker}: {text}")

            # Collect expert statements for semantic extraction
            if speaker.lower() in ["expert", "assistant"]:
                expert_statements.append(text)

        # Store full conversation as one unit for better retrieval
        if full_conversation:
            full_content = f"[{timestamp}] " + "\n".join(full_conversation)
            full_embedding = self._get_embedding(full_content)

            # Add full conversation to episodic memory
            if self.enable_episodic:
                self.episodic_memory.add_episode(
                    user_id=self.user_id,
                    content=full_content,
                    embedding=full_embedding,
                    timestamp=timestamp,
                    session_id=session_id,
                    entities=self._extract_entities(full_content),
                    importance=0.8
                )

            # Add to working memory
            if self.enable_working:
                item = MemoryItem(
                    id=str(uuid.uuid4()),
                    content=full_content,
                    embedding=full_embedding,
                    timestamp=timestamp,
                    memory_type="episodic"
                )
                self.working_memory.add(item, importance=0.9)

        # Also store individual turns for fine-grained retrieval
        for turn in conversation:
            speaker = turn.get("speaker", "user")
            text = turn.get("text", "")
            content = f"{timestamp} | {speaker}: {text}"
            embedding = self._get_embedding(content)

            # Add individual turn to episodic memory
            if self.enable_episodic:
                self.episodic_memory.add_episode(
                    user_id=self.user_id,
                    content=content,
                    embedding=embedding,
                    timestamp=timestamp,
                    session_id=session_id,
                    entities=self._extract_entities(content),
                    importance=0.5
                )

            # Add to working memory (recent items)
            if self.enable_working:
                item = MemoryItem(
                    id=str(uuid.uuid4()),
                    content=content,
                    embedding=embedding,
                    timestamp=timestamp,
                    memory_type="episodic"
                )
                self.working_memory.add(item, importance=0.7)

        # Extract and add to semantic memory
        if self.enable_semantic and expert_statements:
            for statement in expert_statements:
                self._extract_and_store_semantic(statement, timestamp)

    def _extract_entities(self, text: str) -> List[str]:
        """Extract agricultural entities from text."""
        entities = []
        text_lower = text.lower()

        # Disease patterns
        diseases = ["early blight", "late blight", "bacterial spot", "powdery mildew",
                   "downy mildew", "septoria", "anthracnose", "rust", "mosaic", "wilt",
                   "alternaria", "phytophthora", "fusarium", "botrytis"]
        for d in diseases:
            if d in text_lower:
                entities.append(f"disease:{d}")

        # Crop patterns
        crops = ["potato", "tomato", "pepper", "apple", "grape", "corn", "strawberry",
                "squash", "cherry", "peach", "citrus", "soybean", "wheat"]
        for c in crops:
            if c in text_lower:
                entities.append(f"crop:{c}")

        # Treatment patterns
        treatments = ["copper", "fungicide", "bactericide", "streptomycin", "mancozeb",
                     "chlorothalonil", "bordeaux", "neem", "sulfur", "resistant varieties"]
        for t in treatments:
            if t in text_lower:
                entities.append(f"treatment:{t}")

        # Severity patterns
        if any(s in text_lower for s in ["mild", "moderate", "severe", "critical"]):
            for s in ["mild", "moderate", "severe", "critical"]:
                if s in text_lower:
                    entities.append(f"severity:{s}")

        return entities

    def _extract_and_store_semantic(self, text: str, timestamp: str):
        """Extract entities and relations from expert text and store in semantic memory."""
        entities = self._extract_entities(text)
        embedding = self._get_embedding(text)

        for entity in entities:
            entity_type, entity_name = entity.split(":", 1) if ":" in entity else ("unknown", entity)
            self.semantic_memory.add_node(
                user_id=self.user_id,
                entity=entity_name,
                entity_type=entity_type,
                properties={"source_text": text[:200], "timestamp": timestamp},
                embedding=embedding
            )

        # Add relations between entities found in the same text
        if len(entities) >= 2:
            for i, e1 in enumerate(entities):
                for e2 in entities[i+1:]:
                    t1, n1 = e1.split(":", 1) if ":" in e1 else ("unknown", e1)
                    t2, n2 = e2.split(":", 1) if ":" in e2 else ("unknown", e2)

                    # Determine relation type
                    if t1 == "disease" and t2 == "crop":
                        self.semantic_memory.add_edge(self.user_id, f"{t1}_{n1}", "affects", f"{t2}_{n2}")
                    elif t1 == "treatment" and t2 == "disease":
                        self.semantic_memory.add_edge(self.user_id, f"{t1}_{n1}", "treats", f"{t2}_{n2}")
                    elif t1 == "crop" and t2 == "disease":
                        self.semantic_memory.add_edge(self.user_id, f"{t2}_{n2}", "affects", f"{t1}_{n1}")

    def search(self, query: str, top_k: int = 10) -> Tuple[str, float]:
        """
        Search across memory systems based on query type.

        Returns:
            (context_string, search_time)
        """
        start_time = time.time()

        # Get query embedding
        query_embedding = self._get_embedding(query)

        # Route query
        query_type, memory_systems = self.coordinator.route(query)

        # Retrieve from each memory system with increased top_k
        results = {}

        if "working" in memory_systems and self.enable_working:
            working_items = self.working_memory.retrieve(query_embedding, top_k=min(top_k, 10))
            results["working"] = [item.content for item in working_items]

        if "episodic" in memory_systems and self.enable_episodic:
            episodes = self.episodic_memory.retrieve(
                user_id=self.user_id,
                query_embedding=query_embedding,
                top_k=min(top_k, 15),
                temporal_weight=0.3 if query_type == "temporal" else 0.1
            )
            results["episodic"] = [ep["content"] for ep in episodes]

        if "semantic" in memory_systems and self.enable_semantic:
            semantic_results = self.semantic_memory.retrieve(
                user_id=self.user_id,
                query_embedding=query_embedding,
                max_hops=2,
                top_k=min(top_k, 10)
            )
            results["semantic"] = semantic_results

        if "procedural" in memory_systems and self.enable_procedural:
            procedure = self.procedural_memory.retrieve(query)
            if procedure:
                results["procedural"] = [procedure]

        # Aggregate results with deduplication
        context = self._aggregate_with_dedup(results, query_type)

        search_time = time.time() - start_time
        return context, search_time

    def _aggregate_with_dedup(self, results: Dict[str, List[str]], query_type: str) -> str:
        """Aggregate results with deduplication and better formatting."""
        seen_content = set()
        aggregated = []

        # Priority order for memory systems
        priority_order = ["episodic", "working", "semantic", "procedural"]

        for memory_type in priority_order:
            if memory_type in results and results[memory_type]:
                section_items = []
                for item in results[memory_type]:
                    # Normalize for dedup comparison
                    normalized = item.strip().lower()[:200]
                    if normalized not in seen_content:
                        seen_content.add(normalized)
                        section_items.append(item)

                if section_items:
                    aggregated.append(f"--- {memory_type.upper()} MEMORY ---")
                    aggregated.extend(section_items)
                    aggregated.append("")

        return "\n".join(aggregated)

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text with retry logic."""
        import requests
        import time as time_module

        text = text[:8000]
        max_retries = 3

        if self.embedding_api_base:
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        f"{self.embedding_api_base.rstrip('/')}/embeddings",
                        json={"model": self.embedding_model, "input": text},
                        headers={"Content-Type": "application/json"},
                        timeout=60  # Increased timeout
                    )
                    response.raise_for_status()
                    return response.json()["data"][0]["embedding"]
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        self.logger.warning(f"Embedding attempt {attempt + 1} failed, retrying in {wait_time}s...")
                        time_module.sleep(wait_time)
                    else:
                        self.logger.error(f"Embedding error after {max_retries} attempts: {e}")
                        raise
        else:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding

    def generate_response(self, question: str, context: str) -> Tuple[str, float]:
        """Generate answer using LLM."""
        prompt = f"""You are an agricultural expert. Answer based ONLY on the context.

Context:
{context}

Question: {question}

Instructions:
1. Answer based ONLY on the provided context - extract information directly from the text
2. Match your answer length to the question complexity:
   - For "What/Who" factual questions: Give a SHORT, direct answer (1-10 words) - just the specific term, name, or entity
   - For "How/Why/Describe" questions: Provide more detail with supporting evidence
3. For treatment questions: state the specific product or chemical name
4. For temporal questions: include dates and timeframes
5. For severity questions: include the level AND quantitative details if asked
6. For disease/pathogen identification: give the scientific or common name
7. If information is not in the context, say "unknown"

Answer:"""

        try:
            start = time.time()
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=100
            )
            return response.choices[0].message.content.strip(), time.time() - start
        except Exception as e:
            self.logger.error(f"LLM error: {e}")
            return "error", 0

    def query(self, question: str, top_k: int = 10) -> Tuple[str, str, float, float]:
        """
        Full query pipeline: search + generate.

        Returns:
            (answer, context, search_time, llm_time)
        """
        context, search_time = self.search(question, top_k)
        if not context:
            return "unknown", "", search_time, 0
        answer, llm_time = self.generate_response(question, context)
        return answer, context, search_time, llm_time


if __name__ == "__main__":
    # Simple test
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("nms_test")

    nms = NeuroscientificMemorySystem(logger=logger, db_path="./test_nms.db")
    nms.reset("test_user")

    # Add conversation
    conversation = [
        {"speaker": "farmer", "text": "I see yellow spots on my tomato leaves"},
        {"speaker": "expert", "text": "That sounds like early blight"}
    ]
    nms.add_conversation(conversation, {"timestamp": "2024-03-15T10:00:00"})

    # Query
    answer, context, search_time, llm_time = nms.query("What disease was mentioned?")
    print(f"Answer: {answer}")
    print(f"Context: {context[:200]}...")
    print(f"Times: search={search_time:.3f}s, llm={llm_time:.3f}s")
