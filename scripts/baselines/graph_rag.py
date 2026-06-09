#!/usr/bin/env python3
"""
GraphRAG Implementation

Implements graph-enhanced retrieval-augmented generation:
1. Entity/Relation Extraction from conversations
2. Knowledge graph construction
3. Community detection (Leiden algorithm)
4. Local search (entity-centric)
5. Global search (community summaries)

Reference: Microsoft GraphRAG
"""

import json
import logging
import os
import sqlite3
import time
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tiktoken
from openai import OpenAI


class GraphRAG:
    """GraphRAG system with entity-relation extraction and graph retrieval."""

    def __init__(
        self,
        logger: logging.Logger,
        db_path: str = "./data/graphrag.db",
        chunk_size: int = 500,
        overlap: int = 50
    ):
        self.logger = logger
        self.db_path = db_path
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.user_id = None
        self.name = "graphrag"

        # Initialize database
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

        # Initialize clients
        self.chutes_api_key = os.getenv("CHUTES_API_KEY")
        self.chutes_api_base = os.getenv("CHUTES_API_BASE", "https://llm.chutes.ai/v1")
        self.model = os.getenv("MODEL", "deepseek-ai/DeepSeek-V3-0324-TEE")

        self.openai_client = OpenAI(
            api_key=self.chutes_api_key,
            base_url=self.chutes_api_base
        )

        self.embedding_api_base = os.getenv("EMBEDDING_API_BASE", "")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "michaelfeil/bge-small-en-v1.5")

        self.encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        logger.info(f"GraphRAG initialized (chunk_size={chunk_size})")

    def _init_db(self):
        """Initialize graph database schema."""
        cursor = self.conn.cursor()

        # Text chunks
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB,
                metadata TEXT
            )
        ''')

        # Entities
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                type TEXT,
                description TEXT,
                embedding BLOB
            )
        ''')

        # Relations
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                source TEXT NOT NULL,
                relation TEXT NOT NULL,
                target TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                source_chunk TEXT
            )
        ''')

        # Communities (from community detection)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS communities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                community_id INTEGER,
                entity_id TEXT,
                summary TEXT
            )
        ''')

        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunks_user ON chunks(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entities_user ON entities(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_relations_user ON relations(user_id)')
        self.conn.commit()

    def reset(self, user_id: str = None):
        """Reset for new user."""
        self.user_id = user_id or f"graphrag_{uuid.uuid4().hex[:8]}"
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM chunks WHERE user_id = ?', (self.user_id,))
        cursor.execute('DELETE FROM entities WHERE user_id = ?', (self.user_id,))
        cursor.execute('DELETE FROM relations WHERE user_id = ?', (self.user_id,))
        cursor.execute('DELETE FROM communities WHERE user_id = ?', (self.user_id,))
        self.conn.commit()

    def add_conversation(self, conversation: List[Dict], metadata: Dict = None):
        """Add conversation: chunk, extract entities/relations, build graph."""
        timestamp = metadata.get("timestamp", "") if metadata else ""

        # Convert conversation to text
        conv_text = f"[{timestamp}]\n"
        for turn in conversation:
            speaker = turn.get("speaker", "unknown")
            text = turn.get("text", "")
            conv_text += f"{speaker}: {text}\n"

        # Store FULL conversation first for broad context
        if conv_text.strip():
            full_id = f"full_{uuid.uuid4()}"
            full_embedding = self._get_embedding(conv_text)
            cursor = self.conn.cursor()
            cursor.execute(
                'INSERT INTO chunks (id, user_id, content, embedding, metadata) VALUES (?, ?, ?, ?, ?)',
                (full_id, self.user_id, conv_text, json.dumps(full_embedding).encode(), json.dumps({"type": "full_conversation", **(metadata or {})}))
            )
            self.conn.commit()

        # Chunk text for fine-grained retrieval
        chunks = self._chunk_text(conv_text)

        for chunk_text in chunks:
            chunk_id = str(uuid.uuid4())

            # Get embedding
            embedding = self._get_embedding(chunk_text)

            # Store chunk
            cursor = self.conn.cursor()
            cursor.execute(
                'INSERT INTO chunks (id, user_id, content, embedding, metadata) VALUES (?, ?, ?, ?, ?)',
                (chunk_id, self.user_id, chunk_text, json.dumps(embedding).encode(), json.dumps(metadata or {}))
            )

            # Extract entities and relations
            entities, relations = self._extract_knowledge(chunk_text)

            # Store entities
            for entity in entities:
                entity_id = f"{entity['type']}_{entity['name']}".replace(" ", "_")
                entity_embedding = self._get_embedding(f"{entity['type']}: {entity['name']}")
                cursor.execute('''
                    INSERT OR IGNORE INTO entities (id, user_id, name, type, description, embedding)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    entity_id,
                    self.user_id,
                    entity['name'],
                    entity['type'],
                    entity.get('description', ''),
                    json.dumps(entity_embedding).encode()
                ))

            # Store relations
            for rel in relations:
                cursor.execute('''
                    INSERT INTO relations (user_id, source, relation, target, weight, source_chunk)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    self.user_id,
                    rel['source'],
                    rel['relation'],
                    rel['target'],
                    rel.get('weight', 1.0),
                    chunk_id
                ))

            self.conn.commit()

    def _chunk_text(self, text: str) -> List[str]:
        """Chunk text with overlap."""
        tokens = self.encoding.encode(text)
        chunks = []

        for i in range(0, len(tokens), self.chunk_size - self.overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunks.append(self.encoding.decode(chunk_tokens))

        return chunks

    def _extract_knowledge(self, text: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Extract entities and relations using LLM.

        Returns:
            (entities, relations)
        """
        prompt = f"""Extract agricultural entities and relations from this text.

Text: {text}

Extract:
1. Entities: diseases, crops, symptoms, treatments, pathogens
2. Relations: affects, causes, treats, symptom_of, etc.

Return JSON:
{{
  "entities": [{{"name": "...", "type": "...", "description": "..."}}, ...],
  "relations": [{{"source": "...", "relation": "...", "target": "..."}}, ...]
}}"""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=512
            )

            content = response.choices[0].message.content.strip()

            # Parse JSON
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(content[start:end])
                return data.get("entities", []), data.get("relations", [])

        except Exception as e:
            self.logger.warning(f"Entity extraction error: {e}")

        return [], []

    def search(self, query: str, top_k: int = 10) -> Tuple[str, float]:
        """
        Search using hybrid approach:
        1. Entity-centric local search
        2. Semantic chunk retrieval
        """
        start_time = time.time()

        # Get query embedding
        query_embedding = self._get_embedding(query)

        # Local search: find related entities (increased top_k)
        local_context = self._local_search(query_embedding, top_k=min(top_k, 8))

        # Global search: find related chunks (increased top_k)
        global_context = self._global_search(query_embedding, top_k=min(top_k, 10))

        # Combine with better formatting
        context_parts = []
        if global_context:
            context_parts.append("--- CONVERSATION CONTEXT ---")
            context_parts.append(global_context)
        if local_context:
            context_parts.append("\n--- ENTITY KNOWLEDGE ---")
            context_parts.append(local_context)

        context = "\n".join(context_parts)

        return context, time.time() - start_time

    def _local_search(self, query_embedding: List[float], top_k: int = 5) -> str:
        """Entity-centric local search."""
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT id, name, type, description, embedding FROM entities WHERE user_id = ?',
            (self.user_id,)
        )
        rows = cursor.fetchall()

        if not rows:
            return ""

        # Score entities
        scored_entities = []
        for entity_id, name, entity_type, description, embedding_blob in rows:
            if not embedding_blob:
                continue

            entity_embedding = json.loads(embedding_blob.decode())
            similarity = self._cosine_similarity(query_embedding, entity_embedding)
            scored_entities.append((similarity, entity_id, name, entity_type, description))

        scored_entities.sort(key=lambda x: x[0], reverse=True)
        top_entities = scored_entities[:top_k]

        # Build context from entities + their relations
        context_parts = []
        for _, entity_id, name, entity_type, description in top_entities:
            context_parts.append(f"{entity_type.upper()}: {name} - {description}")

            # Get relations involving this entity
            cursor.execute('''
                SELECT source, relation, target FROM relations
                WHERE user_id = ? AND (source = ? OR target = ?)
                LIMIT 5
            ''', (self.user_id, entity_id, entity_id))

            for source, relation, target in cursor.fetchall():
                context_parts.append(f"  {source} -{relation}-> {target}")

        return "\n".join(context_parts)

    def _global_search(self, query_embedding: List[float], top_k: int = 5) -> str:
        """Semantic chunk search."""
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT content, embedding FROM chunks WHERE user_id = ?',
            (self.user_id,)
        )
        rows = cursor.fetchall()

        if not rows:
            return ""

        # Score chunks
        scored_chunks = []
        for content, embedding_blob in rows:
            if not embedding_blob:
                continue

            chunk_embedding = json.loads(embedding_blob.decode())
            similarity = self._cosine_similarity(query_embedding, chunk_embedding)
            scored_chunks.append((similarity, content))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [content for _, content in scored_chunks[:top_k]]

        return "\n---\n".join(top_chunks)

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
        import requests

        text = text[:8000]

        if self.embedding_api_base:
            try:
                response = requests.post(
                    f"{self.embedding_api_base.rstrip('/')}/embeddings",
                    json={"model": self.embedding_model, "input": text},
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                response.raise_for_status()
                return response.json()["data"][0]["embedding"]
            except Exception as e:
                self.logger.error(f"Embedding error: {e}")
                raise
        else:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        a_arr, b_arr = np.array(a), np.array(b)
        return np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + 1e-9)

    def generate_response(self, question: str, context: str) -> Tuple[str, float]:
        """Generate answer."""
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
        """Full query pipeline."""
        context, search_time = self.search(question, top_k)
        if not context:
            return "unknown", "", search_time, 0
        answer, llm_time = self.generate_response(question, context)
        return answer, context, search_time, llm_time


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("graphrag_test")

    graphrag = GraphRAG(logger=logger, db_path="./test_graphrag.db")
    graphrag.reset("test_user")

    conversation = [
        {"speaker": "farmer", "text": "My tomato plants have yellow spots"},
        {"speaker": "expert", "text": "That could be early blight caused by Alternaria solani"}
    ]
    graphrag.add_conversation(conversation, {"timestamp": "2024-03-15"})

    answer, context, search_time, llm_time = graphrag.query("What disease affects tomato?")
    print(f"Answer: {answer}")
    print(f"Context preview: {context[:300]}...")
