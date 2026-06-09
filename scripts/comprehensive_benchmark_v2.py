#!/usr/bin/env python3
"""
Comprehensive Benchmark v2 for AgriMemory Dataset

Integrates all memory systems:
- NMS (Neuroscientific Memory System with 4 components)
- GraphRAG (Entity-relation graph retrieval)
- Standard RAG
- Full-context baseline

Features:
- Statistical significance testing
- Multiple runs with different seeds
- Cross-model evaluation
- Temporal reasoning evaluation
- Multimodal inference support

Usage:
    python comprehensive_benchmark_v2.py \
        --dataset both \
        --techniques nms,graphrag,rag,baseline \
        --n-runs 3 \
        --output results/benchmark_v2.json
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from tqdm import tqdm

load_dotenv()

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "memory_systems"))
sys.path.insert(0, str(Path(__file__).parent / "baselines"))
sys.path.insert(0, str(Path(__file__).parent / "evaluation"))

# Import our modules
try:
    from memory_systems.nms_openmemory import NeuroscientificMemorySystem
    from baselines.graph_rag import GraphRAG
    from evaluation.statistical_tests import StatisticalTester, SignificanceResult
except ImportError as e:
    print(f"Import warning: {e}")
    print("Some modules may not be available. Continuing with available modules...")


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(log_dir: str = "logs", experiment_name: str = "benchmark_v2") -> logging.Logger:
    """Setup logging with file and console handlers."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")

    logger = logging.getLogger("benchmark_v2")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    ))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.info(f"Logging to: {log_file}")

    return logger


# =============================================================================
# Metrics
# =============================================================================

def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison."""
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return ' '.join(text.split())


def calculate_f1(prediction: str, reference: str) -> float:
    """Calculate token-level F1 score."""
    pred_tokens = set(normalize_answer(prediction).split())
    ref_tokens = set(normalize_answer(reference).split())

    if not pred_tokens or not ref_tokens:
        return 0.0

    common = pred_tokens & ref_tokens
    precision = len(common) / len(pred_tokens) if pred_tokens else 0
    recall = len(common) / len(ref_tokens) if ref_tokens else 0

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def calculate_exact_match(prediction: str, reference: str) -> float:
    """Calculate exact match score."""
    return float(normalize_answer(prediction) == normalize_answer(reference))


def calculate_bleu1(prediction: str, reference: str) -> float:
    """Calculate BLEU-1 score."""
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = set(normalize_answer(reference).split())

    if not pred_tokens:
        return 0.0

    matches = sum(1 for t in pred_tokens if t in ref_tokens)
    return matches / len(pred_tokens)


def get_embedding_for_text(text: str) -> List[float]:
    """Get embedding for text using the embedding API."""
    import requests

    embedding_api_base = os.getenv("EMBEDDING_API_BASE", "")
    embedding_model = os.getenv("EMBEDDING_MODEL", "michaelfeil/bge-small-en-v1.5")

    text = text[:8000]

    if embedding_api_base:
        try:
            response = requests.post(
                f"{embedding_api_base.rstrip('/')}/embeddings",
                json={"model": embedding_model, "input": text},
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
        except Exception as e:
            return None
    return None


def calculate_semantic_similarity(prediction: str, reference: str) -> float:
    """Calculate semantic similarity using embeddings (cosine similarity)."""
    pred_emb = get_embedding_for_text(prediction)
    ref_emb = get_embedding_for_text(reference)

    if pred_emb is None or ref_emb is None:
        return 0.0

    # Cosine similarity
    pred_arr = np.array(pred_emb)
    ref_arr = np.array(ref_emb)

    dot_product = np.dot(pred_arr, ref_arr)
    norm_pred = np.linalg.norm(pred_arr)
    norm_ref = np.linalg.norm(ref_arr)

    if norm_pred == 0 or norm_ref == 0:
        return 0.0

    return float(dot_product / (norm_pred * norm_ref))


# =============================================================================
# Prompts
# =============================================================================

ANSWER_PROMPT = """You are an agricultural expert assistant with access to conversation memories.

# Context (Retrieved Memories):
{context}

# Question:
{question}

# Instructions:
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

LLM_JUDGE_PROMPT = """Evaluate if the generated answer is correct compared to the gold answer.

Question: {question}
Gold Answer: {gold_answer}
Generated Answer: {response}

Guidelines:
- Be generous: if the answer captures the key information, mark as CORRECT
- Partial matches that contain the essential facts are CORRECT
- For disease names, treatments, or technical terms, allow synonyms and variations
- For dates/times, flexible matching is OK if contextually correct
- Focus on whether the core information is present, not exact wording

Return ONLY a JSON object: {{"label": "CORRECT"}} or {{"label": "WRONG"}}"""


# =============================================================================
# Base Memory System
# =============================================================================

class BaseMemorySystem(ABC):
    """Base class for all memory systems."""

    def __init__(self, logger: logging.Logger, name: str):
        self.logger = logger
        self.name = name

        # LLM Configuration
        self.chutes_api_key = os.getenv("CHUTES_API_KEY")
        self.chutes_api_base = os.getenv("CHUTES_API_BASE", "https://llm.chutes.ai/v1")
        self.model = os.getenv("MODEL", "deepseek-ai/DeepSeek-V3-0324-TEE")

        if self.chutes_api_key:
            self.openai_client = OpenAI(
                api_key=self.chutes_api_key,
                base_url=self.chutes_api_base
            )
        else:
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Embedding Configuration
        self.embedding_api_base = os.getenv("EMBEDDING_API_BASE", "")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "michaelfeil/bge-small-en-v1.5")

    @abstractmethod
    def reset(self, user_id: str = None):
        """Reset memory for new user/sample."""
        pass

    @abstractmethod
    def add_conversation(self, conversation: List[Dict], metadata: Dict = None):
        """Add conversation to memory."""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> Tuple[str, float]:
        """Search memory and return context."""
        pass

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for text."""
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

    def generate_response(self, question: str, context: str) -> Tuple[str, float]:
        """Generate answer using LLM."""
        prompt = ANSWER_PROMPT.format(context=context, question=question)

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
        """Full query pipeline: search + generate."""
        context, search_time = self.search(question, top_k)
        if not context:
            return "unknown", "", search_time, 0
        answer, llm_time = self.generate_response(question, context)
        return answer, context, search_time, llm_time


# =============================================================================
# RAG System (Enhanced)
# =============================================================================

class RAGSystem(BaseMemorySystem):
    """Enhanced RAG with overlapping chunks, full conversation storage, and better retrieval."""

    def __init__(self, logger: logging.Logger, chunk_size: int = 300, chunk_overlap: int = 100, top_k: int = 10):
        super().__init__(logger, "rag")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.default_top_k = top_k
        self.chunks = []
        self.embeddings = []
        self.full_conversations = []  # Store full conversations for better context
        self.full_embeddings = []

        try:
            import tiktoken
            self.encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        except:
            self.encoding = None

        logger.info(f"RAG system initialized (chunk_size={chunk_size}, overlap={chunk_overlap})")

    def reset(self, user_id: str = None):
        self.chunks = []
        self.embeddings = []
        self.full_conversations = []
        self.full_embeddings = []

    def add_conversation(self, conversation: List[Dict], metadata: Dict = None):
        timestamp = metadata.get("timestamp", "") if metadata else ""

        # Build full conversation text
        conv_text = f"[{timestamp}]\n"
        for turn in conversation:
            speaker = turn.get("speaker", "unknown")
            text = turn.get("text", "")
            conv_text += f"{speaker}: {text}\n"

        # Store full conversation for broad context retrieval
        if conv_text.strip():
            try:
                full_emb = self.get_embedding(conv_text)
                self.full_conversations.append(conv_text)
                self.full_embeddings.append(full_emb)
            except Exception as e:
                self.logger.warning(f"Full conversation embedding error: {e}")

        # Overlapping chunking for fine-grained retrieval
        if self.encoding:
            tokens = self.encoding.encode(conv_text)
            step = max(1, self.chunk_size - self.chunk_overlap)
            for i in range(0, len(tokens), step):
                chunk_tokens = tokens[i:i + self.chunk_size]
                if len(chunk_tokens) < 20:  # Skip very small chunks
                    continue
                chunk = self.encoding.decode(chunk_tokens)
                try:
                    emb = self.get_embedding(chunk)
                    self.chunks.append(chunk)
                    self.embeddings.append(emb)
                except Exception as e:
                    self.logger.warning(f"RAG embedding error: {e}")
        else:
            # Fallback: character-based chunking with overlap
            char_size = self.chunk_size * 4
            char_overlap = self.chunk_overlap * 4
            step = max(1, char_size - char_overlap)
            for i in range(0, len(conv_text), step):
                chunk = conv_text[i:i + char_size]
                if len(chunk.strip()) < 50:
                    continue
                try:
                    emb = self.get_embedding(chunk)
                    self.chunks.append(chunk)
                    self.embeddings.append(emb)
                except Exception as e:
                    self.logger.warning(f"RAG embedding error: {e}")

    def search(self, query: str, top_k: int = None) -> Tuple[str, float]:
        if not self.chunks and not self.full_conversations:
            return "", 0

        k = top_k or self.default_top_k
        start = time.time()

        try:
            query_emb = self.get_embedding(query)

            # Search in full conversations first
            full_results = []
            if self.full_embeddings:
                full_sims = [
                    np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-9)
                    for emb in self.full_embeddings
                ]
                top_full = np.argsort(full_sims)[-min(3, len(full_sims)):][::-1]
                full_results = [self.full_conversations[i] for i in top_full]

            # Search in chunks
            chunk_results = []
            if self.embeddings:
                chunk_sims = [
                    np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-9)
                    for emb in self.embeddings
                ]
                top_chunks = np.argsort(chunk_sims)[-min(k, len(chunk_sims)):][::-1]
                chunk_results = [self.chunks[i] for i in top_chunks]

            # Combine with deduplication
            seen = set()
            combined = []

            # Add full conversations first (higher priority)
            for text in full_results:
                normalized = text.strip().lower()[:100]
                if normalized not in seen:
                    seen.add(normalized)
                    combined.append(text)

            # Add chunks
            for text in chunk_results:
                normalized = text.strip().lower()[:100]
                if normalized not in seen:
                    seen.add(normalized)
                    combined.append(text)

            context = "\n---\n".join(combined)
            return context, time.time() - start
        except Exception as e:
            self.logger.error(f"RAG search error: {e}")
            return "", time.time() - start


# =============================================================================
# Full Context Baseline
# =============================================================================

class FullContextBaseline(BaseMemorySystem):
    """No retrieval - passes all context to LLM."""

    def __init__(self, logger: logging.Logger, max_tokens: int = 15000):
        super().__init__(logger, "baseline")
        self.max_tokens = max_tokens
        self.context = ""

        try:
            import tiktoken
            self.encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        except:
            self.encoding = None

        logger.info("Full-context baseline initialized")

    def reset(self, user_id: str = None):
        self.context = ""

    def add_conversation(self, conversation: List[Dict], metadata: Dict = None):
        timestamp = metadata.get("timestamp", "") if metadata else ""
        for turn in conversation:
            speaker = turn.get("speaker", "unknown")
            text = turn.get("text", "")
            self.context += f"{timestamp} | {speaker}: {text}\n"

    def search(self, query: str, top_k: int = 10) -> Tuple[str, float]:
        if self.encoding:
            tokens = self.encoding.encode(self.context)
            if len(tokens) > self.max_tokens:
                return self.encoding.decode(tokens[:self.max_tokens]), 0
        return self.context, 0


# =============================================================================
# Hybrid NMS+RAG System (NEW)
# =============================================================================

class HybridNMSRAG(BaseMemorySystem):
    """
    Hybrid system combining NMS (neuroscientific) with RAG retrieval.

    Strategy:
    1. Use NMS for structured memory (working, episodic, semantic)
    2. Use RAG for dense retrieval
    3. Combine and rerank results using reciprocal rank fusion
    """

    def __init__(self, logger: logging.Logger, db_path: str = "./data/hybrid.db"):
        super().__init__(logger, "hybrid_nms_rag")

        # Initialize NMS component
        self.nms = NeuroscientificMemorySystem(
            logger=logger,
            db_path=db_path,
            enable_working=True,
            enable_episodic=True,
            enable_semantic=True,
            enable_procedural=True
        )

        # Initialize RAG component with smaller chunks for precision
        self.rag_chunks = []
        self.rag_embeddings = []
        self.full_conversations = []
        self.full_embeddings = []

        try:
            import tiktoken
            self.encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        except:
            self.encoding = None

        logger.info("Hybrid NMS+RAG initialized")

    def reset(self, user_id: str = None):
        self.nms.reset(user_id)
        self.rag_chunks = []
        self.rag_embeddings = []
        self.full_conversations = []
        self.full_embeddings = []

    def add_conversation(self, conversation: List[Dict], metadata: Dict = None):
        # Add to NMS
        self.nms.add_conversation(conversation, metadata)

        # Add to RAG (full conversation + chunks)
        timestamp = metadata.get("timestamp", "") if metadata else ""
        conv_text = f"[{timestamp}]\n"
        for turn in conversation:
            speaker = turn.get("speaker", "unknown")
            text = turn.get("text", "")
            conv_text += f"{speaker}: {text}\n"

        # Store full conversation
        if conv_text.strip():
            try:
                full_emb = self.get_embedding(conv_text)
                self.full_conversations.append(conv_text)
                self.full_embeddings.append(full_emb)
            except:
                pass

        # Chunk for fine-grained retrieval (smaller chunks = more precise)
        chunk_size = 200
        overlap = 50
        if self.encoding:
            tokens = self.encoding.encode(conv_text)
            step = max(1, chunk_size - overlap)
            for i in range(0, len(tokens), step):
                chunk_tokens = tokens[i:i + chunk_size]
                if len(chunk_tokens) < 15:
                    continue
                chunk = self.encoding.decode(chunk_tokens)
                try:
                    emb = self.get_embedding(chunk)
                    self.rag_chunks.append(chunk)
                    self.rag_embeddings.append(emb)
                except:
                    pass

    def _reciprocal_rank_fusion(self, rankings: List[List[Tuple[str, float]]], k: int = 60) -> List[Tuple[str, float]]:
        """Combine multiple rankings using RRF."""
        scores = {}
        for ranking in rankings:
            for rank, (text, _) in enumerate(ranking):
                text_key = text[:200]  # Use prefix as key
                if text_key not in scores:
                    scores[text_key] = {"text": text, "score": 0}
                scores[text_key]["score"] += 1.0 / (k + rank + 1)

        # Sort by RRF score
        results = [(v["text"], v["score"]) for v in scores.values()]
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def search(self, query: str, top_k: int = 10) -> Tuple[str, float]:
        start = time.time()

        rankings = []

        # Get NMS results
        try:
            nms_context, _ = self.nms.search(query, top_k)
            if nms_context:
                # Split by separator and score by position
                nms_parts = nms_context.split("\n---\n")
                nms_ranking = [(part, 1.0 / (i + 1)) for i, part in enumerate(nms_parts) if part.strip()]
                if nms_ranking:
                    rankings.append(nms_ranking)
        except Exception as e:
            self.logger.warning(f"NMS search error: {e}")

        # Get RAG results (dense retrieval)
        try:
            query_emb = self.get_embedding(query)

            # Search full conversations
            if self.full_embeddings:
                sims = [
                    np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-9)
                    for emb in self.full_embeddings
                ]
                top_idx = np.argsort(sims)[-min(5, len(sims)):][::-1]
                full_ranking = [(self.full_conversations[i], sims[i]) for i in top_idx]
                if full_ranking:
                    rankings.append(full_ranking)

            # Search chunks
            if self.rag_embeddings:
                sims = [
                    np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-9)
                    for emb in self.rag_embeddings
                ]
                top_idx = np.argsort(sims)[-min(top_k, len(sims)):][::-1]
                chunk_ranking = [(self.rag_chunks[i], sims[i]) for i in top_idx]
                if chunk_ranking:
                    rankings.append(chunk_ranking)
        except Exception as e:
            self.logger.warning(f"RAG search error: {e}")

        if not rankings:
            return "", time.time() - start

        # Fuse rankings
        fused = self._reciprocal_rank_fusion(rankings)

        # Take top results and deduplicate
        seen = set()
        results = []
        for text, _ in fused[:top_k * 2]:
            key = text.strip().lower()[:100]
            if key not in seen:
                seen.add(key)
                results.append(text)
            if len(results) >= top_k:
                break

        context = "\n---\n".join(results)
        return context, time.time() - start


# =============================================================================
# BM25 + Semantic Reranker (Lightweight)
# =============================================================================

class BM25Reranker(BaseMemorySystem):
    """
    Lightweight memory system using BM25 for initial retrieval
    and semantic similarity for reranking.

    Much faster than dense retrieval but still effective.
    """

    def __init__(self, logger: logging.Logger, top_k_bm25: int = 20, top_k_final: int = 5):
        super().__init__(logger, "bm25_rerank")
        self.top_k_bm25 = top_k_bm25
        self.top_k_final = top_k_final
        self.documents = []
        self.doc_tokens = []
        self.full_conversations = []

        # BM25 parameters
        self.k1 = 1.5
        self.b = 0.75
        self.avgdl = 0
        self.doc_lens = []
        self.idf = {}
        self.doc_freqs = {}

        logger.info("BM25+Reranker initialized")

    def reset(self, user_id: str = None):
        self.documents = []
        self.doc_tokens = []
        self.full_conversations = []
        self.doc_lens = []
        self.idf = {}
        self.doc_freqs = {}
        self.avgdl = 0

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        import re
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        # Remove stopwords
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
                     'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought', 'used',
                     'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                     'through', 'during', 'before', 'after', 'above', 'below', 'between',
                     'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither',
                     'not', 'only', 'own', 'same', 'than', 'too', 'very', 'just', 'i', 'you',
                     'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who', 'whom', 'this',
                     'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'been', 'being'}
        return [t for t in tokens if t not in stopwords and len(t) > 1]

    def _compute_idf(self):
        """Compute IDF for all terms."""
        import math
        N = len(self.documents)
        if N == 0:
            return

        for term, df in self.doc_freqs.items():
            self.idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)

    def add_conversation(self, conversation: List[Dict], metadata: Dict = None):
        timestamp = metadata.get("timestamp", "") if metadata else ""

        conv_text = f"[{timestamp}]\n"
        for turn in conversation:
            speaker = turn.get("speaker", "unknown")
            text = turn.get("text", "")
            conv_text += f"{speaker}: {text}\n"

        self.full_conversations.append(conv_text)

        # Add individual turns as documents
        for turn in conversation:
            text = turn.get("text", "")
            if len(text.strip()) < 10:
                continue

            doc_with_context = f"[{timestamp}] {turn.get('speaker', '')}: {text}"
            tokens = self._tokenize(doc_with_context)

            self.documents.append(doc_with_context)
            self.doc_tokens.append(tokens)
            self.doc_lens.append(len(tokens))

            # Update document frequencies
            seen_terms = set()
            for token in tokens:
                if token not in seen_terms:
                    self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
                    seen_terms.add(token)

        # Update average document length
        if self.doc_lens:
            self.avgdl = sum(self.doc_lens) / len(self.doc_lens)

        # Recompute IDF
        self._compute_idf()

    def _bm25_score(self, query_tokens: List[str], doc_idx: int) -> float:
        """Compute BM25 score for a document."""
        score = 0.0
        doc_tokens = self.doc_tokens[doc_idx]
        doc_len = self.doc_lens[doc_idx]

        # Term frequency in document
        tf = {}
        for token in doc_tokens:
            tf[token] = tf.get(token, 0) + 1

        for term in query_tokens:
            if term not in self.idf:
                continue

            term_tf = tf.get(term, 0)
            idf = self.idf[term]

            # BM25 formula
            numerator = term_tf * (self.k1 + 1)
            denominator = term_tf + self.k1 * (1 - self.b + self.b * doc_len / (self.avgdl + 1e-9))
            score += idf * numerator / (denominator + 1e-9)

        return score

    def search(self, query: str, top_k: int = 10) -> Tuple[str, float]:
        if not self.documents:
            return "", 0

        start = time.time()
        query_tokens = self._tokenize(query)

        # BM25 retrieval
        scores = []
        for i in range(len(self.documents)):
            score = self._bm25_score(query_tokens, i)
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_bm25 = scores[:self.top_k_bm25]

        # Semantic reranking (if we have enough candidates)
        if len(top_bm25) > self.top_k_final:
            try:
                query_emb = self.get_embedding(query)
                reranked = []

                for doc_idx, bm25_score in top_bm25:
                    doc_text = self.documents[doc_idx]
                    doc_emb = self.get_embedding(doc_text[:1000])  # Limit text length

                    # Cosine similarity
                    sem_score = np.dot(query_emb, doc_emb) / (
                        np.linalg.norm(query_emb) * np.linalg.norm(doc_emb) + 1e-9
                    )

                    # Combined score (weighted)
                    combined = 0.3 * bm25_score + 0.7 * sem_score
                    reranked.append((doc_idx, combined))

                reranked.sort(key=lambda x: x[1], reverse=True)
                top_docs = [self.documents[idx] for idx, _ in reranked[:top_k]]
            except Exception as e:
                self.logger.warning(f"Reranking error: {e}")
                top_docs = [self.documents[idx] for idx, _ in top_bm25[:top_k]]
        else:
            top_docs = [self.documents[idx] for idx, _ in top_bm25[:top_k]]

        # Add full conversation context if available
        if self.full_conversations:
            # Find most relevant full conversation
            try:
                query_emb = self.get_embedding(query)
                best_conv = None
                best_score = -1

                for conv in self.full_conversations[:5]:  # Limit search
                    conv_emb = self.get_embedding(conv[:2000])
                    score = np.dot(query_emb, conv_emb) / (
                        np.linalg.norm(query_emb) * np.linalg.norm(conv_emb) + 1e-9
                    )
                    if score > best_score:
                        best_score = score
                        best_conv = conv

                if best_conv and best_score > 0.5:
                    top_docs.insert(0, best_conv)
            except:
                pass

        context = "\n---\n".join(top_docs)
        return context, time.time() - start


# =============================================================================
# MemoryGraph (Graphiti-inspired)
# =============================================================================

class MemoryGraph(BaseMemorySystem):
    """
    Graph-based memory inspired by Graphiti and MemGPT.

    Stores:
    - Entities (diseases, crops, treatments, etc.)
    - Relations between entities
    - Temporal information
    - Episode summaries

    Lightweight SQLite-based implementation.
    """

    def __init__(self, logger: logging.Logger, db_path: str = "./data/memorygraph.db"):
        super().__init__(logger, "memorygraph")
        self.db_path = db_path
        self._init_db()
        self.episode_summaries = []
        self.current_facts = []

        logger.info("MemoryGraph initialized")

    def _init_db(self):
        """Initialize SQLite database for graph storage."""
        import sqlite3
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        # Entities table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE,
                entity_type TEXT,
                attributes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Relations table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY,
                source_id INTEGER,
                target_id INTEGER,
                relation_type TEXT,
                context TEXT,
                timestamp TEXT,
                FOREIGN KEY (source_id) REFERENCES entities(id),
                FOREIGN KEY (target_id) REFERENCES entities(id)
            )
        ''')

        # Episodes table (conversation summaries)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY,
                summary TEXT,
                key_facts TEXT,
                timestamp TEXT,
                embedding BLOB
            )
        ''')

        self.conn.commit()

    def reset(self, user_id: str = None):
        self.cursor.execute("DELETE FROM entities")
        self.cursor.execute("DELETE FROM relations")
        self.cursor.execute("DELETE FROM episodes")
        self.conn.commit()
        self.episode_summaries = []
        self.current_facts = []

    def _extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """Extract entities from text using patterns."""
        entities = []

        # Disease patterns
        disease_patterns = [
            r'(early blight|late blight|bacterial spot|powdery mildew|rice blast|'
            r'septoria leaf spot|downy mildew|fusarium wilt|anthracnose|'
            r'black rot|citrus canker|fire blight)'
        ]

        # Crop patterns
        crop_patterns = [
            r'\b(tomato|potato|pepper|eggplant|rice|wheat|grape|cucumber|'
            r'squash|banana|cotton|spinach|apple|citrus|corn|soybean)\b'
        ]

        # Treatment patterns
        treatment_patterns = [
            r'\b(mancozeb|chlorothalonil|copper fungicide|metalaxyl|'
            r'tricyclazole|sulfur|streptomycin|propiconazole|azoxystrobin|'
            r'copper bactericide|neem oil|solarization)\b'
        ]

        # Pathogen patterns
        pathogen_patterns = [
            r'(Alternaria|Phytophthora|Xanthomonas|Fusarium|Magnaporthe|'
            r'Septoria|Plasmopara|Erysiphe)\s*\w*'
        ]

        import re
        text_lower = text.lower()

        for pattern in disease_patterns:
            for match in re.finditer(pattern, text_lower):
                entities.append((match.group(0), "disease"))

        for pattern in crop_patterns:
            for match in re.finditer(pattern, text_lower):
                entities.append((match.group(0), "crop"))

        for pattern in treatment_patterns:
            for match in re.finditer(pattern, text_lower):
                entities.append((match.group(0), "treatment"))

        for pattern in pathogen_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append((match.group(0).lower(), "pathogen"))

        return list(set(entities))

    def _extract_relations(self, text: str, entities: List[Tuple[str, str]]) -> List[Tuple[str, str, str]]:
        """Extract relations between entities."""
        relations = []
        text_lower = text.lower()

        # Relation patterns
        patterns = [
            (r'(\w+)\s+(?:causes?|caused by)\s+(\w+)', 'causes'),
            (r'(\w+)\s+(?:affects?|infects?)\s+(\w+)', 'affects'),
            (r'(\w+)\s+(?:treat(?:s|ed)?|control(?:s|led)?)\s+(\w+)', 'treats'),
            (r'(\w+)\s+(?:is a|are)\s+(\w+)', 'is_a'),
            (r'(\w+)\s+(?:has|have|shows?)\s+(\w+)', 'has'),
        ]

        entity_names = {e[0] for e in entities}

        for pattern, rel_type in patterns:
            import re
            for match in re.finditer(pattern, text_lower):
                source, target = match.groups()
                if source in entity_names or target in entity_names:
                    relations.append((source, target, rel_type))

        return relations

    def _store_entity(self, name: str, entity_type: str) -> int:
        """Store or get entity ID."""
        try:
            self.cursor.execute(
                "INSERT OR IGNORE INTO entities (name, entity_type) VALUES (?, ?)",
                (name, entity_type)
            )
            self.cursor.execute("SELECT id FROM entities WHERE name = ?", (name,))
            result = self.cursor.fetchone()
            return result[0] if result else None
        except:
            return None

    def add_conversation(self, conversation: List[Dict], metadata: Dict = None):
        timestamp = metadata.get("timestamp", "") if metadata else ""

        # Build full text
        full_text = ""
        for turn in conversation:
            speaker = turn.get("speaker", "")
            text = turn.get("text", "")
            full_text += f"{speaker}: {text}\n"

        # Extract and store entities
        entities = self._extract_entities(full_text)
        entity_ids = {}
        for name, etype in entities:
            eid = self._store_entity(name, etype)
            if eid:
                entity_ids[name] = eid

        # Extract and store relations
        relations = self._extract_relations(full_text, entities)
        for source, target, rel_type in relations:
            source_id = entity_ids.get(source)
            target_id = entity_ids.get(target)
            if source_id and target_id:
                self.cursor.execute(
                    "INSERT INTO relations (source_id, target_id, relation_type, context, timestamp) VALUES (?, ?, ?, ?, ?)",
                    (source_id, target_id, rel_type, full_text[:500], timestamp)
                )

        # Store episode summary
        try:
            summary = f"[{timestamp}] Discussed: {', '.join([e[0] for e in entities[:5]])}. {full_text[:200]}"
            emb = self.get_embedding(summary)
            import pickle
            self.cursor.execute(
                "INSERT INTO episodes (summary, key_facts, timestamp, embedding) VALUES (?, ?, ?, ?)",
                (summary, json.dumps([e[0] for e in entities]), timestamp, pickle.dumps(emb))
            )
            self.episode_summaries.append((summary, emb))
        except Exception as e:
            self.logger.warning(f"Episode storage error: {e}")

        # Store current facts for quick access
        fact_text = f"[{timestamp}]\n{full_text}"
        self.current_facts.append(fact_text)

        self.conn.commit()

    def search(self, query: str, top_k: int = 10) -> Tuple[str, float]:
        start = time.time()
        results = []

        # 1. Entity-based search
        query_entities = self._extract_entities(query)
        entity_names = [e[0] for e in query_entities]

        if entity_names:
            # Find related entities through graph
            placeholders = ','.join(['?' for _ in entity_names])
            try:
                self.cursor.execute(f'''
                    SELECT DISTINCT e2.name, e2.entity_type, r.relation_type, r.context
                    FROM entities e1
                    JOIN relations r ON e1.id = r.source_id OR e1.id = r.target_id
                    JOIN entities e2 ON (e2.id = r.source_id OR e2.id = r.target_id) AND e2.id != e1.id
                    WHERE e1.name IN ({placeholders})
                    LIMIT 10
                ''', entity_names)

                for row in self.cursor.fetchall():
                    name, etype, rel_type, context = row
                    results.append(f"[Graph] {name} ({etype}) - {rel_type}: {context[:200]}")
            except Exception as e:
                self.logger.warning(f"Graph search error: {e}")

        # 2. Semantic search on episodes
        try:
            query_emb = self.get_embedding(query)

            if self.episode_summaries:
                similarities = []
                for summary, emb in self.episode_summaries:
                    sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-9)
                    similarities.append((summary, sim))

                similarities.sort(key=lambda x: x[1], reverse=True)
                for summary, _ in similarities[:3]:
                    results.append(summary)
        except Exception as e:
            self.logger.warning(f"Episode search error: {e}")

        # 3. Add recent facts (recency bias)
        for fact in self.current_facts[-3:]:
            if fact not in results:
                results.append(fact)

        context = "\n---\n".join(results[:top_k])
        return context, time.time() - start


# =============================================================================
# Memory System Factory
# =============================================================================

def create_memory_system(technique: str, logger: logging.Logger) -> Optional[BaseMemorySystem]:
    """Create memory system by name."""
    try:
        if technique == "nms":
            return NeuroscientificMemorySystem(
                logger=logger,
                db_path="./data/nms_memory.db",
                enable_working=True,
                enable_episodic=True,
                enable_semantic=True,
                enable_procedural=True
            )
        elif technique == "nms_working_only":
            return NeuroscientificMemorySystem(
                logger=logger,
                db_path="./data/nms_working.db",
                enable_working=True,
                enable_episodic=False,
                enable_semantic=False,
                enable_procedural=False
            )
        elif technique == "nms_episodic_only":
            return NeuroscientificMemorySystem(
                logger=logger,
                db_path="./data/nms_episodic.db",
                enable_working=False,
                enable_episodic=True,
                enable_semantic=False,
                enable_procedural=False
            )
        elif technique == "hybrid":
            return HybridNMSRAG(logger=logger, db_path="./data/hybrid.db")
        elif technique == "bm25":
            return BM25Reranker(logger=logger)
        elif technique == "memorygraph":
            return MemoryGraph(logger=logger, db_path="./data/memorygraph.db")
        elif technique == "graphrag":
            return GraphRAG(logger=logger, db_path="./data/graphrag.db")
        elif technique == "rag":
            return RAGSystem(logger, chunk_size=500, top_k=5)
        elif technique == "rag_large":
            return RAGSystem(logger, chunk_size=2000, top_k=3)
        elif technique == "baseline":
            return FullContextBaseline(logger)
        else:
            logger.warning(f"Unknown technique: {technique}")
            return None
    except Exception as e:
        logger.error(f"Failed to create {technique}: {e}")
        return None


# =============================================================================
# LLM Judge
# =============================================================================

async def llm_judge(
    client: AsyncOpenAI,
    question: str,
    gold: str,
    response: str,
    model: str = None
) -> bool:
    """Evaluate answer using LLM as judge."""
    model = model or os.getenv("EVAL_MODEL", os.getenv("MODEL", "deepseek-ai/DeepSeek-V3-0324-TEE"))
    prompt = LLM_JUDGE_PROMPT.format(
        question=question,
        gold_answer=gold,
        response=response
    )

    try:
        result = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert grader."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        content = result.choices[0].message.content
        match = re.search(r'\{\s*"label"\s*:\s*"(\w+)"\s*\}', content)
        if match:
            return match.group(1).lower() == "correct"
        return False
    except Exception as e:
        print(f"LLM judge error: {e}")
        return False


async def ensemble_judge(
    question: str,
    gold: str,
    response: str,
    judge_models: List[str] = None
) -> Tuple[bool, Dict[str, bool]]:
    """Ensemble voting with multiple judge models."""
    judge_models = judge_models or [os.getenv("MODEL", "deepseek-ai/DeepSeek-V3-0324-TEE")]

    # Create client using Chutes.ai (DeepSeek) instead of OpenAI
    chutes_client = AsyncOpenAI(
        api_key=os.getenv("CHUTES_API_KEY"),
        base_url=os.getenv("CHUTES_API_BASE", "https://llm.chutes.ai/v1")
    )

    votes = {}
    for model in judge_models:
        try:
            vote = await llm_judge(chutes_client, question, gold, response, model)
            votes[model] = vote
        except Exception as e:
            print(f"Judge {model} failed: {e}")
            votes[model] = False

    # Majority vote
    positive_votes = sum(1 for v in votes.values() if v)
    final_vote = positive_votes > len(votes) / 2

    return final_vote, votes


# =============================================================================
# Data Loading
# =============================================================================

def load_agri_locomo(data_path: str) -> List[Dict]:
    """Load AgriLoCoMo/AgriConvMem dataset."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data.get("samples", data) if isinstance(data, dict) else data


def load_agri_hotpotqa(data_path: str) -> List[Dict]:
    """Load AgriHotpotQA/AgriMultiHop dataset."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("samples", [])


# =============================================================================
# Evaluation
# =============================================================================

async def evaluate_locomo(
    data_path: str,
    memory_system: BaseMemorySystem,
    logger: logging.Logger,
    top_k: int = 10,
    limit: int = None,
    use_llm_judge: bool = True,
    judge_models: List[str] = None
) -> Dict[str, Any]:
    """Evaluate on AgriLoCoMo/AgriConvMem dataset."""
    logger.info(f"Evaluating LoCoMo with {memory_system.name}")
    samples = load_agri_locomo(data_path)

    if limit:
        samples = samples[:limit]

        # Use Chutes.ai (DeepSeek) for LLM judge instead of OpenAI
    async_client = AsyncOpenAI(
        api_key=os.getenv("CHUTES_API_KEY"),
        base_url=os.getenv("CHUTES_API_BASE", "https://llm.chutes.ai/v1")
    )

    results = {
        "predictions": [],
        "f1": [],
        "em": [],
        "semantic_sim": [],
        "bleu1": [],
        "llm_judge": [],
        "by_category": defaultdict(list),
        "timing": {"search": [], "llm": []}
    }

    for sample_idx, sample in enumerate(tqdm(samples, desc=f"LoCoMo ({memory_system.name})")):
        sample_id = sample.get("sample_id", f"sample_{sample_idx}")
        memory_system.reset(f"locomo_{sample_id}")

        # Build memory from conversation sessions
        conversation = sample.get("conversation", {})
        for key, value in conversation.items():
            if key.startswith("session_") and isinstance(value, list):
                timestamp = conversation.get(f"{key}_date_time", "")
                memory_system.add_conversation(value, {"timestamp": timestamp})

        await asyncio.sleep(0.1)  # Rate limiting

        # Evaluate each QA pair
        for qa in sample.get('qa', []):
            question = qa["question"]
            reference = str(qa["answer"])
            category = qa.get("category", 0)
            category_name = qa.get("category_name", str(category))

            # Query memory system
            prediction, context, search_time, llm_time = memory_system.query(question, top_k)

            # Calculate metrics
            f1 = calculate_f1(prediction, reference)
            em = calculate_exact_match(prediction, reference)
            semantic_sim = calculate_semantic_similarity(prediction, reference)
            bleu = calculate_bleu1(prediction, reference)

            # LLM judge
            llm_correct = False
            if use_llm_judge:
                if judge_models and len(judge_models) > 1:
                    llm_correct, _ = await ensemble_judge(question, reference, prediction, judge_models)
                else:
                    llm_correct = await llm_judge(async_client, question, reference, prediction)

            results["predictions"].append({
                "sample_id": sample_id,
                "question": question,
                "reference": reference,
                "prediction": prediction,
                "category": category,
                "category_name": category_name,
                "f1": f1,
                "em": em,
                "semantic_sim": semantic_sim,
                "bleu1": bleu,
                "llm_judge": llm_correct,
                "search_time": search_time,
                "llm_time": llm_time
            })

            results["f1"].append(f1)
            results["em"].append(em)
            results["semantic_sim"].append(semantic_sim)
            results["bleu1"].append(bleu)
            results["llm_judge"].append(1 if llm_correct else 0)
            results["by_category"][category_name].append(f1)
            results["timing"]["search"].append(search_time)
            results["timing"]["llm"].append(llm_time)

    return {
        "dataset": "AgriConvMem",
        "technique": memory_system.name,
        "num_samples": len(samples),
        "num_questions": len(results["f1"]),
        "overall": {
            "f1": np.mean(results["f1"]) if results["f1"] else 0,
            "f1_std": np.std(results["f1"]) if results["f1"] else 0,
            "em": np.mean(results["em"]) if results["em"] else 0,
            "semantic_sim": np.mean(results["semantic_sim"]) if results["semantic_sim"] else 0,
            "bleu1": np.mean(results["bleu1"]) if results["bleu1"] else 0,
            "llm_judge": np.mean(results["llm_judge"]) if results["llm_judge"] else 0
        },
        "by_category": {cat: np.mean(scores) for cat, scores in results["by_category"].items()},
        "timing": {
            "avg_search_time": np.mean(results["timing"]["search"]) if results["timing"]["search"] else 0,
            "avg_llm_time": np.mean(results["timing"]["llm"]) if results["timing"]["llm"] else 0
        },
        "predictions": results["predictions"],
        "raw_scores": {
            "f1": results["f1"],
            "em": results["em"],
            "semantic_sim": results["semantic_sim"],
            "llm_judge": results["llm_judge"]
        }
    }


async def evaluate_hotpotqa(
    data_path: str,
    memory_system: BaseMemorySystem,
    logger: logging.Logger,
    top_k: int = 10,
    limit: int = None,
    use_llm_judge: bool = True
) -> Dict[str, Any]:
    """Evaluate on AgriHotpotQA/AgriMultiHop dataset."""
    logger.info(f"Evaluating HotpotQA with {memory_system.name}")
    samples = load_agri_hotpotqa(data_path)

    if limit:
        samples = samples[:limit]

        # Use Chutes.ai (DeepSeek) for LLM judge instead of OpenAI
    async_client = AsyncOpenAI(
        api_key=os.getenv("CHUTES_API_KEY"),
        base_url=os.getenv("CHUTES_API_BASE", "https://llm.chutes.ai/v1")
    )

    results = {
        "predictions": [],
        "f1": [],
        "em": [],
        "semantic_sim": [],
        "llm_judge": [],
        "by_type": defaultdict(list)
    }

    for sample_idx, sample in enumerate(tqdm(samples, desc=f"HotpotQA ({memory_system.name})")):
        sample_id = sample.get("id", f"hotpot_{sample_idx}")
        memory_system.reset(f"hotpot_{sample_id}")

        # Build memory from context paragraphs
        for para in sample.get("context", []):
            title = para.get("title", "")
            text = para.get("text", "")
            memory_system.add_conversation(
                [{"speaker": "document", "text": f"{title}: {text}" if title else text}],
                {"title": title}
            )

        await asyncio.sleep(0.1)

        question = sample.get("question", "")
        reference = str(sample.get("answer", ""))
        q_type = sample.get("type", "bridge")

        prediction, context, search_time, llm_time = memory_system.query(question, top_k)

        f1 = calculate_f1(prediction, reference)
        em = calculate_exact_match(prediction, reference)
        semantic_sim = calculate_semantic_similarity(prediction, reference)

        llm_correct = False
        if use_llm_judge:
            llm_correct = await llm_judge(async_client, question, reference, prediction)

        results["predictions"].append({
            "sample_id": sample_id,
            "question": question,
            "reference": reference,
            "prediction": prediction,
            "type": q_type,
            "f1": f1,
            "em": em,
            "semantic_sim": semantic_sim,
            "llm_judge": llm_correct
        })

        results["f1"].append(f1)
        results["em"].append(em)
        results["semantic_sim"].append(semantic_sim)
        results["llm_judge"].append(1 if llm_correct else 0)
        results["by_type"][q_type].append(f1)

    return {
        "dataset": "AgriMultiHop",
        "technique": memory_system.name,
        "num_samples": len(samples),
        "overall": {
            "f1": np.mean(results["f1"]) if results["f1"] else 0,
            "f1_std": np.std(results["f1"]) if results["f1"] else 0,
            "em": np.mean(results["em"]) if results["em"] else 0,
            "semantic_sim": np.mean(results["semantic_sim"]) if results["semantic_sim"] else 0,
            "llm_judge": np.mean(results["llm_judge"]) if results["llm_judge"] else 0
        },
        "by_type": {qtype: np.mean(scores) for qtype, scores in results["by_type"].items()},
        "predictions": results["predictions"],
        "raw_scores": {
            "f1": results["f1"],
            "em": results["em"],
            "semantic_sim": results["semantic_sim"],
            "llm_judge": results["llm_judge"]
        }
    }


def print_results(results: Dict, logger: logging.Logger):
    """Print formatted results."""
    output = [
        "\n" + "=" * 60,
        f"  {results['dataset']} - {results['technique']}",
        "=" * 60,
        f"Samples: {results['num_samples']}"
    ]

    if "num_questions" in results:
        output.append(f"Questions: {results['num_questions']}")

    output.append("\nOverall Metrics:")
    for metric, value in results["overall"].items():
        output.append(f"  {metric}: {value:.4f}")

    if results.get("by_category"):
        output.append("\nBy Category (F1):")
        for cat, score in sorted(results["by_category"].items()):
            output.append(f"  {cat}: {score:.4f}")

    if results.get("by_type"):
        output.append("\nBy Type (F1):")
        for qtype, score in sorted(results["by_type"].items()):
            output.append(f"  {qtype}: {score:.4f}")

    output.append("=" * 60)

    for line in output:
        print(line)
        logger.info(line)


# =============================================================================
# Statistical Analysis
# =============================================================================

def run_statistical_analysis(
    all_results: Dict[str, Dict],
    logger: logging.Logger
) -> Dict[str, Any]:
    """Run statistical significance tests on results."""
    tester = StatisticalTester(alpha=0.05, n_bootstrap=1000)

    analysis = {
        "confidence_intervals": {},
        "pairwise_comparisons": {},
        "effect_sizes": {}
    }

    # Extract raw F1 scores
    for dataset in ["locomo", "hotpotqa"]:
        dataset_scores = {}
        for technique, tech_results in all_results.items():
            if dataset in tech_results and "raw_scores" in tech_results[dataset]:
                dataset_scores[technique] = tech_results[dataset]["raw_scores"]["f1"]

        if len(dataset_scores) >= 2:
            # Confidence intervals
            analysis["confidence_intervals"][dataset] = {}
            for technique, scores in dataset_scores.items():
                mean, lower, upper = tester.bootstrap_confidence_interval(scores)
                analysis["confidence_intervals"][dataset][technique] = {
                    "mean": mean,
                    "lower": lower,
                    "upper": upper,
                    "std": np.std(scores)
                }

            # Pairwise comparisons
            comparisons = tester.compare_all_methods(dataset_scores, metric="f1")
            analysis["pairwise_comparisons"][dataset] = [
                {
                    "method_a": c.method_a,
                    "method_b": c.method_b,
                    "mean_diff": c.mean_diff,
                    "p_value": c.p_value,
                    "cohen_d": c.cohen_d,
                    "is_significant": c.is_significant
                }
                for c in comparisons
            ]

            # Generate report
            report = tester.generate_report(dataset_scores, metric="f1")
            logger.info(f"\n{report}")

    return analysis


# =============================================================================
# Main
# =============================================================================

async def main_async():
    """Main async entry point."""
    parser = argparse.ArgumentParser(description="AgriMemory v2 Comprehensive Benchmark")
    parser.add_argument("--dataset", choices=["locomo", "hotpotqa", "both"], default="both")
    parser.add_argument("--data-dir", default="data/text")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="results/benchmark_v2.json")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--techniques", default="nms,rag,baseline",
                        help="Comma-separated: nms,graphrag,rag,baseline")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-llm-judge", action="store_true")
    parser.add_argument("--judge-models", default="deepseek-ai/DeepSeek-V3-0324-TEE",
                        help="Comma-separated judge models for ensemble (uses Chutes.ai)")
    parser.add_argument("--n-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-name", default="benchmark_v2")

    args = parser.parse_args()

    # Setup
    logger = setup_logging(args.log_dir, args.experiment_name)
    logger.info(f"Starting benchmark: {vars(args)}")

    np.random.seed(args.seed)

    techniques = [t.strip() for t in args.techniques.split(",")]
    judge_models = [m.strip() for m in args.judge_models.split(",")]
    logger.info(f"Techniques: {techniques}")
    logger.info(f"Judge models: {judge_models}")

    # Change to script directory
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)

    all_results = {}
    all_runs = []

    for run_id in range(args.n_runs):
        logger.info(f"\n{'='*60}\nRun {run_id + 1}/{args.n_runs}\n{'='*60}")

        run_results = {}

        for technique in techniques:
            logger.info(f"\n{'='*60}\nBenchmarking: {technique}\n{'='*60}")

            memory_system = create_memory_system(technique, logger)
            if not memory_system:
                continue

            technique_results = {}

            # AgriLoCoMo/AgriConvMem (v2) - prefer fixed version
            if args.dataset in ["locomo", "both"]:
                locomo_fixed = Path(args.data_dir) / "agri_locomo_v2" / f"{args.split}_fixed.json"
                locomo_path = Path(args.data_dir) / "agri_locomo_v2" / f"{args.split}.json"

                # Prefer fixed dataset if available
                if locomo_fixed.exists():
                    locomo_path = locomo_fixed
                    logger.info(f"Using fixed LoCoMo dataset: {locomo_fixed}")

                if locomo_path.exists():
                    results = await evaluate_locomo(
                        str(locomo_path), memory_system, logger,
                        args.top_k, args.limit, not args.no_llm_judge, judge_models
                    )
                    technique_results["locomo"] = results
                    print_results(results, logger)
                else:
                    logger.warning(f"LoCoMo path not found: {locomo_path}")

            # AgriHotpotQA/AgriMultiHop (v4 - Hard Multi-Hop)
            if args.dataset in ["hotpotqa", "both"]:
                # Prefer v4 (hard multi-hop) if available
                hotpotqa_v4_path = Path(args.data_dir) / "agri_hotpotqa_v4" / f"{args.split}.json"
                hotpotqa_v3_path = Path(args.data_dir) / "agri_hotpotqa_v3" / f"{args.split}.json"
                hotpotqa_v2_path = Path(args.data_dir) / "agri_hotpotqa_v2" / f"{args.split}.json"

                if hotpotqa_v4_path.exists():
                    hotpotqa_path = hotpotqa_v4_path
                    logger.info(f"Using AgriMultiHop v4 (hard multi-hop): {hotpotqa_path}")
                elif hotpotqa_v3_path.exists():
                    hotpotqa_path = hotpotqa_v3_path
                    logger.info(f"Using AgriMultiHop v3: {hotpotqa_path}")
                elif hotpotqa_v2_path.exists():
                    hotpotqa_path = hotpotqa_v2_path
                    logger.info(f"Using AgriMultiHop v2: {hotpotqa_path}")
                else:
                    hotpotqa_path = None

                if hotpotqa_path and hotpotqa_path.exists():
                    results = await evaluate_hotpotqa(
                        str(hotpotqa_path), memory_system, logger,
                        args.top_k, args.limit, not args.no_llm_judge
                    )
                    technique_results["hotpotqa"] = results
                    print_results(results, logger)
                else:
                    logger.warning(f"HotpotQA path not found: {hotpotqa_path}")

            run_results[technique] = technique_results

        all_runs.append(run_results)

        # Use first run as main results
        if run_id == 0:
            all_results = run_results

    # Statistical analysis (if multiple runs or enough data)
    statistical_analysis = run_statistical_analysis(all_results, logger)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    final_output = {
        "config": vars(args),
        "results": all_results,
        "statistical_analysis": statistical_analysis,
        "all_runs": all_runs if args.n_runs > 1 else None,
        "timestamp": datetime.now().isoformat()
    }

    with open(output_path, 'w') as f:
        json.dump(final_output, f, indent=2, default=float)

    logger.info(f"\nResults saved to: {output_path}")

    # Print summary table
    print("\n" + "=" * 120)
    print("  COMPREHENSIVE BENCHMARK SUMMARY")
    print("=" * 120)
    print(f"\n{'Technique':<15} {'LoCoMo F1':<12} {'LoCoMo Sem':<12} {'LoCoMo LLM':<12} {'HotpotQA F1':<12} {'HotpotQA Sem':<12} {'HotpotQA LLM':<12}")
    print("-" * 120)

    for technique, tech_results in all_results.items():
        loc_f1 = tech_results.get("locomo", {}).get("overall", {}).get("f1", 0)
        loc_sem = tech_results.get("locomo", {}).get("overall", {}).get("semantic_sim", 0)
        loc_llm = tech_results.get("locomo", {}).get("overall", {}).get("llm_judge", 0)
        hot_f1 = tech_results.get("hotpotqa", {}).get("overall", {}).get("f1", 0)
        hot_sem = tech_results.get("hotpotqa", {}).get("overall", {}).get("semantic_sim", 0)
        hot_llm = tech_results.get("hotpotqa", {}).get("overall", {}).get("llm_judge", 0)
        print(f"{technique:<15} {loc_f1:<12.4f} {loc_sem:<12.4f} {loc_llm:<12.4f} {hot_f1:<12.4f} {hot_sem:<12.4f} {hot_llm:<12.4f}")

    print("=" * 120)


def main():
    """Main entry point."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
