"""
FactStore — flat fact store with vector search.

Each fact is a discrete, self-contained piece of information
extracted from conversation. Facts have embeddings for similarity
search, timestamps for recency, and supersession chains for updates.

This is the "shared substrate" both hemispheres read from and write to.
"""

import json
import re
import uuid
import numpy as np
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional
from pathlib import Path

# ── Keyword helpers ──────────────────────────────────────────────────────────

_STOPWORDS = frozenset({
    'i', 'me', 'my', 'myself', 'we', 'our', 'you', 'your', 'he', 'she', 'it',
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has',
    'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
    'what', 'where', 'when', 'how', 'which', 'who', 'that', 'this', 'these',
    'those', 'from', 'by', 'about', 'as', 'into', 'not', 'no', 'can', 'also',
    'just', 'like', 'so', 'if', 'then', 'than', 'up', 'out', 'its', 'more',
})


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokens and numeric tokens, stopwords and single-chars removed."""
    words = {
        t for t in re.findall(r'\b[a-z]\w*\b', text.lower())
        if t not in _STOPWORDS and len(t) > 2
    }
    # Include numeric tokens so counts, amounts, dates, ratios participate
    # in sparse scoring. Pattern captures: 500, 3.14, 3:1, 2024-03-15, etc.
    nums = set(re.findall(r'\b\d[\d.:/-]*\b', text))
    return words | nums


@dataclass
class Fact:
    content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    confidence: float = 1.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_accessed: str = field(default_factory=lambda: datetime.now().isoformat())
    access_count: int = 0
    session_id: str = ""
    superseded_by: Optional[str] = None
    embedding: Optional[list] = None

    def touch(self):
        self.last_accessed = datetime.now().isoformat()
        self.access_count += 1


class FactStore:
    def __init__(self, path: str = "facts.json"):
        self.path = Path(path)
        self.facts: dict[str, Fact] = {}
        self._embeddings_cache: Optional[np.ndarray] = None
        self._ids_cache: Optional[list[str]] = None
        if self.path.exists():
            self.load()

    def add(self, fact: Fact) -> str:
        self.facts[fact.id] = fact
        self._invalidate_cache()
        return fact.id

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.3,
    ) -> list[tuple[Fact, float]]:
        """Find the most relevant active facts by cosine similarity."""
        if not self.facts:
            return []

        embeddings, ids = self._get_embeddings_matrix()
        if embeddings is None:
            return []

        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        emb_norms = embeddings / (
            np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        )
        similarities = emb_norms @ query_norm

        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            sim = float(similarities[idx])
            if sim >= threshold:
                fact = self.facts[ids[idx]]
                fact.touch()
                results.append((fact, sim))

        return results

    def search_hybrid(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.3,
        alpha: float = 0.7,
    ) -> list[tuple["Fact", float]]:
        """
        Hybrid retrieval: alpha * cosine_similarity + (1-alpha) * keyword_recall.

        keyword_recall = fraction of non-trivial query tokens present in the fact.
        This catches named entities and specific terms ("Serenity Yoga", "45 minutes")
        that get diluted in whole-session embeddings but are exact matches in the text.
        alpha=1.0 degrades to pure cosine (same as search()).
        """
        if not self.facts:
            return []

        embeddings, ids = self._get_embeddings_matrix()
        if embeddings is None:
            return []

        # Dense component
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        emb_norms = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = emb_norms @ query_norm

        # Sparse component — fraction of query tokens matched in each fact
        q_tokens = _tokenize(query_text)
        sparse_scores = np.zeros(len(ids))
        if q_tokens:
            for i, fid in enumerate(ids):
                f_tokens = _tokenize(self.facts[fid].content)
                sparse_scores[i] = len(q_tokens & f_tokens) / len(q_tokens)

        combined = alpha * dense_scores + (1 - alpha) * sparse_scores

        top_indices = np.argsort(combined)[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = float(combined[idx])
            if score >= threshold:
                fact = self.facts[ids[idx]]
                fact.touch()
                results.append((fact, score))

        return results

    backend = "flat"

    def get_all_active(self) -> list[Fact]:
        return [f for f in self.facts.values() if f.superseded_by is None]

    def clear_all(self):
        self.facts.clear()
        self._invalidate_cache()
        self.save()

    def supersede(self, old_id: str, new_id: str):
        if old_id in self.facts:
            self.facts[old_id].superseded_by = new_id

    # ── Phase 2 stubs (no-ops on flat backend) ───────────────
    # GraphStore implements these for real; these keep Membrane
    # backend-agnostic without isinstance checks everywhere.

    def elaborate(self, detail_id: str, anchor_id: str):
        pass

    def depends_on(self, dependent_id: str, anchor_id: str):
        pass

    def add_contradicts(self, id_a: str, id_b: str):
        pass

    def get_contradictions(self) -> list[tuple]:
        return []

    def get_neighbors(self, fact_id: str, edge_types: list[str] | None = None) -> list[tuple]:
        return []

    def get_clusters(self, min_children: int = 5) -> list[tuple]:
        return []

    def summarize(self, summary_id: str, original_ids: list[str]):
        pass

    def get_high_centrality_facts(self, min_load: int = 2) -> list[tuple]:
        return []

    def remove_contradicts(self, id_a: str, id_b: str):
        pass

    def save(self):
        data = {}
        for fid, fact in self.facts.items():
            d = asdict(fact)
            if d["embedding"] is not None:
                d["embedding"] = [float(x) for x in d["embedding"]]
            data[fid] = d
        self.path.write_text(json.dumps(data, indent=2))

    def load(self):
        raw = self.path.read_text()
        if not raw.strip():
            return
        data = json.loads(raw)
        for fid, d in data.items():
            self.facts[fid] = Fact(**d)
        self._invalidate_cache()

    def _get_embeddings_matrix(self):
        if self._embeddings_cache is not None:
            return self._embeddings_cache, self._ids_cache

        active = [
            (fid, f)
            for fid, f in self.facts.items()
            if f.embedding is not None and f.superseded_by is None
        ]
        if not active:
            return None, None

        self._ids_cache = [fid for fid, _ in active]
        self._embeddings_cache = np.array([f.embedding for _, f in active])
        return self._embeddings_cache, self._ids_cache

    def _invalidate_cache(self):
        self._embeddings_cache = None
        self._ids_cache = None

    def __len__(self):
        return len([f for f in self.facts.values() if f.superseded_by is None])
