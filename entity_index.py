"""
entity_index.py — Reverse index mapping entities to fact_ids.

At ingest time, each cluster's content is passed through the entity extractor.
The resulting entities are stored on the cluster AND added to a reverse index
keyed by entity string.

At query time, entities are extracted from the question. For each entity, the
reverse index returns the set of clusters that mention it. Results are ranked
by how many query entities each cluster matched (more matches = higher rank).

This is the "precision lever" for aggregation queries:
  v4 aggregation retrieval: ~40 clusters from hybrid search, mixed relevance
  v6 entity-driven retrieval: ~5-15 clusters that all mention the queried entity

The LLM's counting task stops being "find relevant items in a noisy pile" and
becomes "count items in a pre-filtered list."
"""

from __future__ import annotations
from collections import defaultdict


class EntityIndex:
    """
    Maps entity strings to sets of fact_ids. Symmetric: each fact_id stores
    its entity set, each entity stores its fact_id set.

    Memory overhead: for ~50 sessions of ~10 clusters each with ~20 entities
    per cluster = ~10K entity-fact edges. Negligible (~100KB in Python dicts).
    """

    def __init__(self):
        # entity (str) -> set of fact_ids
        self._by_entity: dict[str, set[str]] = defaultdict(set)
        # fact_id (str) -> set of entities
        self._by_fact: dict[str, set[str]] = defaultdict(set)

    def add(self, fact_id: str, entities: set[str]) -> None:
        """Register a cluster's entities. Idempotent."""
        if not entities:
            return
        self._by_fact[fact_id].update(entities)
        for ent in entities:
            self._by_entity[ent].add(fact_id)

    def entities_for(self, fact_id: str) -> set[str]:
        return self._by_fact.get(fact_id, set())

    def facts_for(self, entity: str) -> set[str]:
        return self._by_entity.get(entity, set())

    def lookup(
        self,
        query_entities: set[str],
        min_matches: int = 1,
    ) -> list[tuple[str, int]]:
        """
        Given entities extracted from a query, return fact_ids ranked by how
        many query entities they match.

        Args:
          query_entities: the entity set from the question
          min_matches: minimum number of entities a fact must match to be returned

        Returns:
          list of (fact_id, match_count) sorted by match_count descending.
          Empty list if query_entities is empty or no facts match.
        """
        if not query_entities:
            return []

        # Count matches per fact
        counts: dict[str, int] = defaultdict(int)
        for ent in query_entities:
            for fact_id in self._by_entity.get(ent, ()):
                counts[fact_id] += 1

        # Filter by min_matches and sort
        results = [
            (fid, n) for fid, n in counts.items() if n >= min_matches
        ]
        results.sort(key=lambda x: -x[1])
        return results

    def stats(self) -> dict:
        """For diagnostic printing during smoke tests."""
        n_entities = len(self._by_entity)
        n_facts = len(self._by_fact)
        if n_facts:
            avg_per_fact = sum(len(e) for e in self._by_fact.values()) / n_facts
        else:
            avg_per_fact = 0.0
        if n_entities:
            avg_per_entity = sum(len(f) for f in self._by_entity.values()) / n_entities
        else:
            avg_per_entity = 0.0
        return {
            "n_unique_entities": n_entities,
            "n_facts": n_facts,
            "avg_entities_per_fact": round(avg_per_fact, 2),
            "avg_facts_per_entity": round(avg_per_entity, 2),
        }

    def most_common_entities(self, n: int = 20) -> list[tuple[str, int]]:
        """
        Return the top-N most common entities. Useful for sanity-checking
        extraction quality — if top entities are "user" and "thing", the
        stopword filter is failing.
        """
        counts = [(ent, len(facts)) for ent, facts in self._by_entity.items()]
        counts.sort(key=lambda x: -x[1])
        return counts[:n]
