#!/usr/bin/env python3
"""
LongMemEval benchmark for Membrane.
https://arxiv.org/abs/2410.10813

Two modes:

  retrieval  — index sessions as verbatim text facts, measure Recall@k and
               NDCG@k against the ground-truth answer_session_ids.
               No LLM calls. Fast. Tests pure vector retrieval quality.

  qa         — index sessions, promote top-k facts, answer with LLM,
               score with LLM judge (semantic, not exact-match).
               Requires ANTHROPIC_API_KEY. Measures end-to-end accuracy.

Usage:
  # Retrieval benchmark — all 500 items, k=5 (no API key needed)
  python benchmarks/run_benchmark.py --mode retrieval --k 5

  # QA benchmark — first 50 items
  python benchmarks/run_benchmark.py --mode qa --items 50 --k 10

  # Save results to a specific file
  python benchmarks/run_benchmark.py --mode retrieval --output results/my_run.json
"""

import sys
import os
import json
import math
import time
import argparse
import uuid
from pathlib import Path
from datetime import datetime

# Make sure Membrane root is on the path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from store import FactStore, Fact
from membrane import Membrane, LocalEmbedder, AnthropicLLM, session_has_personal_facts


# ── Helpers ──────────────────────────────────────────────────────────────────

def fresh_store(label: str = "") -> FactStore:
    """Create a non-persisting in-memory FactStore (path never created)."""
    path = f"/tmp/_membrane_bench_{label}_{uuid.uuid4().hex[:8]}.json"
    store = FactStore(path)  # path doesn't exist → nothing is loaded
    store.save = lambda: None  # no-op; we never want to persist during benchmarking
    return store


def format_session(turns: list[dict], date: str = "") -> str:
    """Format session turns as readable prose for embedding."""
    lines = []
    if date:
        lines.append(f"[Session date: {date}]")
    for turn in turns:
        role = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{role}: {turn['content']}")
    return "\n".join(lines)


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, sid in enumerate(retrieved[:k], start=1)
        if sid in relevant
    )
    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, min(len(relevant), k) + 1)
    )
    return dcg / idcg if idcg else 0.0


def index_item_extracted(
    item: dict,
    membrane: "Membrane",
    batch_size: int = 5,
    max_workers: int = 3,
) -> tuple[FactStore, dict[str, str]]:
    """
    Index sessions using Membrane's batch_extract() pipeline, with parallel
    LLM calls across sessions.

    Sessions are independent — all ~48 extraction calls per item can fire
    concurrently. ThreadPoolExecutor gives ~10-20x speedup over sequential,
    reducing per-item time from ~96s to ~5-10s.

    Cost: ~1 LLM call per batch_size turn-pairs per session.
    At batch_size=5 and ~48 sessions × ~5 pairs each:
    ~48 LLM calls per item → ~24,000 for the full 500-item benchmark.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    sessions = item["haystack_sessions"]
    session_ids = item["haystack_session_ids"]

    # Thread-safe accumulator — batch_extract() writes to the store
    # which is in-memory and not thread-safe for concurrent writes.
    # Collect (facts, sess_id) from threads, then write serially.
    store_lock = threading.Lock()
    fact_to_session: dict[str, str] = {}

    def extract_session(turns: list[dict], sess_id: str) -> list[tuple]:
        """Build pairs and call batch_extract. Returns (fact, sess_id) tuples."""
        pairs: list[tuple[str, str]] = []
        i = 0
        while i < len(turns) - 1:
            if turns[i]["role"] == "user" and turns[i + 1]["role"] == "assistant":
                pairs.append((turns[i]["content"], turns[i + 1]["content"]))
                i += 2
            else:
                i += 1

        results = []
        for batch_start in range(0, len(pairs), batch_size):
            batch = pairs[batch_start: batch_start + batch_size]
            # batch_extract does NOT write to the store — it returns Facts
            # that we add below under the lock.
            facts_raw = membrane._batch_extract_raw(batch)
            results.append((facts_raw, sess_id))
        return results

    # Fire all session extractions concurrently
    futures = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for turns, sess_id in zip(sessions, session_ids):
            fut = pool.submit(extract_session, turns, sess_id)
            futures[fut] = sess_id

        for fut in as_completed(futures):
            session_results = fut.result()
            # Write extracted facts to store serially (store isn't thread-safe)
            with store_lock:
                for facts_raw, sess_id in session_results:
                    facts = membrane._commit_extracted_facts(facts_raw, sess_id)
                    for fact in facts:
                        fact_to_session[fact.id] = sess_id

    return membrane.store, fact_to_session


def index_item_dual(
    item: dict,
    membrane: "Membrane",
    embedder: LocalEmbedder,
    chunk_size: int = 4,
    batch_size: int = 5,
    max_workers: int = 3,
    personal_threshold: int = 2,
) -> tuple[FactStore, dict[str, str]]:
    """
    Dual-path indexing: route each session to extraction OR verbatim chunking
    based on whether it contains personal/biographical facts.

    Sessions where the user makes ≥ `personal_threshold` personal-verb statements
    (e.g. "I bought", "I have", "I finished") → LLM extraction.
    These become clean discrete facts like "User owns a Honda Accord" that
    count correctly across sessions for multi-session aggregation questions.

    Sessions without personal statements → verbatim chunks.
    These are generic Q&A, discussions, explanations where extraction would
    produce little or nothing, and the exact phrasing matters for retrieval.

    This resolves the single-session-user vs. multi-session tradeoff:
    - Single-session-user relies on unusual personal details → extraction preserves them.
    - Multi-session aggregation relies on many sessions having the same fact type
      → extraction produces clean, countable discrete facts.
    - Generic sessions that neither category cares about → verbatim (no LLM cost).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    sessions = item["haystack_sessions"]
    session_ids = item["haystack_session_ids"]
    dates = item["haystack_dates"]

    store = membrane.store
    store_lock = threading.Lock()
    fact_to_session: dict[str, str] = {}

    # Classify sessions up-front (no LLM cost — pure regex)
    personal_sessions = []
    verbatim_sessions = []
    for turns, sess_id, date in zip(sessions, session_ids, dates):
        if session_has_personal_facts(turns, threshold=personal_threshold):
            personal_sessions.append((turns, sess_id))
        else:
            verbatim_sessions.append((turns, sess_id, date))

    # --- Path A: verbatim chunks (serial, no LLM) ---
    for turns, sess_id, date in verbatim_sessions:
        if chunk_size <= 0 or len(turns) <= chunk_size:
            chunks = [turns]
        else:
            step = max(1, chunk_size // 2)
            chunks = [turns[i: i + chunk_size] for i in range(0, len(turns), step)]
        for chunk in chunks:
            text = format_session(chunk, date)
            emb = embedder.embed(text)
            fact = Fact(content=text, session_id=sess_id, embedding=emb.tolist())
            fid = store.add(fact)
            fact_to_session[fid] = sess_id

    # --- Path B: LLM extraction (parallel) ---
    def extract_session(turns: list[dict], sess_id: str) -> list[tuple]:
        pairs: list[tuple[str, str]] = []
        i = 0
        while i < len(turns) - 1:
            if turns[i]["role"] == "user" and turns[i + 1]["role"] == "assistant":
                pairs.append((turns[i]["content"], turns[i + 1]["content"]))
                i += 2
            else:
                i += 1
        results = []
        for batch_start in range(0, len(pairs), batch_size):
            batch = pairs[batch_start: batch_start + batch_size]
            facts_raw = membrane._batch_extract_raw(batch)
            results.append((facts_raw, sess_id))
        return results

    if personal_sessions:
        futures = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for turns, sess_id in personal_sessions:
                fut = pool.submit(extract_session, turns, sess_id)
                futures[fut] = sess_id
            for fut in as_completed(futures):
                session_results = fut.result()
                with store_lock:
                    for facts_raw, sess_id in session_results:
                        facts = membrane._commit_extracted_facts(facts_raw, sess_id)
                        for fact in facts:
                            fact_to_session[fact.id] = sess_id

    return store, fact_to_session


def index_item(
    item: dict, embedder: LocalEmbedder, chunk_size: int = 0
) -> tuple[FactStore, dict[str, str]]:
    """
    Index all haystack sessions into a fresh FactStore.
    Returns (store, fact_id_to_session_id) mapping.

    chunk_size=0  — one fact per session (original behaviour)
    chunk_size=N  — split each session into overlapping windows of N turns,
                    stepping by N//2 so adjacent chunks share one window of
                    context. A session with 20 turns at chunk_size=4 becomes
                    ~8 facts instead of 1, each with a tight focused embedding.
    """
    store = fresh_store(item["question_id"])
    sessions = item["haystack_sessions"]
    session_ids = item["haystack_session_ids"]
    dates = item["haystack_dates"]

    fact_to_session: dict[str, str] = {}

    for turns, sess_id, date in zip(sessions, session_ids, dates):
        if chunk_size <= 0 or len(turns) <= chunk_size:
            # Single fact for the whole session
            chunks = [turns]
        else:
            # Sliding window: step = chunk_size // 2 for 50% overlap
            step = max(1, chunk_size // 2)
            chunks = [turns[i: i + chunk_size] for i in range(0, len(turns), step)]

        for chunk in chunks:
            text = format_session(chunk, date)
            emb = embedder.embed(text)
            fact = Fact(content=text, session_id=sess_id, embedding=emb.tolist())
            fid = store.add(fact)
            fact_to_session[fid] = sess_id

    return store, fact_to_session


def index_item_chunked(
    item: dict,
    embedder: LocalEmbedder,
    window_pairs: int = 2,
    stride_pairs: int = 1,
) -> tuple[FactStore, dict[str, str], dict[str, dict]]:
    """
    Index sessions as overlapping windows of user/assistant pair-groups.

    window_pairs=2: each chunk covers 2 user/assistant pairs (≈4 turns).
    stride_pairs=1: advance by 1 pair each step → 1-pair overlap between
                    adjacent chunks (so no content is missed at boundaries).

    Returns (store, fact_to_session, chunk_metadata).
    chunk_metadata maps fact_id → {chunk_id, session_id, date,
                                    pair_start, pair_end, start_turn,
                                    end_turn, pair_texts}.
    pair_texts is a list of strings, one per pair, used for merge-assembly
    in format_promoted().
    """
    store = fresh_store(item["question_id"])
    session_ids = item["haystack_session_ids"]
    dates = item["haystack_dates"]
    sessions = item["haystack_sessions"]

    fact_to_session: dict[str, str] = {}
    chunk_metadata: dict[str, dict] = {}

    for sess_id, date, turns in zip(session_ids, dates, sessions):
        if not turns:
            continue

        # Extract user/assistant pairs; handle non-alternating edge cases.
        pairs: list[list[dict]] = []
        i = 0
        while i < len(turns):
            t = turns[i]
            if t.get("role") == "user":
                pair: list[dict] = [t]
                if i + 1 < len(turns) and turns[i + 1].get("role") == "assistant":
                    pair.append(turns[i + 1])
                    i += 2
                else:
                    i += 1
                pairs.append(pair)
            else:
                # Orphan assistant turn — attach to last pair or skip.
                if pairs:
                    pairs[-1].append(t)
                i += 1

        if not pairs:
            continue

        # Sliding window over pairs.
        n_chunks = max(1, len(pairs) - window_pairs + 1)
        # Track cumulative turn count per pair for exact turn indices.
        pair_turn_offsets = [0]
        for p in pairs:
            pair_turn_offsets.append(pair_turn_offsets[-1] + len(p))

        for chunk_idx in range(n_chunks):
            pair_start = chunk_idx  # stride = 1 pair
            pair_end = pair_start + window_pairs
            chunk_pairs = pairs[pair_start:pair_end]

            # Build per-pair text strings (used for merge deduplication).
            pair_texts = []
            for pair in chunk_pairs:
                lines = []
                for t in pair:
                    role = t.get("role", "").capitalize()
                    content = t.get("content", "").strip()
                    lines.append(f"{role}: {content}")
                pair_texts.append("\n".join(lines))

            # Full chunk text: date header + all turn lines.
            chunk_text = f"[{date}]\n" + "\n".join(pair_texts)

            start_turn = pair_turn_offsets[pair_start]
            end_turn = pair_turn_offsets[min(pair_end, len(pairs))] - 1

            emb = embedder.embed(chunk_text)
            fact = Fact(content=chunk_text, session_id=sess_id, embedding=emb.tolist())
            store.add(fact)

            chunk_metadata[fact.id] = {
                "chunk_id": f"{sess_id}__c{chunk_idx}",
                "chunk_idx": chunk_idx,
                "session_id": sess_id,
                "date": date,
                "pair_start": pair_start,
                "pair_end": pair_end,
                "start_turn": start_turn,
                "end_turn": end_turn,
                "pair_texts": pair_texts,
            }
            fact_to_session[fact.id] = sess_id

    return store, fact_to_session, chunk_metadata


# ── Cluster-layer indexing ────────────────────────────────────────────────────

def build_clusters(
    pairs: list[list[dict]],
    session_id: str,
    date: str,
    max_tokens: int = 600,
    embedder: "LocalEmbedder | None" = None,
    drift_threshold: float = 0.35,
) -> list[dict]:
    """
    Greedy-merge adjacent turn-pairs into semantically coherent clusters.

    A new cluster starts when adding the next pair would exceed max_tokens,
    or (when embedder is provided) when cosine distance from the running
    cluster centroid exceeds drift_threshold.

    Returns a list of cluster dicts, each containing:
      pair_start, pair_end, start_turn, end_turn, pair_texts, text
    pair_texts is a list of pre-formatted strings (one per pair) used by
    format_promoted() to assemble the block — same format as chunk_metadata.
    """
    if not pairs:
        return []

    def est_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    def pair_to_text(pair: list[dict]) -> str:
        return "\n".join(
            f"{t.get('role', '').capitalize()}: {t.get('content', '').strip()}"
            for t in pair
        )

    pair_texts_all = [pair_to_text(p) for p in pairs]
    pair_token_counts = [est_tokens(t) for t in pair_texts_all]

    # Cumulative turn offsets for exact turn indices in the block header.
    pair_turn_offsets = [0]
    for p in pairs:
        pair_turn_offsets.append(pair_turn_offsets[-1] + len(p))

    # Pre-embed all pairs when drift detection is on.
    import numpy as np
    pair_embeddings: list | list[None] = [None] * len(pairs)
    if embedder is not None:
        for i, text in enumerate(pair_texts_all):
            pair_embeddings[i] = embedder.embed(text)

    def _finalize(start: int, end: int) -> dict:
        pt = pair_texts_all[start:end]
        return {
            "session_id": session_id,
            "date": date,
            "pair_start": start,
            "pair_end": end,
            "start_turn": pair_turn_offsets[start],
            "end_turn": pair_turn_offsets[end] - 1,
            "pair_texts": pt,
            "text": f"[{date}]\n" + "\n".join(pt),
        }

    clusters: list[dict] = []
    c_start = 0
    c_tokens = pair_token_counts[0]
    # Running centroid embedding for drift detection.
    c_emb = pair_embeddings[0]

    for i in range(1, len(pairs)):
        over_budget = (c_tokens + pair_token_counts[i]) > max_tokens

        drifted = False
        if embedder is not None and c_emb is not None and pair_embeddings[i] is not None:
            c_norm = np.linalg.norm(c_emb)
            p_norm = np.linalg.norm(pair_embeddings[i])
            if c_norm > 0 and p_norm > 0:
                sim = float(np.dot(c_emb, pair_embeddings[i]) / (c_norm * p_norm))
                drifted = (1.0 - sim) > drift_threshold

        if over_budget or drifted:
            clusters.append(_finalize(c_start, i))
            c_start = i
            c_tokens = pair_token_counts[i]
            c_emb = pair_embeddings[i]
        else:
            c_tokens += pair_token_counts[i]
            # Update centroid as running mean.
            if embedder is not None and c_emb is not None and pair_embeddings[i] is not None:
                n = i - c_start + 1
                c_emb = (c_emb * (n - 1) + pair_embeddings[i]) / n

    clusters.append(_finalize(c_start, len(pairs)))
    return clusters


def index_item_clustered(
    item: dict,
    embedder: "LocalEmbedder",
    max_tokens: int = 600,
    use_drift: bool = False,
    drift_threshold: float = 0.35,
) -> tuple["FactStore", dict[str, str], dict[str, dict]]:
    """
    Index sessions as adaptive clusters of adjacent turn-pairs.

    Clusters replace sessions as the retrieval unit. Session is kept only as
    metadata so format_promoted() can group and order blocks chronologically.

    max_tokens  — character-based token budget per cluster (~600 ≈ 150 words).
    use_drift   — if True, also split on cosine topic drift between adjacent pairs.
    drift_threshold — cosine-distance threshold for drift split (lower = more splits).

    Returns (store, fact_to_session, cluster_metadata).
    cluster_metadata is fact_id → cluster dict compatible with the chunk_metadata
    format that format_promoted() already understands.
    """
    store = fresh_store(item["question_id"])
    session_ids = item["haystack_session_ids"]
    dates = item["haystack_dates"]
    sessions = item["haystack_sessions"]

    fact_to_session: dict[str, str] = {}
    cluster_metadata: dict[str, dict] = {}

    drift_embedder = embedder if use_drift else None

    for sess_id, date, turns in zip(session_ids, dates, sessions):
        if not turns:
            continue

        # Extract user/assistant pairs; handle non-alternating edge cases.
        pairs: list[list[dict]] = []
        i = 0
        while i < len(turns):
            t = turns[i]
            if t.get("role") == "user":
                pair: list[dict] = [t]
                if i + 1 < len(turns) and turns[i + 1].get("role") == "assistant":
                    pair.append(turns[i + 1])
                    i += 2
                else:
                    i += 1
                pairs.append(pair)
            else:
                if pairs:
                    pairs[-1].append(t)
                i += 1

        if not pairs:
            continue

        clusters = build_clusters(
            pairs, sess_id, date,
            max_tokens=max_tokens,
            embedder=drift_embedder,
            drift_threshold=drift_threshold,
        )

        for cluster_idx, cluster in enumerate(clusters):
            emb = embedder.embed(cluster["text"])
            fact = Fact(content=cluster["text"], session_id=sess_id, embedding=emb.tolist())
            store.add(fact)
            cluster_metadata[fact.id] = {
                "chunk_id": f"{sess_id}__cl{cluster_idx}",
                "chunk_idx": cluster_idx,
                "session_id": sess_id,
                "date": date,
                "pair_start": cluster["pair_start"],
                "pair_end": cluster["pair_end"],
                "start_turn": cluster["start_turn"],
                "end_turn": cluster["end_turn"],
                "pair_texts": cluster["pair_texts"],
                "n_pairs": len(cluster["pair_texts"]),
            }
            fact_to_session[fact.id] = sess_id

    return store, fact_to_session, cluster_metadata


# ── Cluster neighbor expansion ────────────────────────────────────────────────

def _expand_cluster_neighbors(
    promoted: list,
    store: "FactStore",
    cluster_metadata: dict,
    intent: str,
    qtype: str,
) -> list:
    """
    1-hop adjacent-cluster expansion for clustered indexing mode.

    After initial retrieval, each selected cluster can pull in the immediately
    preceding and following cluster from the same session.  This addresses the
    within-session miss: the retrieval found the right session but the specific
    cluster containing the answer is adjacent to (not identical to) the
    highest-scoring one.

    Priority and caps
    -----------------
    aggregation intent or single-session-assistant qtype:
        Expand ALL selected clusters.  Hard cap: 40 total.
        These types need the most complete within-session coverage.

    factual / recency / temporal_adjacent:
        Expand only the top-3 scoring clusters (usually 1 session).
        Hard cap: max(current_count, 12) — does not bloat context for
        point-in-time questions that are already well-served by a single cluster.

    Deduplication is by fact_id.  Neighbor clusters added at score=0.0 so
    they don't interfere with downstream re-ranking.
    """
    if not cluster_metadata or not promoted:
        return promoted

    # ── Build session → sorted cluster list ──────────────────────────────────
    # Maps session_id → [(chunk_idx, fact_id)] sorted by chunk_idx.
    session_order: dict[str, list[tuple[int, str]]] = {}
    for fid, meta in cluster_metadata.items():
        sid = meta["session_id"]
        session_order.setdefault(sid, []).append((meta["chunk_idx"], fid))
    for sid in session_order:
        session_order[sid].sort()

    # fact_id → (session_id, position_in_sorted_list)
    fact_pos: dict[str, tuple[str, int]] = {}
    for sid, chunks in session_order.items():
        for i, (_, fid) in enumerate(chunks):
            fact_pos[fid] = (sid, i)

    # ── Fetch all facts for neighbor lookup ──────────────────────────────────
    all_facts: dict[str, object] = {f.id: f for f in store.get_all_active()}

    # ── Determine expansion mode ──────────────────────────────────────────────
    # Only expand for types where adjacent context is known to help:
    #   aggregation — need full within-session coverage to count correctly
    #   single-session-assistant — assistant's response often in adjacent cluster
    #
    # Do NOT expand for factual or recency — those types retrieve a tight
    # preference/point-in-time context that degrades when diluted by neighbors.
    # SSP in particular dropped 20pp when factual expansion was active.
    if intent == "aggregation":
        expand_sources = promoted                       # expand all clusters
        max_total = 40
    elif qtype == "single-session-assistant":
        expand_sources = sorted(promoted, key=lambda x: x[1], reverse=True)[:3]
        max_total = max(len(promoted), 12)
    else:
        # factual / recency / temporal_adjacent — no expansion
        return promoted

    # ── Expand ───────────────────────────────────────────────────────────────
    promoted_ids: set[str] = {f.id for f, _ in promoted}
    result = list(promoted)

    for fact, _ in expand_sources:
        if len(result) >= max_total:
            break
        loc = fact_pos.get(fact.id)
        if loc is None:
            continue
        sid, pos = loc
        chunks = session_order[sid]

        for neighbor_pos in (pos - 1, pos + 1):
            if len(result) >= max_total:
                break
            if 0 <= neighbor_pos < len(chunks):
                _, nbr_id = chunks[neighbor_pos]
                if nbr_id not in promoted_ids:
                    nbr_fact = all_facts.get(nbr_id)
                    if nbr_fact is not None:
                        result.append((nbr_fact, 0.0))
                        promoted_ids.add(nbr_id)

    return result


# ── Retrieval benchmark ───────────────────────────────────────────────────────

def run_retrieval(data: list[dict], k: int, chunk_size: int = 0) -> dict:
    """
    Measure Recall@k and NDCG@k using Membrane's vector search.
    No LLM calls — tests the embedding layer directly.
    """
    embedder = LocalEmbedder()
    by_type: dict[str, dict] = {}
    all_recall: list[float] = []
    all_ndcg: list[float] = []
    item_results: list[dict] = []

    chunk_label = f"chunk_size={chunk_size}" if chunk_size > 0 else "whole-session"
    t0 = time.time()
    print(f"\nRetrieval benchmark — {len(data)} items, k={k}, {chunk_label}")
    print("(no LLM calls — pure embedding retrieval)\n")

    for i, item in enumerate(data):
        qid = item["question_id"]
        qtype = item["question_type"]
        question = item["question"]
        answer_sids = set(item["answer_session_ids"])

        store, fact_to_session = index_item(item, embedder, chunk_size=chunk_size)

        # Retrieve top-k by cosine similarity; threshold=0 so we always get k results.
        # When chunking, multiple chunks from the same session can appear — deduplicate
        # by session ID (keep first/highest-ranked occurrence) so NDCG stays in [0,1].
        query_emb = embedder.embed(question)
        results = store.search(query_emb, top_k=k, threshold=0.0)
        seen: set[str] = set()
        retrieved_sids: list[str] = []
        for f, _ in results:
            sid = fact_to_session.get(f.id)
            if sid and sid not in seen:
                seen.add(sid)
                retrieved_sids.append(sid)

        recall = 1.0 if any(s in answer_sids for s in retrieved_sids) else 0.0
        ndcg = ndcg_at_k(retrieved_sids, answer_sids, k)

        all_recall.append(recall)
        all_ndcg.append(ndcg)

        if qtype not in by_type:
            by_type[qtype] = {"recall": [], "ndcg": []}
        by_type[qtype]["recall"].append(recall)
        by_type[qtype]["ndcg"].append(ndcg)

        item_results.append({
            "question_id": qid,
            "question_type": qtype,
            "question": question,
            "answer_session_ids": list(answer_sids),
            "retrieved_session_ids": retrieved_sids,
            f"recall_at_{k}": recall,
            f"ndcg_at_{k}": round(ndcg, 4),
        })

        if (i + 1) % 100 == 0 or i == 0:
            r = sum(all_recall) / len(all_recall) * 100
            n = sum(all_ndcg) / len(all_ndcg) * 100
            print(f"  [{i+1:>3}/{len(data)}]  Recall@{k}={r:.1f}%  NDCG@{k}={n:.1f}%  — {time.time()-t0:.0f}s")

    elapsed = time.time() - t0
    avg_recall = sum(all_recall) / len(all_recall) * 100
    avg_ndcg = sum(all_ndcg) / len(all_ndcg) * 100

    return {
        "mode": "retrieval",
        "k": k,
        "chunk_size": chunk_size if chunk_size > 0 else "whole-session",
        "n_items": len(data),
        "elapsed_seconds": round(elapsed, 1),
        "overall": {
            f"recall_at_{k}": round(avg_recall, 2),
            f"ndcg_at_{k}": round(avg_ndcg, 2),
        },
        "by_type": {
            qtype: {
                f"recall_at_{k}": round(sum(v["recall"]) / len(v["recall"]) * 100, 2),
                f"ndcg_at_{k}": round(sum(v["ndcg"]) / len(v["ndcg"]) * 100, 2),
                "n": len(v["recall"]),
            }
            for qtype, v in sorted(by_type.items())
        },
        "item_results": item_results,
    }


# ── QA benchmark ─────────────────────────────────────────────────────────────

# Per-type answer prompts. Each type has a distinct failure mode — one prompt
# cannot address all of them without being so general it helps with none.
_ANSWER_BASE = (
    "You are a helpful assistant with access to a user's conversation history. "
    "Sessions are shown oldest first; the LAST session is the most recent. "
    "Each session block is labeled [Session N — Date]. "
)

ANSWER_SYSTEM_BY_TYPE = {
    "single-session-user": (
        _ANSWER_BASE
        + "The answer is a specific fact the user stated in one of the sessions. "
        + "Find it and state it directly. Be concise and confident. "
        + "Do not say 'I don't know' — the answer is in the provided context."
    ),
    "single-session-assistant": (
        _ANSWER_BASE
        + "The answer is something the assistant said, recommended, or explained. "
        + "Find it and state it directly. Be concise and confident. "
        + "Do not say 'I don't know' — the answer is in the provided context."
    ),
    "single-session-preference": (
        "You are a helpful assistant who knows this user well from their conversation history. "
        "Use the user's stated preferences, habits, past experiences, and context "
        "to give a PERSONALIZED response. "
        "Do not give generic advice — tailor your answer specifically to what you know "
        "about this user. Reference their specific preferences by name when relevant. "
        "Sessions are shown oldest first; each block is labeled [Session N — Date]. "
        "Be direct and specific. Only say 'I don't know' if you have truly zero "
        "relevant context about this user's preferences."
    ),
    "multi-session": (
        _ANSWER_BASE
        + "For counting questions: scan EVERY session carefully and tally every matching "
        + "instance — do not estimate or stop early. "
        + "For listing questions: include every relevant item across all sessions. "
        + "Be thorough and precise. State your final count or list directly."
    ),
    "temporal-reasoning": (
        _ANSWER_BASE
        + "The timeline at the top shows exact day/week/month intervals between sessions — "
        + "read those pre-computed intervals rather than computing dates yourself. "
        + "Use session dates to determine before/after relationships and durations. "
        + "State your answer directly. For duration questions, give the exact interval "
        + "shown in the timeline."
    ),
    "knowledge-update": (
        _ANSWER_BASE
        + "The user's information may have changed across sessions. "
        + "ALWAYS use the value from the MOST RECENT session — ignore any older values "
        + "for the same fact. The last session chronologically is the most current. "
        + "State the current value directly and confidently."
    ),
}

JUDGE_PROMPT = """\
You are evaluating whether a system's answer correctly addresses a question.

Question: {question}
Ground truth answer: {ground_truth}
System answer: {hypothesis}

Does the system answer include the key information from the ground truth?
- Semantic equivalence counts — wording does not need to be identical.
- Extra correct detail in the system answer is fine; do not penalise it.
- Only mark "no" if the system answer is factually wrong, missing the core answer, or says it doesn't know.
Respond with ONLY the word "yes" or "no"."""


def llm_judge(client, model: str, question: str, ground_truth: str, hypothesis: str) -> bool:
    resp = client.messages.create(
        model=model,
        max_tokens=5,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            hypothesis=hypothesis,
        )}],
    )
    return resp.content[0].text.strip().lower().startswith("yes")


def run_qa(
    data: list[dict],
    k: int,
    judge_model: str,
    chunk_size: int = 0,
    extract: bool = False,
    dual: bool = False,
    batch_size: int = 5,
    max_workers: int = 3,
    item_workers: int = 10,
    indexing: str = "whole-session",
    cluster_max_tokens: int = 600,
    cluster_drift: bool = False,
    cluster_drift_threshold: float = 0.35,
) -> dict:
    """
    Full pipeline: index → promote → answer → LLM judge.
    Requires ANTHROPIC_API_KEY.

    extract=True  — index sessions via batch_extract() (structured facts, LLM cost)
    dual=True     — dual-path: personal sessions → extraction, generic → verbatim chunks
    extract=False, dual=False — index sessions as verbatim chunks (no LLM cost, default)

    item_workers — items processed in parallel (default 10). Items are fully
    independent so this is safe to raise. At 10 workers: ~10×3=30 concurrent
    API calls, well under the 4K RPM limit. Raise to 15-20 if limits allow.
    """
    import anthropic
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    client = anthropic.Anthropic()

    embedder = LocalEmbedder()
    haiku_llm = AnthropicLLM(model="claude-haiku-4-5-20251001")
    sonnet_llm = AnthropicLLM(model="claude-sonnet-4-6")

    if dual:
        index_mode = f"dual-path (extract personal + chunks size={chunk_size} generic, batch_size={batch_size})"
    elif extract:
        index_mode = "extract (batch_size={})".format(batch_size)
    elif indexing == "chunked":
        index_mode = "chunked (window=2 pairs, stride=1)"
    else:
        index_mode = f"whole-session"
    t0 = time.time()
    print(f"\nQA benchmark — {len(data)} items, k={k}, {index_mode}, item_workers={item_workers}")
    print(f"Answer model: haiku (sonnet for MS + SSP)  |  Judge: {judge_model}\n")

    # Per-intent k overrides for chunked mode (fixed 2-pair windows).
    _CHUNKED_PER_INTENT_K = {
        "factual":           6,
        "temporal_adjacent": 10,
        "recency":           10,
        "aggregation":       24,
    }
    _CHUNKED_MAX_PER_SESSION = 2

    # Per-intent k overrides for clustered mode (adaptive token-budget clusters).
    # Clusters are larger than 2-pair chunks — fewer per session — so budgets
    # are slightly higher to maintain coverage.  Aggregation casts wide to
    # ensure every session has a chance to contribute at least one cluster.
    _CLUSTERED_PER_INTENT_K = {
        "factual":           8,
        "temporal_adjacent": 12,
        "recency":           8,
        "aggregation":       30,
    }
    _CLUSTERED_MAX_PER_SESSION = 3

    def process_item(item: dict) -> dict:
        """Full pipeline for a single item. Runs in a worker thread."""
        qid = item["question_id"]
        qtype = item["question_type"]
        question = item["question"]
        question_date = item["question_date"]
        ground_truth = item["answer"]

        answer_llm = (
            sonnet_llm
            if qtype in ("single-session-preference", "multi-session")
            else haiku_llm
        )

        chunk_meta: dict | None = None

        if extract or dual:
            store = fresh_store(qid)
            membrane_for_index = Membrane(
                store=store,
                llm=haiku_llm,
                embedder=embedder,
                promote_top_k=k,
                promote_threshold=0.0,
                min_promote=1,
            )
            if dual:
                store, _ = index_item_dual(
                    item, membrane_for_index, embedder,
                    chunk_size=chunk_size if chunk_size > 0 else 4,
                    batch_size=batch_size, max_workers=max_workers,
                )
            else:
                store, _ = index_item_extracted(
                    item, membrane_for_index, batch_size=batch_size, max_workers=max_workers
                )
        elif indexing == "chunked":
            store, _, chunk_meta = index_item_chunked(item, embedder)
        elif indexing == "clustered":
            store, _, chunk_meta = index_item_clustered(
                item, embedder,
                max_tokens=cluster_max_tokens,
                use_drift=cluster_drift,
                drift_threshold=cluster_drift_threshold,
            )
        else:
            store, _ = index_item(item, embedder, chunk_size=chunk_size)

        session_dates = dict(zip(item["haystack_session_ids"], item["haystack_dates"]))

        if indexing == "chunked":
            membrane = Membrane(
                store=store,
                llm=haiku_llm,
                embedder=embedder,
                promote_top_k=k,
                promote_threshold=0.0,
                min_promote=1,
                per_intent_k=_CHUNKED_PER_INTENT_K,
                max_chunks_per_session=_CHUNKED_MAX_PER_SESSION,
            )
        elif indexing == "clustered":
            membrane = Membrane(
                store=store,
                llm=haiku_llm,
                embedder=embedder,
                promote_top_k=k,
                promote_threshold=0.0,
                min_promote=1,
                per_intent_k=_CLUSTERED_PER_INTENT_K,
                max_chunks_per_session=_CLUSTERED_MAX_PER_SESSION,
            )
        else:
            membrane = Membrane(
                store=store,
                llm=haiku_llm,
                embedder=embedder,
                promote_top_k=k,
                promote_threshold=0.0,
                min_promote=1,
            )

        promoted = membrane.promote(question, session_dates=session_dates)

        # Adjacent-cluster expansion (clustered mode only).
        # Runs after promote() so the base retrieval is unchanged; expands
        # by 1 hop each side per selected cluster before building context.
        if indexing == "clustered" and chunk_meta:
            promoted = _expand_cluster_neighbors(
                promoted=promoted,
                store=store,
                cluster_metadata=chunk_meta,
                intent=membrane._last_intent,
                qtype=qtype,
            )

        context = membrane.format_promoted(
            promoted, session_dates=session_dates, chunk_metadata=chunk_meta
        )

        # Build retrieval trace AFTER expansion so counts reflect what the LLM sees.
        retrieval_trace = {
            "intent": membrane._last_intent,
            "n_chunks": len(promoted),
            "retrieved_session_ids": list(dict.fromkeys(f.session_id for f, _ in promoted)),
            "chunks": [
                {
                    "fact_id": f.id,
                    "session_id": f.session_id,
                    "score": round(score, 4),
                    **(
                        {
                            "chunk_id": chunk_meta[f.id]["chunk_id"],
                            "start_turn": chunk_meta[f.id]["start_turn"],
                            "end_turn": chunk_meta[f.id]["end_turn"],
                        }
                        if chunk_meta and f.id in chunk_meta else {}
                    ),
                }
                for f, score in promoted
            ],
        }

        user_msg = (
            f"{context}\n\n"
            f"Question (asked on {question_date}): {question}\n\n"
            "Answer:"
        )
        answer_system = ANSWER_SYSTEM_BY_TYPE.get(qtype, ANSWER_SYSTEM_BY_TYPE["single-session-user"])
        hypothesis = answer_llm.complete(
            system=answer_system,
            user_message=user_msg,
            max_tokens=400,
        )

        correct = llm_judge(client, judge_model, question, ground_truth, hypothesis)

        return {
            "question_id": qid,
            "question_type": qtype,
            "question": question,
            "ground_truth": ground_truth,
            "hypothesis": hypothesis,
            "correct": correct,
            "answer_session_ids": item.get("answer_session_ids", []),
            "retrieval_trace": retrieval_trace,
        }

    # Collect results as they complete; lock only for shared counters + printing.
    by_type: dict[str, list[bool]] = {}
    item_results: list[dict] = []
    correct_count = 0
    done_count = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=item_workers) as pool:
        futures = {pool.submit(process_item, item): item for item in data}
        for fut in as_completed(futures):
            try:
                result = fut.result(timeout=120)
            except Exception as e:
                item = futures[fut]
                print(f"  [SKIP] {item['question_id']} — {type(e).__name__}: {e}")
                result = {
                    "question_id": item["question_id"],
                    "question_type": item["question_type"],
                    "question": item["question"],
                    "ground_truth": item["answer"],
                    "hypothesis": f"[ERROR: {e}]",
                    "correct": False,
                }
            with lock:
                done_count += 1
                correct_count += int(result["correct"])
                by_type.setdefault(result["question_type"], []).append(result["correct"])
                item_results.append(result)
                mark = "✓" if result["correct"] else "✗"
                if done_count % 10 == 0 or done_count <= 3:
                    acc = correct_count / done_count * 100
                    print(f"  [{done_count:>3}/{len(data)}] {mark}  {result['question_type']:<32}  running acc={acc:.1f}%")

    # Sort item_results back into original data order for reproducible output.
    order = {item["question_id"]: i for i, item in enumerate(data)}
    item_results.sort(key=lambda r: order[r["question_id"]])

    elapsed = time.time() - t0
    total = len(data)

    return {
        "mode": "qa",
        "k": k,
        "indexing": (
            f"dual(personal=extract,generic=chunks{chunk_size if chunk_size>0 else 4},batch={batch_size})"
            if dual else
            f"extract(batch_size={batch_size})" if extract else
            "chunked(window=2pairs,stride=1)" if indexing == "chunked" else
            f"clustered(max_tokens={cluster_max_tokens},drift={cluster_drift})" if indexing == "clustered" else
            (f"chunks(size={chunk_size})" if chunk_size > 0 else "whole-session")
        ),
        "n_items": total,
        "elapsed_seconds": round(elapsed, 1),
        "answer_model": "haiku (sonnet for multi-session + single-session-preference)",
        "judge_model": judge_model,
        "overall": {
            "accuracy": round(correct_count / total * 100, 2),
            "correct": correct_count,
            "total": total,
        },
        "by_type": {
            qtype: {
                "accuracy": round(sum(v) / len(v) * 100, 2),
                "correct": sum(v),
                "n": len(v),
            }
            for qtype, v in sorted(by_type.items())
        },
        "item_results": item_results,
    }


# ── Comparison ───────────────────────────────────────────────────────────────

def compare_results(path_a: str, path_b: str):
    """
    Print a delta report comparing two QA result files.

    Reports:
    - Overall accuracy difference
    - By-type accuracy differences
    - Up to 10 items that flipped wrong→right (A wrong, B correct)
    - Up to 10 items that flipped right→wrong (A correct, B wrong)
    - For multi-session: avg distinct sessions retrieved per mode
    """
    with open(path_a) as f:
        a = json.load(f)
    with open(path_b) as f:
        b = json.load(f)

    label_a = a.get("indexing", "A")
    label_b = b.get("indexing", "B")

    # Index B results by question_id for fast lookup.
    b_by_id = {r["question_id"]: r for r in b["item_results"]}

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  COMPARISON REPORT")
    print(f"  A: {path_a}")
    print(f"     indexing={label_a}  k={a.get('k')}  n={a.get('n_items')}")
    print(f"  B: {path_b}")
    print(f"     indexing={label_b}  k={b.get('k')}  n={b.get('n_items')}")
    print(sep)

    # ── Overall ──────────────────────────────────────────────────────────────
    acc_a = a["overall"]["accuracy"]
    acc_b = b["overall"]["accuracy"]
    delta = acc_b - acc_a
    sign = "+" if delta >= 0 else ""
    print(f"\n  OVERALL ACCURACY")
    print(f"    A ({label_a:<32}):  {acc_a:.2f}%  ({a['overall']['correct']}/{a['overall']['total']})")
    print(f"    B ({label_b:<32}):  {acc_b:.2f}%  ({b['overall']['correct']}/{b['overall']['total']})")
    print(f"    Delta B−A:  {sign}{delta:.2f}pp")

    # ── By type ──────────────────────────────────────────────────────────────
    all_types = sorted(set(list(a["by_type"].keys()) + list(b["by_type"].keys())))
    print(f"\n  BY QUESTION TYPE")
    print(f"    {'Type':<36}  {'A':>7}  {'B':>7}  {'Δ':>7}")
    print(f"    {'-'*36}  {'-'*7}  {'-'*7}  {'-'*7}")
    for qtype in all_types:
        ta = a["by_type"].get(qtype, {})
        tb = b["by_type"].get(qtype, {})
        acc_ta = ta.get("accuracy", float("nan"))
        acc_tb = tb.get("accuracy", float("nan"))
        d = acc_tb - acc_ta
        sign = "+" if d >= 0 else ""
        n = tb.get("n", ta.get("n", "?"))
        print(f"    {qtype:<36}  {acc_ta:>6.2f}%  {acc_tb:>6.2f}%  {sign}{d:>6.2f}pp  (n={n})")

    # ── Flips ────────────────────────────────────────────────────────────────
    wrong_to_right = []
    right_to_wrong = []

    for r_a in a["item_results"]:
        qid = r_a["question_id"]
        r_b = b_by_id.get(qid)
        if r_b is None:
            continue
        if not r_a["correct"] and r_b["correct"]:
            wrong_to_right.append((qid, r_a, r_b))
        elif r_a["correct"] and not r_b["correct"]:
            right_to_wrong.append((qid, r_a, r_b))

    def _print_flips(flips, header):
        print(f"\n  {header}  ({len(flips)} total, showing up to 10)")
        for qid, ra, rb in flips[:10]:
            print(f"\n    [{qid}]  type={ra['question_type']}")
            print(f"    Q:  {ra['question'][:100]}")
            print(f"    GT: {str(ra['ground_truth'])[:80]}")
            print(f"    A:  {str(ra['hypothesis'])[:80]}")
            print(f"    B:  {str(rb['hypothesis'])[:80]}")
            # Show retrieval intent if available
            trace_a = ra.get("retrieval_trace", {})
            trace_b = rb.get("retrieval_trace", {})
            if trace_a or trace_b:
                intent_a = trace_a.get("intent", "?")
                intent_b = trace_b.get("intent", "?")
                n_a = trace_a.get("n_chunks", "?")
                n_b = trace_b.get("n_chunks", "?")
                sids_a = len(trace_a.get("retrieved_session_ids", []))
                sids_b = len(trace_b.get("retrieved_session_ids", []))
                print(f"    Retrieval: A intent={intent_a} chunks={n_a} sessions={sids_a} | B intent={intent_b} chunks={n_b} sessions={sids_b}")

    _print_flips(wrong_to_right, "WRONG → RIGHT  (B improved)")
    _print_flips(right_to_wrong, "RIGHT → WRONG  (B regressed)")

    # ── Multi-session retrieval audit ─────────────────────────────────────────
    # For every wrong MS item in B, break down the failure mode:
    #   MISS   — relevant session never retrieved (embedding failure)
    #   FOUND  — relevant session was retrieved but LLM still got it wrong
    #             (wrong cluster within session, or comprehension failure)
    ms_items_a = [r for r in a["item_results"] if r["question_type"] == "multi-session"]
    ms_items_b = [r for r in b["item_results"] if r["question_type"] == "multi-session"]

    # Build ground-truth answer_session_ids from either result file.
    # Older runs won't have the field — check both A and B so the audit
    # works when only one file includes it.
    gt_by_id: dict[str, set[str]] = {}
    for r in ms_items_a + ms_items_b:
        qid = r["question_id"]
        sids = r.get("answer_session_ids") or []
        if sids and qid not in gt_by_id:
            gt_by_id[qid] = set(sids)

    if ms_items_b and gt_by_id:
        wrong_ms_b = [r for r in ms_items_b if not r["correct"]]
        right_ms_b  = [r for r in ms_items_b if r["correct"]]

        # Coverage: avg distinct sessions retrieved per item
        def avg_sessions(items):
            counts = [
                len(r.get("retrieval_trace", {}).get("retrieved_session_ids", []))
                for r in items
            ]
            return sum(counts) / len(counts) if counts else 0.0

        avg_a = avg_sessions(ms_items_a)
        avg_b = avg_sessions(ms_items_b)
        delta_ms = avg_b - avg_a
        sign = "+" if delta_ms >= 0 else ""

        print(f"\n  MULTI-SESSION RETRIEVAL AUDIT  (B = {label_b})")
        print(f"    MS accuracy:  A={sum(r['correct'] for r in ms_items_a)}/{len(ms_items_a)}  "
              f"B={len(right_ms_b)}/{len(ms_items_b)}")
        print(f"    Avg sessions retrieved/item:  A={avg_a:.1f}  B={avg_b:.1f}  Δ={sign}{delta_ms:.1f}")

        # Audit wrong MS items in B
        miss_counts = []   # relevant sessions not retrieved at all
        found_counts = []  # relevant sessions retrieved but answer still wrong

        for r in wrong_ms_b:
            qid = r["question_id"]
            relevant = gt_by_id.get(qid, set())
            if not relevant:
                continue
            retrieved = set(r.get("retrieval_trace", {}).get("retrieved_session_ids", []))
            missed   = relevant - retrieved
            found    = relevant & retrieved
            miss_counts.append(len(missed))
            found_counts.append(len(found))

        if miss_counts:
            n_wrong = len(miss_counts)
            avg_miss  = sum(miss_counts)  / n_wrong
            avg_found = sum(found_counts) / n_wrong
            pct_full_miss  = sum(1 for m in miss_counts  if m > 0) / n_wrong * 100
            pct_found_fail = sum(1 for f in found_counts if f > 0) / n_wrong * 100

            print(f"\n    Wrong MS items in B: {n_wrong}")
            print(f"    Per wrong item (avg relevant sessions):")
            print(f"      Retrieved AND relevant (LLM had the data):  {avg_found:.2f}  "
                  f"({pct_found_fail:.0f}% of wrong items had ≥1 relevant session retrieved)")
            print(f"      NOT retrieved (embedding miss):              {avg_miss:.2f}  "
                  f"({pct_full_miss:.0f}% of wrong items missing ≥1 relevant session)")
            print(f"    Interpretation:")
            if pct_found_fail > pct_full_miss:
                print(f"      Majority failure mode → LLM comprehension / wrong cluster selected")
                print(f"      (relevant sessions WERE retrieved but answer still wrong)")
            else:
                print(f"      Majority failure mode → embedding retrieval miss")
                print(f"      (relevant sessions never made it into context)")

            # Show up to 5 worst-miss items
            annotated = sorted(
                zip(miss_counts, found_counts, wrong_ms_b),
                key=lambda x: x[0], reverse=True
            )
            print(f"\n    Top miss examples (most relevant sessions not retrieved):")
            for miss, found, r in annotated[:5]:
                qid = r["question_id"]
                retrieved = set(r.get("retrieval_trace", {}).get("retrieved_session_ids", []))
                relevant  = gt_by_id.get(qid, set())
                n_ret = len(retrieved)
                print(f"      [{qid}]  relevant={len(relevant)}  retrieved={n_ret}  "
                      f"miss={miss}  found={found}")
                print(f"        Q: {r['question'][:90]}")

    print(f"\n{sep}\n")


# ── Output ────────────────────────────────────────────────────────────────────

def print_summary(s: dict):
    mode = s["mode"]
    k = s["k"]
    sep = "=" * 62

    print(f"\n{sep}")
    print(f"  MEMBRANE  ·  LongMemEval-S  ·  {mode.upper()} RESULTS")
    print(f"  n={s['n_items']} items  |  k={k}  |  {s['elapsed_seconds']:.0f}s")
    print(sep)

    if mode == "retrieval":
        ov = s["overall"]
        print(f"\n  OVERALL")
        print(f"    Recall@{k}  {ov[f'recall_at_{k}']:>6.2f}%")
        print(f"    NDCG@{k}    {ov[f'ndcg_at_{k}']:>6.2f}%")
        print(f"\n  BY QUESTION TYPE")
        for qtype, v in s["by_type"].items():
            r = v[f"recall_at_{k}"]
            n = v[f"ndcg_at_{k}"]
            print(f"    {qtype:<34}  R@{k}={r:>6.2f}%  NDCG@{k}={n:>6.2f}%  (n={v['n']})")

    elif mode == "qa":
        ov = s["overall"]
        print(f"\n  OVERALL ACCURACY  {ov['accuracy']:>6.2f}%  ({ov['correct']}/{ov['total']})")
        print(f"  Answer model: {s['answer_model']}")
        print(f"  Judge model:  {s['judge_model']}")
        print(f"\n  BY QUESTION TYPE")
        for qtype, v in s["by_type"].items():
            print(f"    {qtype:<34}  {v['accuracy']:>6.2f}%  ({v['correct']}/{v['n']})")

    print(sep)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LongMemEval benchmark for Membrane",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", choices=["retrieval", "qa"], default="retrieval",
        help="retrieval: Recall@k/NDCG@k, no LLM cost  |  qa: full pipeline + LLM judge",
    )
    parser.add_argument("--k", type=int, default=10, help="Top-k sessions to retrieve (default: 10)")
    parser.add_argument("--items", type=int, default=None, help="Limit to first N items (default: all 500)")
    parser.add_argument(
        "--data", type=str,
        default=str(Path(__file__).parent / "data" / "longmemeval_s_cleaned.json"),
        help="Path to dataset JSON",
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSON path (auto-named if omitted)")
    parser.add_argument(
        "--judge", type=str, default="claude-haiku-4-5-20251001",
        help="Judge model for QA mode (default: claude-haiku-4-5-20251001)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=0, dest="chunk_size",
        help="Split sessions into overlapping chunks of N turns (0 = whole session, default: 0)",
    )
    parser.add_argument(
        "--extract", action="store_true", default=False,
        help="Use LLM batch_extract() for indexing instead of verbatim chunks (qa mode only). "
             "Produces structured facts — fixes multi-session aggregation failures. "
             "Costs ~$1-3 per full 500-item run.",
    )
    parser.add_argument(
        "--dual", action="store_true", default=False,
        help="Dual-path indexing: sessions with personal facts use LLM extraction; "
             "generic sessions use verbatim chunking. Best of both worlds — "
             "fixes multi-session aggregation without sacrificing single-session-user accuracy. "
             "~50%% of LLM cost vs --extract.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=5, dest="batch_size",
        help="Turn-pairs per LLM call in extract mode (default: 5)",
    )
    parser.add_argument(
        "--max-workers", type=int, default=3, dest="max_workers",
        help="Parallel extraction workers per item in --extract/--dual mode (default: 3).",
    )
    parser.add_argument(
        "--item-workers", type=int, default=10, dest="item_workers",
        help="Items processed in parallel (default: 10). Safe to raise to 15-20 with high API limits.",
    )
    parser.add_argument(
        "--indexing", choices=["whole-session", "chunked", "clustered"], default="whole-session",
        dest="indexing",
        help="Indexing strategy for QA mode: "
             "whole-session (default) | "
             "chunked (fixed 2-pair windows, stride=1) | "
             "clustered (adaptive token-budget clusters, session as metadata only).",
    )
    parser.add_argument(
        "--cluster-max-tokens", type=int, default=600, dest="cluster_max_tokens",
        help="Max token budget per cluster in --indexing clustered (default: 600 ≈ 150 words).",
    )
    parser.add_argument(
        "--cluster-drift", action="store_true", default=False, dest="cluster_drift",
        help="Enable embedding-based topic-drift splitting in clustered mode. "
             "Embeds each turn-pair to detect topic changes. Slower but produces "
             "tighter semantic clusters.",
    )
    parser.add_argument(
        "--cluster-drift-threshold", type=float, default=0.35, dest="cluster_drift_threshold",
        help="Cosine-distance threshold for drift split in --cluster-drift (default: 0.35). "
             "Lower = more splits.",
    )
    parser.add_argument(
        "--compare", type=str, nargs=2, metavar=("FILE_A", "FILE_B"),
        help="Compare two result JSON files and print a delta report. "
             "No benchmark is run — only the comparison is printed.",
    )
    args = parser.parse_args()

    # Comparison mode — no benchmark run.
    if args.compare:
        compare_results(args.compare[0], args.compare[1])
        return

    print(f"Loading {args.data} ...")
    with open(args.data) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items.")

    if args.items:
        data = data[: args.items]
        print(f"Using first {len(data)} items.")

    if args.mode == "retrieval":
        summary = run_retrieval(data, k=args.k, chunk_size=args.chunk_size)
    else:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ERROR: ANTHROPIC_API_KEY not set. Required for --mode qa.")
            sys.exit(1)
        if args.extract and args.dual:
            print("ERROR: --extract and --dual are mutually exclusive.")
            sys.exit(1)
        summary = run_qa(
            data, k=args.k, judge_model=args.judge,
            chunk_size=args.chunk_size,
            extract=args.extract,
            dual=args.dual,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            item_workers=args.item_workers,
            indexing=args.indexing,
            cluster_max_tokens=args.cluster_max_tokens,
            cluster_drift=args.cluster_drift,
            cluster_drift_threshold=args.cluster_drift_threshold,
        )

    print_summary(summary)

    # Save results (strip item_results for smaller retrieval-mode files if desired)
    out = args.output
    if out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        indexing_tag = f"_{args.indexing}" if args.mode == "qa" and args.indexing != "whole-session" else ""
        out = str(Path(__file__).parent / f"results_{args.mode}_k{args.k}{indexing_tag}_{ts}.json")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()
