#!/usr/bin/env python3
"""
run_v6.py — v6 benchmark runner.

Architecture: v4 + entity-centric retrieval for aggregation-intent queries.

Hypothesis: v4's MS audit showed 100% of wrong MS items had relevant sessions
retrieved. The answer step miscounts because it's given 40 clusters of mixed
relevance and has to find the ~4 truly relevant ones itself. Entity-centric
retrieval replaces that mixed pile with a tight entity-filtered set (~5-15
clusters, all mentioning the queried entity), reducing the LLM's enumeration
load.

For aggregation-intent queries:
  1. Extract entities from the question via spaCy noun chunks + lemmatization
  2. If ≥1 entity found, look up clusters in the entity index
  3. Use those clusters as retrieval (bypassing v4 hybrid similarity)
  4. Fall back to v4 hybrid if: no entities extracted, or entity lookup returns
     fewer than MIN_ENTITY_RESULTS clusters

For non-aggregation intents: v4 retrieval unchanged.

Ablation lever: --no-entity-index disables entity retrieval; reproduces v4.

Cost overhead: ~30 seconds for entity extraction across ~50 sessions per item.
No additional LLM calls. Total per-run: ~$15, same as v4.
"""

import sys
import os
import json
import time
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# Core infrastructure (v4 unchanged)
from membrane import Membrane, LocalEmbedder, AnthropicLLM
from run_benchmark import (
    index_item_clustered,
    _expand_cluster_neighbors,
    llm_judge,
    compare_results,
)

# v1 procedures
from procedures import (
    PROCEDURE_FOR_TYPE,
    ANSWER_SYSTEM_FALLBACK,
    build_prompt_header,
    anchored_session_ids,
)

# v4 decomposition
from decomposer import decompose, run_multi_query_retrieval

# v6: entity-centric retrieval
from entities import extract_entities, extract_from_batch
from entity_index import EntityIndex


_CLUSTERED_PER_INTENT_K = {
    "factual":           8,
    "temporal_adjacent": 12,
    "recency":           8,
    "aggregation":       30,
}
_CLUSTERED_MAX_PER_SESSION = 3

# v6 entity-retrieval tuning
MIN_ENTITY_RESULTS = 3      # fewer than this -> fall back to v4 hybrid
MAX_ENTITY_RESULTS = 40     # cap to avoid context explosion
MIN_MATCH_COUNT = 1         # cluster must match at least N query entities


def _build_entity_index(
    cluster_metadata: dict,
) -> tuple[EntityIndex, dict[str, set[str]]]:
    """
    After clusters are built, extract entities from each cluster's text.

    Returns:
      (EntityIndex, query_entity_cache_hint)

    cluster_metadata: {fact_id: {..., "pair_texts": list[str], ...}}
    """
    index = EntityIndex()

    fact_ids = list(cluster_metadata.keys())
    texts = [
        "\n".join(cluster_metadata[fid].get("pair_texts", []))
        for fid in fact_ids
    ]

    entity_sets = extract_from_batch(texts)
    if entity_sets is None:
        # spaCy unavailable — return empty index, caller will skip entity retrieval
        return index, {}

    per_fact_entities: dict[str, set[str]] = {}
    for fid, ents in zip(fact_ids, entity_sets):
        index.add(fid, ents)
        per_fact_entities[fid] = ents

    return index, per_fact_entities


def _inject_date_anchored_sessions(
    promoted: list,
    store,
    question: str,
    question_date: str,
    session_dates: dict,
    cluster_metadata: dict,
) -> list:
    """v4's date-based session expansion — only for TR."""
    target_sids = anchored_session_ids(
        question, question_date, session_dates, max_sessions=3
    )
    if not target_sids:
        return promoted

    promoted_ids = {f.id for f, _ in promoted}
    all_facts = {f.id: f for f in store.get_all_active()}
    by_session: dict[str, list[str]] = {}
    for fid, meta in cluster_metadata.items():
        by_session.setdefault(meta["session_id"], []).append(fid)

    result = list(promoted)
    for sid in target_sids:
        for fid in by_session.get(sid, []):
            if fid not in promoted_ids and fid in all_facts:
                result.append((all_facts[fid], 0.0))
                promoted_ids.add(fid)
    return result


def _merge_promoted(a: list, b: list) -> list:
    """Merge two (fact, score) lists, keeping best score per fact, dedup by id."""
    merged: dict[str, tuple] = {}
    for fact, score in a + b:
        if fact.id in merged:
            if score > merged[fact.id][1]:
                merged[fact.id] = (fact, score)
        else:
            merged[fact.id] = (fact, score)
    return sorted(merged.values(), key=lambda x: -x[1])


def _entity_retrieval(
    question: str,
    entity_index: EntityIndex,
    store,
) -> tuple[list, dict]:
    """
    Entity-driven retrieval for aggregation queries.

    Extracts entities from the question, looks them up in the index, returns
    matching clusters ranked by how many query entities they match.

    Returns:
      (results, info)
      results: list of (Fact, score) — score is match_count normalized to [0, 1]
      info: diagnostic dict — {"query_entities": set, "matched_fact_ids": list,
                                "fallback_reason": str|None}

    If fallback_reason is non-None, caller should use v4 hybrid instead.
    """
    info = {"query_entities": set(), "matched_fact_ids": [], "fallback_reason": None}

    query_entities = extract_entities(question)
    if query_entities is None:
        info["fallback_reason"] = "spacy_unavailable"
        return [], info

    info["query_entities"] = query_entities

    if not query_entities:
        info["fallback_reason"] = "no_entities_in_question"
        return [], info

    matches = entity_index.lookup(query_entities, min_matches=MIN_MATCH_COUNT)

    if len(matches) < MIN_ENTITY_RESULTS:
        info["fallback_reason"] = f"too_few_matches:{len(matches)}"
        return [], info

    # Cap the results
    matches = matches[:MAX_ENTITY_RESULTS]
    info["matched_fact_ids"] = [fid for fid, _ in matches]

    # Convert to (Fact, score) — normalize match count to [0, 1]
    all_facts = {f.id: f for f in store.get_all_active()}
    max_count = max(n for _, n in matches) if matches else 1
    results = []
    for fid, n in matches:
        fact = all_facts.get(fid)
        if fact is not None:
            score = n / max_count
            results.append((fact, score))

    return results, info


def process_item(
    item: dict,
    embedder,
    membrane_llm,
    decomposer_llm,
    answer_llm,
    judge_client,
    judge_model: str,
    enable_entity_index: bool = True,
) -> dict:
    """End-to-end pipeline for one item."""
    t0 = time.time()
    qid = item["question_id"]
    qtype = item["question_type"]
    question = item["question"]
    question_date = item["question_date"]
    ground_truth = item["answer"]

    # ── 1. Index ──
    store, _, cluster_meta = index_item_clustered(item, embedder, max_tokens=600)
    session_dates = dict(zip(item["haystack_session_ids"], item["haystack_dates"]))

    # ── 1b. Build entity index (v6 addition) ──
    entity_index = None
    entity_index_stats = None
    if enable_entity_index:
        entity_index, _ = _build_entity_index(cluster_meta)
        entity_index_stats = entity_index.stats()

    membrane = Membrane(
        store=store,
        llm=membrane_llm,
        embedder=embedder,
        promote_top_k=10,
        promote_threshold=0.0,
        min_promote=1,
        per_intent_k=_CLUSTERED_PER_INTENT_K,
        max_chunks_per_session=_CLUSTERED_MAX_PER_SESSION,
    )

    # ── 2. Decompose ──
    def haiku_raw(system, user_message, max_tokens=300):
        return decomposer_llm.complete(
            system=system, user_message=user_message, max_tokens=max_tokens
        )
    decomp_info = decompose(question, haiku_raw)

    # ── 3. Retrieval ──
    entity_info = None
    used_entity_path = False

    # Classify intent (this sets membrane._last_intent as a side effect of promote)
    primary = membrane.promote(question, session_dates=session_dates)
    intent = membrane._last_intent

    # v6 branch: aggregation intent AND entity index available -> try entity retrieval
    if enable_entity_index and entity_index is not None and intent == "aggregation":
        entity_results, entity_info = _entity_retrieval(question, entity_index, store)
        if entity_info["fallback_reason"] is None:
            # Entity retrieval succeeded — use its results as primary
            used_entity_path = True
            primary = entity_results
        # else: entity retrieval fell back, primary keeps v4 result

    # Adjacent-cluster expansion (unchanged from v4)
    primary = _expand_cluster_neighbors(
        promoted=primary,
        store=store,
        cluster_metadata=cluster_meta,
        intent=intent,
        qtype=qtype,
    )

    # Decomposition-driven retrieval union (unchanged from v4)
    if decomp_info.get("is_composite") and len(decomp_info.get("sub_queries", [])) > 1:
        decomposed_hits = run_multi_query_retrieval(
            sub_queries=decomp_info["sub_queries"],
            store=store,
            embedder=embedder,
            per_query_k=6,
            threshold=0.25,
        )
        promoted = _merge_promoted(primary, decomposed_hits)
    else:
        promoted = primary

    max_clusters = 40 if intent == "aggregation" else 20
    promoted = promoted[:max_clusters]

    if qtype == "temporal-reasoning":
        promoted = _inject_date_anchored_sessions(
            promoted, store, question, question_date, session_dates, cluster_meta
        )

    context = membrane.format_promoted(
        promoted, session_dates=session_dates, chunk_metadata=cluster_meta
    )

    # ── 4. Answer — v4's procedure prompts unchanged ──
    answer_system = PROCEDURE_FOR_TYPE.get(qtype, ANSWER_SYSTEM_FALLBACK)
    header = build_prompt_header(qtype, question_date, session_dates)

    user_msg = (
        f"{header}"
        f"{context}\n\n"
        f"Question (asked on {question_date}): {question}\n\n"
        "Execute the procedure above. Emit the structured intermediate "
        "output, then the answer."
    )

    hypothesis = answer_llm.complete(
        system=answer_system,
        user_message=user_msg,
        max_tokens=1500,
    )

    # ── 5. Judge ──
    try:
        correct = llm_judge(
            judge_client, judge_model, question, ground_truth, hypothesis
        )
    except Exception as e:
        correct = False
        hypothesis += f"\n\n[judge error: {e}]"

    elapsed = time.time() - t0
    return {
        "question_id": qid,
        "question_type": qtype,
        "question": question,
        "ground_truth": ground_truth,
        "hypothesis": hypothesis,
        "correct": bool(correct),
        "answer_session_ids": item.get("answer_session_ids", []),
        "elapsed_seconds": round(elapsed, 2),
        "retrieval_intent": intent,
        "n_clusters_promoted": len(promoted),
        "decomposition": {
            "is_composite": decomp_info.get("is_composite", False),
            "n_queries": len(decomp_info.get("sub_queries", [question])),
        },
        "used_entity_path": used_entity_path,
        "entity_info": (
            {
                "query_entities": sorted(entity_info["query_entities"]) if entity_info else [],
                "n_matched_facts": len(entity_info["matched_fact_ids"]) if entity_info else 0,
                "fallback_reason": entity_info["fallback_reason"] if entity_info else None,
            }
            if entity_info else None
        ),
        "entity_index_stats": entity_index_stats,
    }


def run_experiment(
    data: list,
    item_workers: int = 6,
    judge_model: str = "claude-haiku-4-5-20251001",
    membrane_model: str = "claude-haiku-4-5-20251001",
    decomposer_model: str = "claude-haiku-4-5-20251001",
    answer_model: str = "claude-sonnet-4-6",
    enable_entity_index: bool = True,
) -> dict:
    import anthropic

    judge_client = anthropic.Anthropic()
    embedder = LocalEmbedder()
    membrane_llm = AnthropicLLM(model=membrane_model)
    decomposer_llm = AnthropicLLM(model=decomposer_model)
    answer_llm = AnthropicLLM(model=answer_model)

    # Verify spaCy is loadable before starting (fail fast)
    if enable_entity_index:
        from entities import extract_entities
        test_result = extract_entities("How many workshops did I attend?")
        if test_result is None:
            print("\nERROR: spaCy model not available. Entity index disabled.")
            print("To enable entity retrieval, run:")
            print("  pip install spacy")
            print("  python -m spacy download en_core_web_sm")
            print("Continuing with --no-entity-index behavior...\n")
            enable_entity_index = False
        else:
            print(f"spaCy test extraction: {sorted(test_result) if test_result else '(empty)'}")

    print(f"\nv6 experiment on {len(data)} items")
    print(f"  entity_index: enabled={enable_entity_index}")
    print(f"  decompose:    {decomposer_model}")
    print(f"  answer:       {answer_model}")
    print(f"  workers:      {item_workers}")
    print()

    t0 = time.time()
    results: list = []
    correct_count = 0
    entity_path_count = 0
    done_count = 0
    lock = threading.Lock()

    def worker(it):
        return process_item(
            it, embedder, membrane_llm, decomposer_llm, answer_llm,
            judge_client, judge_model, enable_entity_index,
        )

    with ThreadPoolExecutor(max_workers=item_workers) as pool:
        futures = {pool.submit(worker, it): it for it in data}
        for fut in as_completed(futures):
            try:
                r = fut.result(timeout=300)
            except Exception as e:
                it = futures[fut]
                print(f"  [SKIP] {it['question_id']} -- {type(e).__name__}: {e}")
                r = {
                    "question_id": it["question_id"],
                    "question_type": it["question_type"],
                    "question": it["question"],
                    "ground_truth": it["answer"],
                    "hypothesis": f"[ERROR: {e}]",
                    "correct": False,
                    "used_entity_path": False,
                }
            with lock:
                done_count += 1
                correct_count += int(r["correct"])
                if r.get("used_entity_path"):
                    entity_path_count += 1
                results.append(r)
                mark = "v" if r["correct"] else "x"
                ent = "E" if r.get("used_entity_path") else "-"
                if done_count <= 5 or done_count % 25 == 0:
                    acc = correct_count / done_count * 100
                    print(f"  [{done_count:>3}/{len(data)}] {mark} {ent}  "
                          f"{r['question_type']:<32}  running acc={acc:.1f}%")

    order = {it["question_id"]: i for i, it in enumerate(data)}
    results.sort(key=lambda r: order[r["question_id"]])

    elapsed = time.time() - t0
    total = len(data)

    by_type: dict = {}
    for r in results:
        by_type.setdefault(r["question_type"], []).append(r["correct"])

    # Entity path stats
    entity_correct = sum(1 for r in results if r.get("used_entity_path") and r["correct"])
    entity_total = sum(1 for r in results if r.get("used_entity_path"))
    non_entity_correct = sum(1 for r in results if not r.get("used_entity_path") and r["correct"])
    non_entity_total = sum(1 for r in results if not r.get("used_entity_path"))

    # Fallback reason distribution
    fallback_reasons: dict = {}
    for r in results:
        ei = r.get("entity_info")
        if ei and ei.get("fallback_reason"):
            reason = ei["fallback_reason"].split(":")[0]  # strip numeric suffix
            fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1

    summary = {
        "mode": "v6_entity_index",
        "n_items": total,
        "elapsed_seconds": round(elapsed, 1),
        "indexing": "clustered(max_tokens=600) + entity_index",
        "answer_model": answer_model,
        "decomposer_model": decomposer_model,
        "membrane_model": membrane_model,
        "judge_model": judge_model,
        "entity_index_enabled": enable_entity_index,
        "overall": {
            "accuracy": round(correct_count / total * 100, 2) if total else 0.0,
            "correct": correct_count,
            "total": total,
        },
        "by_type": {
            qt: {
                "accuracy": round(sum(v) / len(v) * 100, 2),
                "correct": sum(v),
                "n": len(v),
            }
            for qt, v in sorted(by_type.items())
        },
        "entity_path_stats": {
            "entity_path_items": entity_total,
            "entity_path_correct": entity_correct,
            "entity_path_accuracy": round(entity_correct / entity_total * 100, 2) if entity_total else 0.0,
            "non_entity_items": non_entity_total,
            "non_entity_correct": non_entity_correct,
            "non_entity_accuracy": round(non_entity_correct / non_entity_total * 100, 2) if non_entity_total else 0.0,
            "fallback_reasons": fallback_reasons,
        },
        "item_results": results,
    }
    return summary


def print_summary(s: dict):
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  v6 ENTITY INDEX  ·  n={s['n_items']}  |  {s['elapsed_seconds']:.0f}s")
    print(sep)
    ov = s["overall"]
    print(f"\n  OVERALL  {ov['accuracy']:>6.2f}%  ({ov['correct']}/{ov['total']})")
    print(f"\n  BY TYPE")
    for qt, v in s["by_type"].items():
        print(f"    {qt:<32} {v['accuracy']:>7.2f}%  ({v['correct']}/{v['n']})")

    ep = s.get("entity_path_stats", {})
    print(f"\n  ENTITY PATH vs HYBRID PATH")
    print(f"    Entity-path items: {ep.get('entity_path_items', 0):>3}   "
          f"accuracy: {ep.get('entity_path_accuracy', 0):>6.2f}%  "
          f"({ep.get('entity_path_correct', 0)}/{ep.get('entity_path_items', 0)})")
    print(f"    Hybrid-path items: {ep.get('non_entity_items', 0):>3}   "
          f"accuracy: {ep.get('non_entity_accuracy', 0):>6.2f}%  "
          f"({ep.get('non_entity_correct', 0)}/{ep.get('non_entity_items', 0)})")

    fb = ep.get("fallback_reasons", {})
    if fb:
        print(f"\n  WHEN ENTITY PATH FELL BACK")
        for reason, count in sorted(fb.items(), key=lambda x: -x[1]):
            print(f"    {reason:<30} {count}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, help="Path to longmemeval_s_cleaned.json")
    ap.add_argument("--items", type=int, default=None)
    ap.add_argument("--output", type=str, default="v6_results.json")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-entity-index", action="store_true",
                    help="Disable entity-centric retrieval (reproduces v4)")
    ap.add_argument("--sonnet-model", type=str, default="claude-sonnet-4-6")
    ap.add_argument("--haiku-model", type=str, default="claude-haiku-4-5-20251001")
    ap.add_argument("--judge-model", type=str, default="claude-haiku-4-5-20251001")
    ap.add_argument("--compare", nargs=2, metavar=("FILE_A", "FILE_B"))
    args = ap.parse_args()

    if args.compare:
        compare_results(args.compare[0], args.compare[1])
        return

    if not args.data:
        ap.error("--data required unless using --compare")

    try:
        with open(args.data) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"\nERROR: Dataset JSON parse failure at byte {e.pos}")
        sys.exit(1)

    if args.items:
        data = data[:args.items]

    summary = run_experiment(
        data=data,
        item_workers=args.workers,
        judge_model=args.judge_model,
        membrane_model=args.haiku_model,
        decomposer_model=args.haiku_model,
        answer_model=args.sonnet_model,
        enable_entity_index=not args.no_entity_index,
    )

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print_summary(summary)
    print(f"\nResults saved -> {args.output}")


if __name__ == "__main__":
    main()
