#!/usr/bin/env python3
"""
run_decompose.py — v4 benchmark runner with question decomposition.

Architecture:
  1. Index sessions with clustered indexing (same as v1).
  2. Decompose the question into 1-5 atomic search sub-queries (cheap Haiku).
  3. Run each sub-query as an embedding search. Union the results.
  4. Apply cluster-neighbor expansion (same as v1).
  5. TR-only: inject date-anchored sessions (same as v1).
  6. Build context, call v1's procedure prompt, get answer.
  7. Judge.

The decomposition step is ADDITIVE:
  - Simple questions get 1 sub-query (equivalent to no decomposition).
  - Composite questions get 3-5 sub-queries, giving richer retrieval.
  - Everything downstream of retrieval is unchanged from v1.

If decomposition helps, we'll see it most on:
  - Multi-hop TR (which named events? → one query per event)
  - Aggregation MS with filters (total spent on X in last 4 months → X + cost + time)
  - Cross-attribute KU (age + event date)

Expected no-op or near-no-op on:
  - SSU (single factual lookup)
  - SSA (single factual lookup)
  - KU current-value (single attribute)
  - SSP (preference elicitation, not retrieval-bound)
"""

import sys
import os
import json
import time
import argparse
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "skills"))

# Core infrastructure
from membrane import Membrane, LocalEmbedder, AnthropicLLM
from run_benchmark import (
    index_item_clustered,
    _expand_cluster_neighbors,
    llm_judge,
    compare_results,
)

# v1 procedures — reuse the answer prompts verbatim
from procedures import (
    PROCEDURE_FOR_TYPE,
    ANSWER_SYSTEM_FALLBACK,
    build_prompt_header,
)

# v4 decomposition
from decomposer import decompose, run_multi_query_retrieval


# Per-intent k overrides identical to v1.
_CLUSTERED_PER_INTENT_K = {
    "factual":           8,
    "temporal_adjacent": 12,
    "recency":           8,
    "aggregation":       30,
}
_CLUSTERED_MAX_PER_SESSION = 3


def _inject_date_anchored_sessions(
    promoted: list,
    store,
    question: str,
    question_date: str,
    session_dates: dict,
    cluster_metadata: dict,
) -> list:
    """Reuse v1's date-based session expansion — only for TR."""
    try:
        from procedures import anchored_session_ids
    except ImportError:
        return promoted

    target_sids = anchored_session_ids(question, question_date, session_dates, max_sessions=3)
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


def process_item(
    item: dict,
    embedder,
    membrane_llm,
    decomposer_llm,
    answer_llm,
    judge_client,
    judge_model: str,
    enable_decompose: bool = True,
) -> dict:
    """
    End-to-end pipeline for one item.
    """
    t0 = time.time()
    qid = item["question_id"]
    qtype = item["question_type"]
    question = item["question"]
    question_date = item["question_date"]
    ground_truth = item["answer"]

    # ── 1. Index ──
    store, _, cluster_meta = index_item_clustered(item, embedder, max_tokens=600)
    session_dates = dict(zip(item["haystack_session_ids"], item["haystack_dates"]))

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

    # ── 2. Decompose (v4 addition) ──
    decomp_info = {"is_composite": False, "sub_queries": [question], "n_queries": 1}
    if enable_decompose:
        def haiku_raw(system, user_message, max_tokens=300):
            return decomposer_llm.complete(
                system=system, user_message=user_message, max_tokens=max_tokens
            )
        dinfo = decompose(question, haiku_raw)
        decomp_info.update({
            "is_composite": dinfo.get("is_composite", False),
            "sub_queries": dinfo.get("sub_queries", [question]),
            "n_queries": len(dinfo.get("sub_queries", [question])),
        })

    # ── 3. Retrieval ──
    # Always run v1's primary retrieval (intent-classified, MMR, neighbor
    # expansion). Then if decomposition gave us extra sub-queries, run
    # those as independent embedding searches and union.
    primary = membrane.promote(question, session_dates=session_dates)
    primary = _expand_cluster_neighbors(
        promoted=primary,
        store=store,
        cluster_metadata=cluster_meta,
        intent=membrane._last_intent,
        qtype=qtype,
    )

    if decomp_info["is_composite"] and len(decomp_info["sub_queries"]) > 1:
        # Decomposed queries — fire each as embedding search
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

    # Cap total clusters to prevent context blowup. v1 retrieves up to 40
    # for aggregation; we match that ceiling.
    max_clusters = 40 if membrane._last_intent == "aggregation" else 20
    promoted = promoted[:max_clusters]

    # TR date-anchored injection (same as v1)
    if qtype == "temporal-reasoning":
        promoted = _inject_date_anchored_sessions(
            promoted, store, question, question_date, session_dates, cluster_meta
        )

    # ── 4. Build context — identical to v1 ──
    context = membrane.format_promoted(
        promoted, session_dates=session_dates, chunk_metadata=cluster_meta
    )

    # ── 5. Answer — v1's procedure prompts ──
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

    # ── 6. Judge ──
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
        "decomposition": decomp_info,
        "n_clusters_promoted": len(promoted),
        "retrieval_intent": membrane._last_intent,
    }


def run_experiment(
    data: list,
    item_workers: int = 6,
    judge_model: str = "claude-haiku-4-5-20251001",
    membrane_model: str = "claude-haiku-4-5-20251001",
    decomposer_model: str = "claude-haiku-4-5-20251001",
    answer_model: str = "claude-sonnet-4-6",
    enable_decompose: bool = True,
) -> dict:
    import anthropic

    judge_client = anthropic.Anthropic()
    embedder = LocalEmbedder()
    membrane_llm = AnthropicLLM(model=membrane_model)
    decomposer_llm = AnthropicLLM(model=decomposer_model)
    answer_llm = AnthropicLLM(model=answer_model)

    print(f"\nv4 decomposition experiment on {len(data)} items")
    print(f"  decompose: {decomposer_model} (enabled={enable_decompose})")
    print(f"  answer:    {answer_model}")
    print(f"  membrane:  {membrane_model}")
    print(f"  judge:     {judge_model}")
    print(f"  workers:   {item_workers}")
    print()

    t0 = time.time()
    results: list = []
    correct_count = 0
    done_count = 0
    lock = threading.Lock()

    def worker(it):
        return process_item(
            it, embedder, membrane_llm, decomposer_llm, answer_llm,
            judge_client, judge_model, enable_decompose,
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
                    "decomposition": None,
                }
            with lock:
                done_count += 1
                correct_count += int(r["correct"])
                results.append(r)
                mark = "v" if r["correct"] else "x"
                dc = r.get("decomposition") or {}
                n_q = dc.get("n_queries", 1)
                comp = "C" if dc.get("is_composite") else "-"
                if done_count <= 5 or done_count % 25 == 0:
                    acc = correct_count / done_count * 100
                    print(f"  [{done_count:>3}/{len(data)}] {mark}  "
                          f"{r['question_type']:<32}  {comp}q={n_q}  "
                          f"running acc={acc:.1f}%")

    # Sort results into input order
    order = {it["question_id"]: i for i, it in enumerate(data)}
    results.sort(key=lambda r: order[r["question_id"]])

    elapsed = time.time() - t0
    total = len(data)

    by_type: dict = {}
    for r in results:
        by_type.setdefault(r["question_type"], []).append(r["correct"])

    # Decomposition stats
    decomp_stats = {
        "composite_total": sum(1 for r in results
                                if (r.get("decomposition") or {}).get("is_composite")),
        "avg_sub_queries": (
            sum((r.get("decomposition") or {}).get("n_queries", 1) for r in results)
            / max(total, 1)
        ),
        "composite_accuracy": {
            "composite_correct": sum(
                1 for r in results
                if (r.get("decomposition") or {}).get("is_composite") and r["correct"]
            ),
            "composite_total": sum(
                1 for r in results
                if (r.get("decomposition") or {}).get("is_composite")
            ),
            "simple_correct": sum(
                1 for r in results
                if not (r.get("decomposition") or {}).get("is_composite") and r["correct"]
            ),
            "simple_total": sum(
                1 for r in results
                if not (r.get("decomposition") or {}).get("is_composite")
            ),
        },
    }

    summary = {
        "mode": "v4_decompose",
        "n_items": total,
        "elapsed_seconds": round(elapsed, 1),
        "indexing": "clustered(max_tokens=600)",
        "answer_model": answer_model,
        "decomposer_model": decomposer_model,
        "membrane_model": membrane_model,
        "judge_model": judge_model,
        "decompose_enabled": enable_decompose,
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
        "decomposition_stats": decomp_stats,
        "item_results": results,
    }
    return summary


def print_summary(s: dict):
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  v4 DECOMPOSITION  ·  n={s['n_items']}  |  {s['elapsed_seconds']:.0f}s")
    print(sep)
    ov = s["overall"]
    print(f"\n  OVERALL  {ov['accuracy']:>6.2f}%  ({ov['correct']}/{ov['total']})")
    print(f"\n  BY TYPE")
    for qt, v in s["by_type"].items():
        print(f"    {qt:<32} {v['accuracy']:>7.2f}%  ({v['correct']}/{v['n']})")

    ds = s.get("decomposition_stats", {})
    ca = ds.get("composite_accuracy", {})
    ct = ca.get("composite_total", 0)
    st = ca.get("simple_total", 0)
    print(f"\n  DECOMPOSITION STATS")
    print(f"    Composite items: {ct}/{s['n_items']} ({ct/s['n_items']*100:.1f}%)")
    print(f"    Avg sub-queries: {ds.get('avg_sub_queries', 0):.2f}")
    if ct:
        print(f"    Composite accuracy: {ca['composite_correct']}/{ct} ({ca['composite_correct']/ct*100:.1f}%)")
    if st:
        print(f"    Simple accuracy:    {ca['simple_correct']}/{st} ({ca['simple_correct']/st*100:.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str,
                    help="Path to longmemeval_s_cleaned.json")
    ap.add_argument("--items", type=int, default=None,
                    help="Limit to first N items (smoke test)")
    ap.add_argument("--output", type=str, default="v4_results.json")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-decompose", action="store_true",
                    help="Disable decomposition (runs exactly like v1)")
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
        print(f"\nERROR: Dataset JSON could not be parsed at byte {e.pos}.")
        print(f"  line {e.lineno}, col {e.colno}")
        print(f"  The file '{args.data}' may be corrupted — try redownloading.")
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
        enable_decompose=not args.no_decompose,
    )

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print_summary(summary)
    print(f"\nResults saved -> {args.output}")


if __name__ == "__main__":
    main()
