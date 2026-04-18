#!/usr/bin/env python3
"""
run_procedures.py — plug program-shaped per-category prompts into the
existing clustered retrieval pipeline.

This is NOT a new retrieval system. The clustered baseline at 75.8% is
retrieving correctly in ~95% of failure cases — the audits confirmed this.
The bottleneck is what happens AFTER retrieval: the answer model fails at
enumeration (MS undercounts), picks stale values (KU), and computes wrong
intervals (TR). We replace the "you are a helpful assistant" answer prompt
with a per-category program-shaped template that forces the model to emit
structured intermediate output, from which the answer is a mechanical
function.

Plus two small additive fixes:
  - Temporal anchor tables for TR (no more mental arithmetic)
  - Date-based session expansion for TR retrieval-miss items
  - Optional Haiku-cheap MS count verifier (A/B via --verify-ms)

Usage:
  python run_procedures.py --data path/to/longmemeval_s_cleaned.json
  python run_procedures.py --data ... --items 50
  python run_procedures.py --data ... --verify-ms
  python run_procedures.py --compare procedures.json clustered_75_8.json
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

# Reuse existing infrastructure — do not reimplement retrieval
from membrane import Membrane, LocalEmbedder, AnthropicLLM
from run_benchmark import (
    index_item_clustered,
    _expand_cluster_neighbors,
    llm_judge,
    compare_results,
)

from procedures import (
    PROCEDURE_FOR_TYPE,
    ANSWER_SYSTEM_FALLBACK,
    build_prompt_header,
    anchored_session_ids,
    verify_ms_answer,
)


# Per-intent k overrides identical to the clustered baseline at 75.8%.
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
    session_dates: dict[str, str],
    cluster_metadata: dict,
) -> list:
    """
    For TR questions containing a temporal anchor ("two weeks ago"),
    find sessions close to the implied date and ensure their clusters
    are in the promoted set. Additive — never removes anything.
    """
    target_sids = anchored_session_ids(question, question_date, session_dates)
    if not target_sids:
        return promoted

    promoted_ids = {f.id for f, _ in promoted}
    all_facts = {f.id: f for f in store.get_all_active()}

    # For each target session, pull all its clusters from cluster_metadata
    # and add those whose fact_ids aren't already promoted.
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


def process_item(
    item: dict,
    embedder,
    membrane_llm,
    answer_llm,
    verifier_llm,
    judge_client,
    judge_model: str,
    k: int,
    use_verifier: bool,
) -> dict:
    """
    End-to-end pipeline for one item:
      index (clustered) -> retrieve (identical to 75.8% baseline)
      -> build procedure-shaped prompt -> answer -> optionally verify -> judge
    """
    qid = item["question_id"]
    qtype = item["question_type"]
    question = item["question"]
    question_date = item["question_date"]
    ground_truth = item["answer"]

    # ── 1. Clustered indexing — identical to run_benchmark.py baseline ──
    store, _, cluster_meta = index_item_clustered(item, embedder, max_tokens=600)
    session_dates = dict(zip(item["haystack_session_ids"], item["haystack_dates"]))

    membrane = Membrane(
        store=store,
        llm=membrane_llm,
        embedder=embedder,
        promote_top_k=k,
        promote_threshold=0.0,
        min_promote=1,
        per_intent_k=_CLUSTERED_PER_INTENT_K,
        max_chunks_per_session=_CLUSTERED_MAX_PER_SESSION,
    )
    promoted = membrane.promote(question, session_dates=session_dates)
    promoted = _expand_cluster_neighbors(
        promoted=promoted,
        store=store,
        cluster_metadata=cluster_meta,
        intent=membrane._last_intent,
        qtype=qtype,
    )

    # ── 2. TR-specific: date-anchored session expansion ──
    if qtype == "temporal-reasoning":
        promoted = _inject_date_anchored_sessions(
            promoted=promoted,
            store=store,
            question=question,
            question_date=question_date,
            session_dates=session_dates,
            cluster_metadata=cluster_meta,
        )

    # ── 3. Build context — identical format to baseline ──
    context = membrane.format_promoted(
        promoted, session_dates=session_dates, chunk_metadata=cluster_meta
    )

    # ── 4. Pick procedure by question_type ──
    answer_system = PROCEDURE_FOR_TYPE.get(qtype, ANSWER_SYSTEM_FALLBACK)

    # ── 5. For TR, prepend the pre-computed anchor tables ──
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

    # ── 6. MS verifier (optional) ──
    verified = False
    if use_verifier and qtype == "multi-session":
        ok, hypothesis = verify_ms_answer(verifier_llm, hypothesis)
        verified = not ok

    # ── 7. Judge ──
    correct = llm_judge(
        judge_client, judge_model, question, ground_truth, hypothesis
    )

    return {
        "question_id": qid,
        "question_type": qtype,
        "question": question,
        "ground_truth": ground_truth,
        "hypothesis": hypothesis,
        "correct": correct,
        "answer_session_ids": item.get("answer_session_ids", []),
        "retrieval_trace": {
            "intent": membrane._last_intent,
            "n_chunks": len(promoted),
            "retrieved_session_ids": list(dict.fromkeys(f.session_id for f, _ in promoted)),
        },
        "verifier_corrected": verified,
    }


def run(
    data: list[dict],
    judge_model: str,
    k: int = 10,
    item_workers: int = 10,
    use_verifier: bool = False,
    answer_model: str = "claude-sonnet-4-6",
    membrane_model: str = "claude-haiku-4-5-20251001",
    verifier_model: str = "claude-haiku-4-5-20251001",
) -> dict:
    import anthropic

    judge_client = anthropic.Anthropic()
    embedder = LocalEmbedder()
    membrane_llm = AnthropicLLM(model=membrane_model)
    answer_llm = AnthropicLLM(model=answer_model)
    verifier_llm = AnthropicLLM(model=verifier_model)

    print(f"\nProcedure-prompts experiment on {len(data)} items")
    print(f"  answer: {answer_model}  |  membrane: {membrane_model}  |  judge: {judge_model}")
    print(f"  verifier enabled: {use_verifier}  |  k={k}")
    print()

    t0 = time.time()
    results: list[dict] = []
    correct_count = 0
    done_count = 0
    lock = threading.Lock()

    def worker(it):
        return process_item(
            it, embedder, membrane_llm, answer_llm, verifier_llm,
            judge_client, judge_model, k, use_verifier,
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
                }
            with lock:
                done_count += 1
                correct_count += int(r["correct"])
                results.append(r)
                mark = "v" if r["correct"] else "x"
                if done_count <= 5 or done_count % 25 == 0:
                    acc = correct_count / done_count * 100
                    print(f"  [{done_count:>3}/{len(data)}] {mark}  "
                          f"{r['question_type']:<32}  running acc={acc:.1f}%")

    order = {it["question_id"]: i for i, it in enumerate(data)}
    results.sort(key=lambda r: order[r["question_id"]])

    elapsed = time.time() - t0
    total = len(data)

    by_type: dict[str, list[bool]] = {}
    for r in results:
        by_type.setdefault(r["question_type"], []).append(r["correct"])

    summary = {
        "mode": "procedures",
        "k": k,
        "n_items": total,
        "elapsed_seconds": round(elapsed, 1),
        "indexing": "clustered(max_tokens=600)",
        "answer_model": answer_model,
        "membrane_model": membrane_model,
        "judge_model": judge_model,
        "verifier_enabled": use_verifier,
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
        "item_results": results,
    }
    return summary


def print_summary(s: dict):
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  PROCEDURE-PROMPTS  ·  n={s['n_items']}  |  {s['elapsed_seconds']:.0f}s")
    print(f"  verifier: {s['verifier_enabled']}")
    print(sep)
    ov = s["overall"]
    print(f"\n  OVERALL  {ov['accuracy']:>6.2f}%  ({ov['correct']}/{ov['total']})")
    print(f"\n  BY TYPE")
    for qt, v in s["by_type"].items():
        print(f"    {qt:<32}  {v['accuracy']:>6.2f}%  ({v['correct']}/{v['n']})")
    print(sep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None,
                        help="Path to LongMemEval-S dataset JSON")
    parser.add_argument("--items", type=int, default=None,
                        help="Limit to first N items")
    parser.add_argument("--types", nargs="+", default=None,
                        help="Filter to specific question_types (optional)")
    parser.add_argument("--judge", type=str, default="claude-haiku-4-5-20251001")
    parser.add_argument("--answer-model", type=str, default="claude-sonnet-4-6",
                        dest="answer_model")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--item-workers", type=int, default=10, dest="item_workers")
    parser.add_argument("--verify-ms", action="store_true", default=False,
                        dest="verify_ms",
                        help="Enable the Haiku-cheap MS count verifier pass.")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--compare", type=str, nargs=2, metavar=("A", "B"),
                        help="Compare two result JSON files.")
    args = parser.parse_args()

    if args.compare:
        compare_results(args.compare[0], args.compare[1])
        return

    if not args.data:
        print("ERROR: --data is required.")
        sys.exit(1)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    with open(args.data) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} total items.")

    if args.types:
        data = [it for it in data if it["question_type"] in args.types]
        print(f"Filtered to {len(data)} items matching types={args.types}.")

    if args.items:
        data = data[: args.items]
        print(f"Limited to {len(data)} items.")

    summary = run(
        data,
        judge_model=args.judge,
        k=args.k,
        item_workers=args.item_workers,
        use_verifier=args.verify_ms,
        answer_model=args.answer_model,
    )

    print_summary(summary)

    if args.output is None:
        suffix = "_verify" if args.verify_ms else ""
        args.output = f"procedures_result{suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved -> {args.output}")


if __name__ == "__main__":
    main()
