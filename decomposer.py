"""
decomposer.py — question decomposition for retrieval enrichment.

Core idea: when a question is composite (multi-hop or aggregation with
filters), a single embedding of the question text doesn't retrieve all
relevant sessions. Decomposing the question into atomic sub-queries and
running retrieval on each gives the answer step a richer context to work
with.

Example:
  Question: "How much did I spend on workshops in the last four months?"
  Naive retrieval: embeds the whole question, finds some sessions mentioning
                   workshops. Might miss sessions that mention costs without
                   using the word "workshop", or sessions about specific
                   named workshops.
  Decomposed:
    - "workshop attendance mentions"
    - "workshop costs amounts"
    - "recent dates"
  Each query retrieves independently; union is richer.

Design choice: decomposition enriches retrieval, does NOT split the answer.
The single Sonnet answer step (v1 procedure) still runs on the union of
retrieved clusters. This is additive to v1, not a replacement — v1's answer
mechanism stays intact, we just give it more to work with.
"""

from __future__ import annotations
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable


DECOMPOSE_SYSTEM = """\
You decompose a user's question into atomic search queries.

An atomic search query is a short noun-phrase or keyword string that can be
used for semantic retrieval. Each sub-query should target ONE specific
piece of information the answer needs.

OUTPUT FORMAT (strict JSON, nothing else):
{
  "is_composite": true|false,
  "sub_queries": [
    "<short search phrase>",
    "<another search phrase>",
    ...
  ]
}

RULES:
  1. Sub-queries are SEARCH PHRASES, not questions. "workshop costs" not
     "how much did workshops cost". Keep them 2-6 words.
  2. If the question is simple (single fact, single stated value), set
     is_composite=false and return ONE sub-query that mirrors the question
     keywords.
  3. For composite questions (multi-event, aggregation with filters,
     multi-hop), return 2-5 sub-queries, each targeting a distinct aspect.
  4. For aggregation questions ("how many X"), include both the item type
     and any qualifying attributes as separate sub-queries.
  5. For temporal questions referencing events, include a sub-query for
     each named event.
  6. Output ONLY the JSON object. No commentary.

EXAMPLES:

Question: "How much did I spend on workshops in the last four months?"
{
  "is_composite": true,
  "sub_queries": [
    "workshops attended",
    "workshop cost price fee",
    "registration payment"
  ]
}

Question: "Which three events happened first: preparing nursery, cousin's wedding dress shopping, Maria's baby shower?"
{
  "is_composite": true,
  "sub_queries": [
    "prepare nursery friend",
    "cousin wedding dress shopping",
    "Maria baby shower"
  ]
}

Question: "How many years older am I than when I graduated from college?"
{
  "is_composite": true,
  "sub_queries": [
    "current age",
    "college graduation year"
  ]
}

Question: "What is the name of the playlist I created on Spotify?"
{
  "is_composite": false,
  "sub_queries": [
    "Spotify playlist name"
  ]
}

Question: "What was my last name before I changed it?"
{
  "is_composite": false,
  "sub_queries": [
    "last name change previous"
  ]
}
"""


def _parse_json(raw: str) -> dict:
    """Tolerant JSON parse — handles markdown fences, trailing text, etc."""
    if not raw:
        return {}
    text = raw.strip()
    for fence in ("```json", "```JSON", "```"):
        if text.startswith(fence):
            text = text[len(fence):].lstrip()
        if text.endswith("```"):
            text = text[:-3].rstrip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try first {...} block
    start = text.find("{")
    if start == -1:
        return {}
    depth = 0
    in_str = False
    esc = False
    end = -1
    for i in range(start, len(text)):
        ch = text[i]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        return {}
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}


def decompose(question: str, llm_haiku_raw: Callable) -> dict:
    """
    Break a question into atomic search queries via cheap Haiku call.

    Returns:
      {
        "is_composite": bool,
        "sub_queries": [str, ...],  # always at least 1 query
      }

    On any error, returns a safe fallback: {"is_composite": False,
    "sub_queries": [question]}.
    """
    try:
        raw = llm_haiku_raw(
            system=DECOMPOSE_SYSTEM,
            user_message=f"Question: {question}",
            max_tokens=300,
        )
    except Exception:
        return {"is_composite": False, "sub_queries": [question]}

    parsed = _parse_json(raw)
    if not isinstance(parsed, dict):
        return {"is_composite": False, "sub_queries": [question]}

    queries = parsed.get("sub_queries", [])
    if not isinstance(queries, list) or not queries:
        return {"is_composite": False, "sub_queries": [question]}

    # Clean — strip, dedupe, filter too-short/too-long
    cleaned = []
    seen = set()
    for q in queries:
        if not isinstance(q, str):
            continue
        q = q.strip()
        if len(q) < 2 or len(q) > 120:
            continue
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(q)

    if not cleaned:
        cleaned = [question]

    return {
        "is_composite": bool(parsed.get("is_composite", False)),
        "sub_queries": cleaned[:5],  # cap at 5 to bound cost
    }


def run_multi_query_retrieval(
    sub_queries: list[str],
    store,
    embedder,
    per_query_k: int = 6,
    threshold: float = 0.25,
) -> list:
    """
    Run each sub-query as an independent embedding search. Return deduplicated
    (Fact, score) list sorted by best score per fact.

    Why per_query_k=6: with 3-5 sub-queries, 6 per query gives 18-30 raw hits.
    After dedup (many queries find overlapping sessions), we typically get
    12-20 unique clusters. That's a sweet spot for the answer step — wider
    than v1's single-query retrieval but not so wide that lost-in-the-middle
    kicks in.
    """
    if not sub_queries:
        return []

    all_results: dict[str, tuple] = {}  # fact_id -> (Fact, best_score)

    for q in sub_queries:
        try:
            emb = embedder.embed(q)
        except Exception:
            continue
        try:
            hits = store.search(emb, top_k=per_query_k, threshold=threshold)
        except Exception:
            continue
        for fact, score in hits:
            if fact.id in all_results:
                if score > all_results[fact.id][1]:
                    all_results[fact.id] = (fact, score)
            else:
                all_results[fact.id] = (fact, score)

    # Return sorted by best score desc
    sorted_results = sorted(all_results.values(), key=lambda x: -x[1])
    return sorted_results


def run_multi_query_retrieval_parallel(
    sub_queries: list[str],
    store,
    embedder,
    per_query_k: int = 6,
    threshold: float = 0.25,
    max_workers: int = 4,
) -> list:
    """
    Parallel version — embeds each sub-query concurrently. For 3-5 queries
    with local sentence-transformers the serial version is already fast
    (~50ms total), so parallel is mostly useful if embedder is remote.
    """
    if not sub_queries:
        return []

    def embed_and_search(q: str):
        try:
            emb = embedder.embed(q)
            return store.search(emb, top_k=per_query_k, threshold=threshold)
        except Exception:
            return []

    all_results: dict[str, tuple] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(embed_and_search, q): q for q in sub_queries}
        for fut in as_completed(futures):
            try:
                hits = fut.result(timeout=30)
            except Exception:
                hits = []
            for fact, score in hits:
                if fact.id in all_results:
                    if score > all_results[fact.id][1]:
                        all_results[fact.id] = (fact, score)
                else:
                    all_results[fact.id] = (fact, score)

    return sorted(all_results.values(), key=lambda x: -x[1])
