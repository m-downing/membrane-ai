"""
Membrane — virtual memory for LLMs.

The membrane sits between the user and the model, managing what
the model "knows" at any given moment through three operations:

  PROMOTE  — retrieve relevant facts into context before each turn
  EXTRACT  — pull new facts from conversation after each turn
  REPAIR   — detect contradictions between output and stored facts

The model is unchanged. The membrane manages its working memory
the way an OS manages page tables: swap in what's needed, swap
out what's stale, catch page faults (contradictions) and handle them.

This is the "second hemisphere" — it doesn't do reasoning,
it provides the scaffold that keeps reasoning coherent over time.
"""

import json
import re
import uuid
import numpy as np
from datetime import datetime
from store import FactStore, Fact

# Patterns that signal the user wants to aggregate across multiple facts.
_INTENT_SYSTEM = """\
Classify the user's question into exactly one retrieval category.
Respond with only the category name — no punctuation, no explanation.

Categories:

aggregation — when the answer requires scanning MULTIPLE sessions to count,
              sum, or list things. The clearest signal: the question asks
              HOW MANY or LIST ALL about something that can accumulate
              across sessions.
              This includes both EVENTS ("how many times did I gym") AND
              ACCUMULATED ITEMS ("how many tanks do I own", "how many
              projects have I led") — both require scanning all sessions
              to produce a correct count.
              Key test: HOW MANY (a count) → aggregation.
              WHAT IS / WHICH (a single value) → check recency instead.
              "Currently" alone does not decide — look at whether the
              answer is a count or a single value.
              YES: "how many times did I go to the gym"
                   "how many tanks do I currently have"
                   "how many projects have I led or am leading"
                   "how many fish are in my aquariums"
                   "how many instruments do I own"
                   "list every book I mentioned"
                   "how much total money did I spend on X"
              NO:  "what is my current job"     ← single value, use recency
                   "what car do I drive now"    ← single value, use recency
                   "do I like X", "what do I prefer" ← use factual

temporal_adjacent — the answer lives in the session IMMEDIATELY BEFORE or AFTER
              a specific named anchor event, not in the anchor session itself.
              YES: "what happened the week after I got promoted",
                   "what did I do right after starting therapy".
              Only use this when the question is explicitly about what happened
              adjacent to a named anchor — not general time questions.

recency — the user asks for the MOST CURRENT single value of a fact that
          changes over time: a job title, city, salary, car, relationship
          status. The answer is one thing, from the latest session that
          mentions it.
          YES: "what is my current job", "what car do I drive now",
               "what city do I live in", "my latest salary".
          NO:  "how many X do I have/own" → aggregation (count, not single value)
               preference questions, advice, "what do I like".

factual — everything else: direct recall of a specific event or detail,
          preference questions, advice requests, date/duration questions.
          YES: "what play did I attend", "when did I start therapy",
               "how long between X and Y", "what do I prefer for coffee",
               "tips for my kitchen given my habits".\
"""

# Patterns that indicate a session contains personal/biographical facts.
# Sessions matching this are better indexed via LLM extraction (clean discrete
# facts) rather than verbatim chunks (noisy text). Sessions without these
# patterns are indexed verbatim — extraction would be lossy for generic Q&A.
#
# Heuristic: if the user said something personal (first-person + concrete verb
# + subject), extraction will produce a fact like "User owned a Corolla" that
# counts cleanly across sessions. Without personal statements, extraction
# produces nothing and verbatim is strictly better.
_PERSONAL_FACT_RE = re.compile(
    r"\bI\s+(am|was|have|had|own|owned|had|bought|built|made|finished|"
    r"completed|started|graduated|visited|use|used|like|liked|loved|"
    r"prefer|preferred|remember|got|get|live|lived|work|worked|went|go|"
    r"adopted|rescued|adopted|named|called|grew|ran|run|earned|won|"
    r"attended|took|quit|joined|left|found|lost|sold|gave|ate|tried|"
    r"played|watched|read|wrote|learned|studied|moved|drove|rode|flew|"
    r"hiked|climbed|swam|ran|coached|trained|volunteered|donated|"
    r"published|launched|shipped|deployed|built|created|designed)\b",
    re.IGNORECASE,
)


def _parse_date(date_str: str) -> datetime | None:
    """Parse a date string in common formats to a datetime object."""
    for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


def _build_timeline(session_dates: dict[str, str]) -> str:
    """
    Build a pre-computed timeline string from session_id → date_string mapping.

    Converts raw date strings to human-readable format and computes the exact
    interval between consecutive sessions in Python — so the LLM reads facts
    rather than doing error-prone date arithmetic itself.

    Example output:
        Session timeline (oldest → most recent):
          Session 1 — January 15, 2023
          Session 2 — February 20, 2023  (+36 days / ~5 weeks)
          Session 3 — April 18, 2023  (+57 days / ~2 months)
    """
    if not session_dates:
        return ""

    parsed = []
    for sid, date_str in session_dates.items():
        dt = _parse_date(date_str)
        if dt:
            parsed.append((sid, dt))

    if not parsed:
        return ""

    parsed.sort(key=lambda x: x[1])

    lines = ["Session timeline (oldest → most recent):"]
    prev_dt = None
    for i, (sid, dt) in enumerate(parsed, 1):
        human = dt.strftime("%B %d, %Y")
        if prev_dt is None:
            lines.append(f"  Session {i} — {human}")
        else:
            delta_days = (dt - prev_dt).days
            if delta_days >= 365:
                years = delta_days // 365
                months = (delta_days % 365) // 30
                interval = f"~{years} year{'s' if years != 1 else ''}"
                if months:
                    interval += f" {months} month{'s' if months != 1 else ''}"
            elif delta_days >= 30:
                months = delta_days // 30
                days_rem = delta_days % 30
                interval = f"~{months} month{'s' if months != 1 else ''}"
                if days_rem:
                    interval += f" {days_rem} day{'s' if days_rem != 1 else ''}"
            elif delta_days >= 7:
                weeks = delta_days // 7
                days_rem = delta_days % 7
                interval = f"~{weeks} week{'s' if weeks != 1 else ''}"
                if days_rem:
                    interval += f" {days_rem} day{'s' if days_rem != 1 else ''}"
            else:
                interval = f"{delta_days} day{'s' if delta_days != 1 else ''}"
            lines.append(f"  Session {i} — {human}  (+{delta_days} days / {interval} after Session {i-1})")
        prev_dt = dt

    return "\n".join(lines)


def session_has_personal_facts(turns: list[dict], threshold: int = 2) -> bool:
    """
    Returns True if the session contains enough first-person personal statements
    to make LLM extraction worthwhile.

    Scans user turns only. threshold=2 means at least 2 personal-verb matches.
    Low-threshold intentionally — it's cheaper to extract a session with few
    personal facts than to miss one that has many.
    """
    user_text = " ".join(
        t.get("content", "") for t in turns if t.get("role") == "user"
    )
    matches = _PERSONAL_FACT_RE.findall(user_text)
    return len(matches) >= threshold

# ── Embedder interface ──────────────────────────────────────
# Swap this for Titan, Voyage, OpenAI, etc.
# Default: sentence-transformers (local, fast, no API cost)


class LocalEmbedder:
    """Local embeddings via sentence-transformers. ~90MB model download on first run."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")

    def embed(self, text: str) -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=True)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True)


# ── LLM interface ───────────────────────────────────────────
# Anthropic API. Membrane's LLM layer is duck-typed — any class
# implementing `complete(system, user_message, max_tokens)` and
# `chat(system, messages, max_tokens)` can be passed as the `llm`
# argument to Membrane. Adding a new provider is ~30 lines.


class AnthropicLLM:
    """LLM calls via Anthropic API."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model

    def _call_with_retry(self, **kwargs) -> str:
        import anthropic
        import time
        for attempt in range(6):
            try:
                response = self.client.messages.create(**kwargs, timeout=60.0)
                return response.content[0].text
            except anthropic.RateLimitError:
                if attempt == 5:
                    raise
                wait = 5 * (2 ** attempt)  # 5, 10, 20, 40, 80s
                time.sleep(wait)

    def complete(self, system: str, user_message: str, max_tokens: int = 1000) -> str:
        return self._call_with_retry(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )

    def chat(self, system: str, messages: list[dict], max_tokens: int = 2048) -> str:
        return self._call_with_retry(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )


# ── Membrane ────────────────────────────────────────────────


class Membrane:
    def __init__(
        self,
        store_path: str = "facts.json",
        store=None,
        llm=None,
        embedder=None,
        promote_top_k: int = 10,
        promote_threshold: float = 0.25,
        min_promote: int = 3,
        demotion_min_cluster: int = 5,
        centrality_min_load: int = 2,
        hybrid_alpha: float = 0.7,
        mmr_lambda: float = 0.7,
        per_intent_k: dict | None = None,
        max_chunks_per_session: int | None = None,
    ):
        if store is not None:
            self.store = store
        else:
            self.store = FactStore(store_path)
        self.llm = llm or AnthropicLLM()
        self.embedder = embedder or LocalEmbedder()
        self.promote_top_k = promote_top_k
        self.promote_threshold = promote_threshold
        self.min_promote = min_promote
        self.demotion_min_cluster = demotion_min_cluster
        self.centrality_min_load = centrality_min_load
        self.hybrid_alpha = hybrid_alpha
        self.mmr_lambda = mmr_lambda
        self.per_intent_k = per_intent_k or {}
        self.max_chunks_per_session = max_chunks_per_session
        self.session_id = str(uuid.uuid4())[:8]
        self._traversal_ids: set[str] = set()   # populated by promote(), read by chat.py
        self._centrality_ids: set[str] = set()  # populated by promote(), read by chat.py
        self._last_intent: str = "factual"       # exposed for retrieval trace

    @property
    def backend(self) -> str:
        return getattr(self.store, "backend", "flat")

    # ── INTENT CLASSIFICATION ────────────────────────────────

    def _classify_intent(self, user_message: str) -> str:
        """
        Classify retrieval intent with a single lightweight LLM call.

        Returns one of: 'aggregation', 'temporal_adjacent', 'recency', 'factual'.

        Using the LLM instead of regex handles typos, paraphrasing, unusual
        phrasing, and edge cases that any fixed pattern list will miss.
        Falls back to 'factual' on any error so retrieval always proceeds.
        """
        try:
            resp = self.llm.complete(
                system=_INTENT_SYSTEM,
                user_message=f"Question: {user_message}",
                max_tokens=10,
            )
            intent = resp.strip().lower()
            if "aggregation" in intent:
                return "aggregation"
            if "temporal" in intent:
                return "temporal_adjacent"
            if "recency" in intent:
                return "recency"
            return "factual"
        except Exception:
            return "factual"

    # ── PROMOTE ──────────────────────────────────────────────

    def promote(
        self,
        user_message: str,
        session_dates: dict[str, str] | None = None,
    ) -> list[tuple[Fact, float]]:
        """
        Find facts relevant to the user's current message.
        Makes one small LLM call to classify intent, then routes to the
        appropriate retrieval strategy. Vector math + graph traversal.

        session_dates: optional mapping of session_id → date string (YYYY-MM-DD).
        When provided, enables temporal adjacency expansion (TR) and recency
        top-up (KU). Without it, those strategies are skipped.

        Strategies, in order:
        1. If total facts ≤ promote_top_k, promote ALL ranked by similarity.
        2. Aggregation queries (how many, list all, etc.): return every fact
           ranked by hybrid similarity — no top-k cap. Counting needs full
           coverage; a cap guarantees wrong answers.
        3. All other queries: hybrid vector+keyword search, MMR reranking.
        4. Graph traversal — one hop along ELABORATES/DEPENDS_ON edges.
           No-op on flat backend.
        5. Centrality promotion — load-bearing facts not yet reached.
           No-op on flat backend.
        6. Temporal adjacency expansion (TR) — when the query signals
           "after/before X", pull in the sessions immediately neighbouring
           the top-scoring anchor session. The answer is adjacent, not similar.
        7. Recency top-up (KU) — when the query signals "current/latest",
           append the chronologically newest sessions not already promoted.
        8. Recency fallback — pad to min_promote for meta-queries.
        """
        active = self.store.get_all_active()
        self._traversal_ids = set()
        self._centrality_ids = set()

        # Classify intent once — drives all routing decisions below.
        # LLM-based: handles typos, paraphrasing, and edge cases that
        # regex patterns miss. Falls back to 'factual' on any error.
        intent = self._classify_intent(user_message)
        self._last_intent = intent
        is_aggregation = (intent == "aggregation")
        is_temporal_adjacent = (intent == "temporal_adjacent")
        is_recency = (intent == "recency")
        chunked_mode = self.max_chunks_per_session is not None

        # Per-intent k — driven by per_intent_k overrides (chunked mode) or
        # the built-in heuristics (whole-session mode).
        if is_aggregation:
            # Chunked: fixed budget with two-pass selection.
            # Whole-session: must see every session.
            effective_k = self.per_intent_k.get("aggregation", len(active))
        elif intent == "factual":
            effective_k = self.per_intent_k.get("factual", max(3, self.promote_top_k // 2))
        elif is_temporal_adjacent:
            effective_k = self.per_intent_k.get("temporal_adjacent", self.promote_top_k)
        else:  # recency
            effective_k = self.per_intent_k.get("recency", self.promote_top_k)

        query_emb = self.embedder.embed(user_message)

        # Strategy 1: small store — rank all by similarity
        if len(active) <= effective_k:
            results = []
            for fact in active:
                if fact.embedding is not None:
                    sim = float(np.dot(query_emb, np.array(fact.embedding)))
                else:
                    sim = 0.0
                results.append((fact, sim))
            results = sorted(results, key=lambda x: x[1], reverse=True)
        else:
            # Strategy 2: hybrid vector + keyword search.
            effective_mmr = 1.0 if is_aggregation else self.mmr_lambda
            if is_aggregation and chunked_mode:
                # Over-fetch so two-pass can select best cross-session coverage.
                fetch_k = max(60, effective_k * 3)
            elif effective_mmr < 1.0:
                fetch_k = effective_k * 3
            else:
                fetch_k = effective_k

            if hasattr(self.store, "search_hybrid"):
                results = self.store.search_hybrid(
                    user_message, query_emb,
                    top_k=fetch_k,
                    threshold=self.promote_threshold if not is_aggregation else 0.0,
                    alpha=self.hybrid_alpha,
                )
            else:
                results = self.store.search(
                    query_emb, top_k=fetch_k,
                    threshold=self.promote_threshold if not is_aggregation else 0.0,
                )

            if is_aggregation and chunked_mode:
                # Two-pass: breadth-first coverage then depth fill.
                results = self._two_pass_aggregation(
                    results, effective_k,
                    max_per_session=self.max_chunks_per_session or 2,
                )
            elif effective_mmr < 1.0 and len(results) > 1:
                results = self._mmr_rerank(results, effective_k)
                # Per-session cap applied inside session-aware MMR via post-filter.
                if chunked_mode:
                    results = self._apply_session_cap(
                        results, self.max_chunks_per_session, effective_k
                    )
            else:
                results = results[:effective_k]
                if chunked_mode and not is_aggregation:
                    results = self._apply_session_cap(
                        results, self.max_chunks_per_session, effective_k
                    )

        # Strategy 4: graph traversal (no-op on flat backend)
        promoted_ids = {f.id for f, _ in results}
        traversal_adds = []
        for fact, score in list(results):
            for neighbor, edge_type in self.store.get_neighbors(fact.id):
                if neighbor.id not in promoted_ids:
                    promoted_ids.add(neighbor.id)
                    self._traversal_ids.add(neighbor.id)
                    traversal_adds.append((neighbor, round(score * 0.8, 3)))
        results = results + traversal_adds

        # Strategy 5: centrality promotion (no-op on flat backend)
        promoted_ids = {f.id for f, _ in results}
        for fact, load in self.store.get_high_centrality_facts(
            min_load=self.centrality_min_load
        ):
            if fact.id not in promoted_ids:
                promoted_ids.add(fact.id)
                self._centrality_ids.add(fact.id)
                results.append((fact, min(0.5, round(0.1 * load, 3))))

        # Strategy 6: temporal adjacency expansion (TR)
        # When the query is about what happened after/before an event, the
        # answer lives in the session chronologically adjacent to the anchor —
        # not in the anchor itself. Requires session_dates to know the order.
        if session_dates and is_temporal_adjacent and results:
            promoted_ids = {f.id for f, _ in results}
            sid_to_date = session_dates
            ordered_sids = sorted(sid_to_date, key=lambda s: sid_to_date[s])
            # Build session_id → facts lookup
            facts_by_sid: dict[str, list[Fact]] = {}
            for f in active:
                facts_by_sid.setdefault(f.session_id, []).append(f)
            # Expand from the top-scoring anchor only
            anchor_sid = results[0][0].session_id
            # In chunked/clustered mode cap the neighbor contribution so we
            # don't flood context with every cluster from adjacent sessions.
            topup_cap = self.max_chunks_per_session if chunked_mode else None
            if anchor_sid in ordered_sids:
                anchor_idx = ordered_sids.index(anchor_sid)
                for offset in (-1, 1):
                    ni = anchor_idx + offset
                    if 0 <= ni < len(ordered_sids):
                        nb_sid = ordered_sids[ni]
                        nb_facts = facts_by_sid.get(nb_sid, [])
                        if topup_cap is not None:
                            nb_facts = nb_facts[:topup_cap]
                        for nf in nb_facts:
                            if nf.id not in promoted_ids:
                                promoted_ids.add(nf.id)
                                results.append((nf, 0.0))

        # Strategy 7: recency top-up (KU)
        # When the query asks about current/latest state, ensure the most
        # recent sessions are in context even if they scored low by similarity.
        # Appends up to 3 newest sessions not already promoted.
        # In chunked/clustered mode, cap each session's contribution to
        # max_chunks_per_session clusters so long sessions don't dominate.
        if session_dates and is_recency:
            promoted_ids = {f.id for f, _ in results}
            sid_to_date = session_dates
            ordered_sids = sorted(sid_to_date, key=lambda s: sid_to_date[s])
            facts_by_sid: dict[str, list[Fact]] = {}
            for f in active:
                facts_by_sid.setdefault(f.session_id, []).append(f)
            topup_cap = self.max_chunks_per_session if chunked_mode else None
            sessions_added = 0
            for sid in reversed(ordered_sids):
                if sessions_added >= 3:
                    break
                session_facts = facts_by_sid.get(sid, [])
                new_in_session = [
                    nf for nf in session_facts if nf.id not in promoted_ids
                ]
                if new_in_session:
                    if topup_cap is not None:
                        new_in_session = new_in_session[:topup_cap]
                    for nf in new_in_session:
                        promoted_ids.add(nf.id)
                        results.append((nf, 0.0))
                    sessions_added += 1

        # Strategy 8: recency fallback for meta-queries / vague greetings
        best_score = results[0][1] if results else 0.0
        if len(results) < self.min_promote and best_score < self.promote_threshold:
            padded_ids = {f.id for f, _ in results}
            recent = sorted(
                [f for f in active if f.id not in padded_ids],
                key=lambda f: f.created_at,
                reverse=True,
            )
            for f in recent[: self.min_promote - len(results)]:
                results.append((f, 0.0))

        return results

    def _mmr_rerank(
        self, candidates: list[tuple[Fact, float]], top_k: int
    ) -> list[tuple[Fact, float]]:
        """
        Maximal Marginal Relevance reranking.

        At each step selects the candidate that maximises:
            mmr_lambda * relevance_score
            - (1 - mmr_lambda) * max_cosine_similarity_to_already_selected

        mmr_lambda=1.0  → pure relevance (no diversity, same as no MMR)
        mmr_lambda=0.5  → equal weight on relevance and diversity
        mmr_lambda=0.7  → default: slight diversity bias while preserving ranking

        Directly addresses multi-session failures where the top-k is dominated by
        multiple chunks from the same 2–3 sessions, crowding out other relevant ones.
        """
        if not candidates:
            return []

        selected: list[tuple[Fact, float]] = []
        remaining = list(candidates)
        lam = self.mmr_lambda

        while remaining and len(selected) < top_k:
            if not selected:
                best = max(remaining, key=lambda x: x[1])
            else:
                sel_embs = np.array(
                    [f.embedding for f, _ in selected if f.embedding is not None],
                    dtype=float,
                )
                best_mmr = -float("inf")
                best = remaining[0]
                for fact, rel in remaining:
                    if fact.embedding is None:
                        mmr = rel
                    else:
                        emb = np.array(fact.embedding, dtype=float)
                        redundancy = float(np.max(sel_embs @ emb)) if len(sel_embs) else 0.0
                        mmr = lam * rel - (1 - lam) * redundancy
                    if mmr > best_mmr:
                        best_mmr = mmr
                        best = (fact, rel)

            selected.append(best)
            remaining = [(f, s) for f, s in remaining if f.id != best[0].id]

        return selected

    def _apply_session_cap(
        self,
        results: list[tuple[Fact, float]],
        max_per_session: int,
        budget: int,
    ) -> list[tuple[Fact, float]]:
        """
        Cap how many chunks any single session contributes.
        Iterates in score order (results should already be ranked),
        accepting up to max_per_session chunks per session_id.
        """
        session_counts: dict[str, int] = {}
        selected = []
        for fact, score in results:
            sid = fact.session_id
            if session_counts.get(sid, 0) < max_per_session:
                selected.append((fact, score))
                session_counts[sid] = session_counts.get(sid, 0) + 1
            if len(selected) >= budget:
                break
        return selected

    def _two_pass_aggregation(
        self,
        candidates: list[tuple[Fact, float]],
        budget: int,
        max_per_session: int = 2,
    ) -> list[tuple[Fact, float]]:
        """
        Two-pass selection for aggregation (MS) in chunked/clustered mode.

        Pass 1 — breadth-first: 1 cluster per session, fills at most
                  ⌊2/3 × budget⌋ slots.  Reserving slots means Pass 2 always
                  runs, even when the number of sessions exceeds the budget.

        Pass 2 — depth fill: allow up to max_per_session clusters from any
                  session, filling remaining budget in score order.  High-scoring
                  sessions can now contribute multiple clusters when their later
                  clusters still beat low-scoring sessions' first clusters.

        Reserving budget for Pass 2 matters when n_sessions ≥ budget — without
        the reservation, Pass 1 fills the budget entirely and Pass 2 never fires,
        meaning sessions that need >1 cluster for a complete count are silently
        truncated.
        """
        session_counts: dict[str, int] = {}
        selected_ids: set[str] = set()
        selected: list[tuple[Fact, float]] = []

        # Pass 1: 1 per session, capped at 2/3 of budget so Pass 2 always runs.
        pass1_limit = max(1, budget - max(1, budget // 3))
        for fact, score in candidates:
            if len(selected) >= pass1_limit:
                break
            sid = fact.session_id
            if session_counts.get(sid, 0) < 1:
                selected.append((fact, score))
                selected_ids.add(fact.id)
                session_counts[sid] = 1

        # Pass 2: up to max_per_session per session, fills remaining budget.
        if len(selected) < budget:
            for fact, score in candidates:
                if len(selected) >= budget:
                    break
                if fact.id in selected_ids:
                    continue
                sid = fact.session_id
                if session_counts.get(sid, 0) < max_per_session:
                    selected.append((fact, score))
                    selected_ids.add(fact.id)
                    session_counts[sid] = session_counts.get(sid, 0) + 1

        return selected

    def format_promoted(
        self,
        promoted: list[tuple[Fact, float]],
        session_dates: dict[str, str] | None = None,
        chunk_metadata: dict | None = None,
    ) -> str:
        """
        Format promoted facts as structured context for the model.

        session_dates: optional session_id → date string mapping.
        chunk_metadata: optional fact_id → chunk info dict (chunked mode).
          When provided, adjacent chunks from the same session are merged
          and the header includes the turn range.
          When omitted, falls back to whole-session session blocks.
        """
        if not promoted:
            return ""

        if session_dates:
            all_sids_ordered = sorted(session_dates, key=lambda s: session_dates[s])
            sid_to_idx = {sid: i + 1 for i, sid in enumerate(all_sids_ordered)}

            timeline = _build_timeline(session_dates)

            if chunk_metadata:
                # ── Chunked mode ─────────────────────────────────────────
                # Group chunks by session, sort within session by pair_start,
                # merge adjacent/overlapping chunks, emit one block per merged group.

                session_chunks: dict[str, list[dict]] = {}
                for fact, _ in promoted:
                    meta = chunk_metadata.get(fact.id)
                    if meta is None:
                        # Fact has no chunk info (e.g. recency top-up from
                        # whole-session store) — treat as single-chunk with
                        # the full content.
                        meta = {
                            "pair_start": 0, "pair_end": 999,
                            "start_turn": 0, "end_turn": 999,
                            "pair_texts": [fact.content],
                        }
                    sid = fact.session_id
                    session_chunks.setdefault(sid, []).append(dict(meta))

                blocks = []
                for sid in all_sids_ordered:
                    if sid not in session_chunks:
                        continue
                    chunks = sorted(session_chunks[sid], key=lambda c: c["pair_start"])

                    # Merge adjacent / overlapping chunks within this session.
                    merged: list[dict] = []
                    for c in chunks:
                        if not merged:
                            merged.append({
                                "pair_start": c["pair_start"],
                                "pair_end": c["pair_end"],
                                "start_turn": c["start_turn"],
                                "end_turn": c["end_turn"],
                                "pair_texts": list(c["pair_texts"]),
                            })
                        else:
                            prev = merged[-1]
                            if c["pair_start"] <= prev["pair_end"]:
                                # Overlapping — extend prev, skip duplicate pairs.
                                overlap = prev["pair_end"] - c["pair_start"]
                                prev["pair_texts"] += c["pair_texts"][overlap:]
                                prev["pair_end"] = c["pair_end"]
                                prev["end_turn"] = c["end_turn"]
                            else:
                                merged.append({
                                    "pair_start": c["pair_start"],
                                    "pair_end": c["pair_end"],
                                    "start_turn": c["start_turn"],
                                    "end_turn": c["end_turn"],
                                    "pair_texts": list(c["pair_texts"]),
                                })

                    raw_date = session_dates.get(sid, "")
                    dt = _parse_date(raw_date)
                    human_date = dt.strftime("%B %d, %Y") if dt else raw_date
                    idx = sid_to_idx.get(sid, "?")

                    for m in merged:
                        header = (
                            f"[Session {idx} — {human_date}"
                            f" — turns {m['start_turn']}-{m['end_turn']}]"
                        )
                        content = "\n".join(m["pair_texts"])
                        blocks.append(f"{header}\n{content}")

            else:
                # ── Whole-session mode ────────────────────────────────────
                # One header block per session; multiple facts bullet-listed.
                session_groups: dict[str, list[str]] = {}
                for fact, _ in sorted(promoted, key=lambda fsp: session_dates.get(fsp[0].session_id, "")):
                    sid = fact.session_id
                    session_groups.setdefault(sid, []).append(fact.content)

                blocks = []
                for sid in all_sids_ordered:
                    if sid not in session_groups:
                        continue
                    raw_date = session_dates.get(sid, "")
                    dt = _parse_date(raw_date)
                    human_date = dt.strftime("%B %d, %Y") if dt else raw_date
                    idx = sid_to_idx.get(sid, "?")
                    header = f"[Session {idx} — {human_date}]"
                    contents = session_groups[sid]
                    facts_text = contents[0] if len(contents) == 1 else "\n".join(f"• {c}" for c in contents)
                    blocks.append(f"{header}\n{facts_text}")

            context = (
                "<conversation_history>\n"
                + (timeline + "\n\n" if timeline else "")
                + "Past sessions (oldest → most recent):\n\n"
                + "\n\n".join(blocks)
                + "\n</conversation_history>"
            )
        else:
            # Legacy bullet-list format (chat.py, extract mode, etc.)
            lines = ["- " + fact.content for fact, _ in promoted]
            context = (
                "<established_context>\n"
                "The following facts have been established in previous interactions. "
                "Treat these as ground truth unless the user explicitly updates them.\n"
                + "\n".join(lines)
            )
            context += "\n</established_context>"

        # Surface unresolved contradictions regardless of format
        contradictions = self.store.get_contradictions()
        if contradictions:
            context += "\n\n⚠️ Unresolved contradictions in memory:\n"
            for a, b in contradictions:
                context += f'- "{a.content}" ↔ "{b.content}"\n'
            context += "(These have not been explicitly resolved by the user.)"

        return context

    # ── EXTRACT ──────────────────────────────────────────────

    def extract(self, user_message: str, model_response: str) -> list[Fact]:
        """
        Extract discrete facts from the latest conversation turn.
        The LLM sees existing facts and can explicitly flag replacements via
        {"fact": "...", "replaces": "old_fact_id"}. Vector similarity is a
        fallback for cases the LLM doesn't explicitly tag.
        """
        existing_facts = self.store.get_all_active()
        existing_context = ""
        if existing_facts:
            lines = [f'  [{f.id}] {f.content}' for f in existing_facts]
            existing_context = (
                "\n\nExisting facts already in memory (with their IDs):\n"
                + "\n".join(lines)
                + "\n\nUse structured forms to express relationships with existing facts:\n"
                '- {"fact": "...", "replaces": "id"}   — new fact updates/replaces an existing one\n'
                '- {"fact": "...", "elaborates": "id"} — adds detail to an existing fact without replacing it\n'
                '                                         (e.g. "User uses Python 3.11" elaborates "User works in Python")\n'
                '- {"fact": "...", "depends_on": "id"} — only meaningful given another fact\n'
                '                                         (e.g. "User\'s favorite lib is FastAPI" depends_on "User works in Python")\n'
                "Otherwise use a plain string for new standalone facts."
            )

        raw = self.llm.complete(
            system=(
                "You extract discrete, reusable facts from conversations. "
                "Extract ONLY facts that the USER explicitly stated or confirmed. "
                "Do NOT extract facts from the assistant's response — the assistant "
                "may be wrong, outdated, or speculating. If the assistant says "
                "'You live in Berlin' but the user said nothing about location, "
                "do not extract a location fact. "
                "Focus on: user preferences, decisions, personal details, "
                "project context, stated goals, and concrete information "
                "worth remembering across sessions. "
                "Each fact must be self-contained and understandable in isolation. "
                "Write facts in third person (e.g., 'User prefers dark mode').\n\n"
                "BE CONCISE. Aim for 2–4 facts per turn. "
                "Consolidate related details about the same topic into one fact "
                "rather than splitting them across multiple. "
                "If two candidate facts say essentially the same thing, keep only "
                "the more informative one. Prefer fewer, richer facts over many "
                "granular ones.\n\n"
                "SEPARATE facts only for genuinely distinct entities — e.g., each "
                "pet, each project, each tool is its own fact. Do not merge things "
                "that may need to be independently updated or superseded.\n\n"
                "RELATION TYPES (when existing facts are provided):\n"
                "- 'replaces': new fact changes the value of an existing attribute "
                "(switched language, moved city). Only when the new fact directly contradicts the old. "
                "Only use 'replaces' when the new fact makes the old one factually wrong or obsolete. "
                "Listing a second item does NOT replace the first item. "
                "Adding a feature does NOT replace the tool it's built on. "
                "Two different preferences do NOT replace each other.\n"
                "- 'elaborates': new fact adds specificity to an existing fact without invalidating it. "
                "Use 'elaborates' ONLY when the new fact is fully consistent with and additive to the anchor. "
                "If the new fact is in tension with or partially contradicts the anchor, "
                "use a plain string instead — do not force a contradictory fact into 'elaborates'.\n"
                "- 'depends_on': new fact is only interpretable in context of another.\n"
                "- plain string: genuinely new information with no clear anchor in existing facts.\n\n"
                "Respond ONLY with a JSON array. Each element is a plain string or one of:\n"
                '{"fact": "...", "replaces": "id"} | {"fact": "...", "elaborates": "id"} | '
                '{"fact": "...", "depends_on": "id"}. '
                "If nothing worth remembering, respond with []."
            ),
            user_message=(
                f"User: {user_message}\n\n"
                f"Assistant: {model_response}"
                + existing_context
                + "\n\nExtract facts as a JSON array:"
            ),
        )

        try:
            clean = raw.strip().replace("```json", "").replace("```", "").strip()
            facts_raw = json.loads(clean)
        except (json.JSONDecodeError, IndexError):
            return []

        new_facts = []
        all_fact_ids = set(self.store.facts.keys())

        for item in facts_raw:
            # Normalise: plain string or structured dict
            if isinstance(item, str):
                content = item
                explicit_replaces  = None
                explicit_elaborates = None
                explicit_depends_on = None
            elif isinstance(item, dict) and "fact" in item:
                content = item["fact"]
                explicit_replaces   = item.get("replaces")
                explicit_elaborates = item.get("elaborates")
                explicit_depends_on = item.get("depends_on")
            else:
                continue

            if not isinstance(content, str) or len(content.strip()) < 10:
                continue

            emb = self.embedder.embed(content)

            # ── LLM-declared supersession ─────────────────────
            if explicit_replaces and explicit_replaces in all_fact_ids:
                fact = Fact(content=content, embedding=emb.tolist(), session_id=self.session_id)
                self.store.add(fact)
                self.store.supersede(explicit_replaces, fact.id)
                new_facts.append(fact)
                continue

            # ── LLM-declared typed edges (ELABORATES / DEPENDS_ON) ──
            # These add a new node AND an edge; no supersession.
            if explicit_elaborates and explicit_elaborates in all_fact_ids:
                fact = Fact(content=content, embedding=emb.tolist(), session_id=self.session_id)
                self.store.add(fact)
                self.store.elaborate(fact.id, explicit_elaborates)
                new_facts.append(fact)
                continue

            if explicit_depends_on and explicit_depends_on in all_fact_ids:
                fact = Fact(content=content, embedding=emb.tolist(), session_id=self.session_id)
                self.store.add(fact)
                self.store.depends_on(fact.id, explicit_depends_on)
                new_facts.append(fact)
                continue

            # ── Fallback: vector-based dedup ──────────────────
            # 0.88 — high enough that only genuine rephrases trigger
            # supersession. 0.78 was too low: "User doesn't want X"
            # was matching "User doesn't want Y" on syntax alone.
            existing = self.store.search(emb, top_k=1, threshold=0.88)
            if existing:
                old_fact, sim = existing[0]
                if sim < 0.95:
                    fact = Fact(content=content, embedding=emb.tolist(), session_id=self.session_id)
                    self.store.add(fact)
                    self.store.supersede(old_fact.id, fact.id)
                    new_facts.append(fact)
                # else: near-identical, skip
            else:
                fact = Fact(content=content, embedding=emb.tolist(), session_id=self.session_id)
                self.store.add(fact)
                new_facts.append(fact)

        return new_facts

    _BATCH_EXTRACT_SYSTEM = (
        "You extract discrete, reusable facts from conversations. "
        "You will receive multiple conversation turns. "
        "Extract ONLY facts that the USER explicitly stated or confirmed "
        "across ALL turns. Do NOT extract from assistant responses. "
        "Focus on: personal details, preferences, decisions, events, "
        "quantities, and concrete information worth remembering. "
        "Each fact must be self-contained and understandable in isolation. "
        "Write in third person ('User worked on a Revell F-15 Eagle kit'). "
        "Each distinct item, entity, or event is its own fact — do not merge. "
        "Aim for 3–8 facts total. "
        "Respond ONLY with a JSON array of plain strings, e.g.: "
        '["User worked on a Revell F-15 Eagle kit", "User prefers 1/72 scale"]. '
        "If nothing worth remembering, respond with []."
    )

    def _batch_extract_raw(self, turns: list[tuple[str, str]]) -> list[str]:
        """
        LLM call only — returns raw extracted strings without touching the store.
        Safe to call from multiple threads concurrently.
        Retries on rate-limit errors with exponential backoff.
        """
        import time

        if not turns:
            return []

        turns_text = ""
        for i, (user_msg, asst_msg) in enumerate(turns, start=1):
            turns_text += f"[Turn {i}]\nUser: {user_msg}\nAssistant: {asst_msg}\n\n"

        delay = 5
        for attempt in range(5):
            try:
                raw = self.llm.complete(
                    system=self._BATCH_EXTRACT_SYSTEM,
                    user_message=(
                        f"Extract facts from these {len(turns)} conversation turns:\n\n"
                        + turns_text
                        + "Extract facts as a JSON array:"
                    ),
                )
                break
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    time.sleep(delay)
                    delay *= 2
                else:
                    return []
        else:
            return []

        try:
            clean = raw.strip().replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean)
            if not isinstance(parsed, list):
                return []
            return [s for s in parsed if isinstance(s, str) and len(s.strip()) >= 10]
        except (json.JSONDecodeError, IndexError):
            return []

    def _commit_extracted_facts(
        self, raw_strings: list[str], session_id: str = ""
    ) -> list[Fact]:
        """
        Embed, dedup, and write extracted fact strings to the store.
        Must be called serially (store writes are not thread-safe).
        """
        new_facts = []
        for content in raw_strings:
            content = content.strip()
            emb = self.embedder.embed(content)

            existing = self.store.search(emb, top_k=1, threshold=0.88)
            if existing:
                old_fact, sim = existing[0]
                if sim < 0.95:
                    fact = Fact(content=content, embedding=emb.tolist(), session_id=session_id)
                    self.store.add(fact)
                    self.store.supersede(old_fact.id, fact.id)
                    new_facts.append(fact)
            else:
                fact = Fact(content=content, embedding=emb.tolist(), session_id=session_id)
                self.store.add(fact)
                new_facts.append(fact)

        return new_facts

    def batch_extract(
        self,
        turns: list[tuple[str, str]],
        session_id: str = "",
    ) -> list[Fact]:
        """
        Extract facts from multiple (user, assistant) turn pairs in one LLM call.
        3–5x cheaper than calling extract() once per turn. Designed for bulk
        indexing of conversation history (e.g. benchmarking, backfill ingestion).

        For parallel bulk indexing, use _batch_extract_raw() + _commit_extracted_facts()
        directly — raw() is thread-safe, commit() must be called serially.

        For production turn-by-turn use, prefer extract() — it has the full
        relationship graph, supersession, and existing-facts context.
        """
        raw_strings = self._batch_extract_raw(turns)
        return self._commit_extracted_facts(raw_strings, session_id)

    # ── REPAIR ───────────────────────────────────────────────

    def repair(
        self, model_response: str, promoted: list[tuple[Fact, float]]
    ) -> str | None:
        """
        Check model response for contradictions with established facts.
        Two candidate pools are merged:
          1. Promoted facts — relevant to what the user asked.
          2. Response-matched facts — relevant to what the model said.
        Pool 2 catches cases where the query topic doesn't match the violated
        fact (e.g. "suggest dinner" doesn't embed near "vegetarian," but the
        response mentioning "beef stir-fry" does). One extra embed call per
        turn; LLM called only when candidates exist.
        Returns a correction signal to inject, or None if clean.
        """
        if not self.store.facts:
            return None

        # Pool 1: already-promoted facts
        promoted_ids = {f.id for f, _ in promoted}
        candidates: dict[str, Fact] = {f.id: f for f, _ in promoted}

        # Pool 2: facts matched against the response text itself
        resp_emb = self.embedder.embed(model_response)
        response_matches = self.store.search(resp_emb, top_k=self.promote_top_k, threshold=self.promote_threshold)
        for f, _ in response_matches:
            if f.id not in candidates:
                candidates[f.id] = f

        if not candidates:
            return None

        facts_text = "\n".join([f"- [{f.id}] {f.content}" for f in candidates.values()])
        result_raw = self.llm.complete(
            system=(
                "You detect factual contradictions between a response and "
                "previously established facts. Only flag CLEAR contradictions, "
                "not mere omissions or different phrasings of the same idea. "
                "Respond ONLY with JSON."
            ),
            user_message=(
                f"Established facts:\n{facts_text}\n\n"
                f"Response to check:\n{model_response}\n\n"
                "Is there a clear contradiction? Respond ONLY with:\n"
                '{"contradiction": true/false, "details": "explanation or null", '
                '"contradicts_fact": "id of the fact being violated, or null", '
                '"aligns_with": "id of a fact the response seems consistent with, or null"}'
            ),
            max_tokens=300,
        )

        try:
            clean = result_raw.strip().replace("```json", "").replace("```", "").strip()
            result = json.loads(clean)
            if result.get("contradiction"):
                # Write CONTRADICTS edge only when BOTH ends resolve to real
                # stored facts — guards against the LLM hallucinating IDs.
                contradicts_id = result.get("contradicts_fact")
                aligns_with_id = result.get("aligns_with")
                if (contradicts_id and aligns_with_id
                        and contradicts_id in candidates
                        and aligns_with_id in candidates
                        and contradicts_id != aligns_with_id):
                    self.store.add_contradicts(contradicts_id, aligns_with_id)

                details = result.get("details") or "Please review your response."
                violated = candidates.get(contradicts_id)
                fact_note = (
                    f'\nThe established fact is: "{violated.content}"'
                    if violated else ""
                )
                return (
                    "<membrane_correction>\n"
                    f"Contradiction detected: {details}{fact_note}\n"
                    "Please issue a targeted correction — fix only the specific "
                    "claim that conflicts with the established fact. Do not "
                    "revert or omit other recent updates the user has stated.\n"
                    "</membrane_correction>"
                )
        except (json.JSONDecodeError, IndexError):
            pass

        return None

    # ── CONTRADICTION DETECTION ──────────────────────────────

    def _detect_contradictions(self, new_facts: list[Fact]) -> list[tuple[Fact, Fact]]:
        """
        After extraction, compare each new fact against semantically similar
        existing facts to find unresolved tensions. One LLM call for all
        candidate pairs regardless of how many new facts. Writes CONTRADICTS
        edges for confirmed pairs and returns them.
        """
        if not new_facts:
            return []

        # Build candidate pairs: (new_fact, similar_existing_fact)
        candidates: list[tuple[Fact, Fact]] = []
        seen_pairs: set[frozenset] = set()

        for nf in new_facts:
            if nf.embedding is None:
                continue
            emb = np.array(nf.embedding)
            similar = self.store.search(emb, top_k=5, threshold=0.2)
            for existing, sim in similar:
                if existing.id == nf.id:
                    continue
                pair_key = frozenset([nf.id, existing.id])
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                candidates.append((nf, existing))

        if not candidates:
            return []

        pairs_text = "\n".join(
            f"Pair {i + 1}: [{a.id}] \"{a.content}\"  vs  [{b.id}] \"{b.content}\""
            for i, (a, b) in enumerate(candidates)
        )

        result_raw = self.llm.complete(
            system=(
                "You identify direct factual contradictions between pairs of facts. "
                "A contradiction means both cannot be true simultaneously. "
                "Elaborations, additions, and facts about different contexts "
                "(e.g. local dev vs production) are NOT contradictions. "
                "Only flag CLEAR, direct conflicts. Respond ONLY with JSON."
            ),
            user_message=(
                f"Check these fact pairs for direct contradictions:\n\n{pairs_text}\n\n"
                "Respond with a JSON array of contradicting pairs by their IDs:\n"
                '[[\"id_a\", \"id_b\"], ...]\n'
                "If no contradictions, respond with []."
            ),
            max_tokens=300,
        )

        written: list[tuple[Fact, Fact]] = []
        try:
            clean = result_raw.strip().replace("```json", "").replace("```", "").strip()
            pairs = json.loads(clean)
            fact_index = {}
            for a, b in candidates:
                fact_index[a.id] = a
                fact_index[b.id] = b
            for pair in pairs:
                if isinstance(pair, list) and len(pair) == 2:
                    id_a, id_b = pair[0], pair[1]
                    if id_a in fact_index and id_b in fact_index:
                        self.store.add_contradicts(id_a, id_b)
                        written.append((fact_index[id_a], fact_index[id_b]))
        except (json.JSONDecodeError, ValueError):
            pass

        return written

    # ── DEMOTION ─────────────────────────────────────────────

    def demote(self, min_cluster_size: int | None = None) -> list[dict]:
        """
        Find anchor facts whose child count (ELABORATES + DEPENDS_ON) has
        reached the demotion threshold. For each qualifying cluster, call the
        LLM once to compress anchor + all children into a single summary fact,
        then mark the originals inactive and write SUMMARIZES edges.

        Returns a list of {summary: Fact, replaced: list[Fact]} dicts —
        one per cluster that was compressed.

        Safe to call every turn: get_clusters() returns [] when nothing
        qualifies, so the LLM is only called when compression is needed.
        """
        threshold = min_cluster_size if min_cluster_size is not None else self.demotion_min_cluster
        clusters = self.store.get_clusters(min_children=threshold)
        if not clusters:
            return []

        results = []
        for anchor, children in clusters:
            all_facts = [anchor] + children
            facts_text = "\n".join(f"- {f.content}" for f in all_facts)

            summary_content = self.llm.complete(
                system=(
                    "You compress a cluster of related facts into a single, "
                    "self-contained summary fact. The summary must preserve all "
                    "essential information from every fact in the cluster. "
                    "Write in third person (e.g. 'User ...'). "
                    "Be concise but complete — one to three sentences."
                ),
                user_message=(
                    f"Compress these related facts into one summary:\n{facts_text}\n\n"
                    "Write only the summary fact, no preamble:"
                ),
                max_tokens=200,
            )

            emb = self.embedder.embed(summary_content.strip())
            summary_fact = Fact(
                content=summary_content.strip(),
                embedding=emb.tolist(),
                session_id=self.session_id,
            )
            self.store.add(summary_fact)

            # Deactivate children only — keep the anchor alive so it retains
            # its tight vector score on direct queries. The summary becomes an
            # ELABORATES child of the anchor so traversal pulls it in when the
            # anchor is promoted, giving the model the full picture without
            # replacing the sharpest signal.
            self.store.summarize(
                summary_id=summary_fact.id,
                original_ids=[f.id for f in children],
            )
            self.store.elaborate(summary_fact.id, anchor.id)

            results.append({"summary": summary_fact, "replaced": children})

        return results

    # ── ORCHESTRATION ────────────────────────────────────────

    def before_turn(self, user_message: str) -> tuple[str, list[tuple[Fact, float]]]:
        """
        Call this BEFORE sending to the model.
        Returns (system_prompt, promoted_facts).
        """
        promoted = self.promote(user_message)
        context = self.format_promoted(promoted)

        system = (
            "You are a helpful assistant with persistent memory. "
            "Information from previous sessions is provided in "
            "<established_context> tags — treat it as reliable unless "
            "the user corrects it. Be natural; don't mention the "
            "memory system unless asked.\n\n" + context
        )

        return system, promoted

    def after_turn(
        self,
        user_message: str,
        model_response: str,
        promoted: list[tuple[Fact, float]],
    ) -> dict:
        """
        Call this AFTER receiving model response.
        Returns {new_facts, correction, contradictions, total_facts}.
        """
        new_facts = self.extract(user_message, model_response)
        contradictions = self._detect_contradictions(new_facts)
        correction = self.repair(model_response, promoted)
        demotions = self.demote()
        self.store.save()

        return {
            "new_facts": [f.content for f in new_facts],
            "correction": correction,
            "contradictions": [(a.content, b.content) for a, b in contradictions],
            "demotions": [
                {"summary": d["summary"].content, "replaced_count": len(d["replaced"])}
                for d in demotions
            ],
            "total_facts": len(self.store),
        }
