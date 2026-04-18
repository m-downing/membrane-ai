"""
entities.py — Noun-chunk entity extraction for Membrane v6.

The bet: aggregation-intent queries ("how many workshops", "how many tanks")
need entity-driven retrieval, not similarity-driven. v4's hybrid search blends
keyword and vector with alpha=0.7 — keywords are a 30% tiebreaker. For queries
where the entity IS the question, that's underweighting what matters.

This module extracts content-noun entities from text (not named entities —
noun chunks + lemmatization + filtering). Used at ingestion to tag clusters,
and at query time to find which entities the user is asking about.

Design principles:
  1. Same extractor for ingestion and queries. Guaranteed vocabulary overlap.
  2. Lemmatize aggressively. "tanks" / "tank" / "Tanks" -> "tank".
  3. Drop generic nouns (thing, item, time, way) — they carry no retrieval signal.
  4. Index compounds at multiple granularities. "food delivery service"
     -> {"food delivery service", "delivery service", "service"}.
  5. spaCy first, regex fallback. If spaCy isn't installed or a specific text
     chokes it, regex extraction degrades gracefully.

Cost: spaCy en_core_web_sm runs ~1000 sentences/second on CPU. Full dataset
ingestion adds ~30-60 seconds. Query-time extraction is ~5ms per question.
"""

from __future__ import annotations
import re
from typing import Iterable

# ── Generic nouns that don't carry retrieval signal ───────────────────────
# These appear constantly and would pollute the entity index.
_GENERIC_NOUNS = frozenset({
    # Indefinite reference
    "thing", "things", "something", "anything", "nothing", "everything",
    "stuff", "item", "items", "piece", "pieces", "part", "parts",
    # Quantifiers and abstractions
    "amount", "number", "total", "count", "bunch", "lot", "lots",
    "kind", "kinds", "type", "types", "sort", "sorts", "way", "ways",
    "one", "ones", "other", "others", "some", "any", "many", "few",
    # Temporal fillers (dates handled separately)
    "time", "times", "day", "days", "week", "weeks", "month", "months",
    "year", "years", "moment", "period", "while",
    # Pronouns that slip through
    "i", "me", "my", "myself", "we", "us", "our", "ourselves",
    "you", "your", "yourself", "yours",
    "he", "him", "his", "she", "her", "hers", "it", "its",
    "they", "them", "their", "theirs",
    # Conversation-meta
    "question", "answer", "user", "assistant", "conversation", "chat",
    "session", "sessions", "history",
    # Generic predicates that nominalize
    "something", "anything", "everyone", "someone", "anyone",
    # Too-generic
    "place", "area", "location", "thing", "person", "people",
    "idea", "point", "reason", "fact", "case",
})

# Words that look like entities but are pure noise
_NOISE_PATTERNS = [
    re.compile(r"^\d+$"),                    # bare numbers — dates/amounts extracted elsewhere
    re.compile(r"^[a-z]$"),                  # single letters
    re.compile(r"^(uh|um|oh|ah|ok|okay|yeah|yes|no|hmm)$", re.I),
]


def _looks_like_noise(token: str) -> bool:
    return any(p.match(token) for p in _NOISE_PATTERNS)


# ── spaCy-based extractor (primary path) ──────────────────────────────────

class SpacyExtractor:
    """
    Extracts content-noun entities using spaCy's noun chunk parser + lemmatizer.

    Usage:
        extractor = SpacyExtractor()
        entities = extractor.extract("I bought three fish tanks and a filter.")
        # -> {"fish tank", "tank", "filter"}

    Model: en_core_web_sm. ~12MB. CPU-only is fine.
    """

    def __init__(self, model: str = "en_core_web_sm"):
        import spacy
        try:
            self.nlp = spacy.load(model)
        except OSError:
            raise RuntimeError(
                f"spaCy model '{model}' not installed. Run:\n"
                f"  python -m spacy download {model}\n"
                f"Or: pip install https://github.com/explosion/spacy-models/releases/"
                f"download/{model}-3.8.0/{model}-3.8.0-py3-none-any.whl"
            )

    def extract(self, text: str) -> set[str]:
        """
        Return normalized content-noun entities from text.

        Normalization:
          - Lowercased
          - Lemmatized (plurals -> singular, etc.)
          - Generic nouns removed
          - Compounds expanded to multiple granularities

        Returns a set so a cluster's entity list has no duplicates.
        """
        if not text or not text.strip():
            return set()

        try:
            doc = self.nlp(text)
        except Exception:
            # spaCy occasionally chokes on very long or malformed inputs.
            # Fall back to regex extraction rather than erroring.
            return _regex_extract(text)

        entities: set[str] = set()

        for chunk in doc.noun_chunks:
            # Filter tokens within the chunk:
            # - drop determiners, pronouns, punctuation
            # - keep nouns, proper nouns, adjectives (for compound modifiers)
            meaningful_tokens = [
                t for t in chunk
                if t.pos_ in {"NOUN", "PROPN", "ADJ"}
                and not t.is_stop
                and not t.is_punct
                and len(t.lemma_) > 1
            ]
            if not meaningful_tokens:
                continue

            # Lemmatize and lowercase
            lemmas = [t.lemma_.lower() for t in meaningful_tokens]

            # Filter out generic nouns and noise
            filtered = [
                l for l in lemmas
                if l not in _GENERIC_NOUNS and not _looks_like_noise(l)
            ]
            if not filtered:
                continue

            # Add the head noun (usually last non-adjective token)
            head_noun = None
            for t in reversed(meaningful_tokens):
                if t.pos_ in {"NOUN", "PROPN"}:
                    lemma = t.lemma_.lower()
                    if lemma not in _GENERIC_NOUNS and not _looks_like_noise(lemma):
                        head_noun = lemma
                        break
            if head_noun:
                entities.add(head_noun)

            # Add the full lemmatized phrase (for compound matching)
            if len(filtered) > 1:
                full_phrase = " ".join(filtered)
                entities.add(full_phrase)

                # Also add intermediate granularity for 3+ word compounds
                # "food delivery service" -> also add "delivery service"
                if len(filtered) >= 3:
                    for i in range(1, len(filtered) - 1):
                        entities.add(" ".join(filtered[i:]))

        # Also pull bare proper nouns that noun_chunks might miss
        # (e.g., "Databricks" standing alone)
        for t in doc:
            if t.pos_ == "PROPN" and not t.is_stop:
                lemma = t.lemma_.lower()
                if lemma not in _GENERIC_NOUNS and not _looks_like_noise(lemma) and len(lemma) > 2:
                    entities.add(lemma)

        return entities


# ── Public API ────────────────────────────────────────────────────────────

_DEFAULT_EXTRACTOR: SpacyExtractor | None = None


def get_extractor() -> SpacyExtractor:
    """Lazy singleton. Loading spaCy is ~1 second; do it once per process."""
    global _DEFAULT_EXTRACTOR
    if _DEFAULT_EXTRACTOR is None:
        _DEFAULT_EXTRACTOR = SpacyExtractor()
    return _DEFAULT_EXTRACTOR


def extract_entities(text: str) -> set[str] | None:
    """
    Extract entities from text using spaCy.

    Returns:
      set[str] of normalized entities if extraction succeeded
      None if spaCy is unavailable (caller should fall back to hybrid retrieval,
           NOT to regex extraction — regex is too lossy to be useful)

    Note: an empty set means "spaCy ran but found no content nouns" — that's
    different from None. Empty set is a real answer for queries like "tell me
    about my week" with no concrete entities.
    """
    try:
        return get_extractor().extract(text)
    except (RuntimeError, ImportError):
        return None


def extract_from_batch(texts: Iterable[str]) -> list[set[str]] | None:
    """
    Extract from multiple texts in one pass (shared spaCy pipeline).
    ~3x faster than calling extract_entities per item for large batches.

    Returns:
      list[set[str]], one set per input text, if spaCy is available
      None if spaCy is unavailable (caller should fall back)
    """
    try:
        extractor = get_extractor()
    except (RuntimeError, ImportError):
        return None

    # spaCy's nlp.pipe is ~3x faster than nlp() per document for batch work
    results: list[set[str]] = []
    for doc in extractor.nlp.pipe(list(texts), batch_size=32):
        entities: set[str] = set()
        for chunk in doc.noun_chunks:
            meaningful = [
                t for t in chunk
                if t.pos_ in {"NOUN", "PROPN", "ADJ"}
                and not t.is_stop
                and not t.is_punct
                and len(t.lemma_) > 1
            ]
            if not meaningful:
                continue
            lemmas = [t.lemma_.lower() for t in meaningful]
            filtered = [l for l in lemmas if l not in _GENERIC_NOUNS and not _looks_like_noise(l)]
            if not filtered:
                continue
            for t in reversed(meaningful):
                if t.pos_ in {"NOUN", "PROPN"}:
                    lemma = t.lemma_.lower()
                    if lemma not in _GENERIC_NOUNS and not _looks_like_noise(lemma):
                        entities.add(lemma)
                        break
            if len(filtered) > 1:
                entities.add(" ".join(filtered))
                if len(filtered) >= 3:
                    for i in range(1, len(filtered) - 1):
                        entities.add(" ".join(filtered[i:]))
        for t in doc:
            if t.pos_ == "PROPN" and not t.is_stop:
                lemma = t.lemma_.lower()
                if lemma not in _GENERIC_NOUNS and not _looks_like_noise(lemma) and len(lemma) > 2:
                    entities.add(lemma)
        results.append(entities)
    return results
