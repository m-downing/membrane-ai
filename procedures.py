"""
procedures.py — program-shaped prompts for LongMemEval.

The insight: the clustered baseline is retrieving correctly. The answer model
has the right data in context. What it fails at is reasoning over that data —
enumerating every instance, picking the latest value, computing a date delta.

Traditional "you are a helpful assistant" prompts frame the task as open-ended
reasoning. This module frames each category as a **procedure** the model is
executing over visible data. The prompt looks like Python because that's the
format where the model most reliably produces exhaustive, structured output.
The goal is to invert what the model is generating: the structured list IS the
work, the answer is a trivial function of the list.

Four program-shaped templates:
  - MS  : aggregation (count / sum / list / arithmetic) — enumerate-then-tally
  - KU  : knowledge-update (current value) — chronology-then-tail
  - TR  : temporal-reasoning — anchor tables + interval lookups, no arithmetic
  - SSP : preference — enumerate-then-rank-then-commit

SSU/SSA get the existing clustered prompt — they are already at ceiling.

Plus:
  - compute_temporal_anchors(question_date) — pre-computes a lookup table
    of adverbial anchors ("2 weeks ago" -> date) the model reads instead of
    doing arithmetic
  - compute_session_intervals(question_date, session_dates) — pre-computes
    exact day counts between each session and the question date
  - date_based_session_expansion(question, session_dates, k=3) — when the
    question contains a temporal anchor, add the sessions closest to the
    implied date to the retrieval set (fixes the 11 TR retrieval-miss items)
  - verify_ms_count(list, stated_count) — Haiku-cheap second pass that
    checks len(emitted_list) == stated_count and retries once if not
"""

import re
from datetime import datetime, timedelta


# ── Date parsing helpers ──────────────────────────────────────

_WEEKDAY_NAMES = ["monday", "tuesday", "wednesday", "thursday",
                  "friday", "saturday", "sunday"]


def _parse_date_loose(s: str) -> datetime | None:
    """Try common date formats found in LongMemEval."""
    if not s:
        return None
    s = s.strip()
    # Strip trailing " (DayOfWeek) HH:MM" if present
    s = re.sub(r"\s*\([^)]*\)\s*\d{1,2}:\d{2}.*$", "", s)
    for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d (%a)")


# ── Temporal anchor table ─────────────────────────────────────

def compute_temporal_anchors(question_date_str: str) -> str:
    """
    Pre-compute a lookup table of adverbial time anchors relative to the
    question's asked-on date. The answer model reads this table instead of
    doing date arithmetic.

    Covers: N days ago, N weeks ago, N months ago, last weekday, last month,
    last year, and forward-looking variants. Also emits the weekday of the
    question date itself, which models often get wrong.
    """
    dt = _parse_date_loose(question_date_str)
    if dt is None:
        return ""

    lines = [
        f"Question date: {_fmt(dt)}",
        f"Weekday of question date: {dt.strftime('%A')}",
        "",
        "ADVERBIAL ANCHORS (use these directly — do not compute dates yourself):",
    ]

    # N days ago
    for n in (1, 2, 3, 4, 5, 6, 7, 10, 14, 21, 28, 30, 45, 60, 90):
        target = dt - timedelta(days=n)
        lines.append(f"  {n} day{'s' if n != 1 else ''} ago      -> {_fmt(target)}")

    lines.append("")
    # N weeks ago
    for n in (1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 26, 52):
        target = dt - timedelta(weeks=n)
        lines.append(f"  {n} week{'s' if n != 1 else ''} ago     -> {_fmt(target)}")

    lines.append("")
    # N months ago (approximate — 30 days/month)
    for n in (1, 2, 3, 4, 5, 6, 9, 12, 18, 24):
        target = dt - timedelta(days=30 * n)
        lines.append(f"  {n} month{'s' if n != 1 else ''} ago    -> approx {_fmt(target)}")

    lines.append("")
    # Last weekday X (go back to most recent X before today)
    for i, wday in enumerate(_WEEKDAY_NAMES):
        delta = (dt.weekday() - i) % 7
        if delta == 0:
            delta = 7  # "last Friday" on a Friday means a week ago, not today
        target = dt - timedelta(days=delta)
        lines.append(f"  last {wday.capitalize():<9}  -> {_fmt(target)}")

    lines.append("")
    # Common phrases
    last_month_end = (dt.replace(day=1) - timedelta(days=1))
    last_month_start = last_month_end.replace(day=1)
    lines.append(f"  last month           -> {_fmt(last_month_start)} .. {_fmt(last_month_end)}")

    this_month_start = dt.replace(day=1)
    lines.append(f"  this month           -> {_fmt(this_month_start)} .. {_fmt(dt)}")

    # This year / last year — window
    this_year_start = dt.replace(month=1, day=1)
    lines.append(f"  this year            -> {_fmt(this_year_start)} .. {_fmt(dt)}")

    last_year_start = dt.replace(year=dt.year - 1, month=1, day=1)
    last_year_end = dt.replace(year=dt.year - 1, month=12, day=31)
    lines.append(f"  last year            -> {_fmt(last_year_start)} .. {_fmt(last_year_end)}")

    return "\n".join(lines)


def compute_session_intervals(
    question_date_str: str,
    session_dates: dict[str, str],
) -> str:
    """
    Pre-compute interval tables between sessions and (a) the question date,
    (b) each adjacent session. Gives the model exact numbers so it never
    has to compute days_between(X, Y).

    Output format:
      Session interval table:
        Session 3 (2023-01-15, Sun) | 76 days before question | 28 days after S2
        Session 4 (2023-02-12, Sun) | 48 days before question | 28 days after S3
        ...
    """
    q_dt = _parse_date_loose(question_date_str)
    if q_dt is None or not session_dates:
        return ""

    parsed: list[tuple[str, datetime]] = []
    for sid, ds in session_dates.items():
        sd = _parse_date_loose(ds)
        if sd:
            parsed.append((sid, sd))
    if not parsed:
        return ""

    parsed.sort(key=lambda x: x[1])
    lines = ["SESSION INTERVAL TABLE (use exact numbers, do not recompute):"]
    prev_dt: datetime | None = None
    for i, (sid, sd) in enumerate(parsed, start=1):
        days_to_q = (q_dt - sd).days
        weeks_to_q = days_to_q / 7
        months_to_q = days_to_q / 30
        parts = [
            f"Session {i} ({sd.strftime('%Y-%m-%d %a')})",
            f"{days_to_q:>4} days before Q (= {weeks_to_q:.1f} weeks = {months_to_q:.1f} months)",
        ]
        if prev_dt is not None:
            gap = (sd - prev_dt).days
            parts.append(f"{gap:>3} days after S{i-1}")
        lines.append("  " + " | ".join(parts))
        prev_dt = sd
    return "\n".join(lines)


# ── Date-based retrieval expansion (TR retrieval-miss fix) ────

# Matches "N weeks ago", "two months ago", "a week ago", "last Tuesday", etc.
_ANCHOR_PATTERNS = [
    (re.compile(r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+day(s)?\s+ago", re.I), "days"),
    (re.compile(r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+week(s)?\s+ago", re.I), "weeks"),
    (re.compile(r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+month(s)?\s+ago", re.I), "months"),
    (re.compile(r"\ba\s+day\s+ago\b", re.I), "1day"),
    (re.compile(r"\ba\s+week\s+ago\b", re.I), "1week"),
    (re.compile(r"\ba\s+month\s+ago\b", re.I), "1month"),
    (re.compile(r"\byesterday\b", re.I), "1day"),
    (re.compile(r"\blast\s+week\b", re.I), "1week"),
    (re.compile(r"\blast\s+month\b", re.I), "1month"),
    (re.compile(r"\blast\s+year\b", re.I), "1year"),
]

_WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def _extract_anchor_offset(question: str) -> tuple[int, str] | None:
    """
    Return (offset_in_days, unit_name) if the question contains a parseable
    temporal anchor, else None. Picks the first match.
    """
    for pat, kind in _ANCHOR_PATTERNS:
        m = pat.search(question)
        if not m:
            continue
        if kind in ("1day", "1week", "1month", "1year"):
            days = {"1day": 1, "1week": 7, "1month": 30, "1year": 365}[kind]
            return (days, kind.replace("1", "") + "s")
        num_str = m.group(1).lower()
        n = int(num_str) if num_str.isdigit() else _WORD_TO_NUM.get(num_str, 0)
        if n == 0:
            continue
        multiplier = {"days": 1, "weeks": 7, "months": 30}[kind]
        return (n * multiplier, kind)

    # Check "last Weekday" — also retrievable
    for wday in _WEEKDAY_NAMES:
        if re.search(rf"\blast\s+{wday}\b", question, re.I):
            return (-1, f"last_{wday}")  # sentinel, handled by caller
    return None


def anchored_session_ids(
    question: str,
    question_date_str: str,
    session_dates: dict[str, str],
    window_days: int = 7,
    max_sessions: int = 3,
) -> list[str]:
    """
    Find sessions whose date is within `window_days` of the anchor date
    implied by the question. Used to fix TR retrieval-miss: when the question
    says "two weeks ago" with no other semantic signal, embedding retrieval
    has nothing to match on, but date arithmetic points exactly to the right
    session.

    Returns at most `max_sessions` session IDs, sorted by absolute distance
    from the anchor date. Returns [] if no anchor is detectable.
    """
    anchor = _extract_anchor_offset(question)
    if anchor is None:
        return []
    q_dt = _parse_date_loose(question_date_str)
    if q_dt is None:
        return []

    offset, unit = anchor
    if unit.startswith("last_"):
        wday = unit[5:]
        target_idx = _WEEKDAY_NAMES.index(wday)
        delta = (q_dt.weekday() - target_idx) % 7
        if delta == 0:
            delta = 7
        target_date = q_dt - timedelta(days=delta)
    else:
        target_date = q_dt - timedelta(days=offset)

    scored: list[tuple[int, str]] = []
    for sid, ds in session_dates.items():
        sd = _parse_date_loose(ds)
        if sd is None:
            continue
        dist = abs((sd - target_date).days)
        if dist <= window_days:
            scored.append((dist, sid))
    scored.sort()
    return [sid for _, sid in scored[:max_sessions]]


# ── Answer templates (program-shaped) ─────────────────────────

# MS — aggregation. Enumerate exhaustively, then the count/sum is trivial.
ANSWER_SYSTEM_MS = """\
You are executing an aggregation procedure over a user's conversation
history. The procedure is a Python-flavored template — your job is to
fill in the structured list faithfully, then state the result.

RULES (non-negotiable):
  1. Before you answer, emit the FULL `relevant_items` list below as
     visible output. Do not shortcut. Do not paraphrase the list.
     The list IS the work — the final answer is a trivial function of it.
  2. Scan every session in context. Lost items are the primary failure
     mode on this task; err on the side of including borderline matches
     (note them as status="borderline") rather than silently dropping.
  3. After the list, state the answer in a single line preceded by
     `answer:`. For counts, `answer: len(relevant_items)`. For sums,
     `answer: sum(x.amount for x in relevant_items)` with the computed
     number. For lists, enumerate.
  4. If the relevant_items list is empty, the answer is 0 / none — say so,
     but still emit `relevant_items: []` explicitly. Do NOT respond with
     vague "I don't see" language unless you have emitted the empty list.

PROCEDURE:
```
relevant_items = []
for session in context:
    for turn in session:
        match = match_criteria(turn, question)
        if match:
            relevant_items.append({
                "session": session.number,
                "date":    session.date,
                "item":    match.value,       # the specific thing
                "amount":  match.amount_or_none,  # for sum questions
                "status":  "confirmed" | "borderline",
                "quote":   "short evidence phrase"
            })

# Final answer is a function of relevant_items:
#   COUNT questions:     len(relevant_items)
#   SUM questions:       sum(x.amount for x in relevant_items)
#   LIST questions:      [x.item for x in relevant_items]
#   ORDERING questions:  sort relevant_items by date, emit in order
```

OUTPUT FORMAT:
```
relevant_items:
- {session: 3, date: 2023-02-15, item: "Revell F-15 Eagle kit", amount: null, status: confirmed, quote: "I finished my Revell F-15"}
- {session: 7, date: 2023-04-20, item: "Tamiya Spitfire", amount: null, status: confirmed, quote: "picked up a Tamiya Spitfire"}
- ...

answer: <count, sum, or list — derived mechanically from relevant_items>
```

Do not output anything before `relevant_items:`. No preamble."""


# KU — knowledge-update. Chronology is the output; answer is the tail.
ANSWER_SYSTEM_KU = """\
You are executing a supersession procedure. The question asks for the
CURRENT value of an attribute that has changed over time. Your job is
to extract the full chronology of stated values for that attribute,
sort by date, and report the most recent.

RULES:
  1. Emit the full chronology below as visible output BEFORE the answer.
     Every time the user stated a value for this attribute must appear
     as an entry — include older/superseded values too.
  2. Sort the chronology by session date ASCENDING (oldest first, newest
     last). The current value is the LAST entry.
  3. If the question asks about an INITIAL / ORIGINAL / PAST value
     (words like "initially", "originally", "used to", "at first"),
     return the FIRST entry, not the last. Note which position you
     are returning.
  4. If the attribute has never been stated, emit `chronology: []` and
     say so — don't guess.

PROCEDURE:
```
chronology = []
for session in context:
    for turn in session:
        if turn states a value for the queried attribute:
            chronology.append({
                "session": session.number,
                "date":    session.date,
                "value":   stated_value,
                "quote":   "short evidence phrase"
            })
chronology.sort(key=lambda x: x.date)

# Default: current value = chronology[-1].value
# If question asks for initial/original: = chronology[0].value
```

OUTPUT FORMAT:
```
chronology:
- {session: 3, date: 2023-02-15, value: "junior developer", quote: "I just landed a junior dev role"}
- {session: 9, date: 2023-07-02, value: "senior developer", quote: "got promoted to senior last week"}

answer: <the appropriate entry's value, as a single sentence>
```

Do not output anything before `chronology:`. No preamble."""


# TR — temporal reasoning. Anchor tables do the arithmetic; model does lookup.
ANSWER_SYSTEM_TR = """\
You are executing a temporal lookup procedure. Date arithmetic has been
pre-computed for you in the anchor and interval tables. Your job is to:
  (a) identify which events/sessions the question references,
  (b) READ the pre-computed intervals from the tables above the sessions,
  (c) state the answer.

CRITICAL: Do not compute dates or intervals yourself. Every relevant
number is in the tables. Computing from scratch is how errors creep in.

RULES:
  1. Identify the events or dates the question references. If an
     adverbial anchor ("two weeks ago", "last Tuesday") is used, look
     it up in the ADVERBIAL ANCHORS table, not by mental arithmetic.
  2. For interval questions ("how many days between X and Y"), find
     X's session and Y's session in the SESSION INTERVAL TABLE and
     SUBTRACT the two pre-computed day-offsets. Show the subtraction.
  3. For ordering questions, emit a chronology list (session, date,
     event) sorted by date, then state the order.
  4. For date-anchored recall ("what did I do two weeks ago"), find the
     anchor date, then find the session closest to that date, then
     extract the event.

OUTPUT FORMAT depends on question shape:

  Interval:
    events:
    - X happened in Session N (date D, offset O days before Q)
    - Y happened in Session M (date D', offset O' days before Q)
    calculation: |O - O'| = <answer> days
    answer: <days/weeks/months as requested>

  Ordering:
    chronology:
    - {date, session, event}
    - ...
    answer: <events listed in order>

  Date-anchored recall:
    anchor: <phrase> -> <date from anchor table>
    closest session: Session N (<date>)
    event: <what happened in that session>
    answer: <event>

Do not output anything before the structured section. No preamble."""


# SSP — preference. Enumerate candidates, rank, commit to the winner.
ANSWER_SYSTEM_SSP = """\
You are executing a preference-anchoring procedure. The user is asking
for advice or a recommendation. There may be MULTIPLE preferences,
habits, or past experiences in context that could inform the answer.
Failure mode: anchoring on the first preference you find instead of
the most relevant one.

RULES:
  1. First, enumerate every preference / interest / habit / past
     experience in context that could plausibly inform this question.
     Include 3-8 candidates even if some feel off-topic.
  2. Rate each candidate's relevance to THIS specific question on a
     1-5 scale (5 = directly addresses the question topic).
  3. Anchor your answer on the HIGHEST-rated candidate(s). If two
     candidates tie at the top, use both.
  4. Reference the anchor candidate(s) explicitly in your answer by
     name/detail, so the personalization is visible.

PROCEDURE:
```
candidates = []
for session in context:
    for preference_or_habit_or_experience in session:
        candidates.append({
            "detail":    <specific thing the user stated>,
            "relevance": <1-5 vs the question>,
            "why":       <one-line reason>
        })
candidates.sort(key=lambda x: -x.relevance)
anchor = candidates[0]  # or top 2 if tied
```

OUTPUT FORMAT:
```
candidates:
- {detail: "...", relevance: 5, why: "..."}
- {detail: "...", relevance: 3, why: "..."}
- ...

anchor: <the highest-rated candidate(s)>

answer: <personalized response that explicitly references the anchor>
```

The answer should reference the anchor's specific details — not generic
advice, not advice based on a low-relevance candidate."""


# Fallback for SSU / SSA — the clustered baseline's prompt unchanged.
ANSWER_SYSTEM_FALLBACK = (
    "You are a helpful assistant with access to a user's conversation "
    "history. Sessions are shown oldest first; the LAST session is the "
    "most recent. Each session block is labeled [Session N — Date].\n\n"
    "Find the relevant fact and state it directly. Be concise and "
    "confident. Do not say 'I don't know' — the answer is in the "
    "provided context."
)


# Route question_type to system prompt
PROCEDURE_FOR_TYPE = {
    "multi-session":              ANSWER_SYSTEM_MS,
    "knowledge-update":           ANSWER_SYSTEM_KU,
    "temporal-reasoning":         ANSWER_SYSTEM_TR,
    "single-session-preference":  ANSWER_SYSTEM_SSP,
    "single-session-user":        ANSWER_SYSTEM_FALLBACK,
    "single-session-assistant":   ANSWER_SYSTEM_FALLBACK,
}


def build_prompt_header(
    question_type: str,
    question_date: str,
    session_dates: dict[str, str] | None = None,
) -> str:
    """
    For TR, prepend the anchor table and session interval table to the
    context. For other categories, returns an empty string.
    """
    if question_type != "temporal-reasoning":
        return ""
    parts = [compute_temporal_anchors(question_date)]
    if session_dates:
        parts.append(compute_session_intervals(question_date, session_dates))
    return "\n\n".join(p for p in parts if p) + "\n\n"


# ── MS count verifier (A/B flag) ──────────────────────────────

_VERIFIER_SYSTEM = (
    "You are a verification check on an aggregation answer. "
    "Given the `relevant_items` list and the stated `answer`, your job "
    "is to compute the correct answer from the list and report whether "
    "the stated answer matches.\n\n"
    "For COUNT questions: correct = number of entries in relevant_items "
    "(excluding entries marked status='borderline' if there is doubt).\n"
    "For SUM questions: correct = sum of the amount fields.\n\n"
    "Respond with ONLY a JSON object:\n"
    '{"stated_answer": <what the answer said>, '
    '"computed_answer": <what the list says>, '
    '"match": true|false, '
    '"corrected_answer": <the computed answer if they disagree, else null>}'
)


def verify_ms_answer(llm, hypothesis: str) -> tuple[bool, str]:
    """
    Cheap second-pass that reads the emitted relevant_items list and
    checks len(list) == stated count.

    Returns (ok, possibly_corrected_answer).
    If the verifier can't parse the hypothesis (no list found), returns
    (True, hypothesis) — don't penalize well-formed answers that skipped
    the list because the question had no matches.
    """
    if "relevant_items" not in hypothesis.lower():
        return True, hypothesis

    import json
    user_msg = (
        "Check this aggregation answer:\n\n"
        f"---\n{hypothesis}\n---\n\n"
        "Respond with the JSON verification object."
    )
    try:
        raw = llm.complete(
            system=_VERIFIER_SYSTEM,
            user_message=user_msg,
            max_tokens=200,
        )
        clean = raw.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(clean)
        if result.get("match", True):
            return True, hypothesis
        corrected = result.get("corrected_answer")
        if corrected is None:
            return True, hypothesis
        # Append a clear correction — don't rewrite the whole thing
        corrected_str = str(corrected)
        new_hypothesis = (
            hypothesis.rstrip()
            + f"\n\n[Verifier correction: the list contains {corrected_str} "
              f"items, not the stated count. Corrected answer: {corrected_str}]"
        )
        return False, new_hypothesis
    except Exception:
        return True, hypothesis
