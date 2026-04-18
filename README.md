# Membrane

**A memory layer for LLMs. 85.2% on LongMemEval-S.**

Membrane is a conversational memory system built around one observation: *composite questions need composite retrieval*. A user asking "how much did I spend on workshops in the last four months?" needs the model to see multiple sessions at once — the workshops, the costs, the recent dates. A single embedding of the question text doesn't retrieve all of that. A question decomposer does.

That idea, plus clustered session indexing and intent-routed retrieval, is Membrane v4. It scores **85.2% on LongMemEval-S (cleaned)** with Anthropic's Sonnet 4.6 and Haiku 4.5, at a cost of ~$15 per full benchmark run.

## Why this matters

LongMemEval is the most rigorous public benchmark for long-term conversational memory: 500 questions across six categories, each with ~115K tokens of prior conversation. The challenge is retrieving the right information across ~50 past sessions without flooding the model with context.

Published results on the cleaned dataset, April 2026:

| System | Score | Answer Model | Reproducible |
|---|---|---|---|
| Zep | 71.2% | GPT-4o | Yes |
| TiMem | 76.88% | GPT-4o-mini | Yes |
| RetainDB | 79% | GPT-5.4 | Yes |
| EverMemOS | 83.0% | (varies) | Yes |
| Mastra OM | 84.23% | GPT-4o | Yes |
| **Membrane v4** | **85.20%** | **Sonnet 4.6 + Haiku 4.5** | **Yes** |
| Emergence AI | 86% | (closed) | No |

Membrane v4 outperforms the highest openly-reproducible published result (Mastra's 84.23% with GPT-4o). Caveat: judge model differs across systems; apples-to-apples comparison with GPT-4o as judge is work in progress.

## The architecture in one paragraph

Each user session is chunked into small topically-coherent clusters and embedded. A cheap Haiku call classifies each incoming question's intent (factual, recency, temporal, aggregation) and decomposes composite questions into 2-5 atomic search queries. Each sub-query runs its own embedding search. Results are unioned. Adjacent-cluster expansion pulls neighboring context. The final context goes to Sonnet for a single-pass answer. No pipelines, no multi-stage extraction, no agent loops.

## Reproduce the benchmark

```bash
git clone https://github.com/m-downing/membrane-ai.git
cd membrane-ai

pip install -r requirements.txt
./scripts/fetch_dataset.sh                # pulls 265MB dataset from HuggingFace
export ANTHROPIC_API_KEY=sk-ant-...

python run_decompose.py \
  --data benchmarks/data/longmemeval_s_cleaned.json \
  --output my_run.json --workers 6
```

Takes ~55 minutes at `workers=6`. Costs ~$15 in API calls. Compare against the canonical run:

```bash
python run_decompose.py --compare results/v4_full.json my_run.json
```

## Results, with receipts

Every number comes with the full item-by-item JSON in `results/`:

| File | Score | What it shows |
|---|---|---|
| `baseline_clustered.json` | 75.80% | Clustered retrieval alone, no procedure prompts, no decomposition |
| `v1_procedures.json` | 81.40% | Per-category answer prompts added (v1) |
| `v4_nodecompose.json` | 81.80% | v4 harness with decomposition **disabled** (ablation) |
| `v4_full.json` | **85.20%** | v4 full architecture |

The causal story is in the deltas:

- **Clustered retrieval** contributes the first 75.8pp — the substrate
- **Procedure prompts (v1)** add +5.6pp — better answer generation over the same retrieval
- **Decomposition (v4)** adds +3.4pp overall, with +9.0pp on multi-session and +6.7pp on single-session-preference

The ablation matters. It proves the 85.2% number isn't from wider retrieval, more context, or harness drift — it's decomposition specifically.

## Per-category breakdown (v4 vs ablation)

| Category | Ablation (no decompose) | v4 (decompose on) | Δ |
|---|---|---|---|
| Knowledge Update | 96.15% | 94.87% | -1.28 |
| Multi-Session | 65.41% | 74.44% | **+9.03** |
| Single-Session Assistant | 92.86% | 92.86% | 0.00 |
| Single-Session Preference | 76.67% | 83.33% | **+6.66** |
| Single-Session User | 98.57% | 98.57% | 0.00 |
| Temporal Reasoning | 77.44% | 80.45% | +3.01 |
| **Overall** | **81.80%** | **85.20%** | **+3.40** |

Where decomposition helps:
- **Multi-session aggregation** ("how many X have I done") — atomic sub-queries for each aspect
- **Single-session preference** ("recommend based on what you know about me") — multi-faceted evidence gathering
- **Multi-event temporal** ("order these three events") — one sub-query per named event

Where it's neutral or slightly negative:
- **Knowledge update** ("what's my current X") — decomposition can fragment a focused recency query. A future fix: skip decomposition for recency-intent questions.
- **Single-session lookups** — decomposer correctly classifies these as non-composite and passes through

## How it works

### Storage: clustered indexing

Traditional memory systems index per-session or per-turn. Membrane throws away session boundaries and builds semantically-coherent clusters (max ~600 tokens each) by greedy-merging adjacent turn-pairs. The session metadata is preserved for chronological ordering, but retrieval operates on clusters, not sessions.

Implementation: `run_benchmark.py` → `index_item_clustered()`.

### Retrieval: intent-routed

Before retrieval, a cheap Haiku call classifies the query as `factual`, `recency`, `temporal_adjacent`, or `aggregation`. Each intent drives a different retrieval strategy:

- **factual** — hybrid vector+keyword search, MMR reranking for diversity, ~8 clusters
- **recency** — similarity search + top-up from chronologically newest sessions, ~8 clusters
- **temporal_adjacent** — anchor on the most relevant session, expand to chronological neighbors, ~12 clusters
- **aggregation** — wide retrieval with per-session cap to cover multiple sessions, ~30 clusters

Implementation: `membrane.py` → `Membrane.promote()`.

### The v4 addition: question decomposition

The decomposer reads the question and asks: *is this composite*? If yes, it emits 2-5 atomic search phrases. Each phrase runs its own embedding search. Results union with the primary retrieval.

```python
# Question: "How much did I spend on workshops in the last 4 months?"
{
  "is_composite": true,
  "sub_queries": [
    "workshops attended",
    "workshop cost price fee",
    "registration payment"
  ]
}
```

Crucially, **decomposition enriches retrieval, not reasoning.** The final answer is still one Sonnet call with one prompt. No pipeline branching, no stage-2 narration, no source-access loss. The decomposer's only job is to widen what the answer step sees.

Implementation: `decomposer.py`.

### Answering: single-pass with per-category prompts

The answer step uses question-type-specific prompts (procedures). Each procedure tells the model how to structure its intermediate reasoning — list items, compute intervals, scan for supersession, etc. — but the final answer is one call, one model response.

Implementation: `procedures.py`, `run_decompose.py` → `process_item()`.

## Repository layout

```
membrane-ai/
├── membrane.py               # core Membrane class, intent classification, retrieval
├── store.py                  # FactStore with vector search
├── decomposer.py             # v4 mechanism: question decomposition
├── procedures.py             # per-category answer prompts
├── run_decompose.py          # v4 benchmark harness (current SOTA)
├── run_procedures.py         # v1 harness (baseline, kept for comparison)
├── run_benchmark.py          # shared indexing + comparison utilities
├── chat.py                   # interactive demo (non-benchmark)
├── scripts/fetch_dataset.sh  # fetches LongMemEval-S from HuggingFace
├── benchmarks/data/          # dataset goes here after fetch
└── results/                  # canonical benchmark JSONs
```

## Interactive demo

Not benchmark-related, but shows the memory layer's broader capabilities — fact extraction, contradiction detection, supersession chains, summarization:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python chat.py
```

Commands during chat: `/facts`, `/stats`, `/resolve`, `/demote`, `/session`, `/clear`, `/quit`.

## The LLM interface is pluggable

Membrane's LLM layer is duck-typed. Any class implementing `complete(system, user_message, max_tokens)` and `chat(system, messages, max_tokens)` can be passed as the `llm` argument to `Membrane`. The repo ships `AnthropicLLM` as the single supported backend; adding OpenAI, Gemini, or local model support is ~30 lines per provider.

## What's honest about this

A reader comparing Membrane to Mastra OM, Zep, or Mem0 should know:

- **Per-category answer prompts are benchmark-shaped.** The six procedures in `procedures.py` are tuned to LongMemEval's taxonomy. For production use outside the benchmark, a single adaptive answer prompt is straightforward but hasn't been written yet.
- **The retrieval substrate has headroom the answer step doesn't use.** The ablation audit shows that on wrong multi-session items, relevant sessions are in context 100% of the time — retrieval is delivering the data, the answer step is miscounting. The next architectural bet is not more retrieval; it's a better answer step.
- **Judge model is Haiku 4.5.** Published systems mostly use GPT-4o as judge. Judge variance can shift the overall number by 1-3pp in either direction. Cross-judging with GPT-4o is on the todo list.
- **Graph backend was removed in the v4 cleanup.** Earlier versions of Membrane included a Neo4j-backed graph store for typed edges (`ELABORATES`, `DEPENDS_ON`, `SUMMARIZES`). It wasn't load-bearing for the benchmark result, so it's been cut from v4 to keep the architecture single-path. The stubs remain in `store.py` for future reintegration if warranted.

## How we got here

Membrane's architecture emerged from six failed experiments before v4 landed:

- **Ledger** (structured claim store with per-session extraction) — failed at 37.9%, cost $54/run
- **MS verifier** (count-checking pass on aggregation questions) — null result
- **v2 two-stage** (structured extract → Python reduce → narrate) — failed at ~70%
- **v3 skills framework** (per-category parallel-scan specialists) — failed at 68.2%
- **v3 fixes** (surgical regex + routing tweaks) — failed at 68.8%
- **v4 decomposition** — **85.2%**

Every attempt to *replace* the answer step regressed. Every attempt to *add mechanism* additively worked. v4 is the smallest additive mechanism that moved the number — one cheap LLM call for decomposition, union retrieval, unchanged answer step.

## The analogy

This borrows from how the brain handles memory across hemispheres. The hippocampus doesn't reason — it stores indices into representations distributed across cortex and mediates retrieval. Membrane is the second hemisphere: it doesn't generate language, it manages state, scaffolds retrieval, and provides the context the generating half works with.

The model is the CPU. Membrane is the OS.

## Cost and latency

Per benchmark item (~115K tokens of prior conversation, one question):

- Indexing: ~3-5s (local embeddings, no API calls)
- Intent classification: ~0.3s, ~$0.0002 (one Haiku call)
- Decomposition: ~0.5s, ~$0.0003 (one Haiku call, ~300 tokens out)
- Retrieval: ~50ms (local vector math)
- Answer: ~5-8s, ~$0.025 (one Sonnet call, ~1500 tokens out)

Full benchmark run (500 items at `workers=6`): ~55 minutes, ~$15.

## Requirements

- Python 3.10+
- `anthropic>=0.40.0,<1.0`
- `sentence-transformers>=3.0.0,<4.0`
- `numpy>=1.26.0,<3.0`
- An Anthropic API key
- ~350MB disk (93MB repo + 265MB dataset)

See `requirements.txt`.

## License

See `LICENSE` for terms.

## Status and roadmap

**Where v4 is strong:** Multi-session aggregation, temporal reasoning with multiple events, preference elicitation — the categories where composite retrieval genuinely helps.

**Where it's weak:** Knowledge-update questions are slightly hurt by decomposition (-1.3pp). The answer step is the remaining bottleneck on multi-session — retrieval has already delivered the data.

**Next architectural bets:**
1. Skip decomposition for recency-intent questions (recover the KU regression)
2. Adaptive answer prompting to replace per-category procedures
3. Answer-step improvements for multi-session enumeration — the retrieval is already solved

---

Membrane v4 is reproducible today. `git clone`, `./scripts/fetch_dataset.sh`, `python run_decompose.py`, and you have your own 85.2%.

