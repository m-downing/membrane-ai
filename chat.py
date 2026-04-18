#!/usr/bin/env python3
"""
Membrane v0 — Virtual Memory for LLMs
Interactive demo.

Usage:
    export ANTHROPIC_API_KEY=sk-...
    python chat.py

Commands during chat:
    /facts     — show all stored facts
    /stats     — show membrane statistics
    /clear     — clear all memory
    /session   — start a new session (simulates closing & reopening)
    /quit      — exit

The magic: close the chat, reopen it, and the model remembers you.
"""

import sys
import os
from membrane import Membrane, AnthropicLLM


CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_membrane(msg: str):
    print(f"{DIM}[membrane] {msg}{RESET}")


def print_header():
    print(f"""
{BOLD}{'='*60}
  MEMBRANE v0 — Virtual Memory for LLMs
{'='*60}{RESET}

  This chat has {BOLD}persistent memory{RESET} across sessions.
  The model doesn't know it has memory — the membrane
  manages context on its behalf.

  Commands: /facts /stats /clear /session /demote /resolve /quit
""")


def main():
# ── Setup ────────────────────────────────────────────
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(f"{RED}Set ANTHROPIC_API_KEY environment variable first.{RESET}")
        sys.exit(1)

    print_header()

    store_path = os.environ.get("MEMBRANE_STORE", "facts.json")
    model      = os.environ.get("MEMBRANE_MODEL", "claude-sonnet-4-6")

    membrane = Membrane(store_path=store_path, llm=AnthropicLLM(model=model))
    print_membrane(f"Backend: flat (JSON at {store_path})")

    history: list[dict] = []
    fact_count = len(membrane.store)

    if fact_count > 0:
        print_membrane(f"Loaded {fact_count} facts from previous sessions")
    else:
        print_membrane("Fresh start — no memories yet")
    print()

    # ── Chat loop ────────────────────────────────────────
    while True:
        try:
            user_input = input(f"{BOLD}You:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Goodbye!{RESET}")
            break

        if not user_input:
            continue

        # ── Commands ─────────────────────────────────────
        if user_input.startswith("/"):
            cmd = user_input.lower().split()[0]

            if cmd == "/quit":
                print(f"{DIM}Goodbye!{RESET}")
                break

            elif cmd == "/facts":
                facts = membrane.store.get_all_active()
                if not facts:
                    print_membrane("No facts stored yet.")
                else:
                    print_membrane(f"{len(facts)} facts in memory:")
                    for f in sorted(facts, key=lambda x: x.created_at):
                        accessed = f"accessed {f.access_count}x" if f.access_count > 0 else "never accessed"
                        print(f"  {DIM}[{f.id}]{RESET} {f.content}")
                        print(f"       {DIM}{accessed} | session {f.session_id}{RESET}")
                print()
                continue

            elif cmd == "/stats":
                active = membrane.store.get_all_active()
                superseded = [f for f in membrane.store.facts.values() if f.superseded_by]
                print_membrane(f"Active facts: {len(active)}")
                print_membrane(f"Superseded facts: {len(superseded)}")
                print_membrane(f"Session: {membrane.session_id}")
                print_membrane(f"History turns: {len(history) // 2}")
                print()
                continue

            elif cmd == "/clear":
                confirm = input(f"  {YELLOW}Clear all memory? (y/n): {RESET}").strip().lower()
                if confirm == "y":
                    membrane.store.clear_all()
                    print_membrane("Memory cleared.")
                else:
                    print_membrane("Cancelled.")
                print()
                continue

            elif cmd == "/session":
                # Simulate closing and reopening the chat
                history.clear()
                membrane.session_id = __import__("uuid").uuid4().hex[:8]
                fact_count = len(membrane.store)
                print()
                print(f"  {YELLOW}── New session ──{RESET}")
                print(f"  {DIM}Conversation history cleared.{RESET}")
                print(f"  {DIM}Memory persists: {fact_count} facts available.{RESET}")
                print()
                continue

            elif cmd == "/demote":
                # Parse optional min_cluster arg: /demote 3
                parts = user_input.split()
                try:
                    min_cluster = int(parts[1]) if len(parts) > 1 else membrane.demotion_min_cluster
                except ValueError:
                    min_cluster = membrane.demotion_min_cluster

                demotions = membrane.demote(min_cluster_size=min_cluster)
                membrane.store.save()
                if not demotions:
                    print_membrane(f"No clusters with >= {min_cluster} children found.")
                else:
                    print_membrane(f"Demoted {len(demotions)} cluster(s):")
                    for d in demotions:
                        n = len(d["replaced"])
                        print(f"  {DIM}  ↓ ({n} facts → 1){RESET}")
                        for f in d["replaced"]:
                            print(f"  {DIM}    - {f.content}{RESET}")
                        print(f"  {GREEN}    + {d['summary'].content}{RESET}")
                print()
                continue

            elif cmd == "/resolve":
                pairs = membrane.store.get_contradictions()
                if not pairs:
                    print_membrane("No unresolved contradictions.")
                    print()
                    continue

                print_membrane(f"{len(pairs)} unresolved contradiction(s):\n")
                resolved = 0
                for i, (fact_a, fact_b) in enumerate(pairs, 1):
                    print(f"  {YELLOW}── Contradiction {i} of {len(pairs)} ──{RESET}")
                    print(f"  {BOLD}[1]{RESET} {fact_a.content}")
                    print(f"  {BOLD}[2]{RESET} {fact_b.content}")
                    print()
                    try:
                        choice = input(
                            f"  Which is current? "
                            f"{DIM}(1=keep first, 2=keep second, s=skip, q=quit): {RESET}"
                        ).strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        print()
                        break

                    if choice == "q":
                        remaining = len(pairs) - i
                        if remaining > 0:
                            print_membrane(f"Exiting. {remaining} contradiction(s) remaining.")
                        break
                    elif choice == "1":
                        membrane.store.supersede(fact_b.id, fact_a.id)
                        membrane.store.remove_contradicts(fact_a.id, fact_b.id)
                        membrane.store.save()
                        print(f"  {GREEN}→ Kept: \"{fact_a.content}\"{RESET}")
                        print(f"  {DIM}  Retired: \"{fact_b.content}\"{RESET}")
                        resolved += 1
                    elif choice == "2":
                        membrane.store.supersede(fact_a.id, fact_b.id)
                        membrane.store.remove_contradicts(fact_a.id, fact_b.id)
                        membrane.store.save()
                        print(f"  {GREEN}→ Kept: \"{fact_b.content}\"{RESET}")
                        print(f"  {DIM}  Retired: \"{fact_a.content}\"{RESET}")
                        resolved += 1
                    else:
                        print(f"  {DIM}Skipped.{RESET}")
                    print()

                if resolved:
                    print_membrane(f"Resolved {resolved} contradiction(s). Total facts: {len(membrane.store)}")
                print()
                continue

            else:
                print_membrane(f"Unknown command: {cmd}")
                print()
                continue

        # ── PROMOTE ──────────────────────────────────────
        system, promoted = membrane.before_turn(user_input)

        if promoted:
            n_traversal  = len(membrane._traversal_ids)
            n_centrality = len(membrane._centrality_ids)
            label = f"Promoted {len(promoted)} facts into context"
            tags = []
            if n_traversal:
                tags.append(f"{n_traversal} via traversal")
            if n_centrality:
                tags.append(f"{n_centrality} via centrality")
            if tags:
                label += f" ({', '.join(tags)})"
            print_membrane(label + ":")
            for fact, score in promoted[:5]:
                if fact.id in membrane._centrality_ids:
                    marker = "^"
                elif fact.id in membrane._traversal_ids:
                    marker = "~"
                else:
                    marker = " "
                print(f"  {DIM}  ({score:.2f}{marker}) {fact.content}{RESET}")
            if len(promoted) > 5:
                print(f"  {DIM}  ... and {len(promoted) - 5} more{RESET}")

        # ── GENERATE ─────────────────────────────────────
        history.append({"role": "user", "content": user_input})

        try:
            response = membrane.llm.chat(
                system=system,
                messages=history,
                max_tokens=2048,
            )
        except Exception as e:
            print(f"{RED}API error: {e}{RESET}")
            history.pop()  # remove failed user message
            print()
            continue

        history.append({"role": "assistant", "content": response})
        print(f"\n{CYAN}Claude:{RESET} {response}\n")

        # ── EXTRACT + REPAIR ─────────────────────────────
        result = membrane.after_turn(user_input, response, promoted)

        if result["new_facts"]:
            print_membrane(f"Extracted {len(result['new_facts'])} new facts:")
            for f in result["new_facts"]:
                print(f"  {GREEN}  + {f}{RESET}")

        if result.get("contradictions"):
            print_membrane(f"Contradictions detected ({len(result['contradictions'])}):")
            for a, b in result["contradictions"]:
                print(f"  {YELLOW}  ↔ \"{a}\"")
                print(f"      \"{b}\"{RESET}")

        if result.get("demotions"):
            print_membrane(f"Demoted {len(result['demotions'])} cluster(s):")
            for d in result["demotions"]:
                print(f"  {DIM}  ↓ ({d['replaced_count']} facts → 1) {d['summary']}{RESET}")

        if result["correction"]:
            print(f"\n  {YELLOW}⚠  Contradiction detected — requesting correction...{RESET}")

            # Inject correction and get a new response
            history.append({"role": "user", "content": result["correction"]})
            try:
                corrected = membrane.llm.chat(
                    system=system,
                    messages=history,
                    max_tokens=2048,
                )
                history.append({"role": "assistant", "content": corrected})
                print(f"\n{CYAN}Claude (corrected):{RESET} {corrected}\n")
            except Exception as e:
                print(f"{RED}Correction failed: {e}{RESET}")

        print_membrane(f"Total facts: {result['total_facts']}")
        print()


if __name__ == "__main__":
    main()
