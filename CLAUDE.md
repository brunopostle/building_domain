# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->


## Build & Test

```bash
pytest
```

## Running the Extraction Pipeline

The pipeline must run in a **separate terminal** with `ANTHROPIC_API_KEY` set (cannot be set inside a Claude Code session).

### Recommended command (cost-optimised)

```bash
bsos extract --seed-apl --models claude-haiku-4-5-20251001 \
  --passes 1,2,3,4,5,6,7,8,9,10,11,12 \
  --framings 1 --workers 2
```

- `--framings 1` — single prompt framing per entity (3× cheaper than default 3)
- `--workers 2` — **do not exceed 2**: SQLite WAL shared-memory (-shm) fails with SQLITE_CANTOPEN under 3+ concurrent writers regardless of pool settings
- `--passes` — omit to run all, or specify e.g. `3,4,5,6,7,8,9,10,11,12` to resume after a crash

### Model routing (`bsos/llm/__init__.py`)

| Prefix | Provider | Key needed |
|--------|----------|------------|
| `claude-*` | Anthropic | `ANTHROPIC_API_KEY` |
| `ollama/<name>` | Local Ollama (free) | none |
| anything else | OpenAI-compatible | `OPENAI_API_KEY` |

**Free local option:** `--models ollama/llama3.1` (requires Ollama running on localhost:11434).
**Cheap cloud option:** any OpenAI-compatible provider (Groq, Together AI) via `OPENAI_API_KEY` + `OPENAI_BASE_URL`.

### Known issues

- **Pass 2 threshold:** `CLUSTER_DISTANCE_THRESHOLD = 0.04` in `pass2.py`. Values above ~0.08 merge distinct-but-related concepts (e.g. "High Ceiling Zone" / "Low Ceiling Zone"). Skip pass 2 or keep threshold at 0.04.
- **Pass 1 max_tokens:** Set to 16384 in `anthropic_provider.py`; 4096 is too small for the full concept-discovery response.
- **Crash recovery:** All LLM responses are cached in `bsos.db`. Re-running the same command resumes from where it left off at no extra API cost.
- **Undo bad pass 2 merges:** `UPDATE entities SET status='proposed' WHERE status='merged'; DELETE FROM entity_aliases;` then re-run pass 2.

### Pass 3 missing entities (Roof, Foundation, Beam, Wall)

If pass 3 was interrupted previously, these 4 entities have progress records but no assertions. Patch with:

```bash
bsos extract --seed-apl --models claude-haiku-4-5-20251001 --passes 3 --framings 1 --workers 2
```

Pass 3 skips already-done entities and fills in the gaps.
