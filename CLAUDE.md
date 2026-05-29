# Project Instructions for AI Agents

## Purpose

BSOS is a structured building domain knowledge base that makes implicit architectural and construction knowledge explicitly retrievable by LLM AI agents working with BIM/IFC models. LLM agents already contain broad building domain knowledge, but this knowledge does not reliably emerge at the point of need: an agent asked to sequence construction activities from an IFC model may correctly identify all the building elements but fail to apply rules like *"windows are inserted into masonry walls after the walls are built"* or *"internal finishes cannot start until the roof is watertight"*. BSOS surfaces this knowledge as a queryable graph, accessible via MCP tools.

**Primary consumers are AI agents, not humans.** The knowledge base is queried by agents to navigate, understand, improve, and design IFC building models.

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
- **LLM cache location:** Stored in `bsos_cache.db` (separate from `bsos.db`) using a single persistent connection + threading lock. Do NOT revert this — per-call `sqlite3.connect()` causes SQLITE_CANTOPEN under concurrent workers.
- **Crash recovery:** All LLM responses are cached in `bsos_cache.db`. Re-running the same command resumes from where it left off at no extra API cost.
- **Undo bad pass 2 merges:** `UPDATE entities SET status='proposed' WHERE status='merged'; DELETE FROM entity_aliases;` then re-run pass 2.
- **Groq provider:** Set `OPENAI_API_KEY=gsk_...` and `OPENAI_BASE_URL=https://api.groq.com/openai/v1` then use `--models llama-3.3-70b-versatile`. Quality is significantly lower than Haiku for complex structured output schemas — use Haiku or Sonnet for production runs.
- **Open-weight models (Llama, Groq):** Produce very sparse assertions (1-2 per entity vs 10-20 for Haiku). Not suitable for pass 3+.

### Recovering from phantom progress records

If the pipeline crashes mid-pass, some entities may have `pass_progress` records but no assertions. Clean up before restarting:

```bash
sqlite3 bsos.db "
  DELETE FROM pass_progress
  WHERE pass_number='3'
  AND NOT EXISTS (
    SELECT 1 FROM assertions a WHERE a.subject_id = entity_id
  );
"
```

Replace `'3'` with the relevant pass number. Then restart normally — resume logic fills the gaps.

### Pass 3 missing entities (Roof, Foundation, Beam, Wall)

These 4 entities had their assertions deleted during an earlier cleanup but their progress records persisted. After the current run completes, patch with:

```bash
bsos extract --seed-apl --models claude-haiku-4-5-20251001 --passes 3 --framings 1 --workers 2
```
