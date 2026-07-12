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


## Persisting Database Changes

`bsos.db` (and `bsos_cache.db`, `.bsos_config`) are **git-ignored** — they are too
large for GitHub. The git-tracked form of the knowledge base is the per-table JSON
serialization in `data/snapshot/`. **If a session mutates `bsos.db`, the change is
NOT persisted until you regenerate the affected snapshot file(s) and push them.**
Closing a beads issue without doing this strands the DB work on your local disk.

As part of session completion, whenever you have changed the database:

```bash
# Regenerate only the table(s) you changed (keeps the diff small).
# Directory output writes one <type>.json per type; a trailing slash forces dir mode.
bsos export --format json --type entities --output data/snapshot/
# Repeat --type for each changed table: assertions, constraints, patterns,
# forces, antipatterns, abstraction_nodes, spatial_relations, process_relations.
# Omit --type to regenerate the entire snapshot.

git add data/snapshot/
git commit -m "snapshot: <what changed>"
git push
```

The snapshot includes **all** statuses (proposed/accepted/deprecated/merged) — there is
no status filter — so deprecations and in-place adoptions are captured, not just accepted
rows. `data/snapshot/antipatterns.json` and `patterns.json` are ~75–95 MB; each stays
under GitHub's 100 MB/file limit, which is why the export splits per-table rather than
writing a single `bsos_snapshot.json`.

## Build & Test

```bash
pytest
```

## Running the Extraction Pipeline

The pipeline must run in a **separate terminal** with the provider API key set
(cannot be set inside a Claude Code session).

### Clean-by-construction procedure (from an empty DB)

This is the documented procedure that reproduces a clean knowledge base from
scratch with **no manual SQL cleanup afterwards**. The pipeline's guardrails
(below) keep junk out by construction; run the steps in order:

```bash
# 1. Create the database and run migrations (deterministic, no API key).
bsos init

# 2. Seed canonical IFC classes from the authoritative schema BEFORE extraction,
#    so the ifc_class layer is schema-authoritative and the LLM never invents
#    class names. Add `--schema IFC4X3` (repeatable) for infrastructure classes.
bsos seed-ifc-classes

# 3. (Recommended) populate curated IFC property-set recommendations.
bsos seed-psets

# 4. Smoke-test the provider before committing to a long run (see DeepSeek below).
python scripts/smoke_test_deepseek.py     # exit 0 = HEALTHY

# 5. Run all 12 passes. deepseek-chat (V3) is the production model — it completed
#    the full run at feasible cost (Anthropic is too expensive for all 12 passes).
export OPENAI_API_KEY=sk-<deepseek key>
export OPENAI_BASE_URL=https://api.deepseek.com
bsos extract --seed-apl --models deepseek-chat \
  --passes 1,2,3,4,5,6,7,8,9,10,11,12 \
  --framings 1 --workers 2
```

After this completes the graph is fully normalized and needs no cleanup. The
output is `status='proposed'`; review/accept it separately (building_domain-lfy).

`claude-haiku-4-5-20251001` is a quality-equivalent drop-in for step 5 (measured
~21 assertions/entity, same as V3) if budget allows; it needs `ANTHROPIC_API_KEY`
instead of the OpenAI-compatible vars.

- `--framings 1` — single prompt framing per entity (3× cheaper than default 3)
- `--workers 2` — **do not exceed 2**: SQLite WAL shared-memory (-shm) fails with SQLITE_CANTOPEN under 3+ concurrent writers regardless of pool settings
- `--passes` — omit to run all, or specify e.g. `3,4,5,6,7,8,9,10,11,12` to resume after a crash

### Why the run stays clean (guardrails, no manual cleanup)

These pipeline fixes (epic building_domain-27w) mean the documented run does not
reproduce the junk earlier runs needed manual SQL to remove:

- **Canonical IFC classes** — `seed-ifc-classes` writes one `ifc_class` EntityRow
  per real IFC type (`source_model='ifc-schema'`, `status='accepted'`). Pass 1
  drops any LLM-minted `ifc_class` concepts and Pass 12 maps only onto the seeded
  set, so hallucinated names ('IfcHotel') never enter the DB (building_domain-y30).
- **In-domain concepts** — Pass 1's concept-discovery prompts carry a SCOPE note
  that keeps entities within the built environment and restricts activities to
  on-site construction/installation/commissioning, so loose consumer objects and
  off-site manufacturing activities ('Paintbrush Assembly') are not seeded
  (building_domain-5if, supersedes the proposed Pass 5 bound in building_domain-yms).
- **Scoped passes** — Passes 6 (constraints) and 7 (anti-patterns) skip
  `entity_type='activity'`; passes 8/9 already scope to component/space/system.
  Activity knowledge is captured as ordering by Pass 5, not as constraints
  (building_domain-zu4). No manual scoping needed.
- **No quota-padding** — Passes 6/7/8/9 prompts include quality-over-quantity
  guidance: include only genuinely-applicable items, return few/none when
  appropriate, do not split one item into near-duplicate variants. This stops the
  "exactly 8 patterns per entity" padding (building_domain-8tk). *Caveat: this
  guidance is part of the per-pass cache key, so it only affects a fresh run.*
- **Resume-safe normalization** — Passes 10a/b/c re-run when un-normalized data
  exists rather than trusting a single global "completed" flag, so a multi-stage
  or resumed run normalizes everything with no manual DELETE-from-`pass_progress`
  step (building_domain-5ut).
- **Activity dedup after Pass 5** — Pass 5's `_get_or_create_activity` matches
  existing activities by exact name only, so every LLM wording variant ('Install
  Roof Decking' / 'Roof Decking Installation' / 'Roof Decking / Sheathing')
  became a distinct entity. `run_activity_dedup` runs once after Pass 5 (wired in
  `cli/extract.py`): it embeds active `activity` names and clusters them with the
  same Agglomerative(cosine, average) machinery as Pass 2 at a looser threshold
  (`ACTIVITY_DEDUP_THRESHOLD = 0.08`, calibrated against the production set — 0.04
  is too timid for wording variants, 0.12+ collapses distinct construction
  phases), folding each cluster into one canonical via `merge_entity` (which
  repoints the `process_relations` FKs). Scoped to `entity_type='activity'`
  only (building_domain-e9k).
- **Context-scoped process ordering** — Pass 5 sets `process_relations.subject_id`
  to the entity it is currently processing, so `bsos validate --conflicts`'s cycle
  detection can partition the ordering graph per subject context instead of
  unioning every edge into one graph. Without this, generic shared activities
  ('Concrete Curing', 'Foundation Construction') referenced from many unrelated
  contexts accumulate locally-true-but-globally-incompatible orderings that look
  like one large contradiction (432 edges/16 cycles in the pre-fix DB, 400 of
  them false cross-context artifacts). A fresh run needs no manual reset —
  `subject_id` is populated at write time. `bsos curate backfill-process-context`
  exists only to backfill rows written before this field existed
  (building_domain-eue).

### Model routing (`bsos/llm/__init__.py`)

| Prefix | Provider | Key needed |
|--------|----------|------------|
| `claude-*` | Anthropic | `ANTHROPIC_API_KEY` |
| `ollama/<name>` | Local Ollama (free) | none |
| anything else | OpenAI-compatible | `OPENAI_API_KEY` |

**Free local option:** `--models ollama/llama3.1` (requires Ollama running on localhost:11434).
**Cheap cloud option:** any OpenAI-compatible provider (DeepSeek, Groq, Together AI) via `OPENAI_API_KEY` + `OPENAI_BASE_URL`. See DeepSeek below — it is the recommended cheap cloud route.

### DeepSeek (production model — cheap cloud provider)

The Anthropic API is too expensive to complete the full 12-pass extraction, so the
documented production run (step 5 above) uses **DeepSeek** (`deepseek-chat` = V3).
It is OpenAI-compatible, so it routes through the existing `OpenAIProvider` with
**zero code changes** (it matches neither the `claude-` nor `ollama/` prefix, so it
falls through to OpenAI-compatible handling).

**Smoke-test first** (step 4 above). Before committing to a full run, verify auth, schema validation, and assertion density:

```bash
export OPENAI_API_KEY=sk-<deepseek key>
export OPENAI_BASE_URL=https://api.deepseek.com
python scripts/smoke_test_deepseek.py     # exit 0 = HEALTHY, safe to run the pipeline
```

It drives the real Pass 3 framing + `AssertionExtractionResponse` schema through `OpenAIProvider` and fails (exit 1) if the mean falls below 5 assertions/entity or on an auth error.

Unlike Groq/Llama-70B (which give sparse 1-2 assertions per entity and are unusable for Pass 3+), V3 handles `instructor` tool-calling / structured output cleanly — measured ~21 assertions/entity, comparable to Haiku.

### Known issues

- **Pass 2 threshold:** `CLUSTER_DISTANCE_THRESHOLD = 0.04` in `pass2.py`. Values above ~0.08 merge distinct-but-related concepts (e.g. "High Ceiling Zone" / "Low Ceiling Zone"). Skip pass 2 or keep threshold at 0.04.
- **Pass 1 max_tokens:** Set to 16384 in `anthropic_provider.py`; 4096 is too small for the full concept-discovery response.
- **LLM cache location:** Stored in `bsos_cache.db` (separate from `bsos.db`) using a single persistent connection + threading lock. Do NOT revert this — per-call `sqlite3.connect()` causes SQLITE_CANTOPEN under concurrent workers.
- **Crash recovery:** All LLM responses are cached in `bsos_cache.db`. Re-running the same command resumes from where it left off at no extra API cost.
- **Undo bad pass 2 merges:** `UPDATE entities SET status='proposed' WHERE status='merged'; DELETE FROM entity_aliases;` then re-run pass 2.
- **Groq provider:** Set `OPENAI_API_KEY=gsk_...` and `OPENAI_BASE_URL=https://api.groq.com/openai/v1` then use `--models llama-3.3-70b-versatile`. Quality is significantly lower than Haiku for complex structured output schemas — use Haiku or Sonnet for production runs.
- **Open-weight models (Llama, Groq):** Produce very sparse assertions (1-2 per entity vs 10-20 for Haiku). Not suitable for pass 3+.

### Recovering from a hard crash mid-pass (extraction passes 3–9)

Normalization (passes 10a/b/c) is now resume-safe (building_domain-5ut) and needs
no manual intervention. The extraction passes resume from the LLM cache, but a
process killed mid-write can leave an entity with a `pass_progress` record yet no
assertions. If a restart skips such entities, clear the phantom records first:

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
