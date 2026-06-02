# Building Domain Knowledge Base (BSOS)

BSOS is a structured building domain knowledge base that makes implicit architectural and construction knowledge explicitly retrievable by LLM AI agents working with BIM/IFC models. LLM agents already contain broad building domain knowledge, but this knowledge does not reliably emerge at the point of need: an agent asked to sequence construction activities from an IFC model may correctly identify all the building elements but fail to apply rules like *"windows are inserted into masonry walls after the walls are built"* or *"internal finishes cannot start until the roof is watertight"*. BSOS surfaces this knowledge as a queryable graph, accessible via MCP tools, so agents can explicitly retrieve relevant domain constraints rather than relying on it being spontaneously recalled.

## User Stories

**Construction scheduling** — As an AI agent generating a construction programme from an IFC model, I need sequencing dependencies between building elements so I can order activities correctly without violating construction logic.

**Model completeness review** — As an AI agent reviewing an IFC model, I need to know what components a space or system typically requires so I can identify elements that are present in the knowledge base but missing from the model.

**Model improvement and design** — As an AI agent modifying or extending a building model, I need to understand structural, spatial, and systems relationships so that changes I propose respect established building domain conventions.

**Question answering** — As an AI agent answering questions about a specific building, I need to understand what elements typically connect to, depend on, or contain other elements so I can give contextually grounded answers rather than generic ones.

## Architecture

The knowledge base is populated by an LLM extraction pipeline (`bsos extract`) that runs over a seed vocabulary of building concepts and produces a graph of entities and typed assertions (requires, depends_on, contains, supports, connects_to, etc.). The graph is stored in `bsos.db` (SQLite) and served via an MCP server (`bsos serve`) that exposes query tools to AI agents.

## Quick Start

**Requires Python 3.12+**

### 1. Install and restore the knowledge base

The knowledge base is too large to store as a SQLite file in git. A JSON snapshot
is checked in at `data/bsos_snapshot.json` (~20 MB). To set up a working database:

```bash
git clone https://github.com/brunopostle/building_domain.git
cd building_domain
pip install -e .
bsos init
bsos import --input data/bsos_snapshot.json
bsos query "Kitchen"   # verify: should show ~20 assertions
```

The snapshot contains entities, assertions, and patterns.
It does **not** contain the LLM response cache or embeddings — those stay local.

### 2. Connect to an AI agent via MCP

The project `.mcp.json` is already configured for Claude Code. Enable the server:

```json
// .claude/settings.local.json
{
  "enabledMcpjsonServers": ["bsos"]
}
```

Then restart Claude Code — the `bsos` MCP tools become available immediately.

For Claude Desktop and a full walkthrough of the available tools, see **[MCP_DEMO.md](MCP_DEMO.md)**.

### Development

```bash
pytest          # run tests
bsos status     # show database state
bsos serve      # start MCP server (stdio)
```

See `CLAUDE.md` for extraction pipeline instructions and development notes.

## Contributing Data

The extraction pipeline (`bsos extract`) uses LLM API calls and is expensive to run.
**Do not re-run passes 1–3** — they would generate ~10,000 entities with new random IDs
that cannot be cleanly merged with the existing snapshot.

**What contributors can usefully run:**

- **Passes 4–12** — constraints, failure modes, forces, spatial relations, and process
  sequences. These operate on the existing entity set (stable IDs) and add new assertion
  rows that merge cleanly.
- **Domain extension** — `bsos extract --seed-apl` with additional seed concepts for
  under-represented domains (hospitals, data centres, industrial, heritage buildings).

**Contribution workflow:**

```bash
# 1. Fork and clone, restore from snapshot (preserves entity IDs)
git clone https://github.com/YOUR_FORK/building_domain.git
cd building_domain
pip install -e .
bsos init
bsos import --input data/bsos_snapshot.json

# 2. Run additional extraction passes (needs ANTHROPIC_API_KEY in a separate terminal)
bsos extract --models claude-haiku-4-5-20251001 --passes 4,5,6,7,8,9,10,11,12 \
  --framings 1 --workers 2

# 3. Export and open a PR
bsos export --format json --output data/bsos_snapshot.json
git add data/bsos_snapshot.json
git commit -m "Add passes 4-12 extraction results"
# open pull request
```

**Reviewing a contribution PR:**

```bash
# Preview what the incoming snapshot adds without touching your working database
bsos export --format json --output /tmp/current.json
# (review data/bsos_snapshot.json in the PR diff)

# Merge into your working database (skips records you already have)
bsos import --input data/bsos_snapshot.json --force
```
