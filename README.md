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

```bash
# Query the knowledge base
bsos query "Kitchen"

# Start the MCP server
bsos serve

# Run tests
pytest
```

See `CLAUDE.md` for extraction pipeline instructions and development notes.
