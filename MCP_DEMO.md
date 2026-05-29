# BSOS MCP Server — Integration Demo

This document shows how to connect the BSOS building domain knowledge graph to Claude Code
(or Claude Desktop) via MCP, and demonstrates the tools answering real building design questions.

## Setup

BSOS exposes its knowledge graph as an MCP server over stdio.

### Claude Code (project-level)

The project `.mcp.json` already registers the server:

```json
{
  "mcpServers": {
    "bsos": {
      "type": "stdio",
      "command": "bsos",
      "args": ["serve"]
    }
  }
}
```

Enable it in `.claude/settings.local.json`:

```json
{
  "enabledMcpjsonServers": ["bsos"]
}
```

Restart Claude Code — the `bsos` MCP server connects automatically.

### Claude Desktop

Add to `claude_desktop_config.json` (Linux: `~/.config/claude/`, macOS: `~/Library/Application Support/Claude/`):

```json
{
  "mcpServers": {
    "bsos": {
      "command": "bsos",
      "args": ["serve", "--db", "/path/to/bsos.db"]
    }
  }
}
```

## Available Tools

| Tool | Description |
|------|-------------|
| `search_entities` | Semantic search — find entities by free-text or IFC element name |
| `get_requirements` | What a building element requires (materials, activities, other elements) |
| `get_dependencies` | Dependency graph — what depends on what |
| `get_constraints` | Dimensional and performance constraint rules |
| `get_failure_modes` | Anti-patterns and failure modes with mitigations |
| `get_patterns` | Alexander-style design patterns linked to an entity |
| `get_forces` | Design forces (pressures) acting on an entity |
| `get_spatial_relations` | Spatial topology (above, adjacent, encloses, …) |
| `get_process_sequence` | Construction process ordering for an entity |

## Example Session

The session below shows Claude answering a real design question using the knowledge graph.

---

**User:** What does a foundation need, structurally speaking?

**Claude:** Let me search for the foundation entity and pull its requirements.

```
Tool: search_entities("foundation")

Results:
  1.000  Foundation          [component]  The substructure element that transfers
                                          all building loads safely to the ground.
  0.785  Deep Foundation     [component]  A foundation type that transfers loads to
                                          deeper, more stable soil strata.
  0.744  Foundation Setting  [activity]   Positioning and marking the exact plan
         Out                              location, level, and extent of foundations.
```

```
Tool: get_requirements("Foundation")

Foundation requires:
  • Concrete                      conf=0.95  [engineering]
    Concrete is the primary material for most modern foundation types including
    footings, slabs-on-grade, and piled caps.

  • Formwork                      conf=0.92  [engineering]
    Formwork provides the mould into which concrete is poured to shape foundation
    elements such as strip footings, pad bases, and pile caps.

  • Structural Engineering Design conf=0.98  [engineering]
    Foundation sizing, depth, and reinforcement must be calculated by a structural
    engineer to meet site-specific load and soil conditions.

  • Footing                       conf=0.98  [engineering]
    Footings are the structural elements that distribute foundation loads to the soil.

  • Damp Proof Course             conf=0.93  [physical]  (depends_on)
    Capillary rise of moisture through masonry foundation walls into the superstructure
    causes dampness, decay, and structural degradation.

  • Reinforcement Bar             conf=0.93  [engineering]
    Steel reinforcement is required to provide tensile capacity that plain concrete
    lacks, particularly in resisting bending and shear.

  • Excavation                    conf=0.97  [engineering]
    The ground must be excavated to the required bearing stratum before any foundation
    element can be placed.

  • Soil Investigation            conf=0.98  [engineering]
    Foundation design cannot proceed safely without site-specific knowledge of soil
    bearing capacity, water table, and ground conditions.

  • Concrete Mix Design           conf=0.92  [engineering]
    The concrete mix must be specified to suit the exposure class and durability
    requirements of the below-ground environment.
```

**Claude:** A foundation requires:

1. **Soil Investigation first** (conf 0.98) — ground conditions must be characterised before any design begins.
2. **Structural Engineering Design** (conf 0.98) — sizing and reinforcement are engineer-specified.
3. **Excavation** (conf 0.97) — the ground must be cut to the bearing stratum.
4. **Footing** (conf 0.98) — the load-spreading element at the base.
5. **Concrete + Mix Design** (conf 0.95/0.92) — the primary material, mix specified for below-grade exposure.
6. **Formwork** (conf 0.92) — moulds the concrete before it cures.
7. **Reinforcement Bar** (conf 0.93) — provides tensile capacity.
8. **Damp Proof Course** (conf 0.93) — moisture barrier between ground and superstructure.

---

**User:** What else depends on beams?

**Claude:**
```
Tool: get_dependencies("Beam")

  Column            depends_on  Beam              conf=0.92
  Beam              depends_on  Foundation        conf=0.90
  Beam Bearing Plate depends_on Beam             conf=0.95
```

Beams are depended on by **Columns** and **Beam Bearing Plates**, and themselves depend on the **Foundation**. This reflects the typical load path: Foundation → Beam → Column.

---

## What the Knowledge Graph Contains

The current database (`bsos.db`) holds the output of the Pass 1–3 extraction pipeline:

- **9,934 entities** (components, systems, activities, materials, …)
- **22,548 assertions** across 9 predicate types:
  - `requires` (12,240) — material and activity dependencies
  - `depends_on` (4,472) — structural and process dependencies
  - `contains` (2,478) — composition relationships
  - `supports` (1,683) — load and functional support
  - `connects_to` (1,164) — spatial/mechanical connections
  - `protects_from`, `conflicts_with`, `improves`, `unsuitable_for` — qualitative relations
- **253 Alexander patterns** — summaries of all patterns from *A Pattern Language*

Passes 4–12 (constraints, failure modes, forces, spatial relations) are planned for the next pipeline run and will populate additional tools.

## Integration with IFC Models

The `ifc` MCP server (also configured in `.mcp.json`) allows querying live IFC building models.
A future workflow combines both servers: use `ifc_list` / `ifc_select` to inspect elements in a
loaded model, then `search_entities` + `get_requirements` to retrieve domain knowledge about
those elements from the BSOS knowledge graph.
