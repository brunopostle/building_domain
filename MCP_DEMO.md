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
| `check_model` | Full requirements/constraints/spatial/anti-pattern compliance report for a loaded IFC model file |

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

**User:** What must a cavity wall satisfy, and what typically goes wrong with one?

**Claude:**
```
Tool: get_constraints("Cavity Wall")

Cavity Wall must:
  • have wall ties connecting the inner and outer leaves           conf=0.95
    (unless designed as structurally independent leaves with other lateral restraint)
  • maintain a minimum clear cavity width of at least 50 mm        conf=0.90
  • have weep holes at the base of the outer leaf above flashings  conf=0.90

Cavity Wall must not:
  • have mortar droppings or debris bridging the cavity            conf=0.95
  • have cavity insulation that bridges the cavity completely
    in a way that creates a continuous path for moisture transfer conf=0.85
```

```
Tool: get_failure_modes("Cavity Wall")

"Thermal Bridging via Wall Ties and Penetrations"  conf=0.95  [engineering]
  Conditions: metal wall ties bridging the cavity; uninsulated or partially
  insulated cavity allowing convection currents; service penetrations bridging
  or blocking the cavity; mortar droppings on ties creating conductive paths.
  Consequences: reduced effective R-value; condensation and mould at tie
  locations; increased heating/cooling energy use; detectable cold spots.
  Mitigations: low-conductivity or thermally-broken wall ties; continuous
  cavity insulation without gaps; cavity trays and weep holes for debris
  removal.
```

**Claude:** A cavity wall's binding rules are mostly about the cavity itself: it must stay clear (≥50mm, no mortar bridging) and must drain (weep holes, ties connecting the leaves). The dominant failure mode — thermal bridging through the ties and any debris in the cavity — is a direct consequence of the same cavity that the constraints protect: get the cavity wrong and the wall loses the insulation performance it was built for.

---

**User:** What design pressures shape a cavity wall, and how does it fit into a wall assembly spatially?

**Claude:**
```
Tool: get_forces("Cavity Wall")

  reduced moisture penetration pressure   decrease  conf=0.90
  improved thermal insulation pressure    increase  conf=0.85
  reduced thermal bridging pressure       decrease  conf=0.85
  adequate cavity width pressure          increase  conf=0.80
```

```
Tool: get_spatial_relations("Cavity Wall")

  Cavity Wall  contains      Wall Tie            conf=0.95
  Cavity Wall  above         Foundation          conf=0.90
  Cavity Wall  contains      Insulation          conf=0.90
  Cavity Wall  below         Roof                conf=0.85
  Cavity Wall  connects_to   Foundation          conf=0.85
  Cavity Wall  connects_to   Floor Slab          conf=0.80
  Cavity Wall  adjacent_to   Window Frame        conf=0.80
  Cavity Wall  enclosed_by   Building Envelope   conf=0.75
  … 7 more
```

**Claude:** The four design forces read as two pairs pulling in tension: moisture and thermal-bridging pressures push toward a wider, cleaner cavity, while the thermal-insulation pressure pushes toward filling that same cavity — which is exactly the trade-off the failure mode above warns about. Spatially, the wall sits on the foundation and below the roof, with ties and insulation as its cavity contents and window/door frames as adjacent penetrations — matching `get_patterns("Cavity Wall")`'s "Cavity Wall for Moisture Management" pattern (two leaves separated by a ≥50mm drained air cavity).

---

**User:** What has to happen before and after windows are installed?

**Claude:**
```
Tool: get_process_sequence("Install Windows", max_depth=1)

  Install Flashing and Weather Barrier (House Wrap)
  Frame Rough Opening
  → Install Windows
  Install Exterior Siding and Trim
  Interior Drywall Installation
  Install Window Sills and Interior Trim
```

**Claude:** Windows go in after the rough opening is framed and the weather barrier/flashing is on, and before exterior siding, drywall, and interior trim close the opening up — installing windows after siding or drywall would trap the flashing laps in the wrong order and leave no way to tie the weather barrier in correctly. Note: `max_depth` defaults to 50 and returns the full reachable subgraph, which for a well-connected activity can run to hundreds of loosely-related entries; `max_depth=1` keeps the result to the immediate predecessors/successors, which is usually what you want for a single "what's before/after this" question.

---

## What the Knowledge Graph Contains

The current database (`bsos.db`) holds the output of the full 12-pass extraction pipeline
(`bsos extract --passes 1,2,3,4,5,6,7,8,9,10,11,12`), reviewed and promoted (`bd show building_domain-lfy`):

- **19,174 active entities** (3,706 near-duplicates merged away by Pass 2/normalization and `building_domain-b4p` entity-level curation): 13,499 activities, 2,630 components, 1,005 spaces, 771 IFC schema classes, 667 systems, 602 materials
- **20,732 assertions** across 9 predicate types — `requires` (7,591), `depends_on` (4,700), `supports` (2,873), `connects_to` (2,580), `contains` (2,440), `protects_from` (323), `improves` (109), `conflicts_with` (78), `unsuitable_for` (38) — fully reviewed, 0 left in `proposed`
- **14,004 constraints** (Pass 6) — binary must/must\_not buildability and safety rules with conditions and exceptions
- **23,522 anti-patterns / failure modes** (Pass 7) — documented failure conditions with consequences and mitigations
- **12,401 design patterns** (Pass 8) — 12,148 LLM-synthesised patterns linked to specific entities, plus the original **253 human-curated Alexander patterns** from *A Pattern Language* (`source_model='human'`) used as ground truth
- **12,617 design forces** (Pass 9) — competing pressures driving design decisions per entity
- **8,865 spatial relations** (Pass 4) — topological rules (above, adjacent\_to, contains, connects\_to, …); not yet through manual review, still `status='proposed'`
- **21,609 process relations** (Pass 5) — construction ordering edges; 21,577 `proposed`, 18 `accepted`, 14 `deprecated`, 0 `conflicted` (was 432: `building_domain-eue` scoped cycle detection to per-`subject_id` context partitions so unrelated contexts sharing a generic activity node no longer collide into false cycles, clearing all but 8 genuine same-context reversed-direction edges; those were resolved by hand in `building_domain-g3r`)
- **437 abstraction nodes** (Pass 10c) — synthesised cross-cutting statements over clusters of related assertions

## Composed BSOS + IFC Capabilities

The `ifc` MCP server (also configured in `.mcp.json`) queries live IFC building models. Beyond
using the two servers side by side (`ifc_list` / `ifc_select` to inspect elements, then
`search_entities` + `get_requirements` to retrieve domain knowledge about them), two tools now
compose them directly:

- **`check_model`** (this server) — runs every space in a loaded IFC model against requirements,
  constraints, spatial relations, and anti-patterns in one compliance report, resolving each
  modelled space to a bsos entity via semantic search rather than a fixed space-type list.
- **`scripts/ifc_boq_check.py`** — cross-references a model's costed/quantified elements against
  `get_requirements` to flag likely bill-of-quantities omissions (e.g. a costed foundation with
  no damp-proof-course line item).

Further compositions are tracked under `building_domain-l5w` (`bd show building_domain-l5w`)
and not yet built: clash reports annotated with BSOS rationale, a design-time advisor that checks
`get_process_sequence` prerequisites before an `ifc_edit`/`ifc_new` call, and an Alexander-pattern
spatial critique using `get_patterns` + `get_forces` against derived model facts.
