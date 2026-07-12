# BSOS — Building Domain Knowledge Base

3rd June 2026

## The Problem

AI agents working with BIM (Building Information Modelling) files contain broad architectural knowledge but cannot reliably apply it. An agent asked to sequence construction activities from an IFC model may correctly identify all the building elements yet fail to apply rules like *"windows are inserted into masonry walls after the walls are built"* or *"internal finishes cannot start until the roof is watertight."* This knowledge exists in the model's training data — it does not emerge reliably at the point of need.

The result is that AI agents working with building models produce generic advice, generate construction programmes that violate building logic, or miss glaring omissions in a model.

## What Exists Now

BSOS (Building System of Systems) is an open, queryable building domain knowledge graph, served via the Model Context Protocol (MCP) so AI agents can retrieve relevant domain knowledge on demand rather than hoping it is spontaneously recalled.

**Current scale (all 12 extraction passes complete, reviewed, and promoted):**

- **19,200 active building entities** — 13,499 activities, 2,648 components, 1,007 spaces, 771 IFC schema classes, 668 systems, 607 materials (a further 3,680 near-duplicates were identified and merged away)
- **20,732 typed assertions** across 9 relationship predicates: *requires*, *depends\_on*, *contains*, *supports*, *connects\_to*, *protects\_from*, *conflicts\_with*, *improves*, *unsuitable\_for*
- **14,004 hard constraints**, **23,522 anti-patterns/failure modes**, **12,617 design forces**, and **8,865 spatial relations** — the full building-intelligence layer: not just what an element depends on, but where it sits spatially, what can go wrong, and what design pressures shaped it
- **21,609 process relations** encoding construction ordering, and **437 synthesised abstraction nodes** summarising clusters of related assertions
- **12,401 design patterns**, including the original **253 Christopher Alexander patterns** from *A Pattern Language* used as ground truth — with the first computational connectivity analysis of the full 253-pattern network: 1,754 edges, diameter 6, seven community clusters, and hub patterns identified for the first time quantitatively
- **MCP server** (`bsos serve`) connecting the knowledge graph to Claude Code and Claude Desktop, plus a composed `check_model` tool that runs a full requirements/constraints/spatial/anti-pattern compliance report against a loaded IFC model in one call

This works today. The knowledge base is entirely LLM-synthesised, ships under **ODbL 1.0**, and is hosted at [github.com/brunopostle/building_domain](https://github.com/brunopostle/building_domain).

## What Funding Enables: From Knowledge Base to Agent Capability

The extraction pipeline's job is done — all 12 passes ran, and the output has been reviewed down
to zero items left in `proposed` for assertions, constraints, patterns, forces, and anti-patterns.
What remains is turning a queryable graph into composed agent capability, and closing two review
queues the pipeline surfaced rather than resolved:

- **Compose the BSOS and IFC MCP servers into capabilities beyond lookup** — a design-time advisor
  that checks `get_process_sequence` prerequisites before an IFC edit, an anti-pattern sweep that
  scans a whole loaded model against `get_failure_modes`, an Alexander-pattern spatial critique,
  and clash reports annotated with BSOS rationale. Two compositions are already live (`check_model`,
  a BOQ sanity-check script); four more are scoped and ready to build.
- **Spatial relations review** — 8,865 Pass 4 topological assertions are extracted but not yet
  through the manual accept/reject pass the other five tables received.
- **Process-relation cycle review** — extraction independently produced 432 "X precedes Y" /
  "Y precedes X" ordering conflicts across 16 activity clusters. The tooling gap that made the
  queue unreviewable is fixed, and a follow-up architecture fix (context-scoped ordering, so
  unrelated parts of the graph sharing a generic activity node no longer collide into a false
  conflict) resolved all but 8 edges automatically; the last 8 are small same-context reversals
  awaiting a final human judgement call.
- **Entity deduplication** — 18,429 of the 19,200 active entities remain in `proposed` status
  (this doesn't block MCP queries, but embedding-based dedup would tighten the graph further).

## Why This Is Achievable

The hard problems are already solved. The extraction pipeline, MCP server, IFC integration, data model, test suite, and contributor workflow are all built and in production, and have already processed the full corpus once. What remains is scoped engineering work on the composition tools above, plus review time for the two open queues — not further extraction spend.

Contributors can run additional passes and open pull requests — the merge workflow is already documented and tested. Domain-specific extensions (hospitals, data centres, heritage buildings, industrial facilities) can be added via targeted seed concepts without re-running the full pipeline.
