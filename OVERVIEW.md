# BSOS: Building Science Ontology System

Modern AI assistants are surprisingly good at recognising building elements in an
IFC model, but not at reliably applying the rules that govern them. See
[README.md](README.md) for how BSOS addresses this and the full MCP tool
reference; this document goes deeper on use cases, extraction methodology, and
current data scale.

## Use Cases

BSOS composes its tools into three capabilities beyond single-fact lookup.
**BIM compliance review** (`check_model`) resolves each space in a loaded IFC
model to a knowledge-graph entity via semantic search and checks it against
requirements, constraints, spatial relations, and anti-patterns in one pass.
**Construction sequencing** (`get_process_sequence`, `check_prerequisites`)
generates or gates a construction programme against process-relation ordering.
**Design critique** through the lens of Christopher Alexander's pattern language
(`critique_patterns`, `sweep_failure_modes`, `annotate_clash`) evaluates a
design's human-scale qualities and annotates geometry clashes with the pattern
or failure-mode rationale behind them, rather than a bare "X intersects Y".

## How the Knowledge Base Was Built

The knowledge base was built by asking an AI to systematically enumerate
building concepts — structural and envelope components, building services,
space types, materials, and construction activities — across 12 extraction
passes: concept discovery and deduplication, requirements/dependencies, spatial
relations, process sequencing, hard constraints, anti-patterns, design patterns,
design forces, normalisation, adversarial validation, and IFC schema extraction.
This process drew on the AI's broad synthesis of architectural and construction
knowledge rather than any single reference document or standard. Alexander's
*A Pattern Language* contributed its 253 named patterns as additional seeds,
biasing the concept space towards the human-scale spatial ideas the patterns
describe.

The current graph holds roughly 19,000 active named entities (after merging
near-duplicates), 20,700 typed relationship assertions, 14,000 hard
constraints, 23,500 anti-patterns, 12,600 design forces, 12,400 design
patterns, 8,900 spatial relations, and 21,600 process-ordering relations. None
of this has been individually fact-checked by a human. Most tables have been
bulk-promoted from `proposed` to `accepted` status via a confidence-threshold
database update (≥0.85, after spot-checking a sample at each tier to confirm
the scores were well-calibrated) — process-ordering relations and spatial
relations haven't even had that pass applied and remain almost entirely
`proposed`. (A separate queue of several hundred process-relation ordering
conflicts has since been resolved: scoping ordering to the construction
context that asserted it, rather than comparing globally, cleared all but a
handful of genuine conflicts, which were resolved by hand.) See
[README.md](README.md) for the full data-quality disclaimer. The initial
survey was framed around a generic commercial and residential building, which
means specialist building types are underrepresented: hospitals, data
centres, industrial buildings, heritage structures, and buildings designed
for specific climates all have known gaps. Extending coverage into these
domains is planned follow-on work.
