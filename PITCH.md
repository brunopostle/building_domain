# BSOS — Building Domain Knowledge Base

3rd June 2026

## The Problem

AI agents working with BIM models contain broad architectural knowledge but cannot reliably apply it, producing generic advice, construction programmes that violate building logic, or missed omissions in a model. See [README.md](README.md) for the full framing and a worked example.

## What Exists Now

All 12 extraction passes are complete, and most tables have been bulk-promoted
past a confidence threshold (see the data-quality disclaimer below) — see
[OVERVIEW.md](OVERVIEW.md) for methodology and headline totals, and
[README.md](README.md) for the MCP tool reference. Beyond those top-line
numbers:

- **19,174 active building entities** by type — 13,499 activities, 2,630 components, 1,005 spaces, 771 IFC schema classes, 667 systems, 602 materials (a further 3,706 near-duplicates were identified and merged away)
- **9 relationship predicates**: *requires*, *depends\_on*, *contains*, *supports*, *connects\_to*, *protects\_from*, *conflicts\_with*, *improves*, *unsuitable\_for*
- **437 synthesised abstraction nodes** summarising clusters of related assertions
- The **253 Christopher Alexander patterns** from *A Pattern Language*, used as ground truth, with the first computational connectivity analysis of the full pattern network: 1,754 edges, diameter 6, seven community clusters, and hub patterns identified for the first time quantitatively

This works today. The knowledge base is entirely LLM-synthesised, ships under **ODbL 1.0** (separate from the code's GPL-3.0-or-later license), and is hosted at [github.com/brunopostle/building_domain](https://github.com/brunopostle/building_domain).

## What Funding Enables: From Knowledge Base to Agent Capability

The extraction pipeline's job is done — all 12 passes ran, and assertions, constraints, patterns,
forces, and anti-patterns have all been bulk-promoted out of `proposed` via a confidence-threshold
database update (≥0.85, after spot-checking a sample at each tier), rather than an individual
human review of every row — see [README.md](README.md) for the full data-quality disclaimer. The
process-relation ordering conflicts (432 "X precedes Y" / "Y precedes X" cycles the extraction
produced independently across unrelated parts of the graph) are fully resolved: a context-scoping
fix cleared all but 8 as false cross-context artifacts, and the remaining 8 same-context reversals
were resolved by hand. Entity-level curation (embedding-based dedup of near-duplicate components,
materials, spaces and systems) is also done: 26 more near-duplicates merged in `building_domain-b4p`,
on top of the 3,680 Pass 2/normalization already caught. Composing the BSOS and IFC MCP servers into
capabilities beyond lookup is also done: `check_model`, `check_prerequisites`, `sweep_failure_modes`,
`annotate_clash`, and `critique_patterns` are all live, alongside a BOQ sanity-check script. What
remains is closing the two tables that haven't even had the confidence-threshold pass applied, and
extending the graph into domains the initial survey didn't cover:

- **Two review queues still open** — the 21,566 (of 21,598) process relations and all 8,840 spatial
  relations extracted by Pass 4/5 remain almost entirely at `proposed`, without even the
  bulk-confidence-threshold pass the other five tables received, let alone individual review.
- **Domain expansion** — the initial survey was framed around a generic commercial/residential
  building, so site/infrastructure, climate-specific detailing, data centres, hospitals, industrial
  buildings, and heritage structures are all known-thin. A further Pass 1 seed expansion into these
  domains, followed by Pass 2/3 re-runs, is scoped but currently deferred: we're using the MCP tools
  against real models first to see which entity-graph gaps actually matter in practice, so funded
  extraction work targets the domains agents actually query rather than guessing up front.

## Why This Is Achievable

The hard problems are already solved. The extraction pipeline, MCP server, IFC integration, data model, test suite, and contributor workflow are all built and in production, and have already processed the full corpus once, including the tool-composition layer above lookup. What remains is review time for the two open queues, plus targeted extraction spend on the domain gaps that tool usage surfaces — not re-deriving anything already built.

Contributors can run additional passes and open pull requests — the merge workflow is already documented and tested. Domain-specific extensions (hospitals, data centres, heritage buildings, industrial facilities) can be added via targeted seed concepts without re-running the full pipeline.
