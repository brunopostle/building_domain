# BSOS — Building Domain Knowledge Base

3rd June 2026

## The Problem

AI agents working with BIM (Building Information Modelling) files contain broad architectural knowledge but cannot reliably apply it. An agent asked to sequence construction activities from an IFC model may correctly identify all the building elements yet fail to apply rules like *"windows are inserted into masonry walls after the walls are built"* or *"internal finishes cannot start until the roof is watertight."* This knowledge exists in the model's training data — it does not emerge reliably at the point of need.

The result is that AI agents working with building models produce generic advice, generate construction programmes that violate building logic, or miss glaring omissions in a model.

## What Exists Now

BSOS (Building System of Systems) is an open, queryable building domain knowledge graph, served via the Model Context Protocol (MCP) so AI agents can retrieve relevant domain knowledge on demand rather than hoping it is spontaneously recalled.

**Current scale (passes 1–3 complete):**

- **9,934 building entities** — physical components, spaces, activities, materials, systems, and IFC schema classes, covering standard commercial and residential construction
- **22,548 typed assertions** across 9 relationship predicates: *requires*, *depends\_on*, *contains*, *supports*, *connects\_to*, *protects\_from*, *conflicts\_with*, *improves*, *unsuitable\_for*
- **253 Christopher Alexander patterns** from *A Pattern Language* — with the first computational connectivity analysis of the full 253-pattern network: 1,754 edges, diameter 6, seven community clusters, and hub patterns identified for the first time quantitatively
- **MCP server** (`bsos serve`) connecting the knowledge graph to Claude Code and Claude Desktop — agents query it live during building model analysis

This works today, with passes 1–3 data only. The knowledge base is entirely LLM-synthesised, ships under **ODbL 1.0**, and is hosted at [github.com/brunopostle/building_domain](https://github.com/brunopostle/building_domain).

## What Funding Enables: Passes 4–12

Nine further extraction passes are implemented, tested, and ready to run — they are blocked only by API cost. These transform the current dependency graph into a full building intelligence layer:

| Pass | What it adds |
|------|-------------|
| **4** Spatial relations | Topological rules — what is above, adjacent to, enclosed by, or penetrates what |
| **5** Process sequences | Construction ordering — what must happen before and after each activity or element |
| **6** Hard constraints | Binary safety and buildability rules (must / must\_not) with no valid exceptions |
| **7** Anti-patterns | Documented failure conditions — what goes wrong, why, and how to avoid it |
| **8** Design patterns | Alexander-style recurring solutions linked to entities in the graph |
| **9** Design forces | Competing pressures that drive design decisions for each element |
| **10** Normalisation | Reference resolution, predicate stabilisation, abstraction synthesis |
| **11** Adversarial validation | Multi-model quality filtering — assertions challenged by ≥2 independent models are flagged for review |
| **12** IFC schema extraction | Schema-level relationships and constraints extracted from the official IFC specification |

After passes 4–12, every entity will carry not just *what it depends on* but *where it sits spatially*, *when it is built*, *what can go wrong*, *what design forces act on it*, and *whether the assertions have survived independent challenge*.

The compliance report will gain constraint checking and failure mode warnings on top of the current element-presence checks.

## Why This Is Achievable

The hard problems are already solved. The extraction pipeline, MCP server, IFC integration, data model, test suite, and contributor workflow are all built and in production. What remains is API budget to run passes 4–12 across 9,934 entities and human review time for the adversarial validation findings that pass 11 surfaces for manual decision.

Contributors can run additional passes and open pull requests — the merge workflow is already documented and tested. Domain-specific extensions (hospitals, data centres, heritage buildings, industrial facilities) can be added via targeted seed concepts without re-running the full pipeline.
