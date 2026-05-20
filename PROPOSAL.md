Below is a revised **implementation instruction specification** intended for an LLM coding agent. It encodes the improved architecture as an executable build plan, not a research document.

---

# BSOS — Building Semantic Ontology System

## Implementation Instruction for LLM Coding Agent

---

# 0. Objective

Build a local-first Python system that converts latent building-domain knowledge in LLMs into a structured, evolving semantic knowledge system.

The system extracts, normalizes, and organises building knowledge into:

* atomic assertions
* spatial/topological relations
* construction processes
* architectural patterns (Alexander-style)
* force-based tradeoffs
* anti-patterns / failure modes

The output is a **machine-usable semantic graph + structured database**, not text.

---

# 1. Core Architectural Principle

The system is NOT an ontology-first system.

It is a **semantic distillation pipeline**:

```text
LLM knowledge → extraction → semantic fragments → normalization → compression → structured graph
```

Ontology structures MUST emerge from observed stability in extracted data, not be pre-designed.

---

# 1.1 Epistemic Scope (MANDATORY CONSTRAINT)

This system extracts **universal, physics-grounded building knowledge only**.

In scope:

* physical laws as they apply to buildings (gravity, thermodynamics, water behaviour, material properties)
* engineering fundamentals (structural logic, load paths, thermal performance)
* functional and spatial logic (rooms need light, circulation must be connected)
* construction sequencing (dependency chains derived from physics)
* cross-cultural architectural patterns

Out of scope (future overlay layer):

* regulatory knowledge
* jurisdictional requirements
* building codes
* standards references

Physical reality is the epistemic anchor. "Roof protects from precipitation" is true everywhere. Regulatory overlays vary by jurisdiction and change over time — they are modelled separately and not part of this system.

---

# 2. Required Technology Stack

Use only lightweight open-source Python tools:

* Python 3.12+
* pydantic (data models)
* sqlmodel + sqlite (storage — source of truth)
* networkx (graph — derived query layer only, never primary storage)
* fastapi (optional API layer)
* typer (CLI)
* instructor (structured LLM output)
* sentence-transformers (similarity clustering only) — default model: `all-mpnet-base-v2` (~420MB download on first use; the CLI MUST warn the user and prompt confirmation before the first embedding operation; confirmation is persisted by writing `embedding_model_confirmed = '1'` to the `config` table (Section 12) so subsequent invocations skip the prompt; the first embedding operation occurs during entity clustering in Pass 2 — the prompt fires before Pass 2 begins, when entity clustering first requires embeddings)
* numpy
* scipy (used for `linear_sum_assignment` in cross-prompt consistency alignment — Section 6, Pass 3)
* scikit-learn (used for agglomerative clustering in entity deduplication — Section 8.1)
* structlog (structured logging — required for pipeline observability across 11 passes)
* joblib (graph serialization — `joblib.dump` / `joblib.load` with `compress=3`; see Section 11)
* alembic (schema migrations; see Section 12)

**SQLModel + Alembic compatibility note:** SQLModel's Alembic autogenerate support is incomplete for several column types, particularly JSON-serialized `list[str]` fields and custom enum types. Always review generated migration scripts manually before applying. Declare `list[str]` fields explicitly as `sa_column=Column(Text)` in SQLModel classes so autogenerate treats them as `TEXT` columns rather than inferring a type. Do not rely on SQLModel's default column inference for these fields. The extraction Pydantic models and SQLModel persistence models are two parallel class hierarchies (see Section 12) — autogenerate only sees the SQLModel side, which reduces the risk of inference surprises.

LLM providers must be abstracted behind a `LLMProvider` protocol so the extraction pipeline is provider-agnostic:

```python
from typing import Protocol
from pydantic import BaseModel

class LLMProvider(Protocol):
    def extract(self, prompt: str, schema: type[BaseModel], *, entity_name: str | None = None) -> BaseModel: ...
    def classify(self, prompt: str, options: list[str]) -> str: ...
    @property
    def model_id(self) -> str: ...  # used to populate source_model in ProvenanceMixin
```

Supported back-ends:

* OpenAI-compatible API (via `instructor`)
* local models (Ollama or vLLM, also via `instructor`)

**Ollama / local model fallback:** Not all local models support JSON schema enforcement (function calling or structured output mode), which `instructor` requires. The `LLMProvider` implementation for Ollama MUST detect this at startup by sending a test structured-output request. If the model does not support it, the implementation falls back to prompt-engineering extraction: the prompt includes the full JSON schema as a fenced example and instructs the model to respond with valid JSON only. Responses are parsed with `json.loads`; parse failures are retried once with an explicit repair prompt before being logged and skipped. This fallback path MUST be flagged in `structlog` output so operators know they are not using schema-enforced extraction.

No RDF, OWL, Neo4j, or heavy graph infra.

---

# 2.5 Project Structure

Use a `src/` layout. The project is a single Python package named `bsos`.

```
bsos/                          # repository root
├── src/
│   └── bsos/
│       ├── __init__.py
│       ├── cli/               # Typer command definitions; one module per command group
│       │   ├── __init__.py
│       │   ├── main.py        # top-level app, global flags, command registration
│       │   ├── extract.py
│       │   ├── validate.py
│       │   ├── curate.py
│       │   ├── review.py
│       │   └── ...
│       ├── models/            # Pydantic extraction models (Section 4)
│       │   └── ...
│       ├── persistence/       # SQLModel persistence models + repository layer
│       │   ├── models.py      # SQLModel(table=True) classes mirroring Section 4 models
│       │   ├── database.py    # engine creation, WAL mode, session factory
│       │   └── repos/         # one repository class per table group
│       │       └── ...
│       ├── pipeline/          # extraction pipeline passes (Section 6)
│       │   ├── pass1.py
│       │   ├── pass2.py
│       │   └── ...
│       ├── normalization/     # Section 8: entity clustering, predicate stabilization, etc.
│       ├── graph/             # NetworkX graph construction (Section 11)
│       ├── mcp_server/        # MCP server and query tools (Section 13)
│       ├── llm/               # LLMProvider protocol and concrete implementations
│       │   ├── protocol.py
│       │   ├── openai_provider.py
│       │   └── ollama_provider.py
│       ├── vocab.py           # CORE_PREDICATES, PREDICATE_REGISTRY, SPATIAL_RELATION_TYPES
│       └── config.py          # config table access helpers
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── e2e/                   # gated behind BSOS_E2E=1
│   └── fixtures/
│       ├── fake_responses.py  # FakeLLMProvider dispatch table
│       └── pattern_critique_cases.json
├── alembic/
│   ├── env.py
│   └── versions/
├── pyproject.toml
├── alembic.ini
└── README.md
```

**`pyproject.toml` minimum requirements:**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "bsos"
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.0",
    "sqlmodel",
    "alembic",
    "networkx",
    "typer",
    "instructor",
    "sentence-transformers",
    "numpy",
    "scipy",
    "scikit-learn",
    "structlog",
    "joblib",
    "mcp",
]

[project.scripts]
bsos = "bsos.cli.main:app"

[tool.hatch.build.targets.wheel]
packages = ["src/bsos"]
```

The `bsos` CLI entry point is declared in `[project.scripts]` so that `pip install -e .` makes the `bsos` command available. Do not use a top-level `main.py` invoked via `python -m` — the entry point declaration is the canonical invocation path.

---

# 3. System Data Layers

Implement strict separation between layers:

## 3.1 Entity Layer

Represents building concepts.

Examples:

* roof
* corridor
* pump
* window

---

## 3.2 Assertion Layer (Atomic Knowledge)

Represents single semantic facts. The example below shows resolved display names; internally `subject` and `object` are stored as Entity UUIDs (`subject_id`, `object_id`) and resolved at query time:

```json
{
  "subject": "roof",           // display name resolved from subject_id (Entity UUID)
  "predicate": "protects_from",
  "object": "precipitation"    // display name resolved from object_id (Entity UUID)
}
```

Assertions MUST be:

* atomic
* typed
* normalized
* provenance-tracked

---

## 3.3 Constraint Layer (Hard Rules)

Represents validation-type knowledge:

* must / must not conditions
* engineering requirements
* physical constraints

**Boundary rule — Constraint vs Assertion:** Use a `Constraint` when the rule is a binary precondition for a *valid configuration* (violation makes the design invalid). Use an `Assertion` with predicate `requires` when expressing a typical *physical or functional dependency* that is true in the normal case but admits exceptions. Example: "roof must have a drainage path" is a Constraint (no drainage = invalid). "roof requires structural support" is an Assertion — the dependency is real but the form it takes varies.

Example:

* roof must have a drainage path (Constraint — must)
* basement below sewer invert must include a pumping system (Constraint — must)

**Boundary rule — Pattern vs Assertion:** Use a `Pattern` when the fact is contextual, involves competing forces, and requires a trade-off decision. Use an `Assertion` when the relationship is a direct physical or functional dependency that holds universally without a trade-off. Example: "light on two sides of a room" is a Pattern — it involves competing forces (daylight vs heat gain) and applies based on context and design intent. "window requires structural frame" is an Assertion — it is a universal functional dependency with no trade-off decision.

---

## 3.4 Pattern Layer (Alexander-style)

Represents architectural heuristics:

* contextual
* multi-factor
* non-deterministic
* compositional

Example:

* light on two sides improves spatial quality

Patterns include:

* forces
* tradeoffs
* spatial transformations

---

## 3.5 Force Layer

Represents individual design pressures — single forces acting on a design decision. A `Force` is not itself a tension; tensions between opposing forces are expressed at the Pattern level (Section 3.4), where a Pattern references two or more Force records that pull in different directions.

Examples of individual forces:

* increased daylight
* reduced heat gain
* improved privacy
* reduced cost

A Pattern like "light on two sides" references both the "increased daylight" and "increased heat gain" forces and documents the tradeoff between them.

Patterns MUST reference forces explicitly.

---

## 3.6 Topology Layer

Represents spatial relationships:

* adjacency
* containment
* connectivity
* circulation paths
* accessibility

Example:

* room must be reachable via graph path from entrance

---

## 3.7 Anti-Pattern Layer

Represents failure conditions:

* known bad configurations
* design failures
* pathological layouts

Example:

* internal bathroom without ventilation

---

## 3.8 Process Layer

Represents sequencing and construction logic:

* temporal ordering
* dependency chains
* construction phases

Example:

* waterproofing before finishes

---

## 3.9 Provenance Layer

Every extracted item MUST include:

* source model
* prompt used
* timestamp
* confidence score (bounded 0.0–1.0)
* knowledge origin tag
* optional rationale (free-text explanation of why the item was extracted or accepted)

---

# 4. Core Data Models (Pydantic)

Implement exactly these base structures.

---

## 4.0 Provenance Mixin (REQUIRED BASE)

All extracted models MUST inherit from this mixin. `Entity` (Section 4.1) is the exception — it uses `BaseProvenanceMixin` directly because it has a simpler lifecycle without `confidence`, `knowledge_origin`, or `conflict_evaluated_at`. The two-level hierarchy avoids duplicating the three shared extraction fields while keeping Entity's type distinct.

The sentinel value `source_model = "human"` is reserved for hand-curated ground-truth records (Section 16.1). All code reading `source_model` must treat `"human"` as an authoritative non-LLM origin and exclude such records from auto-promotion logic.

```python
from typing import Annotated, Literal
from datetime import datetime
from pydantic import Field

class BaseProvenanceMixin(BaseModel):
    source_model: str        # LLM model identifier, or "human" for hand-curated records
    source_prompt: str | None  # None for hand-curated records; populated for all LLM-extracted records
    created_at: datetime
    extraction_run_id: str | None = None  # UUID from extraction_runs table; None for hand-curated records

class ProvenanceMixin(BaseProvenanceMixin):
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    status: Literal["proposed", "accepted", "deprecated", "conflicted"] = "proposed"
    knowledge_origin: Literal["physical", "engineering", "cultural", "architectural"]
    rationale: str | None = None  # optional free-text explanation
    conflict_evaluated_at: datetime | None = None  # set by conflict-detection batch (Section 10.1); None = not yet evaluated, ineligible for auto-promotion
```

---

## 4.1 Entity

Represents a building concept. All other knowledge is expressed in relation to entities.

Entities are LLM-extracted and require provenance tracking. Entity does not use `knowledge_origin` (it is derived from the majority origin of its associated assertions at query time) or `status` in the same way as assertions — it uses a simpler lifecycle.

Aliases are not stored inline on the Entity — they live in a separate `entity_aliases` table (entity_id, alias) to support efficient lookup in both directions and avoid JSON serialization in the row.

```python
class Entity(BaseProvenanceMixin):  # NOT ProvenanceMixin — Entity has no confidence, knowledge_origin, or conflict_evaluated_at
    id: str  # UUID
    name: str                  # canonical normalised name, e.g. "roof"
    entity_type: Literal["component", "system", "space", "material", "activity"]
    # "activity" is used for construction activities referenced by ProcessRelation (e.g. "waterproofing", "finishing")
    description: str = ""
    status: Literal["proposed", "accepted", "deprecated"] = "proposed"
    # "conflicted" is deliberately absent: entities are never themselves in conflict.
    # Conflict state is expressed on Assertion records, not on the entities they reference.
    is_entrance: bool = False
    # Marks this entity as a building entrance for topology validation (Section 16.3 / bsos validate --topology).
    # Set via `bsos curate` or manually. Name-based heuristics ("entrance", "lobby") are unreliable
    # across LLM extraction runs — this explicit flag is authoritative.
```

---

## 4.2 Assertion

`subject_id` and `object_id` are Entity UUIDs. Using IDs (not names) means normalization renames do not silently break existing assertions — names are resolved at query time via join to the entities table. `subject_type` and `object_type` mirror the `entity_type` of the referenced entity and are cached here for query performance.

```python
class Assertion(ProvenanceMixin):
    id: str  # UUID

    subject_id: str   # UUID of subject Entity
    predicate: str
    object_id: str    # UUID of object Entity

    subject_type: str  # cached from Entity.entity_type
    object_type: str   # cached from Entity.entity_type

    conditions: list[str] = []   # free-text strings; contexts under which this assertion holds
    exceptions: list[str] = []  # free-text strings; known valid exceptions to this assertion

    applicability: list[str] = []  # free-text strings; building types, climates, or scales where this assertion applies

    cross_prompt_consistency: Annotated[float, Field(ge=0.0, le=1.0)] | None = None  # mean pairwise cosine similarity of assertion embeddings across N prompt framings; set during Pass 2; None means only one framing produced this assertion
    prompt_framing_count: int | None = None  # number of distinct prompt framings that produced this assertion; used to interpret cross_prompt_consistency (lower count = less reliable signal)
```

---

## 4.3 Pattern

```python
class Pattern(ProvenanceMixin):
    id: str
    name: str

    context: list[str]               # free-text strings; conditions under which this pattern applies

    problem: str
    force_descriptions: list[str]    # populated at extraction time (Pass 8): free-text force
                                     # descriptions (e.g. "increased daylight", "reduced heat gain");
                                     # cleared to [] by Pass 10 resolution after all entries are
                                     # matched and moved to force_ids; unresolved entries are written
                                     # to pending_force_refs and then removed from this field
    force_ids: list[str] = []        # Force UUIDs; populated by Pass 10 resolution; empty until then;
                                     # never contains free-text strings — treat any element as a UUID

    solution: str

    consequences: list[str]          # free-text strings; outcomes (positive and negative) of applying this pattern

    related_pattern_names: list[str]  # populated at extraction time (Pass 8): free-text pattern names
                                      # (e.g. "south-facing courtyard"); cleared to [] by Pass 10
                                      # resolution after matching; unresolved names are written to
                                      # pending_pattern_refs and then removed from this field
    related_pattern_ids: list[str] = []  # Pattern UUIDs; populated by Pass 10 resolution; empty
                                         # until then; never contains free-text strings

    emergent_properties: list[str]   # free-text strings; properties that arise when this pattern is applied
```

---

## 4.4 Force

Each `Force` is a single directional pressure. `balance` has been removed — equilibrium between opposing forces is a Pattern-level outcome, not a Force attribute.

```python
from enum import Enum

class ForceDirection(str, Enum):
    increase = "increase"   # e.g. "increased daylight"
    decrease = "decrease"   # e.g. "reduced heat gain"

class Force(ProvenanceMixin):
    id: str  # UUID
    name: str
    direction: ForceDirection

    affects: list[str]  # Entity UUIDs — entities this force acts upon
```

---

## 4.5 AntiPattern

```python
class AntiPattern(ProvenanceMixin):
    id: str  # UUID
    name: str

    conditions: list[str]    # free-text strings; configurations or contexts that constitute this anti-pattern

    consequences: list[str]  # free-text strings; failure outcomes that result from this anti-pattern

    mitigations: list[str]   # free-text strings; design changes that resolve or avoid the anti-pattern
```

---

## 4.6 Process / Sequence

```python
from pydantic import model_validator

class ProcessRelation(ProvenanceMixin):
    id: str  # UUID
    predecessor_id: str  # Entity UUID of the preceding activity or component
    successor_id: str    # Entity UUID of the following activity or component

    hard_constraint: bool

    @model_validator(mode="after")
    def _rationale_required(self) -> "ProcessRelation":
        if not self.rationale:
            raise ValueError("ProcessRelation.rationale must always be populated")
        return self
```

`rationale` is inherited from `ProvenanceMixin` and MUST always be populated for `ProcessRelation` — it explains the physical reason the predecessor must precede the successor (e.g. "waterproofing must precede finishes because moisture trapped behind finishes causes rot and mould").

Both `predecessor_id` and `successor_id` MAY reference entities of `entity_type = "activity"` (e.g. "waterproofing", "finishing") or `entity_type = "component"` (e.g. "foundation", "structure") — relations between components are valid when a clear physical dependency exists. No type-level validation is enforced; the extraction prompt encourages activity-typed entities as the primary case, but component-to-component and activity-to-component relations are accepted.

**Uniqueness constraint:** The SQLModel persistence class for `ProcessRelation` MUST declare a `UniqueConstraint("predecessor_id", "successor_id", "source_model")` on the table. Keying on `source_model` as well as the entity pair allows different models to each insert their version of the same relation — this is required for multi-model consensus (Section 7) and the `INSERT OR IGNORE` deduplication strategy used by concurrent workers (Section 6). Without this constraint, the database engine cannot enforce idempotent insertion across concurrent workers. When two models disagree on `hard_constraint` for the same `(predecessor_id, successor_id)` pair, the divergence MUST be surfaced by the conflict detection batch (`bsos validate --conflicts`) as a reviewable item rather than being silently resolved by whichever model wrote first.

**Within-model divergence:** The unique constraint prevents duplicate rows from the same model for the same pair. However, two workers assigned to the two endpoint entities of the same relation (within the same model's extraction run) may independently extract the same `(predecessor_id, successor_id)` pair with conflicting `hard_constraint` values — only the first insertion succeeds, the second is silently dropped by INSERT OR IGNORE. To detect this, Pass 5 MUST log a warning via `structlog` whenever an INSERT OR IGNORE is triggered. An INSERT OR IGNORE suppression is detected by checking `cursor.rowcount == 0` after the statement executes. **The suppressed insert does not return the existing row** — the worker MUST immediately issue a `SELECT` to retrieve the existing `ProcessRelation` row for `(predecessor_id, successor_id, source_model)` and compare its `hard_constraint` value against the rejected value. If they differ, the warning MUST be elevated to ERROR level and the pair written to a `pending_process_relation_reviews` in-memory list printed in the `bsos extract` completion summary so the operator can investigate.

---

## 4.7 Spatial Relation

```python
class SpatialRelation(ProvenanceMixin):
    id: str  # UUID
    subject_id: str  # Entity UUID
    relation: str    # connects_to, contains, adjacent_to, etc.
    object_id: str   # Entity UUID
```

---

## 4.8 AbstractionNode

Used by the semantic compression system. Represents a higher-level concept that aggregates child assertions. Does NOT replace children.

`knowledge_origin` is derived from the majority origin of child assertions at query time, not independently assigned. `status` starts as `proposed` and requires human acceptance — automated validation of "strictly weaker" is not reliable.

**Queue depth cap:** If the number of `proposed` `AbstractionNode` records awaiting human acceptance exceeds 200, the abstraction synthesis step (Pass 10 / Section 8.4) MUST pause and log a warning rather than adding more. This prevents unbounded queue growth when the review loop is not keeping up with extraction rate.

**`knowledge_origin` is NOT stored on `AbstractionNode`** — it is always computed from the majority `knowledge_origin` value across child assertions at query time. Storing it would require a sentinel value that callers must never use directly, creating a persistent source of bugs. The `abstraction_node_effective_origins` SQLite view (see Implementation safety below) is the authoritative source for all `knowledge_origin` queries against `AbstractionNode` rows.

**Auto-promotion interaction:** When auto-promotion logic (Section 7) evaluates an `AbstractionNode`, it MUST query the `abstraction_node_effective_origins` view for the effective knowledge origin rather than reading the `abstraction_nodes` table directly. The effective origin determines which promotion threshold applies (Section 7).

**Implementation safety:** The REQUIRED implementation is a SQLite view (`abstraction_node_effective_origins`) that computes majority origin via `GROUP BY` + aggregation over child assertions. The view approach keeps resolution logic in the database layer, making it impossible for call sites to bypass it. Any query method that filters or sorts by `knowledge_origin` on `AbstractionNode` records MUST join against this view. SQLite supports views universally — no fallback implementation is provided. The `bsos init` command MUST create this view as part of database initialization (alongside the Alembic-managed tables), and `bsos doctor` MUST verify its existence and fail loudly if it is missing.

```python
class AbstractionNode(ProvenanceMixin):
    # knowledge_origin is NOT a stored field — it is excluded from this class.
    # Always query abstraction_node_effective_origins view for origin; never read from the row directly.
    id: str
    statement: str           # the abstraction itself
    child_ids: list[str]     # IDs of the assertions this aggregates
    abstraction_rationale: str
```

**Multiple parents:** A child assertion UUID MAY appear in the `child_ids` of more than one `AbstractionNode` — the same assertion can be relevant to two distinct higher-level abstractions. This is valid and expected. Cascade logic (Section 10.3) MUST re-evaluate ALL parent `AbstractionNode` records that contain the affected child, not just one.

---

## 4.9 ReviewDecision

Records the outcome of a human review action. Stored in the `review_decisions` table so items are not re-flagged after a decision has been made.

```python
class ReviewDecision(BaseModel):
    id: str  # UUID
    item_id: str
    item_type: str  # "assertion", "abstraction_node", "pending_predicate", etc.
    decision: Literal["accept", "reject", "map_to", "defer"]
    mapped_to: str | None = None  # used when decision is "map_to" (predicate mapping)
    rationale: str | None = None
    reviewer: str  # "human" for manual decisions; model identifier when the conflict-classification pipeline auto-resolved the item without human input
    created_at: datetime
```

---

## 4.10 Constraint

Represents a hard rule or invalid state (Section 3.3). Constraints differ from Assertions in that they are binary — the condition either holds or is violated — and they drive the validation system (Section 16.2).

```python
class Constraint(ProvenanceMixin):
    id: str  # UUID
    subject_id: str  # Entity UUID the constraint applies to
    rule: str        # natural-language description, e.g. "must have a drainage path"
    constraint_type: Literal["must", "must_not"]
    conditions: list[str] = []  # contexts where this constraint applies
    exceptions: list[str] = []  # known valid exceptions
```

---

## 4.11 PendingPredicate

Tracks predicates and spatial relation types that appeared in extraction but are not yet in the core vocabulary. The same model backs both the `pending_predicates` and `pending_spatial_relation_types` tables, distinguished by `vocabulary_type`.

```python
class PendingPredicate(BaseModel):
    id: str  # UUID
    value: str  # the predicate or spatial relation type as extracted
    vocabulary_type: Literal["predicate", "spatial_relation"]
    occurrence_count: int = 1
    first_seen_at: datetime
    last_seen_at: datetime
    flagged_for_review: bool = False  # set True when occurrence_count reaches threshold
```

**SQLModel persistence for `PendingPredicate`:** Because this model backs two physical tables distinguished by `vocabulary_type`, implement two separate SQLModel `(table=True)` classes — `PendingPredicateRow` (table name `pending_predicates`) and `PendingSpatialRelationTypeRow` (table name `pending_spatial_relation_types`) — sharing the same field set. The Pydantic class `PendingPredicate` above is used for validation only; the persistence layer converts to the appropriate SQLModel class based on context. This produces clean independent Alembic migrations for each table and avoids SQLModel polymorphism complexity.

---

# 5. Controlled Vocabulary System (MANDATORY)

Do NOT allow free-form predicates in final storage.

---

## 5.1 Core Predicate Set

```python
CORE_PREDICATES: frozenset[str] = frozenset({
    "requires",        # functional dependency: "corridor requires minimum width"
    "depends_on",      # existence/process dependency: "finishes depends_on waterproofing"
    "protects_from",   # shielding: "roof protects_from precipitation"
    "unsuitable_for",  # incompatibility: "timber unsuitable_for permanent immersion"
    "improves",        # performance enhancement: "insulation improves thermal performance"
    "conflicts_with",  # design incompatibility: "large glazing conflicts_with privacy"
    "contains",        # compositional/material: "wall contains insulation" — NOT spatial containment
    "connects_to",     # functional connectivity: "pipe connects_to pump" — NOT spatial topology
    "supports",        # structural load transfer: "beam supports slab" — NOT general assistance
})
```

`contains` and `connects_to` also appear in `SPATIAL_RELATION_TYPES` (Section 5.4) with distinct semantics: use `Assertion` for compositional or functional relationships, `SpatialRelation` for topological ones. The conflict detection system operates within item types and will not detect cross-layer duplication — rely on the semantic distinction to prevent it.

---

## 5.2 Pending Predicate Queue

During extraction, the LLM may use any predicate. Unknown predicates are not rejected — they are written to a `pending_predicates` table with an occurrence count.

Rules:

* when a pending predicate reaches occurrence threshold N, flag it for human review
* threshold N scales with corpus size: `min(50, round(5 + total_assertions * 0.005))` — linear growth capped at 50 (5 at 0 assertions, 10 at 1000, 30 at 5000, 50 at 9000+). The cap prevents important predicates from requiring thousands of occurrences before review at large scale. The `max(5, N * 0.01)` formulation is rejected because it lacks an upper cap — at 50,000 assertions it would require 500 occurrences before a predicate is reviewed, making infrequent but important predicates effectively invisible. `total_assertions` is queried at the time of the threshold check — not cached at extraction start. If `pending_predicate_threshold_override` is set in the `config` table, its integer value replaces the formula output entirely.
* human review decision: add to core set, or map to an existing core predicate
* the extended pool is not a permanent state — it drains over time into the core set

---

## 5.3 Predicate Registry

Each core predicate has a `PredicateDefinition` record. The registry is a module-level constant (`PREDICATE_REGISTRY: dict[str, PredicateDefinition]`) and is authoritative source code — not stored in the database.

```python
from typing import Literal

EntityType = Literal["component", "system", "space", "material", "activity"]

class PredicateDefinition(BaseModel):
    predicate: str
    meaning: str
    allowed_subject_types: list[EntityType]  # empty list = any type allowed
    allowed_object_types: list[EntityType]   # empty list = any type allowed
    directional: bool  # True = (A pred B) ≠ (B pred A)
```

Example entries (abbreviated):

```python
PREDICATE_REGISTRY: dict[str, PredicateDefinition] = {
    "requires": PredicateDefinition(
        predicate="requires",
        meaning="functional dependency — subject cannot operate correctly without object",
        allowed_subject_types=[],
        allowed_object_types=[],
        directional=True,
    ),
    "supports": PredicateDefinition(
        predicate="supports",
        meaning="structural load transfer — subject transmits loads from or to object",
        allowed_subject_types=["component", "system"],
        allowed_object_types=["component", "system"],
        directional=True,
    ),
    # one entry per CORE_PREDICATES item
}
```

The `curate add` command validates the chosen predicate against `PREDICATE_REGISTRY` keys and rejects unknowns. Extraction prompts for Passes 3–9 include the predicate meanings from this registry to calibrate the LLM.

---

## 5.4 Spatial Relation Controlled Vocabulary

`SpatialRelation.relation` is subject to a separate controlled vocabulary. Unknown spatial relation types follow the same pending queue and threshold rules as predicates (Section 5.2).

```python
SPATIAL_RELATION_TYPES = [
    "adjacent_to",
    "contains",
    "connects_to",
    "accessible_from",
    "above",
    "below",
    "enclosed_by",
]
```

---

# 6. Extraction Pipeline (MULTI-PASS REQUIRED)

Implement sequential extraction pipeline. Cache all LLM responses keyed by `(model, prompt_hash)` to avoid redundant API calls. Cache implementation: a dedicated SQLite table `llm_response_cache` with columns `(model TEXT, prompt_hash TEXT, response_json TEXT, cached_at DATETIME)`, keyed on `(model, prompt_hash)` where `prompt_hash` is the SHA-256 of the prompt string. This persists the cache across CLI invocations and is consistent with the rest of the storage layer.

**Full pass execution order:**

```
Pass 1  — Concept Discovery (all models, sequential)
Pass 2  — Entity Deduplication (runs once, across all models' entity pools)
Pass 3  — Relationship Extraction      ┐
Pass 4  — Spatial Relation Extraction  │ Stage C: parallelisable across entities
Pass 5  — Process / Sequence           │ sequential per entity
Pass 6  — Constraint Extraction        │
Pass 7  — Anti-Pattern Extraction      │
Pass 8  — Pattern Extraction           │
Pass 9  — Force Extraction             ┘
Pass 10 — Normalization (ref resolution, predicate stabilization, abstraction synthesis)
          [Cross-model Matching runs automatically after Pass 10 completes for all models]
Pass 11 — Adversarial Validation
```

Pass 1 MUST be run for all participating models before any other pass begins. Pass 10 MUST be run per-model (once for each model's extracted data) before cross-model matching begins — see Section 7.

All entities from all models are committed to a single shared entities table. Entity deduplication (Pass 2 / Section 8.2) runs across the combined pool first, producing stable canonical UUIDs. Only after deduplication is complete do Passes 3–9 run — they store UUIDs as foreign keys and cannot proceed without a stable entity set. Cross-model assertion matching (Section 7) requires normalization (Pass 10) to have run on all models' assertions first, so it executes after Pass 10 has completed for every participating model.

**Resume semantics:** Before processing any entity in Passes 3–9, the pipeline checks the `pass_progress` table for an existing `completed` row matching `(pass_number, entity_id, model)`. If found, the entity is skipped — no LLM call is made and no DB write is attempted. This makes each pass idempotent: a crashed or interrupted run can be restarted with the same `bsos extract` command and will continue from where it left off without re-processing completed entities. Entities with status `skipped` (timed out or parse-failed) are also skipped on resume; re-processing them requires explicit re-seeding with `--seed <entity>`. The `bsos status` command displays per-pass completion counts derived from this table.

**Pass parallelism:** Passes 3–9 are logically independent *across entities* and MAY be parallelised across entities using a thread pool. Cap the thread pool at 4 workers **globally across all concurrent model extraction runs** to avoid SQLite write contention — see Section 12 for the WAL mode requirement. If two models are being extracted concurrently, they share the same 4-worker pool, not 4 workers each. Use a single `concurrent.futures.ThreadPoolExecutor(max_workers=4)` — its internal queue naturally serializes tasks beyond the worker cap, so no explicit queuing mechanism is required. The orchestrator MUST create this executor before dispatching any model's Passes 3–9 and MUST pass it as an argument to each model's extraction worker; having each worker create its own executor would violate the 4-worker global cap. Passes MUST run sequentially for the *same entity* to prevent race conditions on entity records. `ProcessRelation` insertions (Pass 5) link two entities and may be attempted by workers assigned to each endpoint independently; the insertion MUST use INSERT OR IGNORE keyed on `(predecessor_id, successor_id, source_model)` to prevent duplicates — see Section 4.6.

---

## Pass 1 — Concept Discovery

Identify relevant building concepts.

**Inputs:** Pass 1 accepts a seed topic — either:

* a user-supplied list of building concepts (e.g. `["roof", "foundation", "corridor", "HVAC system"]`), or
* a free-text domain description (e.g. "commercial office building systems and components")

If no input is supplied, the system bootstraps with a default seed prompt asking the LLM to enumerate the major systems, components, spaces, materials, and **construction activities** found in typical buildings, producing an initial concept list. This default seed list is then expanded by asking the LLM for sub-concepts and related terms for each top-level concept before committing entities to the database. Construction activities (e.g. "waterproofing", "finishing", "formwork removal", "curing") MUST be included in the seed to ensure that `ProcessRelation` records extracted in Pass 5 can reference existing entity UUIDs rather than requiring inline entity creation.

**Pass 1 model parallelism:** Pass 1 MAY run concurrently across participating models using separate threads within the same `bsos extract` invocation — one thread per model. (Because `bsos extract` holds a process-level file lock at startup, two separate `bsos extract` processes cannot overlap; "concurrent" always means within-process threads.) Each thread independently writes its discovered entities to the shared entities table via WAL-mode SQLite. Concurrent writes will produce duplicate entity names across models; this is expected and resolved by Pass 2 deduplication. The `source_model` field distinguishes each model's contributions. Pass 2 MUST NOT begin until Pass 1 has completed for all participating models.

---

## Pass 2 — Entity Deduplication

Run entity clustering (Section 8.1) and canonical concept formation (Section 8.2) across the combined entity pool from all models. Produces stable canonical UUIDs used as foreign keys in Passes 3–9. This pass MUST complete before any relationship or knowledge extraction begins.

---

## Pass 3 — Relationship Extraction

Extract assertions between concepts.

Use multiple prompt framings per concept (minimum 3 distinct phrasings). Semantic consistency across framings is a reliability signal.

To compute `cross_prompt_consistency`:

1. Run N ≥ 3 prompt variants for the same concept; each variant produces a set of assertions
2. Embed all extracted assertion texts using `all-mpnet-base-v2`
3. For each pair of framings (framing i, framing j), build a cosine similarity matrix over all assertion embeddings. Use the **linear sum assignment** algorithm (`scipy.optimize.linear_sum_assignment`) to find the one-to-one (bijective) maximum-weight matching. Do NOT use greedy nearest-neighbour — it allows multiple assertions in framing i to match the same assertion in framing j, inflating the consistency score. Only matched pairs with cosine similarity ≥ 0.70 are considered aligned; unmatched assertions are treated as framing-unique.
4. For each aligned group (assertions matched across k ≥ 2 framings), compute pairwise cosine similarity across the k matched embeddings
5. Store the mean pairwise similarity as `cross_prompt_consistency` on the canonical assertion (range 0.0–1.0); assertions that appear in only one framing get `None`. Store the count of framings that produced this assertion in `prompt_framing_count`.

**Interim embedding handling:** Assertion texts across N prompt framings for a single entity do not have stable `item_id` values until persisted. Compute their embeddings into a temporary in-memory NumPy array for the duration of the consistency calculation, then discard them. Only the winning canonical assertion is persisted; its embedding is stored in the `embeddings` table (Section 8.5) in the normal way. This avoids writing embeddings for rejected or transient framing variants into the cache.

Partial matches (assertion appears in k framings where 1 < k < N) receive the mean pairwise cosine similarity across those k embeddings using the same formula, with `prompt_framing_count = k`. The score is valid but less reliable than a full N-framing match; review tooling MUST display `prompt_framing_count` alongside the score so reviewers can judge confidence accordingly.

A score ≥ 0.85 with `prompt_framing_count` ≥ 3 indicates the assertion is stable across prompt framings.

---

## Pass 4 — Spatial Relation Extraction

Extract spatial and topological relationships between entities, producing `SpatialRelation` records (Section 3.6 / Section 4.7).

For each entity, ask the LLM: "what are the spatial and topological relationships between X and other building entities?" Map extracted relations to the controlled vocabulary in Section 5.4. Unknown relation types are written to `pending_spatial_relation_types` (Section 5.4) rather than rejected, following the same occurrence-threshold and human-review rules as pending predicates (Section 5.2).

Pass 4 runs in the same parallelism envelope as Passes 3 and 5–9: parallelisable across entities, sequential per entity.

---

## Pass 5 — Process / Sequence Extraction

Extract temporal ordering and construction dependency chains, producing `ProcessRelation` records (Section 3.8 / Section 4.6).

Ask the LLM: "what must happen before X, and what can only happen after X?" for each entity. Populate `predecessor_id`, `successor_id`, and `hard_constraint`. The `rationale` field MUST always be populated (Section 4.6).

**Activity entity creation:** When the LLM names a predecessor or successor activity that has no matching entity in the database, Pass 5 MUST create a new `Entity` with `entity_type = "activity"` and `status = "proposed"` before inserting the `ProcessRelation`. Creation is idempotent — check for an exact case-insensitive name match before inserting. Activity entities created inline inherit `source_model` and `source_prompt` from the current extraction context. (This is a fallback for activities not discovered in Pass 1; the default Pass 1 seed should enumerate common construction activities to minimise inline creation.) Pass 5 MUST emit a structlog WARNING each time it creates an activity entity inline, including the entity name and the entity currently being processed, so operators can identify gaps in the Pass 1 seed and add the missing activities for future runs.

Pass 5 runs in the same parallelism envelope as Passes 3–4 and 6–9: parallelisable across entities, sequential per entity. See the INSERT OR IGNORE deduplication requirement in the parallelism note above.

---

## Pass 6 — Constraint Extraction

Extract hard rules and invalid states, producing `Constraint` records (Section 3.3 / Section 4.10).

For each entity, ask the LLM: "What conditions MUST or MUST NOT hold for a valid building design involving X? List only binary constraints where violation makes the design physically invalid, unsafe, or inoperable — not typical functional dependencies or preferences."

Structured output must include:
* `rule` — natural-language constraint statement (e.g. "must have a drainage path")
* `constraint_type` — `"must"` or `"must_not"`
* `conditions` — contexts where this constraint applies (list of strings)
* `exceptions` — known valid exceptions (list of strings)
* `confidence` — certainty that this is a universal physical requirement (not a code or standard)
* `knowledge_origin` — one of `"physical"`, `"engineering"`, `"cultural"`, `"architectural"`
* `rationale` — why this constraint is a hard binary rule rather than a soft dependency

Apply the boundary rule from Section 3.3: if the LLM produces an item that would be equally valid as an `Assertion` with predicate `requires`, prefer the `Assertion` unless the constraint is genuinely binary (violation = invalid configuration). Borderline items MUST be stored as `Assertions`, not `Constraints`. The extraction prompt MUST include one example of each type to calibrate the LLM:

> **Constraint example:** "roof must have a drainage path" — a flat roof with no drainage will pond and eventually fail; there is no valid exception for a permanent occupied roof.
>
> **Assertion example (not a constraint):** "roof requires structural support" — true in general, but the form of support varies; a tension membrane roof transfers loads differently from a timber frame. Use Assertion with predicate `requires`.

Pass 6 runs in the same parallelism envelope as Passes 3–5 and 7–9.

---

## Pass 7 — Anti-Pattern Extraction

Extract failure conditions, producing `AntiPattern` records (Section 3.7 / Section 4.5).

For each entity, ask the LLM: "What are known failure conditions, pathological configurations, or design mistakes involving X that consistently lead to physical, functional, or performance failures? Describe the configuration, its consequences, and how it can be avoided or corrected."

Structured output must include:
* `name` — short label for the anti-pattern (e.g. "internal bathroom without ventilation")
* `conditions` — the configuration or context that constitutes the anti-pattern (list of strings)
* `consequences` — failure outcomes that result (list of strings)
* `mitigations` — design changes that resolve or avoid the anti-pattern (list of strings)
* `confidence` — certainty that this failure mode is universally applicable
* `knowledge_origin` — one of `"physical"`, `"engineering"`, `"cultural"`, `"architectural"`
* `rationale` — why this configuration consistently fails

Pass 7 runs in the same parallelism envelope as Passes 3–6 and 8–9.

---

## Pass 8 — Pattern Extraction

Extract Alexander-style patterns, producing `Pattern` records (Section 3.4 / Section 4.3).

For each entity, ask the LLM: "What architectural or spatial patterns involving X improve building quality? For each pattern, describe: the context where it applies, the problem it addresses, the competing forces at play, the solution, and what consequences follow from applying it."

Structured output must include:
* `name` — short pattern label (e.g. "light on two sides")
* `context` — conditions under which this pattern applies (list of strings)
* `problem` — the design problem being solved
* `force_descriptions` — force names as free-text strings (list); these are matched to `Force` records after Pass 9 completes (see note below)
* `solution` — the spatial or design move that resolves the problem
* `consequences` — outcomes of applying this pattern, positive and negative (list of strings)
* `emergent_properties` — properties that arise when this pattern is applied (list of strings)
* `confidence` — certainty that this is a cross-cultural architectural pattern, not a local preference
* `knowledge_origin` — one of `"physical"`, `"engineering"`, `"cultural"`, `"architectural"`

**Force resolution (post-Pass 9, global):** At extraction time, `Pattern.force_descriptions` contains free-text force descriptions (e.g. "increased daylight", "reduced heat gain"). After Pass 9 has finished processing all entities across all workers, a resolution step matches each description to a `Force` record by name (exact match first, then cosine similarity ≥ 0.85 using `all-mpnet-base-v2`). Matched descriptions are removed from `force_descriptions` and their corresponding UUIDs appended to `force_ids`. Unresolved descriptions are written to `pending_force_refs` and then removed from `force_descriptions`; after resolution, `force_descriptions` MUST be empty (all entries either resolved to `force_ids` or sent to `pending_force_refs`).

**Related-pattern resolution (post-Pass 9, global):** At extraction time, `Pattern.related_pattern_names` contains free-text pattern names (e.g. "south-facing courtyard"). After Pass 9 has finished processing all entities across all workers, a resolution step matches each name against `Pattern.name` records (exact match first, then cosine similarity ≥ 0.85 using `all-mpnet-base-v2`). Matched names are removed from `related_pattern_names` and their UUIDs appended to `related_pattern_ids`. Unresolved names are written to `pending_pattern_refs` and then removed from `related_pattern_names`; after resolution, `related_pattern_names` MUST be empty. Unresolved `Force.affects` entity names from Pass 9 are written to `pending_entity_refs`. Each table has a dedicated resolution path — see Section 12.

Pass 8 runs in the same parallelism envelope as Passes 3–7 and 9.

---

## Pass 9 — Force Extraction

Extract individual design pressures acting on building decisions, producing `Force` records (Section 3.5 / Section 4.4).

For each entity, ask the LLM: "What individual design pressures act on decisions involving X? For each force, specify: (1) the force name, phrased as a directional pressure (e.g. 'increased daylight', 'reduced heat gain', 'improved privacy'), (2) whether the force acts to increase or decrease its target quality, and (3) which entities it acts upon."

Structured output must include:
* `name` — directional force label; MUST include an explicit directional qualifier (see validation below)
* `direction` — `"increase"` or `"decrease"` (from `ForceDirection`)
* `affects` — entity names as free-text strings (resolved to UUIDs against the entities table; unresolved names are written to `pending_entity_refs` for manual review — they are not discarded)
* `confidence` — certainty that this is a real universal design pressure
* `knowledge_origin` — one of `"physical"`, `"engineering"`, `"cultural"`, `"architectural"`
* `rationale` — what design decisions this force acts upon and why

**ForceDirection consistency validation:** The extraction prompt MUST instruct the LLM that the force name and direction must be consistent, and MUST include the exact permitted qualifier terms so the LLM selects from them rather than inventing novel phrasing. The prompt MUST include wording equivalent to:

> "For `direction = increase`, the force name MUST contain one of: increased, improved, enhanced, maximised/maximized, greater, higher, more, better, stronger, expanded, adequate, sufficient, optimised/optimized. For `direction = decrease`, the name MUST contain one of: reduced, minimised/minimized, decreased, limited, lower, less, fewer, smaller, restricted, constrained, avoided, prevented, eliminated. Use British or American spelling. Do not use other directional qualifiers."

The persistence layer then validates each extracted `Force` before storage using case-insensitive substring matching against the same two lists:
* `direction = "increase"` — name must contain at least one of: `increased`, `improved`, `enhanced`, `maximised`, `maximized`, `greater`, `higher`, `more`, `better`, `stronger`, `expanded`, `adequate`, `sufficient`, `optimised`, `optimized`
* `direction = "decrease"` — name must contain at least one of: `reduced`, `minimised`, `minimized`, `decreased`, `limited`, `lower`, `less`, `fewer`, `smaller`, `restricted`, `constrained`, `avoided`, `prevented`, `eliminated`

Because the extraction prompt pre-constrains the LLM to these terms, validation failures should be rare in practice and indicate prompt non-compliance or an edge-case phrasing. Forces that fail validation are logged via `structlog` at WARNING level and written to `pending_force_refs` with `failure_type = "validation_failure"` for manual review rather than being silently discarded or stored inconsistently. The `--fix` option of `bsos doctor` does not auto-resolve validation failures; they require human correction via `bsos review-pending --type force`.

Pass 9 runs in the same parallelism envelope as Passes 3–8.

---

## Pass 10 — Predicate Stabilization, Ref Resolution, and Abstraction Synthesis

**Scope:** Pass 10 MUST be run once per participating model — once for each model's extracted data — before cross-model assertion matching begins (Section 7). This ensures all assertions are on a normalized predicate basis before cross-model cosine similarity comparison is performed. After Pass 10 has completed for every model, cross-model matching runs automatically.

**Pass 10 synchronization with concurrent model extraction:** When Passes 3–9 are running concurrently for multiple models within the same `bsos extract` invocation, Pass 10 for model A MAY begin as soon as model A's Passes 3–9 are complete, without waiting for model B's Passes 3–9 to finish. Cross-model matching MUST NOT begin until Pass 10 has completed for every participating model. The `bsos extract` orchestrator tracks per-model Pass 10 completion via `pass_progress` and starts cross-model matching only after the final model's Pass 10 is confirmed complete.

Pass 10 runs three steps in sequence:

1. **Force/pattern/entity ref resolution** — resolve all pending free-text refs left by Passes 3–9: `Pattern.force_descriptions` from Pass 8 (matched to `Force` records, moving matched entries to `Pattern.force_ids`), `Pattern.related_pattern_names` from Pass 8 (matched to `Pattern` records, moving matched entries to `Pattern.related_pattern_ids`), and entity name refs from Pass 9 `Force.affects`. Unresolved entries are written to their respective pending tables and removed from the source fields. This step is explicitly part of Pass 10, not an implicit post-Pass-9 step, so it is covered by pass-completion tracking and can be resumed if interrupted. On completion, set `config` key `passes_3_9_refs_resolved = "1"`. **Per-pattern atomicity:** the resolution loop MUST commit each pattern's changes in a separate database transaction before moving to the next. A pattern whose `force_descriptions` is already `[]` is skipped — this makes the step resumable at the per-pattern level: if the process crashes mid-loop, only the in-progress pattern needs re-processing. The same per-item commit discipline applies to `Pattern.related_pattern_names` and `Force.affects` resolution.
2. **Predicate stabilization** — map all extracted predicates to canonical forms (Section 8.3).
3. **Abstraction synthesis** — run abstraction synthesis across the full assertion set (Section 8.4).

**Sub-pass resume:** Pass 10's three steps are tracked independently in `pass_progress` using keys `"10a"`, `"10b"`, and `"10c"` — see Section 12. A crashed run that completed ref resolution but not predicate stabilization resumes at step 2 on restart. The `bsos extract --passes 10` command reports which sub-passes have already completed before executing, and `bsos status` displays sub-pass completion in the same format as full passes.

See Section 8 for the detailed normalization algorithm. All three steps must complete before cross-model matching begins (Section 7).

**Known limitation — per-model abstractions:** Abstraction synthesis runs on per-model data before cross-model matching, so two models may independently produce equivalent `AbstractionNode` records for the same assertion cluster. These duplicates are resolved by the conflict detection batch (`bsos validate --conflicts`), which applies the same embedding similarity + LLM classification algorithm to `AbstractionNode` records as it does to assertions. Implementors MUST ensure the conflict detection batch covers `AbstractionNode` items, not only `Assertion` items.

---

## Pass 11 — Adversarial Validation

Pass 11 operates across the full accepted and proposed assertion set — it is not a per-entity loop. It runs after Pass 10 has completed so all assertions are on a normalized predicate basis before adversarial review.

**Prompts:** For each batch of assertions (grouped by subject entity, up to 20 per LLM call to stay within context limits), ask the LLM:

> "Review these building assertions. For each one, identify: (1) known exceptions where the assertion does not hold, (2) contexts or building types that limit its applicability, (3) any factual error or physical implausibility. Report only findings you are confident about."

Run adversarial validation across multiple models where available to reduce single-model bias. Each model reviews the same assertion set independently; findings are merged.

**Structured output schema:**

```python
class AdversarialFinding(BaseModel):
    item_id: str           # UUID of the assertion, constraint, or pattern being reviewed
    item_type: str         # "assertion" | "constraint" | "pattern" | "antipattern"
    finding_type: Literal["exception", "context_limitation", "potential_error", "scope_restriction"]
    detail: str            # free-text description of the finding
    suggested_action: Literal["add_exception", "add_condition", "flag_for_review", "deprecate"]
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]  # LLM confidence in this finding
```

**State machine integration — how findings update item status:**

| `finding_type` | `suggested_action` | Effect |
|---|---|---|
| `exception` | `add_exception` | Appends `detail` to `Assertion.exceptions` (or equivalent field); item status unchanged |
| `context_limitation` | `add_condition` | Appends `detail` to `Assertion.applicability`; item status unchanged |
| `potential_error` | `flag_for_review` | Sets item `status = "conflicted"`; adds to human review queue; writes `ReviewDecision` with `decision = "defer"` and `reviewer = <model_id>` |
| `scope_restriction` | `deprecate` | Does NOT auto-deprecate — writes a `ReviewDecision` with `decision = "defer"` and `rationale = detail`; a human must confirm via `review-pending` before status changes |

**Confidence threshold:** Findings with `confidence < 0.70` are logged via `structlog` but do not trigger any state change. This prevents low-confidence adversarial hallucinations from polluting the review queue.

**Multi-model merge:** When the same `item_id` receives the same `finding_type` from ≥ 2 models, the finding is promoted: `flag_for_review` findings from a single model are logged only; from ≥ 2 models they update status to `conflicted`. Exception and condition findings are appended regardless of model count but duplicates (identical `detail` strings) are deduplicated before storage.

**Epistemic scope of Pass 11:** Pass 11 is a supplementary triage step, not an independent validation mechanism. All major LLMs share training data provenance and will systematically miss the same domain-specific errors. Multi-model consensus reduces noise from single-model hallucination but does not break the circularity of validating LLM-extracted knowledge with LLMs. The hand-curated ground-truth set (Section 16.1) provides the only genuinely independent validation signal and is the authoritative check on extraction quality. Pass 11 findings should be treated as triage hints for the human review queue, not as confirmation of correctness.

---

## 6.1 Extraction Confidence

Each LLM extraction call (Passes 3–9) MUST request a `confidence` score alongside the structured output. Using `instructor`, include `confidence: float` in the extraction schema and instruct the LLM to rate its certainty that the extracted item represents a true universal building fact (not a regulatory rule, cultural preference, or edge case). Scale: 0.0 = no confidence, 1.0 = certain.

The LLM-assigned score is the initial value stored in `ProvenanceMixin.confidence`. It is subsequently updated by:

* multi-model consensus logic (Section 7): agreement raises it (mean), divergence lowers it (min)
* cross-prompt consistency (Pass 3): `cross_prompt_consistency` ≥ 0.85 is a positive signal but does not directly overwrite `confidence` — it informs the auto-promotion threshold check

Ground-truth records (`source_model = "human"`) are stored with `confidence = 1.0`.

---

## 6.2 LLM API Resilience

All LLM provider calls MUST be wrapped with retry logic and timeout handling.

**Retry policy:** Use exponential backoff with jitter for transient failures (HTTP 429, 500, 502, 503, 504, connection errors). Default: 3 attempts, initial delay 1 s, multiplier 2.0, max delay 60 s. The `instructor` library does not provide built-in retry logic across all providers — implement retries at the `LLMProvider` adapter layer, not inside `instructor` calls.

**Timeout:** Each LLM call MUST have a configurable timeout (default: 120 s). Set via `config` key `llm_timeout_seconds`. Timed-out calls are logged via `structlog` with the entity name and pass number; the entity is skipped for that pass and written to a `skipped_entities` in-memory list so the operator can re-run with `--passes <N> --seed <entity>`.

**Rate limiting:** `LLMProvider` implementations MUST respect API rate limits. For providers that return rate-limit headers (`Retry-After`, `X-RateLimit-Reset`), use the header value as the delay instead of the backoff formula.

**Non-retryable errors:** Authentication failures (HTTP 401, 403) and malformed request errors (HTTP 400) MUST NOT be retried — they indicate a configuration problem and MUST cause `bsos extract` to exit immediately with a clear error message.

**Structured output parse failures (Ollama fallback path):** When the Ollama fallback path (Section 2) produces a response that `json.loads` cannot parse, retry once with an explicit repair prompt: `"Your previous response was not valid JSON. Respond with only a JSON object matching this schema: [schema]."` If the repair attempt also fails, log and skip the item. Do not retry more than once — repeated parse failures indicate the model is not capable of the extraction task for this entity.

**Concurrency lock:** `bsos extract` acquires a file lock (`bsos.lock` in the same directory as the database file) at startup and releases it on exit. If the lock is already held by another process, the command exits with: `"Another bsos extract process is running. If this is stale, delete bsos.lock and retry."` This prevents two concurrent invocations from colliding on Pass 1/2 writes outside the thread-pool envelope.

---

# 7. Multi-Model Consensus (REQUIRED)

To reduce epistemic dependence on a single LLM:

* extract the same assertions from at least 2 different models where available
* convergence across models raises confidence score; divergence flags for review and lowers it
* `physical` and `engineering` assertions: auto-promote to `accepted` when ≥ 2 models agree and mean confidence ≥ 0.80
* `cultural` and `architectural` assertions: require ≥ 3 models agreeing with mean confidence ≥ 0.90, or explicit human acceptance
* **Agreement case:** confidence = mean(per-model extraction confidence scores)
* **Divergence case:** confidence = min(per-model extraction confidence scores); item added to human review queue regardless of the minimum value. "Divergence" means at least one model produced a classification (duplicate/contradictory/complementary) that differs from the majority

**Audit trail for auto-promoted items:** When an item is auto-promoted (not reviewed by a human), a `ReviewDecision` record MUST be written with `decision = "accept"` and `reviewer = <model_id>` (the identifier of the model whose consensus triggered promotion, or `"consensus"` when multiple models contributed). This ensures the review queue and audit log remain complete regardless of whether promotion was manual or automatic. Auto-promotion records are distinguishable from human decisions by the `reviewer` field — `"human"` is reserved for human-initiated decisions only (Section 4.9).

**Cross-model assertion matching:** Assertions from different models will not share UUIDs or identical text. Equivalence is established after normalization has been applied to both extraction sets, using the following algorithm:

1. Normalize all assertions from all models (Pass 10 must be run per-model before comparison)
2. For each assertion from model A, find candidates from model B with cosine similarity ≥ 0.85 (same embedding model, `all-mpnet-base-v2`)
3. Among candidates, require the same normalized predicate — similarity alone cannot distinguish `requires` from `depends_on`
4. If a match is found, the pair is treated as equivalent across models (contributing to consensus); if no match exists, the assertion is model-unique and does not contribute to cross-model consensus

This breaks the circularity of validating LLM output with the same LLM.

## 7.1 Single-Model Fallback

When only one model is available (e.g. local Ollama), the system MUST still function. Apply these rules:

* use cross-prompt consistency score as the primary confidence signal in place of cross-model consensus
* `physical` and `engineering` assertions with cross-prompt consistency ≥ 0.85 across 3+ prompt framings are eligible for auto-promotion
* `cultural` and `architectural` assertions always require explicit human acceptance when only one model is available
* the `source_model` field records which path was taken; review tooling MUST surface single-model assertions distinctly so they can be re-validated if a second model becomes available

## 7.2 Incremental Model Addition

When a new model's extraction is added after an initial run has already completed cross-model matching:

1. Run Passes 1–10 for the new model against the existing shared entity set. If the new model introduces entities not yet in the database, re-run Pass 2 deduplication across the expanded entity pool before proceeding to Passes 3–9.
2. Run cross-model matching between the new model's normalized assertions and all *existing* normalized assertions from prior models. Do not re-run matching across prior model pairs — their pairwise results are already stable.
3. For assertions where the new model agrees with an existing `accepted` assertion: apply the consensus confidence update (mean) and write a `ReviewDecision` with `reviewer = "consensus"`. The assertion's `status` remains `accepted`.
4. For assertions where the new model introduces a new disagreement with an `accepted` assertion: set `status = "conflicted"` on the accepted assertion and add it to the review queue. This may re-open previously resolved items — this is correct behaviour.

**CLI workflow:** `bsos extract --models <new-model-id> --passes 1,2,3,4,5,6,7,8,9,10` followed by `bsos validate --conflicts` covers the incremental addition workflow. The operator is responsible for running Pass 2 explicitly (via `--passes 2`) if new entities were introduced.

---

# 8. Semantic Normalization Pipeline

Implement:

## 8.1 Entity Clustering

Use embeddings only for similarity grouping. Algorithm:

1. Compute `all-mpnet-base-v2` embeddings for all entity names in the combined pool from Pass 1.
2. Build a pairwise cosine similarity matrix.
3. Run agglomerative clustering with average linkage and a distance threshold of 0.20 (cosine similarity ≥ 0.80). Use `sklearn.cluster.AgglomerativeClustering(metric="cosine", linkage="average", distance_threshold=0.20, n_clusters=None)`. Do not pre-specify the number of clusters.
4. Clusters of size 1 pass through unchanged — no merging needed.
5. Clusters of size ≥ 2 proceed to Section 8.2 for canonical concept formation.

**Threshold rationale:** 0.20 distance (cosine similarity ≥ 0.80) is conservative — it groups clear synonyms ("corridor", "hallway") while keeping distinct concepts ("roof", "ceiling") apart. The threshold is overridable via `bsos config set entity_cluster_threshold <float>` for corpora where the default produces over- or under-merging.

---

## 8.2 Canonical Concept Formation

Merge clusters into stable concepts. Merging N entities into one canonical entity MUST follow this algorithm, executed in a single database transaction:

1. Designate one entity as canonical (prefer the entity with the most associated assertions; break ties by `created_at` ascending)
2. Insert all non-canonical entity names and aliases into `entity_aliases` keyed to the canonical entity's UUID (preserving lookup of old names)
3. `UPDATE assertions SET subject_id = <canonical_uuid> WHERE subject_id IN (<deprecated_uuids>)`
4. `UPDATE assertions SET object_id = <canonical_uuid> WHERE object_id IN (<deprecated_uuids>)`
5. If `entity_type` changed: `UPDATE assertions SET subject_type = <new_type> WHERE subject_id = <canonical_uuid>`; same for `object_type`
6. Apply equivalent UUID replacement updates to `spatial_relations`, `process_relations`, `constraints`, and `forces.affects`
7. Mark all non-canonical entities as `status = "deprecated"` in the entities table
8. Roll back the entire transaction if any step fails — partial merges leave the graph in an inconsistent state

If an entity is only renamed (no merge), steps 3–6 are skipped; only step 2 (alias insertion) and the `entity_type` cache update (step 5) are needed.

**Atomicity note for JSON list fields:** Steps 3–6 handle UUID substitution in SQL columns via plain `UPDATE`. However, `Force.affects`, `Pattern.force_ids`, `Pattern.related_pattern_ids`, and `AbstractionNode.child_ids` are stored as JSON-encoded text (Section 12, list field serialization). SQL `UPDATE` cannot substitute UUIDs inside a JSON string. The persistence layer MUST handle these fields separately: for each table with a JSON list column containing entity or assertion UUIDs, deserialize the list in Python, substitute deprecated UUIDs with the canonical UUID, and re-serialize. These Python-level operations happen within the same SQLite transaction as the SQL UPDATEs — open the transaction before the first SQL UPDATE and commit after the final re-serialization write. If the Python-level substitution raises an exception, the transaction is rolled back and no changes are persisted. This is the same transaction that covers all eight steps above; the Python-level writes are included in it, not separate.

---

## 8.3 Predicate Stabilization

Map extracted predicates to canonical core predicates using a two-phase algorithm:

**Phase 1 — Embedding-based auto-mapping:**

For each non-core predicate P appearing in extracted assertions:

1. Compute the cosine similarity between the embedding of P and the embedding of each core predicate name (Section 5.1), using `all-mpnet-base-v2`
2. If the best-matching core predicate has similarity ≥ 0.85: auto-map P → that core predicate; update all assertions using P; log the mapping
3. If the best match is in the range [0.60, 0.85): proceed to Phase 2 (LLM disambiguation)
4. If the best match is below 0.60: write P to `pending_predicates` (Section 5.2); do not attempt auto-mapping

**Phase 2 — LLM disambiguation for ambiguous predicates:**

For predicates in the [0.60, 0.85) similarity range, run a single LLM classification call. The LLM provider used is determined as follows: when predicate stabilization runs as part of standalone `bsos normalize`, use the provider configured in `default_llm_model`. When predicate stabilization runs inside `bsos extract` (as Pass 10), use the first model in the `--models` list (the primary model). If neither is available, exit with an error directing the user to set `default_llm_model`. This provider resolution applies only to Phase 2 disambiguation — it is not used for any other step of Pass 10.

> "In the building domain, which of these core predicates best captures the meaning of '{P}'? Options: [CORE_PREDICATES list with one-line descriptions]. If none apply precisely, respond 'pending'."

* If the LLM returns a core predicate: map P → that predicate; update assertions; log the mapping with `source = "llm_disambiguation"`
* If the LLM returns `"pending"`: write P to `pending_predicates`

**Deduplication:** Before writing a pending predicate, check whether it already exists in `pending_predicates`. If it does, increment `occurrence_count` and update `last_seen_at` rather than inserting a duplicate row.

**Mapping persistence:** All auto-mappings and LLM-resolved mappings are recorded in a `predicate_mappings` table `(original TEXT, canonical TEXT, method TEXT, mapped_at DATETIME)` so that re-runs of the normalization pass do not re-classify already-resolved predicates.

---

## 8.4 Abstraction Synthesis

Merge repeated assertions into higher-level concepts. See section 9 for the required algorithm.

---

## 8.5 Embedding Storage

Embeddings are computed using `all-mpnet-base-v2` and stored in the dedicated `embeddings` table (Section 12), keyed by `(item_type, item_id)`, as a BLOB. The BLOB is the raw bytes of a float32 NumPy array (`ndarray.astype("float32").tobytes()`), dimension 768 for `all-mpnet-base-v2`. The `embeddings` table includes a `dim` integer column storing the vector dimension so that retrieval code can reconstruct the array with `numpy.frombuffer(blob, dtype="float32").reshape(dim)` and validate against the expected dimension.

Embeddings are computed once on insertion and cached; they are recomputed only when the underlying text or embedding model changes. Do not recompute on every conflict-detection pass — use the cached value.

**Invalidation triggers:** The embedding input text for an `Assertion` is `"{subject_name} {predicate} {object_name}"`, where names are resolved from the entities table. When entity deduplication or renaming changes an entity's canonical name, all assertions referencing that entity via `subject_id` or `object_id` have stale embeddings even though the assertion row itself has not changed. The `embeddings` table MUST include:

* a `content_hash TEXT` column storing the SHA-256 of the embedding input string at the time the embedding was computed
* a `model TEXT` column storing the embedding model identifier (e.g. `"all-mpnet-base-v2"`) used to produce the vector

On each use, the storage layer checks whether the current resolved input string's hash matches `content_hash` AND the configured embedding model matches `model`; if either differs, it recomputes the embedding and updates the BLOB, hash, and model columns. This check MUST also run after any entity merge or rename operation (Section 8.2). If the embedding model is changed in configuration, a migration pass (triggered via `bsos normalize --reembed`) MUST recompute all stored embeddings before any similarity search is run — mixing vectors from different models produces meaningless cosine similarities.

**Threshold recalibration:** All cosine similarity thresholds in this specification (entity clustering distance threshold 0.20, cross-model matching threshold 0.85, predicate stabilization thresholds 0.85/0.60, ground-truth fuzzy match threshold 0.90) are calibrated to `all-mpnet-base-v2`'s distance geometry. Changing the embedding model invalidates these thresholds — different models have different absolute distance distributions. When `bsos normalize --reembed` is run after a model change, the CLI MUST emit a warning listing all affected threshold config keys and directing the operator to recalibrate them before running extraction or validation. The `config` table MUST persist an `embedding_model_at_last_calibration` key (written by `--reembed`) so subsequent commands can detect when the active embedding model has changed since thresholds were last reviewed.

---

# 9. Semantic Compression System (NON-DESTRUCTIVE)

The system MUST detect repeated patterns and produce abstractions. Compression is **additive**: original assertions are never deleted or replaced.

Algorithm:

1. Group assertions by `subject_id` (exact UUID match — no embeddings needed at this stage). Within each group, use embeddings to cluster semantically similar assertion texts. Only clusters above size threshold N (default: 3 semantically similar assertions) proceed to step 2.
2. For each qualifying cluster, ask LLM-A: "what single statement captures all of these without introducing new information?"
3. Validate using a separate adversarial prompt to LLM-B (a different model from LLM-A): "does this abstraction assert anything not already present in these source assertions?" If LLM-B is not available or is the same model as LLM-A, skip automated validation and set status to `proposed` — human acceptance required. Using the same model for both synthesis and validation provides no independence and MUST NOT be treated as equivalent to two-model validation.
4. If validation passes (or single-model path), create an `AbstractionNode` with status `proposed` and `child_ids` pointing to the source assertions
5. Store the abstraction as a new graph node with `aggregates` edges to children
6. Queries return both abstractions and their children

Example:

Input assertions (children, retained):

* roof prevents water ingress
* roof protects from rain
* roof shelters interior

Output abstraction node (new, additive):

* roof → environmental separation system

The children remain queryable. Over-abstraction is auditable by inspecting child assertions.

---

# 10. Conflict Resolution State Machine (REQUIRED)

The description below is the **logical model** — what must happen before an item can be promoted. The **default implementation** is the deferred batch path (Section 10.1): conflict detection does not block ingestion. An implementing agent MUST default to deferred mode and MUST NOT implement synchronous blocking conflict detection unless explicitly instructed.

On ingestion of any new extracted item:

1. Run embedding similarity search against existing items on the same subject to find candidates
2. For each candidate above similarity threshold, run a dedicated LLM classification call:
   * prompt: given assertion A and assertion B, classify their relationship as one of: `duplicate | complementary | contradictory | unrelated`
   * use the classification result, not embedding distance alone, to determine the action
   * embedding similarity is only a cheap pre-filter; it cannot distinguish contradiction from complementarity
3. If classified **duplicate** → merge: increment confidence, append provenance entry, do not create duplicate
4. If classified **contradictory** → mark both as `conflicted`, add to human review queue
5. If classified **complementary** or **unrelated** → treat as novel
6. If **novel** → status remains `proposed`; promote to `accepted` when confidence threshold is reached or human accepts

For `physical` knowledge, contradictions are strong signals of either a bad extraction prompt or a genuine edge case. Edge cases MUST be captured as explicit exceptions on the relevant assertion rather than being silently resolved.

## 10.1 Deferred Conflict Detection

LLM classification calls during bulk ingestion are expensive. Conflict detection MAY run as a deferred batch job rather than blocking each ingestion:

* items stay in `proposed` status until the conflict batch runs
* the batch job processes all `proposed` items where `conflict_evaluated_at IS NULL`; on completion it sets `conflict_evaluated_at = now()` on each evaluated item
* when two items are classified as `contradictory`, the batch writes a row to the `conflict_pairs` table (Section 12) recording both item IDs, item types, and the classification before updating their status to `conflicted`; this persists the pairing so `get_conflicts()` (Section 13.3) and `review-pending` (Section 17) can surface items as pairs rather than as individually-flagged items with no visible counterpart
* provide a CLI trigger: `bsos validate --conflicts`
* **the conflict batch MUST complete before any `proposed` item is eligible for auto-promotion** — auto-promotion logic (Section 7) MUST check `conflict_evaluated_at IS NOT NULL AND status = 'proposed'` before applying promotion rules; items where `conflict_evaluated_at IS NULL` remain `proposed` regardless of confidence score; items with `status = 'conflicted'` are never auto-promoted even if `conflict_evaluated_at` is set — they require explicit human acceptance via `review-pending`
* **conflict queue cap:** when the number of `conflicted` items across all item types exceeds 500, the conflict detection batch MUST pause further `conflicted` status transitions and emits a structlog warning on every subsequent `bsos extract`, `bsos normalize`, and `bsos validate --conflicts` invocation: `"Conflict queue has [N] items awaiting review. Conflict promotion suspended until queue drops below 500. Run bsos review-pending."` When the cap is active and a pair is classified as `contradictory`, the batch MUST still write the `conflict_pairs` row (to record the classification) but MUST NOT set `conflict_evaluated_at` on either item and MUST NOT update their status. This preserves re-evaluation eligibility: on the next `bsos validate --conflicts` run when the queue has dropped below 500, those items are found via `conflict_evaluated_at IS NULL` and the batch re-uses the already-written `conflict_pairs` row (skipping re-classification) to apply the `conflicted` status transition. The re-use check: before running LLM classification for a candidate pair, check for an existing `conflict_pairs` row for those two item IDs — if found, skip classification and proceed directly to status update. Auto-promotion of non-conflicted items is not suspended by this cap — only the transition to `conflicted` status is throttled.

## 10.2 ProcessRelation Divergence Detection

`ProcessRelation` rows are not subject to the embedding-similarity conflict detection used for assertions (their semantic content is two entity UUIDs and a boolean, not a free-text assertion). The conflict detection batch handles them separately:

For each `(predecessor_id, successor_id)` pair that has rows from ≥ 2 distinct `source_model` values, check whether `hard_constraint` is consistent across models. If any two rows for the same pair disagree on `hard_constraint` (one `True`, one `False`), mark all rows for that pair as `status = "conflicted"` and add them to the review queue. LLM classification is not needed — the divergence is a schema-level boolean comparison. The review queue entry MUST display the disagreeing models and their respective `hard_constraint` values alongside the `rationale` from each row so the reviewer can judge which is correct. A `ReviewDecision` with `decision = "defer"` is written for the pair; no auto-resolution is applied.

`ProcessRelation` rows where all models agree on `hard_constraint` are treated as `complementary` (the same dependency is affirmed by multiple sources) and follow the same auto-promotion rules as assertions (Section 7), adjusted for `ProcessRelation`'s `knowledge_origin` field.

## 10.3 AbstractionNode Cascade

When a child assertion's status changes, ALL `AbstractionNode` records whose `child_ids` contain that assertion's UUID MUST be re-evaluated. A child assertion may appear in more than one parent (see Section 4.8); the cascade implementation MUST query `abstraction_nodes WHERE json_each(child_ids).value = <child_id>` to find every affected parent rather than assuming a one-to-one relationship.

For each affected parent `AbstractionNode`:

* child becomes `conflicted` → if ≥ 2 of that parent's children are now `conflicted`, set the parent's status to `conflicted` and add it to the review queue; if only 1 child is `conflicted` out of N ≥ 3 active children, set the parent's status to `proposed` (flagging it for human review without claiming the abstraction is fully invalidated). The parent is re-evaluated after the child conflict is resolved via `review-pending`.
* child becomes `deprecated` → if fewer than 2 of that parent's children remain active, the parent is `deprecated`; otherwise re-run abstraction validation without the deprecated child
* child is newly `accepted` → if all of that parent's children are now `accepted` and the parent `AbstractionNode` has `status = "proposed"`, add the parent to the review queue so a human can confirm it; if not all children are `accepted` yet, no cascade is required

**Conflict pair staleness:** Conflict detection operates on pairs. When one member of a conflicted pair is subsequently marked `deprecated`, the pair becomes stale. The conflict detection batch (`bsos validate --conflicts`) MUST skip pairs where either member has `status = 'deprecated'` and log these as resolved-by-deprecation rather than surfacing them in the review queue. If the reviewer depreciates item B which appears in both an A–B pair and a B–C pair, both pairs are marked stale independently — the A–C relationship is not evaluated automatically (transitive closure is not computed).

---

# 11. Graph Construction Layer

Use NetworkX to build semantic graph as a **derived query layer**. The graph is rebuilt from SQLite on startup. Never write to the graph directly.

**Performance note:** Full graph rebuild on startup is suitable for development and knowledge bases up to ~50,000 nodes. Beyond that threshold, switch to lazy loading: build only the subgraph reachable from the queried entity on demand, caching subgraphs between requests. The rebuild threshold should be a configurable setting.

**CLI invocation cost:** For CLI usage (where the process exits after each command), the graph is rebuilt on every invocation. Do not perform a full graph build in CLI context — use the lazy-loading path for all CLI queries regardless of graph size, building only the subgraph needed to answer the query. The `build-graph` CLI command is the exception: it performs a full build and serialises the result to disk for use by the API layer. The API layer loads the serialised graph at startup and holds it in memory across requests. Use `joblib.dump` / `joblib.load` (with `compress=3`) rather than raw `pickle` — the serialised file MUST only be loaded from the same machine that wrote it, and MUST NOT be transmitted or loaded from untrusted sources (pickle is arbitrary code execution on load). The serialised file MUST include a `schema_version` string matching the current Alembic revision head at write time. On load, the API layer MUST compare this value against the live revision; if they differ, it MUST refuse to use the cached graph and instead re-run `build-graph` automatically before serving requests. The `build-graph` command MUST also write a SHA-256 content hash of the serialised file to a companion file (same path with `.sha256` suffix); on load the API layer MUST verify this hash before deserializing — a mismatch indicates corruption or tampering and MUST cause a clean rebuild rather than loading the file. Any code path that writes the serialised graph file — including `bsos build-graph`, the `POST /rebuild-graph` endpoint, and any `--fix` automatic rebuild — MUST always write the companion `.sha256` file atomically in the same operation. "Atomically in the same operation" means: write both the graph file and the `.sha256` file to temporary paths (e.g. `.graph.tmp` and `.graph.sha256.tmp`) then rename both into their final paths in sequence. A rename is atomic on POSIX filesystems; writing to the final path directly would leave a window where the graph file exists but the hash file does not (or is stale from a previous run), causing a spurious integrity failure on load.

**Known limitation — in-memory graph staleness:** The API layer holds the graph in memory for the lifetime of the process. If CLI commands write new data to SQLite while the API is running, the in-memory graph will be stale — it does not auto-reload on database changes. This is an accepted trade-off for a local-first system. The API MUST expose a `POST /rebuild-graph` endpoint that re-runs `build-graph` and reloads the in-memory graph on demand. **`POST /rebuild-graph` is a mandatory step after every `bsos extract` run when the API layer is deployed** — it is not an edge-case recovery action but a required part of the standard extraction workflow when the API is active. The API documentation MUST remind the operator to call it. The `bsos extract` completion output MUST print this reminder only when the API layer is active (i.e. when `api_enabled = "1"` is set in the `config` table). The schema-version and SHA-256 checks only apply at startup; mid-session staleness is the operator's responsibility.

Node types:

* Entity
* Assertion
* AbstractionNode
* Pattern
* Force
* AntiPattern

Note: `ProcessRelation` has no node type — it is an edge (`precedes`) between Entity nodes, not a node itself. `SpatialRelation` likewise has no node type — each row becomes a directed edge between the two referenced Entity nodes (see spatial edge construction below).

Edge types fall into two categories:

**Assertion edges** — one edge type per core predicate (Section 5.1), applied between Entity nodes via their Assertion rows:

* requires
* depends_on
* protects_from
* unsuitable_for
* improves
* conflicts_with
* contains
* connects_to
* supports

**Structural edges** — not assertion predicates; represent system-level relationships:

* aggregates  (AbstractionNode → child Assertions)
* precedes    (Entity → Entity, represents construction sequencing via ProcessRelation rows)
* acts_on     (Force → Entity, from Force.affects)

**SpatialRelation edge construction:** For each `SpatialRelation` row, add a directed edge from the Entity node `subject_id` to the Entity node `object_id` with attributes `edge_type = relation` (e.g. `"accessible_from"`, `"adjacent_to"`) and `edge_category = "spatial"`. The `relation` value becomes the edge label. Only rows with `status IN ("accepted", "proposed")` are included by default; the graph builder accepts a `min_status` parameter to restrict to `accepted` only. These spatial edges are what topology queries (`get_spatial_relations`, `bsos validate --topology`) traverse — the `accessible_from` edge type is specifically required for entrance reachability checks (Section 16.3).

Graph MUST support traversal queries.

---

# 12. Storage Layer (SQLite — Source of Truth)

SQLite is the authoritative store. The NetworkX graph is always derived from it.

Tables:

* entities
* entity_aliases      — (entity_id, alias); junction table replacing Entity.aliases field
* assertions
* constraints
* abstraction_nodes
* patterns
* forces
* antipatterns
* process_relations
* spatial_relations
* provenance_log      — audit log of status transitions: (item_id, item_type, old_status, new_status, changed_at, changed_by); provenance fields on each row record initial extraction state; this table records subsequent lifecycle changes
* embeddings          — (item_type, item_id, model TEXT, dim INT, content_hash TEXT, vector BLOB); caches sentence-transformer vectors; invalidated on text change or embedding model change (see Section 8.5 for serialisation and invalidation details)
* pending_predicates
* pending_spatial_relation_types  — mirrors pending_predicates for spatial relation vocabulary (Section 5.4)
* pending_force_refs  — two failure categories distinguished by `failure_type`; columns: (id TEXT, source_id TEXT, value TEXT, failure_type TEXT CHECK(failure_type IN ('unresolved_ref','validation_failure')), logged_at DATETIME); `failure_type = 'unresolved_ref'`: force description from `Pattern.force_descriptions` (Pass 8) that could not be matched to a `Force` record — `source_id` is the Pattern UUID; `failure_type = 'validation_failure'`: a `Force` record that failed ForceDirection consistency validation (Pass 9) — `source_id` is the Force UUID; resolution for `unresolved_ref`: match `value` against `Force.name` (exact, then cosine similarity ≥ 0.85); `validation_failure` entries require human correction
* pending_pattern_refs  — unresolved pattern name references from `Pattern.related_pattern_names` (Pass 8); columns: (id TEXT, source_id TEXT, value TEXT, logged_at DATETIME); `source_id` is the UUID of the Pattern that referenced the unresolved name; resolution: match `value` against `Pattern.name` (exact, then cosine similarity ≥ 0.85)
* pending_entity_refs  — unresolved entity names from `Force.affects` (Pass 9); columns: (id TEXT, source_id TEXT, value TEXT, logged_at DATETIME); `source_id` is the UUID of the Force record that referenced the unresolved entity name; resolution: match `value` against `Entity.name` then `entity_aliases` (exact match only — entity names are not similarity-searched to avoid false merges)
* predicate_mappings  — (original TEXT, canonical TEXT, method TEXT, mapped_at DATETIME); records all auto-mappings and LLM-disambiguated mappings from Pass 10 / Section 8.3
* conflict_pairs       — (id TEXT, item_a_id TEXT, item_a_type TEXT, item_b_id TEXT, item_b_type TEXT, detected_at DATETIME, classification TEXT); one row per conflicting pair written by the conflict detection batch; used by `get_conflicts()` (Section 13.3) and `review-pending` (Section 17) to surface item pairs; rows are not deleted when a `ReviewDecision` resolves the pair — the decision record is the authoritative resolution state
* extraction_runs      — (id TEXT PRIMARY KEY, started_at DATETIME, completed_at DATETIME, models TEXT, passes TEXT, seed TEXT, entity_count_before INT, entity_count_after INT, assertion_count_before INT, assertion_count_after INT); one row per `bsos extract` invocation; `models` and `passes` are JSON arrays; `*_before` counts are snapshotted at extraction start, `*_after` at completion; enables comparison between runs and provides an audit trail for corpus evolution. `bsos status` displays the most recent run's metadata.
* review_decisions
* llm_response_cache  — (model TEXT, prompt_hash TEXT, entity_name TEXT, response_json TEXT, cached_at DATETIME); keyed on (model, prompt_hash); see Section 6. `entity_name` is populated from the `entity_name` kwarg passed to `LLMProvider.extract()` (NULL for `classify` calls and for any call where `entity_name` is not provided). Storing `entity_name` explicitly enables exact-match cache invalidation by `bsos cache clear --entity` without substring-searching `response_json`.
* config              — (key TEXT PRIMARY KEY, value TEXT); stores runtime flags and user settings; see Section 17 (`bsos config`) for documented keys
* pass_progress       — (pass_number TEXT, entity_id TEXT, model TEXT, completed_at DATETIME, status TEXT CHECK(status IN ('completed','skipped'))); one row per (pass, entity, model) triple; used for resume-within-pass logic; `skipped` indicates the entity was attempted but timed out or produced a parse failure after retries. Passes 1–9 and 11 use string keys `"1"`–`"9"` and `"11"`. Pass 10 uses sub-pass keys `"10a"` (ref resolution), `"10b"` (predicate stabilization), `"10c"` (abstraction synthesis) — this allows a crashed run that completed ref resolution to resume at predicate stabilization rather than re-running all three steps. The `pass_number` column type is TEXT (not INT) to accommodate these sub-pass keys.

**List field serialization:** All `list[str]` fields (`Assertion.conditions`, `Assertion.exceptions`, `Assertion.applicability`, `AbstractionNode.child_ids`, `Pattern.force_descriptions`, `Pattern.force_ids`, `Pattern.related_pattern_names`, `Pattern.related_pattern_ids`, `Pattern.context`, `Pattern.consequences`, `Pattern.emergent_properties`, `Force.affects`, `AntiPattern.conditions`, `AntiPattern.consequences`, `AntiPattern.mitigations`, `Constraint.conditions`, `Constraint.exceptions`, `extraction_runs.models`, `extraction_runs.passes`) are stored as JSON arrays in TEXT columns in SQLite. The persistence layer handles serialization/deserialization transparently. Referential integrity for UUID lists (e.g. `Force.affects`, `Pattern.force_ids`, `Pattern.related_pattern_ids`) is enforced at the application layer, not the database layer. When entity deduplication (Section 8.2) replaces deprecated UUIDs, the application layer MUST deserialize each JSON array, substitute UUIDs, and re-serialize — a plain SQL UPDATE is not sufficient for JSON-encoded lists.

**Model layering:** The Pydantic `BaseModel` classes defined in Section 4 are *extraction models* — used with `instructor` for structured LLM output and for validation logic. They do not map directly to SQLite tables. The *persistence models* are a parallel set of `SQLModel(table=True)` classes that mirror the same fields and are used exclusively by the storage layer. Conversion between extraction and persistence models happens at the repository boundary. This separation keeps the extraction pipeline independent of SQLAlchemy.

**Repository boilerplate:** The parallel class hierarchy produces conversion code at every boundary. To contain this, implement a `BaseRepository` class in `src/bsos/persistence/repos/` that provides generic `to_persistence(extraction_model)` and `from_persistence(row)` methods using `model.model_dump()` and `PersistenceModel(**data)` — both Pydantic and SQLModel support this pattern. Each concrete repository subclass only needs to specify the extraction model class and persistence model class; the field-by-field conversion is handled by the base. List fields requiring JSON serialization (Section 12, list field serialization) are the exception — these MUST be handled explicitly in each subclass `to_persistence` override.

**`AbstractionNode` exception — `knowledge_origin` is not a stored column:** `BaseRepository` MUST declare a class-level flag `_knowledge_origin_stored: bool = True`. `AbstractionNodeRepository` MUST override this to `False`. The base `filter_by_knowledge_origin()` method MUST check this flag and raise `NotImplementedError` when it is `False`, forcing callers to use the view-aware override. `AbstractionNodeRepository` MUST override `list()` and any filter method that could return results sorted or filtered by `knowledge_origin` to join against the `abstraction_node_effective_origins` SQLite view before returning results. No call site should query the `abstraction_nodes` table directly for `knowledge_origin` — all such queries MUST go through the repository layer.

**WAL mode (REQUIRED):** The database MUST be opened with `PRAGMA journal_mode=WAL` at connection time. WAL mode is required for the parallel extraction pipeline (Passes 3–9) where multiple threads write concurrently to the same database. Without WAL mode, concurrent writers will collide and the pipeline will deadlock or corrupt data. The thread pool cap of 4 workers (Section 6) reduces contention further but does not remove the WAL requirement.

**Throughput note:** WAL still serializes writers — concurrent workers queue at the write lock. For a typical corpus of 50–200 concepts with 3 prompt framings per entity, expect 500–5000 assertion insertions per model; extraction wall-clock time is dominated by LLM API latency, not SQLite write throughput. If write contention becomes a bottleneck (observable via `structlog` timing output), reduce the thread pool to 2 workers or batch writes: accumulate a worker's results in memory and flush in a single transaction at the end of each entity's passes.

**Schema migrations:** Use Alembic (via its SQLModel integration) for all schema changes. Migrations are versioned and applied with `alembic upgrade head`. Do not modify the schema by hand or drop and recreate tables — all changes must be reversible. Note: SQLModel's Alembic autogenerate support is incomplete for some column types. Always review generated migration scripts manually before applying — do not rely on autogenerate as a final authority.

**Protecting the `abstraction_node_effective_origins` view from migrations:** This view is created by `bsos init` outside Alembic's versioned migration system. Alembic autogenerate will not drop or modify it during normal operation — SQLite views are not reflected by SQLModel's Alembic integration. However, any manually-authored migration script that drops or recreates the `assertions` or `abstraction_nodes` tables MUST include explicit `DROP VIEW IF EXISTS abstraction_node_effective_origins` and `CREATE VIEW abstraction_node_effective_origins AS ...` statements in the `upgrade()` and `downgrade()` functions respectively. Add the following comment to `alembic/env.py` as a persistent reminder: `# NOTE: abstraction_node_effective_origins view is not Alembic-managed. Any migration that drops assertions or abstraction_nodes must recreate it manually.`

---

# 13. Query System (MCP Server — Primary Deliverable)

The query system MUST be exposed as an MCP (Model Context Protocol) server using the `mcp` Python SDK. This is not optional infrastructure — it is the primary consumption path that enables agents using ifcmcp to access building domain knowledge at query time. The FastAPI layer (Section 18) is a secondary interface; the MCP server ships with the initial build.

## 13.1 MCP Server Specification

Each query function is registered as an MCP tool. All tools accept `entity: str` (resolved against `Entity.name` and `entity_aliases` — case-insensitive, alias-aware) plus shared filter arguments:

**Concurrency safety:** The MCP server MAY receive concurrent tool calls from clients that pipeline requests. Each tool call MUST build its own subgraph from SQLite reads into a local NetworkX object that is not shared across requests. NetworkX graphs are not thread-safe; sharing a single in-memory graph across concurrent requests without locking will produce data races. The lazy-loading path (Section 11) naturally satisfies this requirement when implemented correctly — each call constructs a new subgraph object. The `bsos build-graph` / FastAPI path (Section 18) uses a single in-memory graph; that layer MUST acquire a read lock before traversal if it is ever accessed from concurrent request handlers.

**Entity not found:** When a tool call is made for an entity that cannot be resolved against `Entity.name` or `entity_aliases`, the tool MUST return a structured error in the MCP tool result `content` field — `{"error": "entity_not_found", "message": "Entity '<name>' not found. Run 'bsos status' to confirm extraction has completed."}` — rather than raising an exception. The MCP server MUST NOT crash or close the connection on unrecognized entity names; the client may be querying an entity that has not yet been extracted.

* `min_confidence: float = 0.0` — exclude items below this score
* `max_results: int = 20` — hard cap on returned items; default is configurable via `bsos config set query_max_results <N>`
* `include_proposed: bool = True` — whether to include `proposed` items alongside `accepted` ones

**Result ranking:** all results are sorted by `confidence DESC`, then knowledge_origin priority (`physical` > `engineering` > `architectural` > `cultural`), then `created_at ASC` as tiebreak. Only items with `status IN ("accepted", "proposed")` are returned. Deprecated and conflicted items are excluded unless explicitly requested.

## 13.2 Return Types

All query results use these structured types. Fields correspond directly to database columns; names are resolved from entity UUIDs at query time.

```python
class AssertionResult(BaseModel):
    subject: str; predicate: str; object: str
    confidence: float; knowledge_origin: str; status: str
    conditions: list[str]; exceptions: list[str]; applicability: list[str]
    cross_prompt_consistency: float | None

class ConstraintResult(BaseModel):
    rule: str; constraint_type: str  # "must" | "must_not"
    conditions: list[str]; exceptions: list[str]
    confidence: float; knowledge_origin: str; status: str

class AntiPatternResult(BaseModel):
    name: str
    conditions: list[str]; consequences: list[str]; mitigations: list[str]
    confidence: float; knowledge_origin: str; status: str

class PatternResult(BaseModel):
    name: str; problem: str; solution: str
    context: list[str]; forces: list[str]; consequences: list[str]; emergent_properties: list[str]
    confidence: float; knowledge_origin: str; status: str

class ForceResult(BaseModel):
    name: str; direction: str  # "increase" | "decrease"
    confidence: float; knowledge_origin: str; rationale: str | None; status: str

class SpatialRelationResult(BaseModel):
    subject: str; relation: str; object: str
    confidence: float; knowledge_origin: str; status: str

class ProcessSequenceResult(BaseModel):
    entity: str
    sequence: list[str]    # entity names in topological order, predecessors first;
                           # includes all entities reachable in both directions via
                           # precedes edges — not just immediate predecessors/successors.
    truncated: bool         # True when max_depth was reached and the sequence is partial
    has_cycle: bool         # True when a cycle was detected; sequence is partial
    cycle_description: str | None

class AbstractionResult(BaseModel):
    statement: str; abstraction_rationale: str; child_count: int
    effective_knowledge_origin: str  # majority origin from children — never "derived"
    confidence: float; status: str

class ConflictResult(BaseModel):
    item_type: str
    item_a_id: str; item_a_text: str
    item_b_id: str; item_b_text: str
```

## 13.3 Query Functions

* `get_requirements(entity)` → `list[AssertionResult]` — assertions where subject matches entity and predicate is `requires` or `depends_on`
* `get_constraints(entity)` → `list[ConstraintResult]` — Constraint records where `subject_id` matches entity
* `get_failure_modes(entity)` → `list[AntiPatternResult]` — AntiPattern records associated with entity
* `get_patterns(entity)` → `list[PatternResult]` — Pattern records associated with entity; `PatternResult.forces` is populated from `Pattern.force_ids` resolved to Force names; if `force_ids` is empty and `force_descriptions` is non-empty (resolution not yet run), the raw descriptions are used with a warning in the result
* `get_forces(entity)` → `list[ForceResult]` — traverses `acts_on` edges in reverse to find Force nodes that point to the entity
* `get_dependencies(entity)` → `list[AssertionResult]` — assertions where predicate is `depends_on` and either subject or object matches entity
* `get_spatial_relations(entity)` → `list[SpatialRelationResult]` — SpatialRelation records where subject or object matches entity
* `get_process_sequence(entity, max_depth: int | None = 50)` → `ProcessSequenceResult` — extracts the subgraph reachable from entity via `precedes` edges in both directions up to `max_depth` hops (default: 50; pass `None` for unlimited); calls `nx.is_directed_acyclic_graph` before `nx.topological_sort`; if a cycle is detected the function returns `has_cycle=True` with a partial sequence and descriptive `cycle_description` rather than raising — agents querying at runtime must not crash on a corrupted process graph; BFS MUST NOT be used as a substitute for topological sort; cycle information is also surfaced by `bsos validate --conflicts`; if the traversal reaches `max_depth` before exhausting the graph, `truncated=True` is set on the result so callers know the sequence is partial
* `get_abstractions(entity)` → `list[AbstractionResult]` — AbstractionNode records whose child_ids include assertions about entity; `effective_knowledge_origin` is resolved from majority child origin (never `"derived"`)
* `get_conflicts(entity)` → `list[ConflictResult]` — queries the `conflict_pairs` table for rows where `item_a_id` or `item_b_id` references an item whose subject or object entity matches the given entity; returns the populated `ConflictResult` pairs; does NOT scan for items with `status = "conflicted"` without a matching `conflict_pairs` row — orphaned conflicted items (if any) are a data integrity issue surfaced by `bsos doctor`, not by this query

---

# 14. Lifecycle Awareness

Each extracted item uses the status field from `ProvenanceMixin`:

```python
status: Literal["proposed", "accepted", "deprecated", "conflicted"]
```

Promotion thresholds (confidence levels, model count requirements) are defined in Section 7. The conflict detection prerequisite for promotion is defined in Section 10.1.

---

# 15. Knowledge Origin Tagging

Each extracted item MUST be tagged via `ProvenanceMixin.knowledge_origin`:

* physical
* engineering
* cultural
* architectural

Regulatory is NOT a valid tag in this system. Regulatory knowledge is handled by a future overlay.

**Enforcement limitation:** The epistemic scope boundary (Section 1.1) is enforced at prompt level only — the extraction prompt instructs the LLM to rate its certainty that an item is a universal physical requirement rather than a regulatory rule. No downstream filter exists. LLMs will occasionally extract regulatory assertions with high confidence and assign them `knowledge_origin = "engineering"`. Operators should treat a high `knowledge_origin = "engineering"` score as a necessary condition for physical grounding, not a sufficient one, and watch for regulatory language ("must comply with", "per code", "as required by") in assertion text during curation.

Acceptance thresholds by origin:

* `physical`, `engineering`: auto-accept on multi-model consensus
* `cultural`, `architectural`: higher threshold or explicit human acceptance

---

# 16. Evaluation System (REQUIRED)

## 16.1 Hand-curated ground truth

Maintain a set of verifiably true universal building facts, curated by hand and independent of any LLM. This breaks evaluation circularity. Use the `bsos curate` CLI command (Section 17) to add, review, and export ground-truth records. Ground-truth assertions are stored with `source_model: "human"` and are excluded from auto-promotion rules — they are authoritative by definition.

**Ground-truth density:** Start with 50–100 assertions; this is sufficient for a corpus up to ~5,000 extracted assertions. As the corpus grows, the coverage check becomes less representative. A target density of at least 1 ground-truth assertion per 100 extracted accepted assertions is recommended, capped at 500 to remain manageable for human curation. `bsos status` MUST display ground-truth density (ground_truth_count / total_accepted_assertions) so operators can monitor whether the ground-truth set is keeping pace with corpus growth. A density below 1% on a corpus larger than 5,000 accepted assertions MUST be displayed as a `[WARN]` line in `bsos status` output.

**Scope of the coverage test:** The 50–100 assertion set validates the system's ability to extract well-known canonical building facts at high confidence. It does not measure coverage of the full extracted corpus. A high consensus coverage score (Section 16.6) confirms the system reliably extracts obvious facts; it does not validate assertions outside the ground-truth set. This is a targeted precision check, not a recall measure across all possible building knowledge. Treat high scores as a necessary condition, not a sufficient one.

## 16.2 Constraint reasoning test

Can the system detect invalid configurations?

**Implementation:** For each `Constraint` record with `status = "accepted"`, generate a synthetic assertion that directly violates the constraint (e.g. for "roof must have a drainage path", generate "roof has no drainage provision"). Submit both to `bsos validate --constraints <entity>` and verify the violation is detected. Track: detection rate (violations correctly flagged / total synthetic violations) and false positive rate (valid assertions incorrectly flagged as violations). A detection rate ≥ 0.80 is the passing threshold. Violations that fall below the LLM confidence threshold (Section 11) are counted as misses.

---

## 16.3 Topology reasoning test

Can the system detect accessibility violations?

**Implementation:** Using `bsos validate --topology`, check that the system correctly identifies space entities with no `accessible_from` path to any entrance node. Requires at least one entity with `is_entrance = True` (set via `bsos curate set-entrance`). Test against a fixture spatial graph with a known set of accessible and inaccessible spaces. Verify: (1) all truly inaccessible spaces are reported as violations, and (2) all accessible spaces are not reported. False negatives (missed inaccessible spaces) are more harmful than false positives — weight recall over precision. If no entrance node is configured, the test is skipped and a warning is printed.

---

## 16.4 Pattern critique test

Can the system surface relevant patterns for a suboptimal layout?

**Implementation:** For a set of fixture entity configurations known to violate common patterns (e.g. a room with no windows violates "light on two sides"), run `bsos query <entity> --type pattern` and verify that relevant patterns are returned. Because relevance is subjective, this test is intentionally qualitative — it does not have an automated pass/fail oracle. It functions as a regression check: after re-running extraction, the reviewer manually confirms that the returned patterns are still relevant to the fixture entity context. The fixture set MUST be documented in `tests/fixtures/pattern_critique_cases.json` alongside human-written expected patterns for reference comparison.

**Fixture file schema** (`tests/fixtures/pattern_critique_cases.json`):

```json
[
  {
    "entity": "room",
    "context_description": "A room with no windows on external walls",
    "expected_patterns": [
      {
        "name": "light on two sides",
        "rationale": "Single-aspect room violates the two-sided lighting pattern"
      }
    ],
    "notes": "Free-text reviewer notes for this case (optional)"
  }
]
```

Each object in the top-level array is one test case. `entity` is the concept queried. `expected_patterns` is a list of patterns the reviewer expects to find in the query results — each with a `name` (matched case-insensitively against `Pattern.name`) and a `rationale` explaining why it applies. The test runner prints, for each case, which expected patterns were found and which were missing, but does not fail — the reviewer reads the output. The file MUST contain at least three cases covering different entity types (space, component, system) before Phase 4 is considered complete. **This file and its `expected_patterns` entries are a human-authored deliverable** — they encode domain judgements about which patterns are relevant to specific entity contexts and cannot be generated by the implementing agent. A human domain expert must populate `tests/fixtures/pattern_critique_cases.json` during Phase 4 before the pattern critique test can be run.

---

## 16.5 Exception reasoning test

Can the system surface valid exceptions to universal rules?

**Implementation:** For a set of accepted assertions known to have documented exceptions (populated in `Assertion.exceptions` by Pass 11 adversarial validation or by `bsos curate`), verify that querying the assertion returns non-empty exception text alongside the base assertion. Automated scoring is not applicable — this is a coverage check confirming that adversarial validation is populating exceptions rather than leaving them empty. The test passes if ≥ 80% of fixture assertions with known exceptions have at least one exception string recorded. Empty exception lists on assertions that are known to have exceptions indicate that Pass 11 is not running, the confidence threshold is filtering out valid findings, or the adversarial model lacks domain knowledge for that assertion.

## 16.6 Consensus coverage test

What proportion of ground-truth assertions are recovered with high confidence?

**Implementation:** Run `bsos curate verify` (Section 17, curate spec). For each ground-truth assertion (records with `source_model = "human"`), attempt to find a matching record in the extracted corpus using a two-step strategy:

1. **Exact match** — look up by `(subject_id, predicate, object_id)` tuple. Subject and object UUIDs are resolved from the ground-truth assertion's entity names using the same lookup order as `curate add` (exact name → alias → error).
2. **Fuzzy match fallback** — if no exact match, compute cosine similarity between the ground-truth assertion embedding and all extracted assertion embeddings; treat the best match as a hit if similarity ≥ 0.90 (configurable via `ground_truth_match_threshold` in the `config` table — see Section 17, config spec).

A ground-truth assertion is *covered* if a match is found with `confidence ≥ 0.80` and `status IN ("accepted", "proposed")`.

**Reporting:** Output four numbers:

* exact-match coverage: covered (exact) / total ground-truth assertions
* fuzzy-match coverage: covered (exact + fuzzy) / total ground-truth assertions
* mean confidence of matched assertions
* count of ground-truth assertions with no match at any similarity level

**Passing threshold:** fuzzy-match coverage ≥ 0.80. The command exits with code 0 on PASS and code 1 on FAIL, enabling use in CI via `bsos validate --ground-truth`.

**Scope note:** This measures extraction of well-known canonical facts, not recall across all possible building knowledge. A high score is a necessary condition for system correctness, not a sufficient one — see Section 16.1.

---

## 16.7 Software Testing Strategy

The system MUST have test coverage at three levels:

**Unit tests** — test each component in isolation with no LLM or SQLite calls:

* Pydantic model validation (field constraints, Literal enforcement, mixin inheritance)
* Predicate registry lookup and validation
* Threshold formula for pending predicate queue
* Entity merge algorithm (Section 8.2) using a fixture in-memory database
* Embedding content hash invalidation logic

**Integration tests** — test against a real SQLite database using a temporary file, with LLM calls replaced by a `FakeLLMProvider` fixture that returns deterministic structured responses. The `FakeLLMProvider` MUST implement the `LLMProvider` protocol (Section 2) and return a fixed, hand-authored response set dispatched by `(type[BaseModel], entity_name)` key — for example, `(AssertionExtractionResponse, "roof")` returns a canned list of 5 assertions covering `roof protects_from precipitation`, `roof requires structural_support`, etc. The schema class (`type[BaseModel]`) uniquely identifies the extraction pass (each pass uses a distinct Pydantic response schema), replacing the need for an explicit pass number. `entity_name` is passed explicitly by the extraction pipeline as the `entity_name` keyword argument to `LLMProvider.extract()` (Section 2); `FakeLLMProvider` dispatches directly on this value without parsing the prompt string. `FakeLLMProvider` raises `ValueError` if `entity_name` is `None` — this catches pipeline code that omits the parameter. When the dispatch key `(type[BaseModel], entity_name)` is present in the table, the canned response is returned. When the key is absent (entity not covered by the fixture set), `FakeLLMProvider` returns a minimal valid empty response (empty lists, `confidence = 0.5`) rather than raising — this prevents test failures when the pipeline processes entities not listed in the fixture table. Production `LLMProvider` implementations ignore `entity_name`. The fixture dispatch table is a plain `dict[tuple[type[BaseModel], str], BaseModel]` defined in `tests/fixtures/fake_responses.py`. The fixture response set MUST include at least one duplicate pair, one contradictory pair, and one process relation so that conflict detection and sequencing tests have meaningful data to operate on.

* Full single-model extraction pipeline (Passes 1 → 9) on a small seed corpus; corpus size is not prescribed — choose the smallest set that exercises all passes and produces at least one record of each item type
* Conflict detection batch (`bsos validate --conflicts`) on a fixture set containing known duplicates and contradictions
* Entity deduplication (Pass 2) produces stable UUIDs and updates all FK references
* Graph rebuild from SQLite produces the expected node and edge counts
* `get_process_sequence` returns nodes in topological order on a fixture DAG
* MCP server: register the server using stdio transport, send a `get_requirements` tool call for entity `"roof"` against a fixture database, and assert that at least one `AssertionResult` is returned with the correct fields populated; verify that a second concurrent tool call to the same server instance does not corrupt the first result (concurrency safety check)

**End-to-end tests** — use a real LLM provider (gated behind an environment flag `BSOS_E2E=1`) against the actual API:

* Extract and normalize a single concept ("roof") end-to-end; assert that at least one assertion with confidence ≥ 0.80 is produced
* Ground-truth coverage test: `bsos curate verify` recovers ≥ 80% of a fixture ground-truth set with exact match

All tests MUST pass without a network connection when `BSOS_E2E` is not set.

---

# 17. CLI TOOLING (Typer)

Commands:

* init
* extract
* normalize
* compress
* build-graph
* serve
* query
* validate
* review-pending
* curate
* export
* config
* cache
* status
* history
* doctor
* purge

**Global flags** (accepted by all commands):

* `--verbose` / `-v` — increase log verbosity; sets structlog output level to DEBUG, written to stderr
* `--quiet` / `-q` — suppress all non-error log output; sets structlog output level to ERROR

Default log level is INFO.

### init spec

Initialises the BSOS database and configuration. MUST be run before any other command.

```
bsos init [--db <path>]
```

* Creates the SQLite database at `<path>` (default: `./bsos.db`; the path is persisted to `config` table key `db_path` so all subsequent commands resolve the database without requiring `--db`)
* Runs `alembic upgrade head` to apply the current schema
* Writes the `graph_rebuild_threshold` default value to the `config` table
* Exits with a clear error if the database already exists at the target path, unless `--force` is passed (which re-applies migrations without dropping data)

All other commands resolve the database path using this lookup order (first match wins):

1. `--db <path>` flag on the current invocation
2. `BSOS_DB` environment variable
3. `.bsos_config` file in the current directory (a single-line file containing only the database path, written by `bsos init`)

If none of the above is set, the command exits with: `"No database configured. Pass --db, set BSOS_DB, or run 'bsos init' first."` All settings other than the database path live in the `config` SQLite table and are read after the database is opened. `bsos init` automatically appends `.bsos_config` to `.gitignore` in the current directory (creating the file if it does not exist) unless `--no-gitignore` is passed. Use `BSOS_DB` in scripts and CI environments to avoid relying on a working-directory file.

### extract spec

Runs the extraction pipeline (Passes 1–11 or a subset) against one or more LLM providers.

Flags:

* `--seed <text|file>` — free-text domain description or path to a concept list file (one concept per line); if omitted, uses the default bootstrap seed (Section 6, Pass 1)
* `--passes <N,M,...>` — comma-separated pass numbers to run (default: all); allows resuming a partial run. Pass `10` is automatically expanded to include all three sub-passes (`10a`, `10b`, `10c`) — there is no way to run an individual sub-pass via `--passes`; sub-pass resume is handled automatically by the `pass_progress` table (Section 12). The `bsos extract` completion output and `bsos status` display sub-pass completion using the `10a`/`10b`/`10c` keys.
* `--models <id,...>` — comma-separated LLM model identifiers; first is primary, remainder participate in consensus (Section 7)
* `--dry-run` — run the pipeline without writing to the database; prints a count of items that would be extracted per pass and item type (e.g. `Pass 3: 12 assertions would be written`). Does not print individual items. Cache interaction: reads from `llm_response_cache` if responses are already cached (to avoid redundant API calls during testing), but does not write new responses to the cache and does not write any extracted data to any other table. **First-run note:** on the first invocation when the LLM response cache is empty, `--dry-run` will still make live API calls since nothing is cached. Use `--dry-run` on subsequent runs to preview re-extraction without additional API cost.

**Model resolution:** If `--models` is omitted, the command reads `default_llm_model` from the `config` table. If that key is also absent, the command MUST exit with: `"No LLM model specified. Pass --models or run: bsos config set default_llm_model <model-id>"`.

**Embedding model consistency check:** Before Pass 2 begins (entity clustering is the first operation requiring embeddings), `bsos extract` checks whether any existing rows in the `embeddings` table carry a `model` value that differs from the currently configured `embedding_model`. If a mismatch is found, the command MUST exit with: `"Embedding model mismatch: existing embeddings were computed with '<old_model>' but current config specifies '<new_model>'. Run 'bsos normalize --reembed' before extracting with the new model."` This prevents mixing vectors from different embedding models in Pass 2 clustering, which would produce meaningless cosine similarities. If the `embeddings` table is empty (first run), the check is skipped.

**Pass dependency enforcement:** When `--passes` is supplied, the CLI MUST verify that prerequisite passes have completed before running the requested subset. Pass completion is checked as follows:

| Pass(es) requested | Prerequisite check |
|---|---|
| 3–9 | At least one row exists in the `entities` table (Pass 2 has run) |
| 10 | At least one row exists in the `assertions` table (Passes 3–9 have run) |
| 11 | At least one row exists in the `predicate_mappings` table (Pass 10 has run) |

If a prerequisite is not met, the command exits with a descriptive error naming the missing pass and the command to run it.

### normalize spec

Runs Pass 10 (ref resolution, predicate stabilization, abstraction synthesis) against all extracted data.

```
bsos normalize [--reembed]
```

* Executes the three steps of Pass 10 in sequence (Section 6, Pass 10): force/pattern/entity ref resolution, predicate stabilization, and abstraction synthesis.
* `--reembed` — recompute all stored embeddings from scratch using the configured embedding model. Required when `embedding_model` is changed in config (Section 8.5). Deletes all rows in the `embeddings` table, then recomputes embeddings for all items before any similarity operation runs. Prevents mixing vectors from different models, which produces meaningless cosine similarities.
* Pass dependency: requires at least one row in the `assertions` table. Exits with a descriptive error if the table is empty.

### export spec

Exports the knowledge base to a structured format for interoperability.

Flags:

* `--format json|csv` — output format (default: json)
* `--status accepted|proposed|all` — filter by item status (default: accepted)
* `--type assertion|pattern|constraint|antipattern|force|spatial-relation|process-relation` — filter by item type; may be repeated; omit to export all types
* `--output <path>` — write to file instead of stdout

JSON output is a single object keyed by item type; each value is a list of serialized records including all provenance fields.

### compress spec

Runs the semantic compression system (Section 9) across the full assertion set, producing `AbstractionNode` records.

```
bsos compress [--min-cluster-size N] [--dry-run]
```

* Executes the three steps of Section 9 in sequence: embedding-based cluster detection, LLM-A abstraction synthesis, LLM-B adversarial validation.
* `--min-cluster-size N` — override the default cluster size threshold of 3 (the minimum number of semantically similar assertions required to trigger abstraction for a subject group).
* `--dry-run` — run clustering and synthesis but do not write any `AbstractionNode` records; prints candidate abstractions and their child assertion counts to stdout.
* Pass dependency: requires at least one row in the `assertions` table with `status IN ("accepted", "proposed")`. Exits with a descriptive error if none exist.
* Obeys the 200-node queue cap from Section 4.8 — if the number of `proposed` `AbstractionNode` records already exceeds 200, the command exits with: `"AbstractionNode review queue is at [N]/200. Run 'bsos review-pending --type abstraction' before compressing further."` The `--dry-run` flag bypasses this check.
* LLM provider used: `default_llm_model` for synthesis (LLM-A) and `constraint_validation_model` for adversarial validation (LLM-B), falling back to `default_llm_model` with a structlog warning if LLM-B is not configured. When both resolve to the same model, automated validation is skipped and all generated nodes are stored with `status = "proposed"` — a structlog warning is emitted noting that single-model abstraction provides no independent validation.

### serve spec

Starts the MCP server, making all query tools (Section 13) available to MCP-compatible clients such as ifcmcp.

```
bsos serve [--transport stdio|sse] [--port PORT] [--host HOST]
```

* `--transport stdio` (default) — communicates over stdin/stdout; suitable for direct MCP client subprocess invocation (the standard Claude Code / ifcmcp integration pattern).
* `--transport sse` — starts an HTTP server using Server-Sent Events transport; requires `--port` (default: 8765) and `--host` (default: `127.0.0.1`).

**Authentication:** the MCP server requires no authentication. It is a local-only tool; network exposure is the operator's responsibility if SSE transport is used on a non-loopback address.
* Pass dependency: requires at least one row in the `entities` table. Exits with a descriptive error if the database is empty, directing the user to run `bsos extract` first.
* On startup, logs the transport mode, database path, and entity/assertion counts via structlog at INFO level so the operator can confirm the correct database is loaded.
* The MCP server rebuilds the query graph lazily per request (Section 11 CLI path) — it does NOT perform a full graph build at startup. Use `bsos build-graph` followed by the FastAPI layer (Section 18) for a persistent in-memory graph.
* Configuration in Claude Code / ifcmcp: add to MCP server config as a subprocess command: `["bsos", "serve"]` with `--transport stdio`. No port configuration is needed for the stdio transport.

### validate spec

Subcommands:

* `bsos validate --conflicts [--limit N]` — runs the conflict detection batch (Sections 10.1 and 10.2): (1) processes all `proposed` assertion/constraint/pattern items where `conflict_evaluated_at IS NULL`, runs LLM classification for each candidate pair above the embedding similarity threshold, writes a row to `conflict_pairs` for each `contradictory` pair, updates item status (`conflicted`, `accepted`, or remains `proposed`), and sets `conflict_evaluated_at = now()` on each evaluated item; (2) checks all `ProcessRelation` pairs for `hard_constraint` boolean divergence across models (Section 10.2) and marks divergent pairs as `conflicted`, writing corresponding rows to `conflict_pairs`; (3) checks for directed cycles in the full ProcessRelation graph — for each strongly connected component of size ≥ 2, all ProcessRelation edges in the cycle are marked `status = "conflicted"` and added to the review queue with a note identifying the cycle members. This detects extraction-time cycles (where Pass 5 independently extracted "X precedes Y" and "Y precedes X") before they are encountered at query time. Outputs a summary: items evaluated, conflicts found, items auto-promoted, ProcessRelation divergences flagged, process graph cycles detected. `--limit N` stops after N LLM classification calls for that invocation (default: unlimited); items above the limit retain `conflict_evaluated_at IS NULL` and are processed on the next invocation. Use `--limit` to control API costs on large corpora — a corpus of 5,000 assertions can produce O(N²) candidate pairs before embedding pre-filtering reduces the count.
* `bsos validate --constraints <entity>` — checks whether the current assertion set for the given entity violates any `Constraint` records for that entity (Section 16.2). Implementation: for each `Constraint` with `subject_id` matching the entity, retrieve all `accepted` assertions involving that entity (as subject or object). Submit a single LLM classification call per constraint: `"Constraint: [rule text]. Assertions about [entity name]: [list]. Does any assertion directly violate this constraint? If so, identify the assertion and explain the violation."` The LLM response must use the structured schema `{violated: bool, violating_assertion_id: str | None, explanation: str | None}`. Violations are printed with the assertion text and explanation; non-violations are summarised as a count. Constraints with no associated assertions are reported as "unverifiable — no assertions loaded." This command uses the LLM provider configured in `default_llm_model` (or `--models` override). **Circularity note:** validating LLM-extracted constraints with the same LLM that produced them offers limited independence. Set `constraint_validation_model` in the config to a different model than was used for extraction; `bsos validate --constraints` will use that model automatically without requiring a `--models` flag on every invocation. When `constraint_validation_model` is unset and any `Constraint` record being validated has `source_model` matching `default_llm_model`, the command MUST emit a single `[WARN]` line before running: `"constraint_validation_model is not set — validating constraints produced by <model> with the same model provides limited independence. Set a different model: bsos config set constraint_validation_model <model-id>."` This warning fires once per invocation regardless of how many constraints match.
* `bsos validate --topology` — checks the spatial relation graph for accessibility violations: Entity nodes with `entity_type = "space"` that are not reachable from an **entrance node** via `accessible_from` edges (Section 16.3). An entrance node is any Entity with `is_entrance = True` (Section 4.1). Name-based heuristics (matching "entrance", "lobby", "entry", etc.) are NOT used — LLM extraction produces inconsistent names across runs and across models. Entrance nodes MUST be designated explicitly via `bsos curate set-entrance <entity>` (or its alias `bsos config set-entrance <entity>`). If no entrance node exists in the graph, the command reports a warning and skips the check rather than treating all spaces as unreachable.
* `bsos validate --ground-truth` — alias for `bsos curate verify` (Section 16.6).

Running `bsos validate` without flags (or with `--all`) runs checks in this order: `--conflicts`, then `--topology`, then `--ground-truth`. When `--conflicts` runs as part of `--all`, it applies a default `--limit 100` to bound LLM API costs — on a large corpus, unlimited conflict detection can produce O(N²) candidate pairs and run for hours. **The command MUST print a banner before the conflict detection step begins:**

```
Note: --all mode caps conflict detection at 100 LLM calls.
For complete conflict detection on large corpora run: bsos validate --conflicts
```

This banner is printed unconditionally in `--all` mode, regardless of corpus size. Operators who want unlimited conflict detection MUST run `bsos validate --conflicts` explicitly without `--all`. `--constraints` is intentionally asymmetric: it requires an entity argument and is always skipped in the no-flag / `--all` case; a note is printed directing the user to run `bsos validate --constraints <entity>` for per-entity constraint verification. `--ground-truth` (alias for `bsos curate verify`, Section 16.6) runs unconditionally in `--all` mode — it operates across all ground-truth assertions rather than a specific entity, so no entity argument is required. The `--help` output MUST document this asymmetry explicitly so users understand why `--all` runs `--ground-truth` but not `--constraints`.

---

### review-pending spec

Lists items requiring human decision in priority order. Conflict pairs are sourced from the `conflict_pairs` table (Section 12), not by scanning for items with `status = "conflicted"` — this ensures items are always displayed with their counterpart visible:

1. `conflicted` assertions (pairs from `conflict_pairs`) — sorted by lowest confidence first, then oldest `detected_at`
2. `proposed` AbstractionNodes awaiting acceptance — sorted by oldest `created_at` first
3. pending predicates that have reached the occurrence threshold — sorted by highest `occurrence_count` first (most-used unknown predicates are most likely to be worth adding to core)
4. pending spatial relation types that have reached the occurrence threshold — sorted by highest `occurrence_count` first

**Flags:**

* `--type <assertion|abstraction|predicate|spatial-relation|force>` — filter to one item type only
* `--limit N` — stop after N items (default: 20); prevents unbounded review sessions when the queue is large
* `--re-review` — re-surface items that already have a `ReviewDecision`
* `--stats` — print queue depth by type and exit without prompting

For each item, displays the item and prompts for a decision:

* assertions: `accept` / `reject` / `defer`
* abstraction nodes: `accept` / `reject` / `defer`
* predicates: `add-to-core` / `map-to <existing>` / `defer`
* spatial relation types: `add-to-core` / `map-to <existing>` / `defer`
* forces (`--type force`): two sub-cases distinguished by `failure_type`:
  * `unresolved_ref` — a `Pattern.force_description` that could not be matched to a `Force` record; decisions: `map-to <force-id>` (link to an existing Force) / `create` (create a new Force from the description) / `dismiss` (discard the ref) / `defer`
  * `validation_failure` — a `Force` record whose name is inconsistent with its `direction`; decisions: `correct <new-name>` (update the Force name to a consistent value) / `correct-direction <increase|decrease>` (update the direction instead) / `dismiss` (deprecate the Force) / `defer`
  The display MUST show `failure_type` prominently so reviewers know which action set applies. Because the two `failure_type` values carry different action vocabularies, the display MUST render them as distinct sections — do NOT interleave them. Print all `unresolved_ref` items first under an `Unresolved Force References (N)` header, then all `validation_failure` items under an `Invalid Force Names (N)` header.

Each decision writes a `ReviewDecision` record. Items with an existing `ReviewDecision` are skipped unless `--re-review` flag is passed.

**Conflicted chains:** Conflict detection operates on pairs. If item A conflicts with B, and B separately conflicts with C, these are surfaced as two independent pairs (A–B and B–C) — transitive closure is not computed. Each pair is reviewed independently; resolving A–B does not automatically affect B–C. The reviewer must evaluate each pair on its own merits. When a conflicted item appears in multiple pairs, `review-pending` MUST display all pairs involving that item grouped together so the reviewer understands the full conflict context before making a decision.

### curate spec

Manages the hand-curated ground-truth assertion set (Section 16.1):

* `bsos curate add` — interactive prompt to define a ground-truth assertion; stored with `source_model: "human"` and `status: "accepted"`
* `bsos curate list` — lists all ground-truth assertions
* `bsos curate export` — exports ground-truth set to JSON for use in evaluation runs
* `bsos curate verify` — runs Section 16.6 consensus coverage test: what proportion of ground-truth assertions are matched in the DB with confidence ≥ 0.80. Match strategy: exact tuple `(subject_id, predicate, object_id)` first; if no exact match, use embedding nearest-neighbour (cosine similarity ≥ 0.90) as a fallback to catch paraphrased equivalents. Report both exact-match and fuzzy-match coverage separately. The 0.90 threshold is intentionally higher than the 0.85 cross-model matching threshold (Section 7): ground-truth verification is a precision check — a false positive match inflates the coverage score and undermines the evaluation's independence from the extraction process. Cross-model matching accepts more fuzzy matches because divergence between models is surfaced by the conflict system regardless of match fuzziness. Override with `bsos config set ground_truth_match_threshold <float>` if the corpus vocabulary diverges significantly from the ground-truth phrasing.
* `bsos curate set-entrance <entity>` — sets `is_entrance = True` on the named entity (resolved via the same lookup order as `curate add`: exact name match, then alias match, then error). This is the **canonical interface** for designating entrance nodes used by `bsos validate --topology`. `bsos config set-entrance <entity>` is an alias that delegates to this subcommand.

**Entity resolution in `curate add`:** the command accepts subject and object as plain text names. Resolution order:

1. Exact match against `Entity.name` in the DB → use that entity's UUID
2. Exact match (case-insensitive) against `entity_aliases` — aliases are stored as exact strings with no embeddings, so only exact lookup is possible here; if a match is found, present the matched entity to the user for confirmation before proceeding
3. No match → offer to create a new Entity with status `accepted` and `source_model: "human"`

The predicate must be a member of `CORE_PREDICATES`. If not, the command rejects it and lists valid options.

### config spec

Manages runtime configuration stored in the `config` table. All settings are key/value pairs persisted to SQLite and are available across CLI invocations.

```
bsos config get <key>               # print current value, or "(not set)"
bsos config set <key> <value>       # set a key
bsos config unset <key>             # remove a key
bsos config list                    # print all current settings
bsos config set-entrance <entity>   # alias for `bsos curate set-entrance` — delegates to curate subcommand
```

**Documented config keys:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `db_path` | string | `./bsos.db` | Absolute path to the SQLite database; written by `bsos init` for reference and verified by `bsos doctor`. **Not used for path resolution** — the `config` table can only be read after the database is already open, so this key cannot bootstrap the connection. All commands use the flag → `BSOS_DB` env var → `.bsos_config` file lookup order (defined in `init` spec). Treat this as a human-readable audit record of where the database lives, not a configuration input. |
| `embedding_model_confirmed` | `"1"` / unset | unset | Set after user confirms sentence-transformer download |
| `embedding_model` | string | `"all-mpnet-base-v2"` | Embedding model identifier |
| `default_llm_model` | string | — | Primary LLM model identifier used by `bsos extract` when `--models` is omitted; if unset, `bsos extract` exits with an error directing the user to set this key |
| `graph_rebuild_threshold` | integer | `50000` | Node count above which CLI queries use lazy loading (Section 11) |
| `pending_predicate_threshold_override` | integer | unset | Replaces the dynamic threshold formula output entirely (Section 5.2) — the set value is used as-is instead of `min(50, round(5 + total_assertions * 0.005))`; use for small corpora where the formula produces too-high a threshold |
| `auto_promote_enabled` | `"1"` / `"0"` | `"1"` | Disable auto-promotion when human review is preferred for all items |
| `constraint_validation_model` | string | — | LLM model used by `bsos validate --constraints`; SHOULD differ from the extraction model to reduce circularity (validating LLM-extracted constraints with a different model provides partial independence); if unset, falls back to `default_llm_model` with a structlog warning |
| `embedding_model_at_last_calibration` | string | — | Embedding model identifier at the time similarity thresholds were last reviewed; written by `bsos normalize --reembed`; if it differs from `embedding_model`, all commands warn that thresholds may be uncalibrated |
| `query_max_results` | integer | `20` | Default cap on MCP query tool result counts (Section 13.1) |
| `llm_timeout_seconds` | integer | `120` | Per-call LLM timeout in seconds (Section 6.2) |
| `api_enabled` | `"1"` / `"0"` | `"0"` | When `"1"`, `bsos extract` completion output reminds operator to call `POST /rebuild-graph` (Section 11) |
| `ground_truth_match_threshold` | float | `0.90` | Cosine similarity threshold for fuzzy ground-truth matching in `bsos curate verify` (Section 16.6); intentionally higher than the 0.85 cross-model matching threshold |
| `log_format` | `"text"` / `"json"` | `"text"` | structlog output format; `"json"` emits newline-delimited JSON objects — use in CI and production environments where log aggregators expect structured input |

Unknown keys are accepted without error but logged as a warning — this allows forward compatibility when new keys are introduced without requiring a migration.

### cache spec

Inspects and manages the LLM response cache (`llm_response_cache` table).

```
bsos cache stats                         # print cache size (row count, estimated size in MB)
bsos cache list [--model <id>] [--limit N]  # list cached entries (model, prompt_hash, cached_at)
bsos cache clear                         # delete all cache entries (prompts for confirmation)
bsos cache clear --entity <name>         # delete entries whose prompt contains the entity name
bsos cache clear --model <id>            # delete entries for a specific model
bsos cache clear --before <ISO-date>     # delete entries cached before this date
bsos cache clear --yes                   # skip confirmation prompt (non-interactive use)
```

**Purpose:** The LLM response cache prevents redundant API calls across `bsos extract` invocations. During iteration — when adjusting extraction prompts or fixing a specific entity — operators need to invalidate stale cache entries without clearing everything. `bsos cache clear --entity <name>` is the primary tool for re-extracting a single entity without re-running all API calls.

**`--entity` matching:** Matches against the `entity_name` column in `llm_response_cache` (case-insensitive exact match). This targets only rows explicitly associated with the named entity — `classify` calls and rows with a NULL `entity_name` are not affected. The count of matching rows is shown before the confirmation prompt.

**Safety:** `bsos cache clear` (no flags) prints a count of rows to be deleted and requires explicit confirmation or `--yes`. Partial clears (`--entity`, `--model`, `--before`) also print a count before acting. Cache rows do not affect the extracted knowledge base — clearing the cache only means future invocations will make live API calls rather than using stored responses.

### query spec

Queries the knowledge base for a single entity and prints results to stdout. Wraps the same query functions exposed by the MCP server (Section 13.3).

```
bsos query <entity> [--type <assertion|constraint|pattern|antipattern|force|spatial|process|abstraction>] [--min-confidence <float>] [--include-proposed] [--json]
```

* `<entity>` — resolved against `Entity.name` and `entity_aliases` (case-insensitive)
* `--type` — restrict output to one item type; may be repeated; omit to show all types
* `--min-confidence` — exclude items below this score (default: 0.0)
* `--include-proposed` — include `proposed` items alongside `accepted` ones (default: false for CLI, true for MCP)
* `--json` — output as JSON instead of formatted text

When `--type` is omitted, output is grouped by type with a header line per section. If the entity is not found, the command exits with: `"Entity '<name>' not found. Run 'bsos status' to check if extraction has completed."` and exit code 1.

### status spec

Displays a summary of the current database state for monitoring extraction progress.

```
bsos status [--json]
```

Output (formatted as a table):

| Item Type          | proposed | accepted | conflicted | deprecated | Total |
|--------------------|----------|----------|------------|------------|-------|
| entities           | N        | N        | —          | N          | N     |
| assertions         | N        | N        | N          | N          | N     |
| constraints        | N        | N        | N          | N          | N     |
| patterns           | N        | N        | N          | N          | N     |
| forces             | N        | N        | N          | N          | N     |
| antipatterns       | N        | N        | N          | N          | N     |
| process_relations  | N        | N        | N          | N          | N     |
| spatial_relations  | N        | N        | N          | N          | N     |
| abstraction_nodes  | N        | N        | N          | N          | N     |

Additional lines below the table:

* `Passes completed:` — last completed pass number per model, derived from the `pass_progress` table
* `Entities skipped (pass failures): N`
* `Pending predicates awaiting review: N`
* `Pending spatial relation types awaiting review: N`
* `Unresolved refs — force: N | pattern: N | entity: N`
* `LLM response cache: N entries`
* `Conflict queue: N items` — with a note if the 500-item cap is active (Section 10.1)

Exits with code 0 always. `--json` outputs the same data as a JSON object.

### history spec

Displays the status transition history of a specific item from the `provenance_log` table.

```
bsos history <item-id> [--json]
```

Output: chronological list of `(old_status → new_status, changed_at, changed_by)` tuples for the given UUID. `changed_by` is either a model identifier (automated transitions) or `"human"` (decisions made via `review-pending`). If no history exists for the UUID, exits with: `"No history found for <item-id>"` and exit code 1. `--json` outputs a JSON array.

**Initial state:** `provenance_log` records only *subsequent* status transitions — the initial extraction state is not written to the log. The `history` command MUST synthesize a leading entry `(None → proposed, <item.created_at>, <item.source_model>)` derived from the item's own row before displaying logged transitions, so the full lifecycle is visible without requiring the user to cross-reference the item table. The synthesized entry is labelled `(initial)` in text output and has `old_status: null` in JSON output.

---

### doctor spec

Checks database and system integrity. Reports problems found and suggested remediation commands. Exits with code 0 if all checks pass, code 1 if any check fails.

```
bsos doctor [--check <check-name>] [--fix]
```

* `--check <check-name>` — run only the named check (see below); omit to run all checks
* `--fix` — automatically apply safe remediations where possible (currently: recompute stale embeddings, rebuild effective-origins view if missing); does not apply remediations that modify or delete user data

**Checks performed (in order):**

| Check name | What it verifies | Failure action |
|---|---|---|
| `db-exists` | Database file exists at the configured path | Error with path |
| `view-origins` | `abstraction_node_effective_origins` view exists in SQLite | Prints CREATE VIEW statement; `--fix` recreates it |
| `schema-version` | Alembic revision head matches the version the database was last migrated to | Directs user to run `alembic upgrade head` |
| `embedding-model` | `embedding_model` config key matches `model` column in all `embeddings` rows | Directs user to run `bsos normalize --reembed`; reports count of stale rows |
| `embedding-hashes` | Content hashes in the `embeddings` table match the current resolved text for each item | Reports count of stale hashes; `--fix` recomputes stale embeddings |
| `orphaned-fks` | No `subject_id` / `object_id` in assertions/constraints/spatial_relations/forces references a non-existent entity UUID | Reports count and example UUIDs |
| `json-lists` | All JSON-encoded `list[str]` columns deserialize without error | Reports table and column of any unparseable row |
| `conflict-pairs` | Every row in `conflict_pairs` references item IDs that exist in their respective tables | Reports stale pairs (items deleted or deprecated after the pair was written) |
| `pending-refs` | Reports counts of unresolved entries in `pending_force_refs`, `pending_pattern_refs`, `pending_entity_refs` | Informational only — not a failure; guides the operator to run resolution steps |
| `pass-progress` | No entity has `status = "skipped"` in `pass_progress` without a corresponding entry in the `entities` table | Reports dangling progress rows |
| `lock-file` | `bsos.lock` is not present (no extraction is running) | Reports stale lock file; directs user to delete it if the process is no longer running |
| `force-desc-resolved` | When `passes_3_9_refs_resolved = "1"` is set in `config`, no `Pattern` row has a non-empty `force_descriptions` JSON array | Reports count of patterns still carrying unresolved descriptions; `--fix` re-runs Pass 10a for affected patterns only |

**Output format:** one line per check — `[ OK ]`, `[WARN]`, or `[FAIL]` — followed by a detail line for failures and warnings. `--fix` outputs `[ FIX ]` for each remediation applied. Summary line at the end: `N checks passed, N warnings, N failures`.

---

### purge spec

Deprecates all items produced by a specific extraction run. Use this when a run produced bad results (wrong seed, wrong model, hallucination-heavy output) and the items need to be removed from active use without deleting database rows.

```
bsos purge --run <extraction_run_id> [--dry-run] [--yes]
```

* `--run <id>` — UUID from the `extraction_runs` table (see `bsos status` or `bsos history` for IDs)
* `--dry-run` — print a count of items that would be deprecated per type, without making any changes
* `--yes` — skip the confirmation prompt; required for non-interactive use

**Behaviour:**

1. Look up the `extraction_run_id` in the `extraction_runs` table; exit with an error if not found.
2. Find all items (entities, assertions, constraints, patterns, forces, antipatterns, process_relations, spatial_relations) where `extraction_run_id` matches the specified run UUID. **Fallback for legacy rows:** items with `extraction_run_id = NULL` (inserted before this field was added) are identified by the secondary method — `source_model` matches the run's model AND `created_at` falls within the run's `started_at`–`completed_at` window. This fallback is a known limitation: if the same model was run twice in quick succession, items from adjacent runs may overlap in the time window. The fallback path MUST emit a `structlog` WARNING listing the count of rows identified by the fallback so operators know the precision is reduced.
3. Without `--dry-run`: set `status = "deprecated"` on all matched items and write a `provenance_log` row for each transition (`changed_by = "bsos purge"`). Does NOT delete rows.
4. Print a summary: count deprecated per item type.

**Constraints:** `purge` will not deprecate items with `source_model = "human"` — ground-truth records are never touched by automated cleanup. Items already `deprecated` are skipped silently (counted but not re-written). `purge` does not re-run Pass 2 deduplication or update the canonical entity set — run `bsos normalize` after purging if entity coverage has changed significantly. **Entity purge caveat:** purging a run that introduced a canonical entity (one elected by Pass 2 deduplication) will deprecate that entity row but leave assertions from other runs that reference it via `subject_id` / `object_id` intact — those assertions pass `bsos doctor --check orphaned-fks` because the UUID still exists (as deprecated). Run `bsos query <entity> --include-proposed` to verify whether cross-run assertions still reference the entity before deciding whether to deprecate those assertions manually via `review-pending`.

---

# 18. API LAYER (Optional FastAPI — out of scope for initial build)

The FastAPI layer is intentionally left as a stub. Implement only after the CLI and extraction pipeline are complete and tested. Endpoints when implemented:

* `POST /extract` — trigger extraction pipeline run
* `GET /query/{entity}` — return all knowledge about an entity
* `GET /graph` — return serialized graph (loads from joblib cache built by `bsos build-graph`)
* `POST /validate` — run validation checks

The API layer loads the serialized graph at startup (Section 11) and holds it in memory across requests. No further API design is specified here — derive it from the CLI and query layer once those are stable.

---

# 19. NON-GOALS (IMPORTANT)

Do NOT implement:

* regulatory knowledge extraction (future overlay)
* IFC geometry engines
* BIM validation systems
* scheduling engines
* CAD tools
* semantic web stacks (RDF/OWL)
* distributed systems
* production-scale infra

---

# 19.5 Implementation Phases

Build in phases. Each phase is independently deliverable and useful. An implementing agent MUST complete each phase's tests before beginning the next.

## Phase 1 — Working Extraction Core (MVP)

Implement:

* `bsos init` and `bsos config`
* `LLMProvider` protocol with one concrete implementation (OpenAI-compatible API via `instructor`)
* Passes 1–3 (concept discovery, entity deduplication, relationship extraction) with single-model support only
* Storage tables: `entities`, `entity_aliases`, `assertions`, `embeddings`, `pass_progress`, `llm_response_cache`, `extraction_runs`, `config`
* `abstraction_node_effective_origins` view (created by `bsos init` — required even in Phase 1 so migrations are consistent)
* `bsos status` and `bsos query --type assertion`
* MCP server exposing `get_requirements` and `get_dependencies`
* Unit tests and integration tests using `FakeLLMProvider`

Phase 1 end state: given a seed concept, the system extracts and stores normalized assertions, and exposes them to AI agents via MCP with no other dependencies.

## Phase 2 — Full Knowledge Layer

Implement:

* Passes 4–9 (spatial, process, constraints, anti-patterns, patterns, forces)
* Storage tables: `constraints`, `patterns`, `forces`, `antipatterns`, `process_relations`, `spatial_relations`, `pending_predicates`, `pending_spatial_relation_types`, `pending_force_refs`, `pending_pattern_refs`, `pending_entity_refs`, `predicate_mappings`
* Controlled vocabulary system (Sections 5.1–5.4)
* Full MCP query layer (all tools in Section 13.3)
* `bsos validate --topology` and `bsos validate --constraints`
* `bsos curate` and `bsos review-pending`
* Ollama / local model fallback path (Section 2)

Phase 2 end state: the full knowledge schema is populated and queryable; single-model extraction covers all eight knowledge types.

## Phase 3 — Normalization and Conflict Detection

Implement:

* Pass 10 with sub-pass resume semantics (`"10a"`, `"10b"`, `"10c"`)
* Pass 11 (adversarial validation)
* Deferred conflict detection batch (`bsos validate --conflicts`)
* Auto-promotion logic (Section 7, single-model path via cross-prompt consistency)
* Storage tables: `review_decisions`, `conflict_pairs`, `provenance_log`
* `bsos normalize`, `bsos compress`
* `bsos history`

Phase 3 end state: extracted knowledge is normalized, conflicts are surfaced, items are promoted or flagged for review, and the full lifecycle is auditable.

## Phase 4 — Multi-Model Consensus and Evaluation

> **BLOCKING DEPENDENCY:** Phase 4 cannot be declared complete without a human domain expert populating `tests/fixtures/pattern_critique_cases.json` (Section 16.4). This file encodes domain judgements about pattern relevance that cannot be generated by an implementing agent. Schedule a domain-expert review session before beginning Phase 4 evaluation work — the pattern critique test cannot run without it.

Implement:

* Multi-model consensus matching (Section 7): cross-model assertion matching, confidence updates, `ReviewDecision` audit trail for auto-promoted items
* Single-model fallback rules (Section 7.1) — formally distinct from Phase 3's cross-prompt path
* Incremental model addition (Section 7.2)
* Ground-truth evaluation suite (Sections 16.1–16.6): `bsos curate verify`, constraint reasoning test, topology test, pattern critique fixture
* `bsos export`
* FastAPI layer (Section 18, optional)

Phase 4 end state: epistemic independence through multi-model consensus; system validated against ground truth.

**Blocking dependency:** Phase 4 cannot be declared complete without a human domain expert populating `tests/fixtures/pattern_critique_cases.json` (Section 16.4). This file encodes domain judgements about which patterns are relevant to specific entity contexts and cannot be generated by the implementing agent. Plan a domain-expert review session before Phase 4 is closed.

---

# 20. SUCCESS CRITERION

System is successful when:

Given a concept (e.g. "roof"), it can produce:

* normalized constraints
* dependencies
* spatial relations
* process constraints
* anti-patterns
* architectural patterns
* force interactions
* exceptions
* confidence scores
* provenance
* abstraction hierarchy

All results MUST be:

* structured
* queryable
* graph-integrated
* consistent under normalization
* grounded in multi-model consensus where possible

---

# 21. FINAL IMPLEMENTATION PRINCIPLE

The system is a:

```text
semantic distillation and compression engine for universal building intelligence
```

NOT:

* an ontology database
* a rules engine
* a BIM validator
* a regulatory knowledge base

All implementation decisions should optimize for:

* semantic stability
* abstraction emergence
* normalization quality
* graph coherence
* epistemic honesty (confidence scores reflect actual consensus, not extraction count)

---
