"""SQLModel persistence models — parallel hierarchy to src/bsos/models/ Pydantic classes.

List[str] fields are stored as JSON-encoded TEXT. Conversion happens at the repository boundary.
NOTE: abstraction_node_effective_origins view is not Alembic-managed.
Any migration that drops assertions or abstraction_nodes must recreate it manually.
"""
from datetime import datetime
from typing import Optional
from sqlalchemy import Column, Text, UniqueConstraint
from sqlmodel import Field, SQLModel


class ConfigRow(SQLModel, table=True):
    __tablename__ = "config"

    key: str = Field(primary_key=True)
    value: str


class EntityRow(SQLModel, table=True):
    __tablename__ = "entities"

    id: str = Field(primary_key=True)
    name: str = Field(index=True)
    entity_type: str
    description: str = ""
    status: str = "proposed"
    is_entrance: bool = False
    source_model: str
    source_prompt: Optional[str] = None
    created_at: datetime
    extraction_run_id: Optional[str] = None


class EntityAliasRow(SQLModel, table=True):
    __tablename__ = "entity_aliases"

    id: Optional[int] = Field(default=None, primary_key=True)
    entity_id: str = Field(index=True)
    alias: str = Field(index=True)


class AssertionRow(SQLModel, table=True):
    __tablename__ = "assertions"

    id: str = Field(primary_key=True)
    subject_id: str = Field(index=True)
    predicate: str = Field(index=True)
    object_id: str = Field(index=True)
    subject_type: str
    object_type: str
    conditions: str = Field(default="[]", sa_column=Column(Text))
    exceptions: str = Field(default="[]", sa_column=Column(Text))
    applicability: str = Field(default="[]", sa_column=Column(Text))
    cross_prompt_consistency: Optional[float] = None
    prompt_framing_count: Optional[int] = None
    source_model: str
    source_prompt: Optional[str] = None
    created_at: datetime
    extraction_run_id: Optional[str] = None
    confidence: float
    status: str = "proposed"
    knowledge_origin: str
    rationale: Optional[str] = None
    conflict_evaluated_at: Optional[datetime] = None


class EmbeddingRow(SQLModel, table=True):
    __tablename__ = "embeddings"

    item_type: str = Field(primary_key=True)
    item_id: str = Field(primary_key=True)
    model: str = Field(primary_key=True)
    dim: int
    content_hash: str
    vector: bytes


class PassProgressRow(SQLModel, table=True):
    __tablename__ = "pass_progress"

    pass_number: str = Field(primary_key=True)
    entity_id: str = Field(primary_key=True)
    model: str = Field(primary_key=True)
    completed_at: datetime
    status: str  # "completed" | "skipped"


class LLMResponseCacheRow(SQLModel, table=True):
    __tablename__ = "llm_response_cache"

    model: str = Field(primary_key=True)
    prompt_hash: str = Field(primary_key=True)
    entity_name: Optional[str] = None
    response_json: str = Field(sa_column=Column(Text))
    cached_at: datetime


class ExtractionRunRow(SQLModel, table=True):
    __tablename__ = "extraction_runs"

    id: str = Field(primary_key=True)
    started_at: datetime
    completed_at: Optional[datetime] = None
    models: str = Field(default="[]", sa_column=Column(Text))  # JSON array
    passes: str = Field(default="[]", sa_column=Column(Text))  # JSON array
    seed: Optional[str] = None
    entity_count_before: Optional[int] = None
    entity_count_after: Optional[int] = None
    assertion_count_before: Optional[int] = None
    assertion_count_after: Optional[int] = None


# ---------------------------------------------------------------------------
# Phase 2 tables
# ---------------------------------------------------------------------------

class ConstraintRow(SQLModel, table=True):
    __tablename__ = "constraints"

    id: str = Field(primary_key=True)
    subject_id: str = Field(index=True)
    rule: str
    constraint_type: str
    conditions: str = Field(default="[]", sa_column=Column(Text))
    exceptions: str = Field(default="[]", sa_column=Column(Text))
    source_model: str
    source_prompt: Optional[str] = None
    created_at: datetime
    extraction_run_id: Optional[str] = None
    confidence: float
    status: str = "proposed"
    knowledge_origin: str
    rationale: Optional[str] = None
    conflict_evaluated_at: Optional[datetime] = None


class PatternRow(SQLModel, table=True):
    __tablename__ = "patterns"

    id: str = Field(primary_key=True)
    name: str = Field(index=True)
    subject_id: Optional[str] = Field(default=None, index=True)
    context: str = Field(default="[]", sa_column=Column(Text))
    problem: str
    force_descriptions: str = Field(default="[]", sa_column=Column(Text))
    force_ids: str = Field(default="[]", sa_column=Column(Text))
    solution: str
    consequences: str = Field(default="[]", sa_column=Column(Text))
    related_pattern_names: str = Field(default="[]", sa_column=Column(Text))
    related_pattern_ids: str = Field(default="[]", sa_column=Column(Text))
    emergent_properties: str = Field(default="[]", sa_column=Column(Text))
    source_model: str
    source_prompt: Optional[str] = None
    created_at: datetime
    extraction_run_id: Optional[str] = None
    confidence: float
    status: str = "proposed"
    knowledge_origin: str
    rationale: Optional[str] = None
    conflict_evaluated_at: Optional[datetime] = None


class ForceRow(SQLModel, table=True):
    __tablename__ = "forces"

    id: str = Field(primary_key=True)
    name: str = Field(index=True)
    direction: str
    affects: str = Field(default="[]", sa_column=Column(Text))
    source_model: str
    source_prompt: Optional[str] = None
    created_at: datetime
    extraction_run_id: Optional[str] = None
    confidence: float
    status: str = "proposed"
    knowledge_origin: str
    rationale: Optional[str] = None
    conflict_evaluated_at: Optional[datetime] = None


class AntiPatternRow(SQLModel, table=True):
    __tablename__ = "antipatterns"

    id: str = Field(primary_key=True)
    name: str = Field(index=True)
    subject_id: Optional[str] = Field(default=None, index=True)
    conditions: str = Field(default="[]", sa_column=Column(Text))
    consequences: str = Field(default="[]", sa_column=Column(Text))
    mitigations: str = Field(default="[]", sa_column=Column(Text))
    source_model: str
    source_prompt: Optional[str] = None
    created_at: datetime
    extraction_run_id: Optional[str] = None
    confidence: float
    status: str = "proposed"
    knowledge_origin: str
    rationale: Optional[str] = None
    conflict_evaluated_at: Optional[datetime] = None


class ProcessRelationRow(SQLModel, table=True):
    __tablename__ = "process_relations"
    __table_args__ = (
        UniqueConstraint("predecessor_id", "successor_id", "source_model"),
    )

    id: str = Field(primary_key=True)
    predecessor_id: str = Field(index=True)
    successor_id: str = Field(index=True)
    hard_constraint: bool
    source_model: str
    source_prompt: Optional[str] = None
    created_at: datetime
    extraction_run_id: Optional[str] = None
    confidence: float
    status: str = "proposed"
    knowledge_origin: str
    rationale: str
    conflict_evaluated_at: Optional[datetime] = None


class SpatialRelationRow(SQLModel, table=True):
    __tablename__ = "spatial_relations"

    id: str = Field(primary_key=True)
    subject_id: str = Field(index=True)
    relation: str
    object_id: str = Field(index=True)
    source_model: str
    source_prompt: Optional[str] = None
    created_at: datetime
    extraction_run_id: Optional[str] = None
    confidence: float
    status: str = "proposed"
    knowledge_origin: str
    rationale: Optional[str] = None
    conflict_evaluated_at: Optional[datetime] = None


class PendingPredicateRow(SQLModel, table=True):
    __tablename__ = "pending_predicates"

    id: Optional[int] = Field(default=None, primary_key=True)
    value: str = Field(index=True, unique=True)
    vocabulary_type: str = "predicate"
    occurrence_count: int = 1
    first_seen_at: datetime
    last_seen_at: datetime
    flagged_for_review: bool = False


class PendingSpatialRelationTypeRow(SQLModel, table=True):
    __tablename__ = "pending_spatial_relation_types"

    id: Optional[int] = Field(default=None, primary_key=True)
    value: str = Field(index=True, unique=True)
    vocabulary_type: str = "spatial_relation"
    occurrence_count: int = 1
    first_seen_at: datetime
    last_seen_at: datetime
    flagged_for_review: bool = False


class PendingForceRefRow(SQLModel, table=True):
    __tablename__ = "pending_force_refs"

    id: Optional[int] = Field(default=None, primary_key=True)
    description: str = Field(sa_column=Column(Text))
    failure_type: str
    pattern_id: Optional[str] = None
    force_id: Optional[str] = None
    created_at: datetime


class PendingPatternRefRow(SQLModel, table=True):
    __tablename__ = "pending_pattern_refs"

    id: Optional[int] = Field(default=None, primary_key=True)
    pattern_name: str
    source_pattern_id: str
    created_at: datetime


class PendingEntityRefRow(SQLModel, table=True):
    __tablename__ = "pending_entity_refs"

    id: Optional[int] = Field(default=None, primary_key=True)
    entity_name: str
    source_force_id: str
    created_at: datetime


class PredicateMappingRow(SQLModel, table=True):
    __tablename__ = "predicate_mappings"

    id: Optional[int] = Field(default=None, primary_key=True)
    from_predicate: str = Field(index=True)
    to_predicate: str
    created_at: datetime
    reviewer: Optional[str] = None
