"""SQLModel persistence models — parallel hierarchy to src/bsos/models/ Pydantic classes.

List[str] fields are stored as JSON-encoded TEXT. Conversion happens at the repository boundary.
NOTE: abstraction_node_effective_origins view is not Alembic-managed.
Any migration that drops assertions or abstraction_nodes must recreate it manually.
"""
from datetime import datetime
from typing import Optional
from sqlalchemy import Column, Text
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
