"""phase3_tables

Revision ID: b6fd228e79cb
Revises: ada30a7ca5e3
Create Date: 2026-05-22 22:14:31.543165

NOTE: Phase 2 tables (constraints, patterns, forces, antipatterns, process_relations,
spatial_relations, pending_*, predicate_mappings) were created via SQLModel.metadata.create_all()
and are not repeated here — they already exist in the database.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision: str = 'b6fd228e79cb'
down_revision: Union[str, None] = 'ada30a7ca5e3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'abstraction_nodes',
        sa.Column('id', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('statement', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('child_ids', sa.Text(), nullable=True),
        sa.Column('abstraction_rationale', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('source_model', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('source_prompt', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('extraction_run_id', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('status', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('rationale', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('conflict_evaluated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_table(
        'conflict_pairs',
        sa.Column('id', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('item_a_id', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('item_a_type', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('item_b_id', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('item_b_type', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('detected_at', sa.DateTime(), nullable=False),
        sa.Column('classification', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_conflict_pairs_item_a_id', 'conflict_pairs', ['item_a_id'], unique=False)
    op.create_index('ix_conflict_pairs_item_b_id', 'conflict_pairs', ['item_b_id'], unique=False)
    op.create_table(
        'provenance_log',
        sa.Column('id', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('item_id', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('item_type', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('old_status', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('new_status', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('changed_at', sa.DateTime(), nullable=False),
        sa.Column('changed_by', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_provenance_log_item_id', 'provenance_log', ['item_id'], unique=False)
    op.create_table(
        'review_decisions',
        sa.Column('id', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('item_id', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('item_type', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('decision', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('mapped_to', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('rationale', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('reviewer', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_review_decisions_item_id', 'review_decisions', ['item_id'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_review_decisions_item_id', table_name='review_decisions')
    op.drop_table('review_decisions')
    op.drop_index('ix_provenance_log_item_id', table_name='provenance_log')
    op.drop_table('provenance_log')
    op.drop_index('ix_conflict_pairs_item_b_id', table_name='conflict_pairs')
    op.drop_index('ix_conflict_pairs_item_a_id', table_name='conflict_pairs')
    op.drop_table('conflict_pairs')
    op.drop_table('abstraction_nodes')
