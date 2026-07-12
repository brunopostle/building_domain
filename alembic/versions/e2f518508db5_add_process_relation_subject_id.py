"""add_process_relation_subject_id

Revision ID: e2f518508db5
Revises: add_schema_ifc_class_values
Create Date: 2026-07-12 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision: str = 'e2f518508db5'
down_revision: Union[str, None] = 'add_schema_ifc_class_values'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('process_relations',
        sa.Column('subject_id', sqlmodel.sql.sqltypes.AutoString(), nullable=True)
    )
    op.create_index(op.f('ix_process_relations_subject_id'), 'process_relations', ['subject_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_process_relations_subject_id'), table_name='process_relations')
    op.drop_column('process_relations', 'subject_id')
