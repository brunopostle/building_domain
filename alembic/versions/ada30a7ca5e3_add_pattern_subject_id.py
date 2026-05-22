"""add_pattern_subject_id

Revision ID: ada30a7ca5e3
Revises: 539fc33e63b4
Create Date: 2026-05-22 21:40:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision: str = 'ada30a7ca5e3'
down_revision: Union[str, None] = '539fc33e63b4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('patterns',
        sa.Column('subject_id', sqlmodel.sql.sqltypes.AutoString(), nullable=True)
    )
    op.create_index(op.f('ix_patterns_subject_id'), 'patterns', ['subject_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_patterns_subject_id'), table_name='patterns')
    op.drop_column('patterns', 'subject_id')
