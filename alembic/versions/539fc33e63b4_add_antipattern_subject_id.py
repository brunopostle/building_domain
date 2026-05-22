"""add_antipattern_subject_id

Revision ID: 539fc33e63b4
Revises: 5b0f3e266c92
Create Date: 2026-05-22 21:22:32.233106

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision: str = '539fc33e63b4'
down_revision: Union[str, None] = '5b0f3e266c92'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('antipatterns',
        sa.Column('subject_id', sqlmodel.sql.sqltypes.AutoString(), nullable=True)
    )
    op.create_index(op.f('ix_antipatterns_subject_id'), 'antipatterns', ['subject_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_antipatterns_subject_id'), table_name='antipatterns')
    op.drop_column('antipatterns', 'subject_id')
