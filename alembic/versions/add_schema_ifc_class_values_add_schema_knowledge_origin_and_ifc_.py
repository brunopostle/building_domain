"""add schema knowledge_origin and ifc_class entity_type

Revision ID: add_schema_ifc_class_values
Revises: b6fd228e79cb
Create Date: 2026-05-23 11:13:54.014004

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision: str = 'add_schema_ifc_class_values'
down_revision: Union[str, None] = 'b6fd228e79cb'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # No-op: knowledge_origin and entity_type columns are plain VARCHAR with no CHECK
    # constraints, so adding 'schema' and 'ifc_class' as valid values requires no DDL.
    pass


def downgrade() -> None:
    pass
