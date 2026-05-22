"""Config table access helpers."""
from typing import Optional
from sqlmodel import Session, select


def get_config(session: Session, key: str) -> Optional[str]:
    from bsos.persistence.models import ConfigRow
    row = session.exec(select(ConfigRow).where(ConfigRow.key == key)).first()
    return row.value if row else None


def set_config(session: Session, key: str, value: str) -> None:
    from bsos.persistence.models import ConfigRow
    row = session.exec(select(ConfigRow).where(ConfigRow.key == key)).first()
    if row:
        row.value = value
    else:
        row = ConfigRow(key=key, value=value)
        session.add(row)
    session.commit()
