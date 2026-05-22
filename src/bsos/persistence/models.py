"""SQLModel persistence models."""
from sqlmodel import Field, SQLModel


class ConfigRow(SQLModel, table=True):
    __tablename__ = "config"

    key: str = Field(primary_key=True)
    value: str
