"""Database engine creation, WAL mode, session factory."""
from sqlmodel import SQLModel, create_engine, Session
from sqlalchemy import event, text


def create_db_engine(db_path: str = "bsos.db"):
    engine = create_engine(f"sqlite:///{db_path}", echo=False)

    @event.listens_for(engine, "connect")
    def set_wal_mode(dbapi_conn, _):
        dbapi_conn.execute("PRAGMA journal_mode=WAL")

    SQLModel.metadata.create_all(engine)
    return engine


def get_session(engine):
    return Session(engine)
