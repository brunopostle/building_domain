"""Database engine creation, WAL mode, session factory."""
from sqlmodel import SQLModel, create_engine, Session
from sqlalchemy import event, text

# View created by bsos init (not Alembic-managed).
ABSTRACTION_NODE_EFFECTIVE_ORIGINS_VIEW = """
CREATE VIEW IF NOT EXISTS abstraction_node_effective_origins AS
SELECT
    an.id AS abstraction_node_id,
    a.knowledge_origin,
    COUNT(*) AS origin_count
FROM abstraction_nodes an
JOIN json_each(an.child_ids) child ON 1=1
JOIN assertions a ON a.id = child.value
GROUP BY an.id, a.knowledge_origin
"""


def create_db_engine(db_path: str = "bsos.db"):
    engine = create_engine(f"sqlite:///{db_path}", echo=False)

    @event.listens_for(engine, "connect")
    def set_wal_mode(dbapi_conn, _):
        dbapi_conn.execute("PRAGMA journal_mode=WAL")
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    SQLModel.metadata.create_all(engine)
    return engine


def create_views(engine) -> None:
    """Create non-Alembic-managed views. Called by bsos init."""
    with engine.connect() as conn:
        conn.execute(text(ABSTRACTION_NODE_EFFECTIVE_ORIGINS_VIEW))
        conn.commit()


def verify_views(engine) -> bool:
    """Return True if required views exist. Called by bsos doctor."""
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='view' AND name='abstraction_node_effective_origins'")
        ).fetchone()
    return row is not None


def get_session(engine) -> Session:
    return Session(engine)
