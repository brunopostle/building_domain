"""bsos curate command."""
import json
import uuid
from datetime import datetime, timezone
from typing import Optional

import typer
from sqlmodel import Session, select

from bsos.persistence.models import AssertionRow, EntityAliasRow, EntityRow

app = typer.Typer()


@app.command("merge")
def merge(
    source: str = typer.Argument(..., help="Entity to merge (will be marked merged)"),
    target: str = typer.Argument(..., help="Canonical entity to keep"),
    db: str = typer.Option(None, "--db"),
) -> None:
    """Merge one entity into another."""
    from bsos.cli.db_context import open_db
    _, session = open_db(db)

    with session:
        def find(name: str) -> EntityRow | None:
            row = session.exec(
                select(EntityRow).where(EntityRow.name.ilike(name))  # type: ignore[attr-defined]
            ).first()
            if row:
                return row
            alias = session.exec(
                select(EntityAliasRow).where(EntityAliasRow.alias.ilike(name))  # type: ignore[attr-defined]
            ).first()
            if alias:
                return session.get(EntityRow, alias.entity_id)
            return None

        src_row = find(source)
        tgt_row = find(target)

        if src_row is None:
            typer.echo(f"Source entity '{source}' not found.", err=True)
            raise typer.Exit(1)
        if tgt_row is None:
            typer.echo(f"Target entity '{target}' not found.", err=True)
            raise typer.Exit(1)
        if src_row.id == tgt_row.id:
            typer.echo("Source and target are the same entity.", err=True)
            raise typer.Exit(1)

        src_row.status = "merged"
        session.add(EntityAliasRow(entity_id=tgt_row.id, alias=src_row.name))
        session.commit()
        typer.echo(f"Merged '{src_row.name}' → '{tgt_row.name}'")


@app.command("set-entrance")
def set_entrance(
    entity: str = typer.Argument(..., help="Space entity to mark as entrance"),
    db: str = typer.Option(None, "--db"),
    unset: bool = typer.Option(False, "--unset", help="Remove entrance designation"),
) -> None:
    """Mark (or unmark) a space entity as an entrance node for topology validation."""
    from bsos.cli.db_context import open_db
    _, session = open_db(db)

    with session:
        row = session.exec(
            select(EntityRow).where(EntityRow.name.ilike(entity))  # type: ignore[attr-defined]
        ).first()
        if row is None:
            alias = session.exec(
                select(EntityAliasRow).where(EntityAliasRow.alias.ilike(entity))  # type: ignore[attr-defined]
            ).first()
            if alias:
                row = session.get(EntityRow, alias.entity_id)

        if row is None:
            typer.echo(f"Entity '{entity}' not found.", err=True)
            raise typer.Exit(1)

        if row.entity_type != "space":
            typer.echo(
                f"Warning: '{row.name}' has type '{row.entity_type}', not 'space'. "
                "Entrance designation is normally for space entities.",
                err=True,
            )

        row.is_entrance = not unset
        session.commit()
        action = "Unset entrance for" if unset else "Set as entrance:"
        typer.echo(f"{action} '{row.name}'")


def _find_entity(session: Session, name: str) -> EntityRow | None:
    """Resolve entity by exact name then alias (case-insensitive)."""
    row = session.exec(
        select(EntityRow).where(EntityRow.name.ilike(name), EntityRow.status != "merged")  # type: ignore[attr-defined]
    ).first()
    if row:
        return row
    alias = session.exec(
        select(EntityAliasRow).where(EntityAliasRow.alias.ilike(name))  # type: ignore[attr-defined]
    ).first()
    if alias:
        entity = session.get(EntityRow, alias.entity_id)
        if entity and entity.status != "merged":
            return entity
    return None


@app.command("add")
def add_assertion(
    subject: str = typer.Argument(..., help="Subject entity name"),
    predicate: str = typer.Argument(..., help="Predicate (must be in PREDICATE_REGISTRY)"),
    object_: str = typer.Argument(..., metavar="OBJECT", help="Object entity name"),
    condition: list[str] = typer.Option([], "--condition", help="Condition (repeatable)"),
    exception: list[str] = typer.Option([], "--exception", help="Exception (repeatable)"),
    db: str = typer.Option(None, "--db"),
) -> None:
    """Add a ground-truth assertion (source_model=human, status=accepted)."""
    from bsos.cli.db_context import open_db
    from bsos.vocab import PREDICATE_REGISTRY

    if predicate not in PREDICATE_REGISTRY:
        known = ", ".join(sorted(PREDICATE_REGISTRY.keys()))
        typer.echo(f"Unknown predicate '{predicate}'. Known: {known}", err=True)
        raise typer.Exit(1)

    _, session = open_db(db)
    with session:
        subj_row = _find_entity(session, subject)
        if subj_row is None:
            typer.echo(f"Subject entity '{subject}' not found.", err=True)
            raise typer.Exit(1)

        obj_row = _find_entity(session, object_)
        if obj_row is None:
            typer.echo(f"Object entity '{object_}' not found.", err=True)
            raise typer.Exit(1)

        row = AssertionRow(
            id=str(uuid.uuid4()),
            subject_id=subj_row.id,
            predicate=predicate,
            object_id=obj_row.id,
            subject_type=subj_row.entity_type,
            object_type=obj_row.entity_type,
            conditions=json.dumps(condition),
            exceptions=json.dumps(exception),
            confidence=1.0,
            status="accepted",
            knowledge_origin="human",
            source_model="human",
            created_at=datetime.now(timezone.utc),
        )
        session.add(row)
        session.commit()
        typer.echo(f"Added: {subj_row.name} {predicate} {obj_row.name}")


@app.command("list")
def list_assertions(
    entity: Optional[str] = typer.Option(None, "--entity", help="Filter by entity name"),
    db: str = typer.Option(None, "--db"),
) -> None:
    """List ground-truth assertions (source_model=human)."""
    from bsos.cli.db_context import open_db

    _, session = open_db(db)
    with session:
        stmt = select(AssertionRow).where(AssertionRow.source_model == "human")
        rows = session.exec(stmt).all()

        if entity:
            ent_row = _find_entity(session, entity)
            if ent_row is None:
                typer.echo(f"Entity '{entity}' not found.", err=True)
                raise typer.Exit(1)
            rows = [r for r in rows if r.subject_id == ent_row.id or r.object_id == ent_row.id]

        entity_ids = {r.subject_id for r in rows} | {r.object_id for r in rows}
        entities = {
            e.id: e.name
            for e in session.exec(select(EntityRow).where(EntityRow.id.in_(entity_ids))).all()  # type: ignore[attr-defined]
        }

    if not rows:
        typer.echo("No ground-truth assertions found.")
        return

    for r in rows:
        subj = entities.get(r.subject_id, r.subject_id)
        obj = entities.get(r.object_id, r.object_id)
        conds = json.loads(r.conditions) if r.conditions else []
        excs = json.loads(r.exceptions) if r.exceptions else []
        line = f"  {subj} {r.predicate} {obj}"
        if conds:
            line += f"  [conditions: {', '.join(conds)}]"
        if excs:
            line += f"  [exceptions: {', '.join(excs)}]"
        typer.echo(line)

    typer.echo(f"\n{len(rows)} ground-truth assertion(s)")


@app.command("export")
def export_assertions(
    output: Optional[str] = typer.Option(None, "--output", help="Output path (default: stdout)"),
    db: str = typer.Option(None, "--db"),
) -> None:
    """Export ground-truth assertions to JSON."""
    from bsos.cli.db_context import open_db

    _, session = open_db(db)
    with session:
        rows = session.exec(
            select(AssertionRow).where(AssertionRow.source_model == "human")
        ).all()

        entity_ids = {r.subject_id for r in rows} | {r.object_id for r in rows}
        entities = {
            e.id: e.name
            for e in session.exec(select(EntityRow).where(EntityRow.id.in_(entity_ids))).all()  # type: ignore[attr-defined]
        }

        data = [
            {
                "id": r.id,
                "subject": entities.get(r.subject_id, r.subject_id),
                "predicate": r.predicate,
                "object": entities.get(r.object_id, r.object_id),
                "conditions": json.loads(r.conditions) if r.conditions else [],
                "exceptions": json.loads(r.exceptions) if r.exceptions else [],
                "status": r.status,
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ]

    payload = json.dumps(data, indent=2)
    if output:
        with open(output, "w") as f:
            f.write(payload)
        typer.echo(f"Exported {len(data)} assertion(s) to {output}")
    else:
        typer.echo(payload)


@app.command("verify")
def verify_coverage(
    db: str = typer.Option(None, "--db"),
    threshold: float = typer.Option(0.90, "--threshold", help="Fuzzy match threshold"),
    target: float = typer.Option(0.80, "--target", help="Coverage target (default 0.80)"),
) -> None:
    """Verify ground-truth coverage against extracted corpus (Section 16.6)."""
    import numpy as np
    from bsos.cli.db_context import open_db
    from bsos.config import get_config

    _, session = open_db(db)
    with session:
        gt_rows = session.exec(
            select(AssertionRow).where(AssertionRow.source_model == "human")
        ).all()

        if not gt_rows:
            typer.echo("No ground-truth assertions found. Nothing to verify.")
            return

        corpus_rows = session.exec(
            select(AssertionRow).where(
                AssertionRow.source_model != "human",
                AssertionRow.status == "accepted",
            )
        ).all()

        all_entity_ids = (
            {r.subject_id for r in gt_rows} | {r.object_id for r in gt_rows}
            | {r.subject_id for r in corpus_rows} | {r.object_id for r in corpus_rows}
        )
        entities = {
            e.id: e.name
            for e in session.exec(select(EntityRow).where(EntityRow.id.in_(all_entity_ids))).all()  # type: ignore[attr-defined]
        }
        embedding_model = get_config(session, "embedding_model") or "all-mpnet-base-v2"

    corpus_keys = {(r.subject_id, r.predicate, r.object_id) for r in corpus_rows}

    def _text(r: AssertionRow) -> str:
        return f"{entities.get(r.subject_id, r.subject_id)} {r.predicate} {entities.get(r.object_id, r.object_id)}"

    matched = 0
    unmatched: list[str] = []

    if corpus_rows:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(embedding_model)
        corpus_texts = [_text(r) for r in corpus_rows]
        corpus_vecs = np.array(st.encode(corpus_texts, show_progress_bar=False), dtype=np.float32)
    else:
        corpus_vecs = np.zeros((0, 1), dtype=np.float32)

    for gt in gt_rows:
        key = (gt.subject_id, gt.predicate, gt.object_id)
        if key in corpus_keys:
            matched += 1
            continue

        if len(corpus_rows) == 0:
            unmatched.append(_text(gt))
            continue

        from sentence_transformers import SentenceTransformer  # already loaded above
        gt_vec = np.array(st.encode([_text(gt)], show_progress_bar=False), dtype=np.float32)[0]  # type: ignore[possibly-undefined]
        norms = np.linalg.norm(corpus_vecs, axis=1)
        gt_norm = float(np.linalg.norm(gt_vec))
        with np.errstate(divide="ignore", invalid="ignore"):
            sims = (corpus_vecs @ gt_vec) / (norms * gt_norm)
        sims = np.where(np.isfinite(sims), sims, 0.0)
        best_sim = float(np.max(sims)) if len(sims) > 0 else 0.0

        if best_sim >= threshold:
            matched += 1
        else:
            unmatched.append(_text(gt))

    total = len(gt_rows)
    coverage = matched / total if total > 0 else 0.0
    typer.echo(f"Coverage: {matched}/{total} ({coverage:.0%})")

    if unmatched:
        typer.echo("\nUnmatched ground-truth assertions:")
        for txt in unmatched:
            typer.echo(f"  - {txt}")

    if coverage < target:
        typer.echo(f"\nWARNING: coverage {coverage:.0%} below target {target:.0%}", err=True)
        raise typer.Exit(1)
    else:
        typer.echo(f"\nPASS: coverage meets target ({target:.0%})")
