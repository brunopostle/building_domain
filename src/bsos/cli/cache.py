"""bsos cache subcommand group — LLM response cache management."""
import json
from datetime import datetime, timezone
from typing import Optional

import typer
from sqlmodel import select

app = typer.Typer(help="Manage the LLM response cache.")


@app.command("stats")
def cmd_stats(
    db: Optional[str] = typer.Option(None, "--db"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Show cache statistics: entry count, total size, date range."""
    from bsos.cli.db_context import open_db
    from bsos.persistence.models import LLMResponseCacheRow

    _, session = open_db(db)
    with session:
        rows = session.exec(select(LLMResponseCacheRow)).all()

    if not rows:
        if json_out:
            typer.echo(json.dumps({"entries": 0, "total_bytes": 0, "oldest": None, "newest": None}))
        else:
            typer.echo("Cache is empty.")
        return

    total_bytes = sum(len(r.response_json.encode("utf-8")) for r in rows)
    dates = [r.cached_at for r in rows]
    oldest = min(dates)
    newest = max(dates)

    by_model: dict[str, int] = {}
    for r in rows:
        by_model[r.model] = by_model.get(r.model, 0) + 1

    if json_out:
        typer.echo(json.dumps({
            "entries": len(rows),
            "total_bytes": total_bytes,
            "oldest": oldest.isoformat(),
            "newest": newest.isoformat(),
            "by_model": by_model,
        }))
    else:
        typer.echo(f"Entries:     {len(rows)}")
        typer.echo(f"Total size:  {total_bytes / 1024:.1f} KB")
        typer.echo(f"Oldest:      {oldest.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        typer.echo(f"Newest:      {newest.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        typer.echo("By model:")
        for model, count in sorted(by_model.items()):
            typer.echo(f"  {model}: {count}")


@app.command("list")
def cmd_list(
    entity: Optional[str] = typer.Option(None, "--entity", "-e", help="Filter by entity name"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Filter by model name"),
    limit: int = typer.Option(50, "--limit", "-n", help="Max rows to show"),
    db: Optional[str] = typer.Option(None, "--db"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """List cache entries, optionally filtered by entity or model."""
    from bsos.cli.db_context import open_db
    from bsos.persistence.models import LLMResponseCacheRow

    _, session = open_db(db)
    with session:
        stmt = select(LLMResponseCacheRow)
        if model:
            stmt = stmt.where(LLMResponseCacheRow.model == model)
        if entity:
            stmt = stmt.where(LLMResponseCacheRow.entity_name == entity)
        rows = session.exec(stmt).all()

    rows = sorted(rows, key=lambda r: r.cached_at, reverse=True)[:limit]

    if json_out:
        typer.echo(json.dumps([
            {
                "model": r.model,
                "prompt_hash": r.prompt_hash,
                "entity_name": r.entity_name,
                "cached_at": r.cached_at.isoformat(),
                "response_bytes": len(r.response_json.encode("utf-8")),
            }
            for r in rows
        ], indent=2))
        return

    if not rows:
        typer.echo("No matching cache entries.")
        return

    col_w = 36
    typer.echo(f"{'Model':<30}  {'Entity':<30}  {'Cached at':<20}  {'Hash':<{col_w}}")
    typer.echo("-" * (30 + 2 + 30 + 2 + 20 + 2 + col_w))
    for r in rows:
        entity_name = r.entity_name or "(none)"
        cached = r.cached_at.strftime("%Y-%m-%d %H:%M:%S")
        typer.echo(f"{r.model:<30}  {entity_name:<30}  {cached:<20}  {r.prompt_hash[:col_w]}")

    if len(rows) == limit:
        typer.echo(f"\n(showing first {limit}; use --limit to see more)")


@app.command("clear")
def cmd_clear(
    entity: Optional[str] = typer.Option(None, "--entity", "-e", help="Delete entries for this entity name"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Delete entries for this model"),
    before: Optional[str] = typer.Option(
        None, "--before",
        help="Delete entries cached before this date (YYYY-MM-DD or ISO datetime)",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
    db: Optional[str] = typer.Option(None, "--db"),
) -> None:
    """Delete cache entries matching the given filters."""
    from bsos.cli.db_context import open_db
    from bsos.persistence.models import LLMResponseCacheRow

    if not entity and not model and not before:
        typer.echo(
            "No filters specified. To clear the entire cache pass --yes with at least one filter, "
            "or use: bsos cache clear --before 9999-01-01 --yes",
            err=True,
        )
        raise typer.Exit(1)

    before_dt: Optional[datetime] = None
    if before:
        try:
            before_dt = datetime.fromisoformat(before)
            if before_dt.tzinfo is None:
                before_dt = before_dt.replace(tzinfo=timezone.utc)
        except ValueError:
            typer.echo(f"Cannot parse date '{before}'. Use YYYY-MM-DD or ISO datetime.", err=True)
            raise typer.Exit(1)

    _, session = open_db(db)
    with session:
        stmt = select(LLMResponseCacheRow)
        if model:
            stmt = stmt.where(LLMResponseCacheRow.model == model)
        if entity:
            stmt = stmt.where(LLMResponseCacheRow.entity_name == entity)
        if before_dt:
            stmt = stmt.where(LLMResponseCacheRow.cached_at < before_dt)
        rows = session.exec(stmt).all()

        if not rows:
            typer.echo("No matching cache entries.")
            return

        typer.echo(f"Found {len(rows)} matching cache entries.")

        if not yes:
            confirm = typer.confirm("Delete these entries?")
            if not confirm:
                typer.echo("Aborted.")
                return

        for r in rows:
            session.delete(r)
        session.commit()

    typer.echo(f"Deleted {len(rows)} cache entries.")
