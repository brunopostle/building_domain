"""bsos validate command."""
import json
import typer
from sqlmodel import Session, select

from bsos.persistence.models import (
    AssertionRow, ConstraintRow, EntityRow, SpatialRelationRow,
)

app = typer.Typer()


@app.command("topology")
def validate_topology(db: str = typer.Option(None, "--db")) -> None:
    """Check space reachability from entrance nodes via accessible_from edges."""
    import networkx as nx
    from bsos.cli.db_context import open_db
    _, session = open_db(db)

    with session:
        spaces = session.exec(
            select(EntityRow).where(
                EntityRow.entity_type == "space",
                EntityRow.status != "merged",
            )
        ).all()
        space_ids = {e.id for e in spaces}
        space_names = {e.id: e.name for e in spaces}

        if not space_ids:
            typer.echo("No space entities found.")
            return

        entrance_ids = {e.id for e in spaces if e.is_entrance}
        if not entrance_ids:
            typer.echo(
                "Warning: no entrance nodes defined. "
                "Use 'bsos curate set-entrance <entity>' to mark entrance spaces.",
                err=True,
            )
            return

        accessible_rows = session.exec(
            select(SpatialRelationRow).where(
                SpatialRelationRow.relation == "accessible_from",
            )
        ).all()

    g = nx.DiGraph()
    for s in space_ids:
        g.add_node(s)
    for row in accessible_rows:
        if row.subject_id in space_ids and row.object_id in space_ids:
            g.add_edge(row.object_id, row.subject_id)  # object → subject (accessible_from)

    reachable: set[str] = set()
    for entrance_id in entrance_ids:
        if entrance_id in g:
            reachable.update(nx.descendants(g, entrance_id))
            reachable.add(entrance_id)

    unreachable = space_ids - reachable
    if not unreachable:
        typer.echo(f"All {len(space_ids)} spaces are reachable from entrance nodes.")
        return

    typer.echo(f"Topology check: {len(unreachable)} unreachable space(s):\n")
    for sid in sorted(unreachable, key=lambda i: space_names.get(i, i)):
        typer.echo(f"  {space_names.get(sid, sid)}")
    raise typer.Exit(1)


@app.command("constraints")
def validate_constraints(
    entity: str = typer.Argument(..., help="Entity name to validate constraints for"),
    db: str = typer.Option(None, "--db"),
    model: str = typer.Option(None, "--model", help="LLM model ID (overrides config)"),
) -> None:
    """Check entity constraints against accepted assertions using LLM classification."""
    from pydantic import BaseModel
    from bsos.cli.db_context import open_db
    from bsos.config import get_config
    from bsos.llm import make_provider
    from bsos.mcp_server.server import resolve_entity

    _, session = open_db(db)

    class ConstraintCheckResult(BaseModel):
        violated: bool
        violating_assertion_id: str | None = None
        explanation: str | None = None

    with session:
        entity_row = resolve_entity(session, entity)
        if entity_row is None:
            typer.echo(f"Entity '{entity}' not found.", err=True)
            raise typer.Exit(1)

        constraints = session.exec(
            select(ConstraintRow).where(ConstraintRow.subject_id == entity_row.id)
        ).all()
        if not constraints:
            typer.echo(f"No constraints found for '{entity}'.")
            return

        assertions = session.exec(
            select(AssertionRow).where(
                (AssertionRow.subject_id == entity_row.id)
                | (AssertionRow.object_id == entity_row.id),
                AssertionRow.status == "accepted",
            )
        ).all()

        assertion_summaries = [
            f"[{a.id[:8]}] {a.predicate}: {a.rationale or '(no rationale)'}"
            for a in assertions
        ]
        assertions_text = "\n".join(assertion_summaries) if assertion_summaries else "(none)"

        model_id = model or get_config(session, "constraint_validation_model")

    if not model_id:
        typer.echo(
            "No model specified. Pass --model or set: bsos config set constraint_validation_model <id>",
            err=True,
        )
        raise typer.Exit(1)

    provider = make_provider(model_id)

    violations = []
    for constraint in constraints:
        prompt = (
            f"Entity: {entity_row.name}\n"
            f"Constraint ({constraint.constraint_type}): {constraint.rule}\n\n"
            f"Accepted assertions for this entity:\n{assertions_text}\n\n"
            "Does any assertion violate the constraint? "
            "Reply with violated=true/false, the violating_assertion_id if any (first 8 chars), "
            "and a brief explanation."
        )
        try:
            result = provider.extract(prompt, ConstraintCheckResult, entity_name=entity_row.name)
            if result.violated:
                violations.append((constraint, result))
        except Exception as exc:
            typer.echo(f"  Warning: LLM error for constraint '{constraint.rule}': {exc}", err=True)

    if not violations:
        typer.echo(f"No constraint violations found for '{entity}'.")
        return

    typer.echo(f"{len(violations)} violation(s) found for '{entity}':\n")
    for constraint, result in violations:
        typer.echo(f"  [{constraint.constraint_type.upper()}] {constraint.rule}")
        if result.violating_assertion_id:
            typer.echo(f"    violating assertion: {result.violating_assertion_id}")
        if result.explanation:
            typer.echo(f"    explanation: {result.explanation}")
        typer.echo("")
    raise typer.Exit(1)
