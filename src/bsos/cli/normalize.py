"""bsos normalize command — Pass 10 orchestration with sub-pass resume."""
import typer
from sqlmodel import Session, func, select

app = typer.Typer()

_REEMBED_THRESHOLD_KEYS = [
    "pass10a similarity threshold (0.85)",
    "pass10b auto-map threshold (0.85)",
    "pass10b ambiguous band low (0.60)",
    "pass10c cluster distance threshold (0.25)",
]


def _check_assertions(session: Session) -> int:
    from bsos.persistence.models import AssertionRow
    count = session.exec(select(func.count()).select_from(AssertionRow)).one()
    return count


def _delete_embeddings(engine) -> int:
    from bsos.persistence.models import EmbeddingRow
    with Session(engine) as s:
        rows = s.exec(select(EmbeddingRow)).all()
        count = len(rows)
        for r in rows:
            s.delete(r)
        s.commit()
    return count


def _sub_pass_status(session: Session, embedding_model: str) -> dict[str, bool]:
    from bsos.persistence.models import PassProgressRow
    result = {}
    for sub in ("10a", "10b", "10c"):
        row = session.get(PassProgressRow, (sub, "__global__", embedding_model))
        result[sub] = row is not None and row.status == "completed"
    return result


def run_normalize(
    engine,
    model_list: list[str],
    reembed: bool,
    dry_run: bool,
    embedding_model: str = "all-mpnet-base-v2",
) -> None:
    """Shared normalize logic used by both bsos normalize and bsos extract --passes 10."""
    from bsos.llm import make_provider
    from bsos.normalization.pass10a import run_pass10a
    from bsos.normalization.pass10b import run_pass10b
    from bsos.normalization.pass10c import run_pass10c

    with Session(engine) as session:
        assertion_count = _check_assertions(session)
    if assertion_count == 0:
        typer.echo(
            "No assertions found. Run 'bsos extract' to populate the knowledge base before normalizing.",
            err=True,
        )
        raise typer.Exit(1)

    if reembed:
        typer.echo("Warning: --reembed deletes all cached embeddings. The following thresholds")
        typer.echo("will affect re-run results and may differ from original runs:")
        for key in _REEMBED_THRESHOLD_KEYS:
            typer.echo(f"  • {key}")
        # Also clear pass_progress for 10a/10b/10c so they re-run from scratch.
        with Session(engine) as session:
            from bsos.persistence.models import PassProgressRow
            for sub in ("10a", "10b", "10c"):
                row = session.get(PassProgressRow, (sub, "__global__", embedding_model))
                if row:
                    session.delete(row)
            session.commit()
        deleted = _delete_embeddings(engine)
        typer.echo(f"Deleted {deleted} embedding rows.")

    with Session(engine) as session:
        status = _sub_pass_status(session, embedding_model)

    # Pass 10a — ref resolution (embedding-only, no LLM needed)
    if status.get("10a"):
        typer.echo("Pass 10a: already completed, skipping.")
    else:
        typer.echo("Pass 10a: resolving force/pattern/entity refs…")
        result = run_pass10a(engine, embedding_model=embedding_model, dry_run=dry_run)
        if dry_run:
            typer.echo(
                f"Pass 10a (dry-run): {result.get('patterns_with_force_descriptions', 0)} patterns "
                f"with unresolved force refs, {result.get('pending_entity_refs', 0)} pending entity refs"
            )
        else:
            typer.echo(
                f"Pass 10a complete: "
                f"{result.get('force_descriptions_resolved', 0)} force refs resolved, "
                f"{result.get('pattern_names_resolved', 0)} pattern refs resolved, "
                f"{result.get('entity_refs_resolved', 0)} entity refs resolved."
            )

    # Pass 10b — predicate stabilization
    if not dry_run:
        with Session(engine) as session:
            status = _sub_pass_status(session, embedding_model)
    if status.get("10b"):
        typer.echo("Pass 10b: already completed, skipping.")
    else:
        typer.echo("Pass 10b: stabilizing predicates…")
        provider_a = (make_provider(model_list[0]) if model_list else None) if not dry_run else None
        result = run_pass10b(
            engine,
            embedding_model=embedding_model,
            provider=provider_a,
            dry_run=dry_run,
        )
        if dry_run:
            typer.echo(
                f"Pass 10b (dry-run): {result.get('non_core_predicate_count', 0)} non-core predicates found"
            )
        else:
            typer.echo(
                f"Pass 10b complete: "
                f"{result.get('auto_mapped', 0)} auto-mapped, "
                f"{result.get('phase2_processed', 0)} LLM-disambiguated, "
                f"{result.get('pending_predicates', 0)} pending review."
            )

    # Pass 10c — abstraction synthesis
    if not dry_run:
        with Session(engine) as session:
            status = _sub_pass_status(session, embedding_model)
    if status.get("10c"):
        typer.echo("Pass 10c: already completed, skipping.")
    else:
        typer.echo("Pass 10c: synthesising abstractions…")
        if not dry_run and not model_list:
            typer.echo(
                "Pass 10c requires at least one LLM model. "
                "Pass --models or set default_llm_model in config.",
                err=True,
            )
            raise typer.Exit(1)
        if dry_run:
            provider_a, provider_b = None, None
        else:
            provider_a = make_provider(model_list[0])
            provider_b = make_provider(model_list[1]) if len(model_list) >= 2 else None
        result = run_pass10c(
            engine,
            provider_a=provider_a,
            provider_b=provider_b,
            embedding_model=embedding_model,
            dry_run=dry_run,
        )
        if dry_run:
            typer.echo(
                f"Pass 10c (dry-run): {result.get('eligible_subject_groups', 0)} eligible subject groups"
            )
        else:
            cap_note = " (queue cap reached)" if result.get("cap_reached") else ""
            typer.echo(
                f"Pass 10c complete: "
                f"{result.get('nodes_created', 0)} abstraction nodes created"
                f"{cap_note}."
            )


@app.callback(invoke_without_command=True)
def normalize(
    reembed: bool = typer.Option(False, "--reembed", help="Delete all embeddings and recompute from scratch"),
    models: str = typer.Option(None, "--models", help="Comma-separated LLM model IDs (first=synthesis, second=validation)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without writing"),
    db: str = typer.Option(None, "--db"),
) -> None:
    """Run normalization passes (10a → 10b → 10c) with sub-pass resume."""
    from bsos.cli.db_context import open_db, resolve_db_path
    from bsos.persistence.database import create_db_engine
    from bsos.config import get_config

    engine, session = open_db(db)

    with session:
        model_ids_str = models or get_config(session, "default_llm_model")

    if not model_ids_str:
        typer.echo(
            "No LLM model specified. Pass --models or run: "
            "bsos config set default_llm_model <model-id>",
            err=True,
        )
        raise typer.Exit(1)

    model_list = [m.strip() for m in model_ids_str.split(",") if m.strip()]
    db_path = resolve_db_path(db)
    engine = create_db_engine(db_path)

    run_normalize(engine, model_list, reembed=reembed, dry_run=dry_run)
