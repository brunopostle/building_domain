"""bsos extract command."""
import sys
from pathlib import Path
import typer
from sqlmodel import Session

app = typer.Typer()


@app.callback(invoke_without_command=True)
def extract(
    seed: str = typer.Option(None, "--seed", help="Free-text domain description or path to concept list file"),
    models: str = typer.Option(None, "--models", help="Comma-separated LLM model identifiers"),
    passes: str = typer.Option(None, "--passes", help="Comma-separated pass numbers (default: all)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without writing to database"),
    db: str = typer.Option(None, "--db"),
) -> None:
    """Run the extraction pipeline."""
    from bsos.cli.db_context import open_db
    from bsos.config import get_config, set_config
    from bsos.pipeline.lock import ExtractionLock
    from bsos.pipeline.run import start_run, complete_run
    from bsos.pipeline.pass1 import run_pass1
    from bsos.llm import make_provider
    from bsos.llm.cache import LLMResponseCache

    engine, session = open_db(db)
    db_path = engine.url.database

    # Resolve model IDs
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

    # Parse pass list
    requested_passes = [p.strip() for p in passes.split(",")] if passes else None

    # Resolve seed
    seed_text: str | None = None
    seed_is_file = False
    if seed:
        seed_path = Path(seed)
        if seed_path.exists() and seed_path.is_file():
            seed_text = seed_path.read_text()
            seed_is_file = True
        else:
            seed_text = seed

    with ExtractionLock(db_path):
        for model_id in model_list:
            cache = LLMResponseCache(db_path)
            provider = make_provider(model_id, cache=cache)

            if requested_passes is None or "1" in requested_passes:
                engine2, session2 = open_db(db)
                with session2:
                    if dry_run:
                        concepts = run_pass1(
                            session2, provider, "__dry_run__",
                            seed=seed_text, seed_is_file_contents=seed_is_file, dry_run=True,
                        )
                        typer.echo(f"Pass 1 (dry-run): {len(concepts)} concepts would be written")
                    else:
                        run_id = start_run(session2, [model_id], ["1"], seed_text)
                        run_pass1(
                            session2, provider, run_id,
                            seed=seed_text, seed_is_file_contents=seed_is_file,
                        )
                        complete_run(session2, run_id)
                        typer.echo(f"Pass 1 complete for model {model_id}.")

        if requested_passes is None or "2" in requested_passes:
            from bsos.pipeline.pass2 import run_pass2, EMBEDDING_MODEL
            engine2, session2 = open_db(db)
            with session2:
                confirmed = get_config(session2, "embedding_model_confirmed")
                if not confirmed:
                    typer.echo(
                        f"Pass 2 requires downloading the '{EMBEDDING_MODEL}' "
                        "sentence-transformers model (~420 MB)."
                    )
                    if not typer.confirm("Download now?", default=True):
                        typer.echo("Skipping Pass 2.", err=True)
                    else:
                        set_config(session2, "embedding_model_confirmed", "true")
                        session2.commit()
                        confirmed = "true"

                if confirmed:
                    if dry_run:
                        result = run_pass2(session2, "__dry_run__", dry_run=True)
                        typer.echo(
                            f"Pass 2 (dry-run): {result['clusters_found']} clusters found, "
                            f"{result['entities_merged']} entities would be merged"
                        )
                    else:
                        run_id = start_run(session2, [], ["2"], seed_text)
                        result = run_pass2(session2, run_id)
                        complete_run(session2, run_id)
                        typer.echo(
                            f"Pass 2 complete: {result['entities_merged']} entities merged."
                        )

        if requested_passes is None or "3" in requested_passes:
            from bsos.pipeline.pass3 import run_pass3
            from bsos.persistence.database import create_db_engine
            from bsos.cli.db_context import resolve_db_path
            engine3 = create_db_engine(resolve_db_path(db))
            for model_id in model_list:
                cache = LLMResponseCache(db_path)
                provider = make_provider(model_id, cache=cache)
                if dry_run:
                    result3 = run_pass3(engine3, provider, "__dry_run__", dry_run=True)
                    typer.echo(
                        f"Pass 3 (dry-run): {result3['entities_processed']} entities would be processed"
                    )
                else:
                    run_id = start_run(
                        Session(engine3), [model_id], ["3"], seed_text
                    )
                    result3 = run_pass3(engine3, provider, run_id)
                    complete_run(Session(engine3), run_id)
                    typer.echo(
                        f"Pass 3 complete for model {model_id}: "
                        f"{result3['assertions_written']} assertions written."
                    )

        if requested_passes is None or "4" in requested_passes:
            from bsos.pipeline.pass4 import run_pass4
            from bsos.persistence.database import create_db_engine
            from bsos.cli.db_context import resolve_db_path
            engine4 = create_db_engine(resolve_db_path(db))
            for model_id in model_list:
                cache = LLMResponseCache(db_path)
                provider = make_provider(model_id, cache=cache)
                if dry_run:
                    result4 = run_pass4(engine4, provider, "__dry_run__", dry_run=True)
                    typer.echo(
                        f"Pass 4 (dry-run): {result4['entities_processed']} entities would be processed"
                    )
                else:
                    run_id = start_run(
                        Session(engine4), [model_id], ["4"], seed_text
                    )
                    result4 = run_pass4(engine4, provider, run_id)
                    complete_run(Session(engine4), run_id)
                    typer.echo(
                        f"Pass 4 complete for model {model_id}: "
                        f"{result4['relations_written']} spatial relations written."
                    )

        if requested_passes is None or "5" in requested_passes:
            from bsos.pipeline.pass5 import run_pass5
            from bsos.persistence.database import create_db_engine
            from bsos.cli.db_context import resolve_db_path
            engine5 = create_db_engine(resolve_db_path(db))
            for model_id in model_list:
                cache = LLMResponseCache(db_path)
                provider = make_provider(model_id, cache=cache)
                if dry_run:
                    result5 = run_pass5(engine5, provider, "__dry_run__", dry_run=True)
                    typer.echo(
                        f"Pass 5 (dry-run): {result5['entities_processed']} entities would be processed"
                    )
                else:
                    run_id = start_run(
                        Session(engine5), [model_id], ["5"], seed_text
                    )
                    result5 = run_pass5(engine5, provider, run_id)
                    complete_run(Session(engine5), run_id)
                    typer.echo(
                        f"Pass 5 complete for model {model_id}: "
                        f"{result5['relations_written']} process relations written"
                        + (f" ({result5['hard_constraint_divergences']} divergences)" if result5['hard_constraint_divergences'] else "")
                        + "."
                    )

        if requested_passes is None or "6" in requested_passes:
            from bsos.pipeline.pass6 import run_pass6
            from bsos.persistence.database import create_db_engine
            from bsos.cli.db_context import resolve_db_path
            engine6 = create_db_engine(resolve_db_path(db))
            for model_id in model_list:
                cache = LLMResponseCache(db_path)
                provider = make_provider(model_id, cache=cache)
                if dry_run:
                    result6 = run_pass6(engine6, provider, "__dry_run__", dry_run=True)
                    typer.echo(
                        f"Pass 6 (dry-run): {result6['entities_processed']} entities would be processed"
                    )
                else:
                    run_id = start_run(
                        Session(engine6), [model_id], ["6"], seed_text
                    )
                    result6 = run_pass6(engine6, provider, run_id)
                    complete_run(Session(engine6), run_id)
                    typer.echo(
                        f"Pass 6 complete for model {model_id}: "
                        f"{result6['constraints_written']} constraints written."
                    )
