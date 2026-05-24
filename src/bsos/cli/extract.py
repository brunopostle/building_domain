"""bsos extract command."""
import sys
from pathlib import Path
import typer
from sqlmodel import Session

app = typer.Typer()


@app.callback(invoke_without_command=True)
def extract(
    seed: str = typer.Option(None, "--seed", help="Free-text domain description or path to concept list file"),
    seed_apl: bool = typer.Option(False, "--seed-apl", help="Augment Pass 1 seeds with Alexander's 253 pattern names from data/apl_patterns.json"),
    models: str = typer.Option(None, "--models", help="Comma-separated LLM model identifiers"),
    passes: str = typer.Option(None, "--passes", help="Comma-separated pass numbers (default: all)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without writing to database"),
    db: str = typer.Option(None, "--db"),
    framings: int = typer.Option(3, "--framings", help="Number of prompt framings per entity in Pass 3 (fewer = faster/cheaper, default: 3)"),
    workers: int = typer.Option(4, "--workers", help="Parallel workers for passes 3-9 (fewer = less rate-limit pressure, default: 4)"),
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

    # Load APL pattern names if requested
    apl_pattern_names: list[str] | None = None
    if seed_apl:
        import json
        apl_path = Path(__file__).parent.parent.parent.parent / "data" / "apl_patterns.json"
        if not apl_path.exists():
            typer.echo(f"--seed-apl: {apl_path} not found", err=True)
            raise typer.Exit(1)
        raw = json.loads(apl_path.read_text())
        apl_pattern_names = [" ".join(w.capitalize() for w in p["name"].split()) for p in raw]
        typer.echo(f"Loaded {len(apl_pattern_names)} APL pattern names for Pass 1 seeding.")

    _LLM_PASSES = {"1", "3", "4", "5", "6", "7", "8", "9", "11", "12"}
    _needs_llm = requested_passes is None or bool(_LLM_PASSES.intersection(requested_passes))

    with ExtractionLock(db_path):
        for model_id in model_list:
            cache = LLMResponseCache(db_path)
            provider = make_provider(model_id, cache=cache) if _needs_llm else None

            if requested_passes is None or "1" in requested_passes:
                engine2, session2 = open_db(db)
                with session2:
                    if dry_run:
                        concepts = run_pass1(
                            session2, provider, "__dry_run__",
                            seed=seed_text, seed_is_file_contents=seed_is_file,
                            apl_patterns=apl_pattern_names, dry_run=True,
                        )
                        typer.echo(f"Pass 1 (dry-run): {len(concepts)} concepts would be written")
                    else:
                        run_id = start_run(session2, [model_id], ["1"], seed_text)
                        run_pass1(
                            session2, provider, run_id,
                            seed=seed_text, seed_is_file_contents=seed_is_file,
                            apl_patterns=apl_pattern_names,
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
                    result3 = run_pass3(engine3, provider, "__dry_run__", dry_run=True, n_framings=framings, max_workers=workers)
                    typer.echo(
                        f"Pass 3 (dry-run): {result3['entities_processed']} entities would be processed"
                    )
                else:
                    run_id = start_run(
                        Session(engine3), [model_id], ["3"], seed_text
                    )
                    result3 = run_pass3(engine3, provider, run_id, n_framings=framings, max_workers=workers)
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
                    result4 = run_pass4(engine4, provider, "__dry_run__", dry_run=True, max_workers=workers)
                    typer.echo(
                        f"Pass 4 (dry-run): {result4['entities_processed']} entities would be processed"
                    )
                else:
                    run_id = start_run(
                        Session(engine4), [model_id], ["4"], seed_text
                    )
                    result4 = run_pass4(engine4, provider, run_id, max_workers=workers)
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
                    result5 = run_pass5(engine5, provider, "__dry_run__", dry_run=True, max_workers=workers)
                    typer.echo(
                        f"Pass 5 (dry-run): {result5['entities_processed']} entities would be processed"
                    )
                else:
                    run_id = start_run(
                        Session(engine5), [model_id], ["5"], seed_text
                    )
                    result5 = run_pass5(engine5, provider, run_id, max_workers=workers)
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
                    result6 = run_pass6(engine6, provider, "__dry_run__", dry_run=True, max_workers=workers)
                    typer.echo(
                        f"Pass 6 (dry-run): {result6['entities_processed']} entities would be processed"
                    )
                else:
                    run_id = start_run(
                        Session(engine6), [model_id], ["6"], seed_text
                    )
                    result6 = run_pass6(engine6, provider, run_id, max_workers=workers)
                    complete_run(Session(engine6), run_id)
                    typer.echo(
                        f"Pass 6 complete for model {model_id}: "
                        f"{result6['constraints_written']} constraints written."
                    )

        if requested_passes is None or "7" in requested_passes:
            from bsos.pipeline.pass7 import run_pass7
            from bsos.persistence.database import create_db_engine
            from bsos.cli.db_context import resolve_db_path
            engine7 = create_db_engine(resolve_db_path(db))
            for model_id in model_list:
                cache = LLMResponseCache(db_path)
                provider = make_provider(model_id, cache=cache)
                if dry_run:
                    result7 = run_pass7(engine7, provider, "__dry_run__", dry_run=True, max_workers=workers)
                    typer.echo(
                        f"Pass 7 (dry-run): {result7['entities_processed']} entities would be processed"
                    )
                else:
                    run_id = start_run(
                        Session(engine7), [model_id], ["7"], seed_text
                    )
                    result7 = run_pass7(engine7, provider, run_id, max_workers=workers)
                    complete_run(Session(engine7), run_id)
                    typer.echo(
                        f"Pass 7 complete for model {model_id}: "
                        f"{result7['anti_patterns_written']} anti-patterns written."
                    )

        if requested_passes is None or "8" in requested_passes:
            from bsos.pipeline.pass8 import run_pass8
            from bsos.persistence.database import create_db_engine
            from bsos.cli.db_context import resolve_db_path
            engine8 = create_db_engine(resolve_db_path(db))
            for model_id in model_list:
                cache = LLMResponseCache(db_path)
                provider = make_provider(model_id, cache=cache)
                if dry_run:
                    result8 = run_pass8(engine8, provider, "__dry_run__", dry_run=True, max_workers=workers)
                    typer.echo(
                        f"Pass 8 (dry-run): {result8['entities_processed']} entities would be processed"
                    )
                else:
                    run_id = start_run(
                        Session(engine8), [model_id], ["8"], seed_text
                    )
                    result8 = run_pass8(engine8, provider, run_id, max_workers=workers)
                    complete_run(Session(engine8), run_id)
                    typer.echo(
                        f"Pass 8 complete for model {model_id}: "
                        f"{result8['patterns_written']} patterns written."
                    )

        if requested_passes is None or "9" in requested_passes:
            from bsos.pipeline.pass9 import run_pass9
            from bsos.persistence.database import create_db_engine
            from bsos.cli.db_context import resolve_db_path
            engine9 = create_db_engine(resolve_db_path(db))
            for model_id in model_list:
                cache = LLMResponseCache(db_path)
                provider = make_provider(model_id, cache=cache)
                if dry_run:
                    result9 = run_pass9(engine9, provider, "__dry_run__", dry_run=True, max_workers=workers)
                    typer.echo(
                        f"Pass 9 (dry-run): {result9['entities_processed']} entities would be processed"
                    )
                else:
                    run_id = start_run(
                        Session(engine9), [model_id], ["9"], seed_text
                    )
                    result9 = run_pass9(engine9, provider, run_id, max_workers=workers)
                    complete_run(Session(engine9), run_id)
                    msg = f"Pass 9 complete for model {model_id}: {result9['forces_written']} forces written"
                    if result9["validation_failures"]:
                        msg += f" ({result9['validation_failures']} validation failures)"
                    if result9["unresolved_refs"]:
                        msg += f" ({result9['unresolved_refs']} unresolved refs)"
                    typer.echo(msg + ".")

        if requested_passes is None or "10" in requested_passes:
            from bsos.persistence.database import create_db_engine
            from bsos.cli.db_context import resolve_db_path
            from bsos.cli.normalize import run_normalize
            engine10 = create_db_engine(resolve_db_path(db))
            run_normalize(engine10, model_list, reembed=False, dry_run=dry_run)

        if requested_passes is None or "12" in requested_passes:
            from bsos.pipeline.pass12 import run_pass12
            from bsos.persistence.database import create_db_engine
            from bsos.cli.db_context import resolve_db_path
            engine12 = create_db_engine(resolve_db_path(db))
            for model_id in model_list:
                cache = LLMResponseCache(db_path)
                provider = make_provider(model_id, cache=cache)
                if dry_run:
                    result12 = run_pass12(engine12, provider, "__dry_run__", dry_run=True)
                    typer.echo(
                        f"Pass 12 (dry-run): {result12['chunks_processed']} chunks would be processed"
                    )
                else:
                    run_id = start_run(
                        Session(engine12), [model_id], ["12"], seed_text
                    )
                    result12 = run_pass12(engine12, provider, run_id)
                    complete_run(Session(engine12), run_id)
                    typer.echo(
                        f"Pass 12 complete for model {model_id}: "
                        f"{result12['entities_written']} IFC entities, "
                        f"{result12['assertions_written']} assertions, "
                        f"{result12['constraints_written']} constraints written."
                    )

        if requested_passes is None or "11" in requested_passes:
            from bsos.pipeline.pass11 import run_pass11
            from bsos.persistence.database import create_db_engine
            from bsos.cli.db_context import resolve_db_path
            engine11 = create_db_engine(resolve_db_path(db))
            providers11 = [make_provider(m, cache=LLMResponseCache(db_path)) for m in model_list]
            if dry_run:
                result11 = run_pass11(engine11, providers11, "__dry_run__", dry_run=True)
                typer.echo(
                    f"Pass 11 (dry-run): {result11.get('findings_collected', 0)} findings collected "
                    f"from {result11.get('assertion_count', '?')} assertions"
                )
            else:
                run_id11 = start_run(Session(engine11), model_list, ["11"], seed_text)
                result11 = run_pass11(engine11, providers11, run_id11)
                complete_run(Session(engine11), run_id11)
                typer.echo(
                    f"Pass 11 complete: {result11.get('findings_applied', 0)} findings applied "
                    f"({result11.get('exceptions_appended', 0)} exceptions, "
                    f"{result11.get('conditions_appended', 0)} conditions, "
                    f"{result11.get('conflicted', 0)} conflicted, "
                    f"{result11.get('deferred', 0)} deferred)."
                )
