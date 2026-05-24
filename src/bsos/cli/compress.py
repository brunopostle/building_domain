"""bsos compress command: operator-triggered semantic compression."""
import uuid
import warnings
from datetime import datetime, timezone

import numpy as np
import typer
from sqlmodel import Session

from bsos.normalization.pass10c import (
    ABSTRACTION_QUEUE_CAP,
    CLUSTER_DISTANCE_THRESHOLD,
    EMBEDDING_MODEL,
    MIN_CLUSTER_SIZE,
    _SynthesisResponse,
    _assertion_text,
    _build_synthesis_prompt,
    _build_validation_prompt,
    _cluster,
    _group_by_label,
    _load_entity_names,
    _load_grouped_assertions,
)

app = typer.Typer()


@app.callback(invoke_without_command=True)
def compress(
    min_cluster_size: int = typer.Option(MIN_CLUSTER_SIZE, "--min-cluster-size", help="Minimum assertions per cluster"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Cluster and synthesize without writing records"),
    model_a: str = typer.Option(None, "--model-a", help="Synthesis model (overrides config default_llm_model)"),
    model_b: str = typer.Option(None, "--model-b", help="Adversarial validation model (overrides config constraint_validation_model)"),
    db: str = typer.Option(None, "--db"),
) -> None:
    """Semantically compress the knowledge base by synthesizing abstraction nodes."""
    from bsos.cli.db_context import open_db, resolve_db_path
    from bsos.config import get_config
    from bsos.llm import make_provider
    from bsos.models.abstraction import AbstractionNode
    from bsos.persistence.database import create_db_engine
    from bsos.persistence.repos.abstraction import AbstractionNodeRepository

    engine, session = open_db(db)

    with session:
        model_a_id = model_a or get_config(session, "default_llm_model")
        model_b_id = model_b or get_config(session, "constraint_validation_model")

    if not model_a_id:
        typer.echo(
            "No synthesis model specified. Pass --model-a or set: bsos config set default_llm_model <id>",
            err=True,
        )
        raise typer.Exit(1)

    if not model_b_id:
        typer.echo(
            "Warning: no adversarial validation model configured. "
            "Set constraint_validation_model or pass --model-b to enable validation. "
            "Abstractions will be created with status='proposed' without adversarial check.",
            err=True,
        )
    elif model_b_id == model_a_id:
        typer.echo(
            f"Warning: model-b is the same as model-a ({model_a_id}). "
            "Adversarial validation requires a distinct model — skipping validation.",
            err=True,
        )
        model_b_id = None

    provider_a = make_provider(model_a_id)
    provider_b = make_provider(model_b_id) if model_b_id else None

    db_path = resolve_db_path(db)
    full_engine = create_db_engine(db_path)

    # Queue cap check (skipped in dry-run)
    if not dry_run:
        with Session(full_engine) as s:
            repo = AbstractionNodeRepository(s)
            proposed_count = repo.count_proposed()
        if proposed_count >= ABSTRACTION_QUEUE_CAP:
            typer.echo(
                f"Abstraction queue cap reached ({proposed_count}/{ABSTRACTION_QUEUE_CAP} proposed nodes). "
                "Run 'bsos review-pending --type abstraction' to clear the queue before compressing.",
                err=True,
            )
            raise typer.Exit(1)

    with Session(full_engine) as s:
        subject_groups = _load_grouped_assertions(s)
        entity_names = _load_entity_names(s)

    eligible_groups = {
        sid: rows for sid, rows in subject_groups.items()
        if len(rows) >= min_cluster_size
    }

    if not eligible_groups:
        typer.echo("No eligible subject groups found (need ≥ {} assertions per entity).".format(min_cluster_size))
        return

    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(EMBEDDING_MODEL)

    run_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    nodes_created = 0
    nodes_rejected = 0
    clusters_processed = 0
    cap_reached = False

    for subject_id, rows in eligible_groups.items():
        if not dry_run:
            with Session(full_engine) as s:
                repo = AbstractionNodeRepository(s)
                proposed_count = repo.count_proposed()
            if proposed_count >= ABSTRACTION_QUEUE_CAP:
                typer.echo(
                    f"\nAbstraction queue cap ({ABSTRACTION_QUEUE_CAP}) reached mid-run. "
                    "Run 'bsos review-pending --type abstraction' to continue.",
                    err=True,
                )
                cap_reached = True
                break

        texts = [_assertion_text(r) for r in rows]
        vecs = np.array(st.encode(texts, show_progress_bar=False), dtype=np.float32)

        labels = _cluster(vecs, CLUSTER_DISTANCE_THRESHOLD)
        cluster_groups = _group_by_label(labels)

        for label, indices in cluster_groups.items():
            if len(indices) < min_cluster_size:
                continue

            if not dry_run:
                with Session(full_engine) as s:
                    repo = AbstractionNodeRepository(s)
                    proposed_count = repo.count_proposed()
                if proposed_count >= ABSTRACTION_QUEUE_CAP:
                    typer.echo(
                        f"\nAbstraction queue cap ({ABSTRACTION_QUEUE_CAP}) reached. "
                        "Run 'bsos review-pending --type abstraction' to continue.",
                        err=True,
                    )
                    cap_reached = True
                    break

            cluster_rows = [rows[i] for i in indices]
            clusters_processed += 1
            synthesis_prompt = _build_synthesis_prompt(cluster_rows, entity_names)

            try:
                response = provider_a.extract(synthesis_prompt, _SynthesisResponse)
            except Exception as exc:
                typer.echo(f"  Warning: synthesis failed for cluster of {len(cluster_rows)}: {exc}", err=True)
                continue

            # Adversarial validation
            accepted = True
            if provider_b is not None:
                cluster_lines = "\n".join(
                    f"  {i+1}. {r.predicate}: {r.rationale or ''}" for i, r in enumerate(cluster_rows)
                )
                val_prompt = _build_validation_prompt(cluster_lines, response.statement)
                try:
                    verdict = provider_b.classify(val_prompt, ["yes", "no"])
                    if verdict.strip().lower() == "yes":
                        accepted = False
                except Exception as exc:
                    typer.echo(f"  Warning: validation failed: {exc}", err=True)

            entity_name = entity_names.get(subject_id, subject_id)
            if dry_run:
                status_label = "ACCEPTED" if accepted else "REJECTED"
                typer.echo(
                    f"[{status_label}] {entity_name} — {len(cluster_rows)} assertions\n"
                    f"  {response.statement}"
                )
                if not accepted:
                    nodes_rejected += 1
                else:
                    nodes_created += 1
                continue

            if not accepted:
                nodes_rejected += 1
                continue

            child_ids = [r.id for r in cluster_rows]
            node = AbstractionNode(
                id=str(uuid.uuid4()),
                statement=response.statement,
                child_ids=child_ids,
                abstraction_rationale=response.abstraction_rationale,
                source_model=provider_a.model_id,
                source_prompt=synthesis_prompt,
                created_at=now,
                extraction_run_id=run_id,
                confidence=response.confidence,
                status="proposed",
                rationale=None,
                conflict_evaluated_at=None,
            )
            with Session(full_engine) as s:
                repo = AbstractionNodeRepository(s)
                repo.add(node)
                s.commit()
            nodes_created += 1

        if cap_reached:
            break

    if dry_run:
        typer.echo(
            f"\nDry run complete: {clusters_processed} clusters evaluated, "
            f"{nodes_created} would be created, {nodes_rejected} rejected by validation."
        )
    else:
        typer.echo(
            f"Compress complete: {clusters_processed} clusters evaluated, "
            f"{nodes_created} abstraction nodes created, {nodes_rejected} rejected."
        )
        if cap_reached:
            typer.echo("Queue cap reached — run 'bsos review-pending --type abstraction' to continue.")
