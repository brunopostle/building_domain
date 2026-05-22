"""bsos status command."""
import json
from sqlmodel import select, func
from bsos.persistence.models import (
    EntityRow, AssertionRow, PassProgressRow, LLMResponseCacheRow,
    ConstraintRow, PatternRow, ForceRow, AntiPatternRow,
    ProcessRelationRow, SpatialRelationRow,
    PendingPredicateRow, PendingSpatialRelationTypeRow,
    PendingForceRefRow, PendingPatternRefRow, PendingEntityRefRow,
)

ORIGIN_PRIORITY = {"physical": 0, "engineering": 1, "architectural": 2, "cultural": 3}

_PHASE2_MODELS = [
    ("constraints", ConstraintRow),
    ("patterns", PatternRow),
    ("forces", ForceRow),
    ("antipatterns", AntiPatternRow),
    ("process_relations", ProcessRelationRow),
    ("spatial_relations", SpatialRelationRow),
]


def _count_by_status(session, model, statuses):
    return {
        s: session.exec(
            select(func.count()).where(model.status == s)  # type: ignore[attr-defined]
        ).one()
        for s in statuses
    }


def _count_all(session, model) -> int:
    return session.exec(select(func.count()).select_from(model)).one()  # type: ignore[arg-type]


def run_status(session, json_out: bool) -> None:
    entity_counts = _count_by_status(session, EntityRow, ["proposed", "accepted", "deprecated"])
    assertion_counts = _count_by_status(
        session, AssertionRow, ["proposed", "accepted", "conflicted", "deprecated"]
    )

    phase2_counts: dict[str, int] = {}
    for label, model in _PHASE2_MODELS:
        phase2_counts[label] = _count_all(session, model)

    pending_predicates = _count_all(session, PendingPredicateRow)
    pending_spatial_types = _count_all(session, PendingSpatialRelationTypeRow)
    pending_force_refs = _count_all(session, PendingForceRefRow)
    pending_pattern_refs = _count_all(session, PendingPatternRefRow)
    pending_entity_refs = _count_all(session, PendingEntityRefRow)

    pass_rows = session.exec(select(PassProgressRow)).all()
    passes_by_model: dict[str, set[str]] = {}
    skipped_count = 0
    for row in pass_rows:
        if row.status == "skipped":
            skipped_count += 1
        else:
            passes_by_model.setdefault(row.model, set()).add(row.pass_number)

    cache_count = session.exec(select(func.count(LLMResponseCacheRow.model))).one()

    data = {
        "entities": {**entity_counts, "total": sum(entity_counts.values())},
        "assertions": {**assertion_counts, "total": sum(assertion_counts.values())},
        "phase2": phase2_counts,
        "pending": {
            "predicates": pending_predicates,
            "spatial_relation_types": pending_spatial_types,
            "force_refs": pending_force_refs,
            "pattern_refs": pending_pattern_refs,
            "entity_refs": pending_entity_refs,
        },
        "passes_completed": {m: sorted(ps) for m, ps in passes_by_model.items()},
        "entities_skipped_pass_failures": skipped_count,
        "llm_response_cache_entries": cache_count,
    }

    if json_out:
        print(json.dumps(data, indent=2))
        return

    # Text table — Phase 1 rows
    p1_rows_data = [
        ("entities", entity_counts.get("proposed", 0), entity_counts.get("accepted", 0),
         "—", entity_counts.get("deprecated", 0), sum(entity_counts.values())),
        ("assertions", assertion_counts.get("proposed", 0), assertion_counts.get("accepted", 0),
         assertion_counts.get("conflicted", 0), assertion_counts.get("deprecated", 0),
         sum(assertion_counts.values())),
    ]
    # Phase 2 rows (total only; no status breakdown yet)
    p2_rows_data = [
        (label, phase2_counts[label], "—", "—", "—", phase2_counts[label])
        for label, _ in _PHASE2_MODELS
    ]
    all_rows = p1_rows_data + p2_rows_data

    headers = ["Item Type", "proposed", "accepted", "conflicted", "deprecated", "Total"]
    col_widths = [max(len(h), max(len(str(r[i])) for r in all_rows)) for i, h in enumerate(headers)]
    sep = "  "
    header_line = sep.join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    print(header_line)
    print("-" * len(header_line))
    for row in all_rows:
        print(sep.join(str(row[i]).ljust(col_widths[i]) for i in range(len(headers))))

    print("")
    if passes_by_model:
        for model, passes in passes_by_model.items():
            print(f"Passes completed ({model}): {', '.join(sorted(passes))}")
    else:
        print("Passes completed: none")
    print(f"Entities skipped (pass failures): {skipped_count}")
    print(f"LLM response cache: {cache_count} entries")

    print("")
    print(f"Pending predicates awaiting review: {pending_predicates}")
    print(f"Pending spatial relation types awaiting review: {pending_spatial_types}")
    if pending_force_refs or pending_pattern_refs or pending_entity_refs:
        print(
            f"Unresolved refs — force: {pending_force_refs} | "
            f"pattern: {pending_pattern_refs} | entity: {pending_entity_refs}"
        )
