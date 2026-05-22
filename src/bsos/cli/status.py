"""bsos status command."""
import json
from sqlmodel import select, func
from bsos.persistence.models import (
    EntityRow, AssertionRow, PassProgressRow, LLMResponseCacheRow,
)

ORIGIN_PRIORITY = {"physical": 0, "engineering": 1, "architectural": 2, "cultural": 3}


def _count_by_status(session, model, statuses):
    return {
        s: session.exec(
            select(func.count()).where(model.status == s)  # type: ignore[attr-defined]
        ).one()
        for s in statuses
    }


def run_status(session, json_out: bool) -> None:
    entity_counts = _count_by_status(session, EntityRow, ["proposed", "accepted", "deprecated"])
    assertion_counts = _count_by_status(
        session, AssertionRow, ["proposed", "accepted", "conflicted", "deprecated"]
    )

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
        "passes_completed": {m: sorted(ps) for m, ps in passes_by_model.items()},
        "entities_skipped_pass_failures": skipped_count,
        "llm_response_cache_entries": cache_count,
    }

    if json_out:
        print(json.dumps(data, indent=2))
        return

    headers = ["Item Type", "proposed", "accepted", "conflicted", "deprecated", "Total"]
    rows_data = [
        ("entities", entity_counts.get("proposed", 0), entity_counts.get("accepted", 0),
         "—", entity_counts.get("deprecated", 0), sum(entity_counts.values())),
        ("assertions", assertion_counts.get("proposed", 0), assertion_counts.get("accepted", 0),
         assertion_counts.get("conflicted", 0), assertion_counts.get("deprecated", 0),
         sum(assertion_counts.values())),
    ]

    col_widths = [max(len(h), max(len(str(r[i])) for r in rows_data)) for i, h in enumerate(headers)]
    sep = "  "
    header_line = sep.join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    print(header_line)
    print("-" * len(header_line))
    for row in rows_data:
        print(sep.join(str(row[i]).ljust(col_widths[i]) for i in range(len(headers))))

    print("")
    if passes_by_model:
        for model, passes in passes_by_model.items():
            print(f"Passes completed ({model}): {', '.join(sorted(passes))}")
    else:
        print("Passes completed: none")
    print(f"Entities skipped (pass failures): {skipped_count}")
    print(f"LLM response cache: {cache_count} entries")
