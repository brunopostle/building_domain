"""bsos history command."""
import json
import sys
from datetime import datetime
from typing import Optional
from sqlmodel import select
from bsos.persistence.models import (
    ProvenanceLogRow,
    EntityRow, AssertionRow,
    ConstraintRow, PatternRow, ForceRow, AntiPatternRow,
    ProcessRelationRow, SpatialRelationRow,
)

_ITEM_TABLES = [
    EntityRow, AssertionRow,
    ConstraintRow, PatternRow, ForceRow, AntiPatternRow,
    ProcessRelationRow, SpatialRelationRow,
]


def _lookup_item(session, item_id: str):
    """Return (created_at, source_model) for the given item UUID, or None."""
    for model in _ITEM_TABLES:
        row = session.get(model, item_id)
        if row is not None:
            return row.created_at, row.source_model
    return None


def run_history(session, item_id: str, json_out: bool) -> None:
    meta = _lookup_item(session, item_id)
    if meta is None:
        print(f"No history found for {item_id}", file=sys.stderr)
        sys.exit(1)

    created_at, source_model = meta

    log_rows = session.exec(
        select(ProvenanceLogRow)
        .where(ProvenanceLogRow.item_id == item_id)
        .order_by(ProvenanceLogRow.changed_at)
    ).all()

    initial = {
        "old_status": None,
        "new_status": "proposed",
        "changed_at": created_at.isoformat(),
        "changed_by": source_model,
        "label": "(initial)",
    }

    transitions = [initial] + [
        {
            "old_status": r.old_status,
            "new_status": r.new_status,
            "changed_at": r.changed_at.isoformat(),
            "changed_by": r.changed_by,
        }
        for r in log_rows
    ]

    if json_out:
        print(json.dumps(transitions, indent=2))
        return

    for entry in transitions:
        old = entry["old_status"] or "None"
        new = entry["new_status"]
        when = entry["changed_at"]
        by = entry["changed_by"]
        label = entry.get("label", "")
        suffix = f"  {label}" if label else ""
        print(f"{when}  {old} → {new}  (by {by}){suffix}")
