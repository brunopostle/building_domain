"""Shared entity-merge utility.

Merging one entity into another must repoint **every** table that references
entities, not just assertions. ``pass2._merge_cluster`` historically only
repointed ``assertions.subject_id/object_id``, which strands constraints,
antipatterns, spatial/process relations, forces and IFC pset rows on the
duplicate (which then becomes invisible because ``status='merged'`` is excluded
from all queries).

``merge_entity`` repoints all of these, deduplicates the collisions that
repointing can create (e.g. two assertions that become identical, or a
process_relations unique-constraint violation), records an alias, and marks the
duplicate ``status='merged'``. It does not commit — callers control the
transaction boundary.
"""
from __future__ import annotations

import json
from sqlmodel import Session, select

from bsos.persistence.models import (
    AntiPatternRow,
    AssertionRow,
    ConstraintRow,
    EntityAliasRow,
    EntityRow,
    ForceRow,
    IFCPropertySetRow,
    PatternRow,
    ProcessRelationRow,
    SpatialRelationRow,
)


def merge_entity(
    session: Session,
    canonical_id: str,
    duplicate_id: str,
    *,
    add_alias: bool = True,
    update_types: bool = True,
) -> dict[str, int]:
    """Merge ``duplicate_id`` into ``canonical_id``, repointing all FK tables.

    Repoints, in order, every column that references an entity id:

    - ``assertions.subject_id`` / ``object_id`` (+ subject_type/object_type)
    - ``constraints.subject_id``
    - ``patterns.subject_id``
    - ``antipatterns.subject_id``
    - ``spatial_relations.subject_id`` / ``object_id``
    - ``process_relations.predecessor_id`` / ``successor_id``
    - ``forces.affects`` (JSON list of entity ids)
    - ``ifc_pset_recommendations.entity_id``
    - ``entity_aliases.entity_id`` (aliases already pointing at the duplicate)

    After repointing, rows that have become exact duplicates of an existing
    canonical-pointing row are deleted so unique keys are not violated and the
    graph is not littered with self-relations.

    Args:
        canonical_id: entity to keep.
        duplicate_id: entity to fold in; marked ``status='merged'``.
        add_alias: record the duplicate's name as an alias on the canonical.
        update_types: rewrite ``subject_type``/``object_type`` on repointed
            assertions to the canonical entity's ``entity_type`` (used when the
            merge changes the conceptual type, e.g. a prose ifc_class folded
            into a schema seed).

    Returns:
        dict of per-table counts: keys ``repointed`` (total FK references moved)
        and ``deleted`` (duplicate/self rows removed).
    """
    if canonical_id == duplicate_id:
        raise ValueError("canonical_id and duplicate_id are the same entity")

    canonical = session.get(EntityRow, canonical_id)
    duplicate = session.get(EntityRow, duplicate_id)
    if canonical is None:
        raise ValueError(f"canonical entity {canonical_id!r} not found")
    if duplicate is None:
        raise ValueError(f"duplicate entity {duplicate_id!r} not found")

    repointed = 0
    deleted = 0

    # --- assertions (subject_id, object_id) -------------------------------
    existing_keys = {
        (a.subject_id, a.predicate, a.object_id)
        for a in session.exec(
            select(AssertionRow).where(
                (AssertionRow.subject_id == canonical_id)
                | (AssertionRow.object_id == canonical_id)
            )
        ).all()
    }
    for a in session.exec(
        select(AssertionRow).where(
            (AssertionRow.subject_id == duplicate_id)
            | (AssertionRow.object_id == duplicate_id)
        )
    ).all():
        new_subj = canonical_id if a.subject_id == duplicate_id else a.subject_id
        new_obj = canonical_id if a.object_id == duplicate_id else a.object_id
        key = (new_subj, a.predicate, new_obj)
        # Drop assertions that collapse to a self-loop or collide with an
        # existing canonical-pointing assertion.
        if new_subj == new_obj or key in existing_keys:
            session.delete(a)
            deleted += 1
            continue
        a.subject_id = new_subj
        a.object_id = new_obj
        if update_types:
            if new_subj == canonical_id:
                a.subject_type = canonical.entity_type
            if new_obj == canonical_id:
                a.object_type = canonical.entity_type
        existing_keys.add(key)
        repointed += 1

    # --- single-subject tables: constraints, patterns, antipatterns -------
    for model in (ConstraintRow, PatternRow, AntiPatternRow):
        for row in session.exec(
            select(model).where(model.subject_id == duplicate_id)
        ).all():
            row.subject_id = canonical_id
            repointed += 1

    # --- spatial_relations (subject_id, object_id) ------------------------
    spatial_keys = {
        (s.subject_id, s.relation, s.object_id)
        for s in session.exec(
            select(SpatialRelationRow).where(
                (SpatialRelationRow.subject_id == canonical_id)
                | (SpatialRelationRow.object_id == canonical_id)
            )
        ).all()
    }
    for s in session.exec(
        select(SpatialRelationRow).where(
            (SpatialRelationRow.subject_id == duplicate_id)
            | (SpatialRelationRow.object_id == duplicate_id)
        )
    ).all():
        new_subj = canonical_id if s.subject_id == duplicate_id else s.subject_id
        new_obj = canonical_id if s.object_id == duplicate_id else s.object_id
        key = (new_subj, s.relation, new_obj)
        if new_subj == new_obj or key in spatial_keys:
            session.delete(s)
            deleted += 1
            continue
        s.subject_id = new_subj
        s.object_id = new_obj
        spatial_keys.add(key)
        repointed += 1

    # --- process_relations (predecessor_id, successor_id) -----------------
    # Has a UNIQUE(predecessor_id, successor_id, source_model) constraint, so
    # collisions must be deleted, not repointed.
    proc_keys = {
        (p.predecessor_id, p.successor_id, p.source_model)
        for p in session.exec(
            select(ProcessRelationRow).where(
                (ProcessRelationRow.predecessor_id == canonical_id)
                | (ProcessRelationRow.successor_id == canonical_id)
            )
        ).all()
    }
    for p in session.exec(
        select(ProcessRelationRow).where(
            (ProcessRelationRow.predecessor_id == duplicate_id)
            | (ProcessRelationRow.successor_id == duplicate_id)
        )
    ).all():
        new_pred = canonical_id if p.predecessor_id == duplicate_id else p.predecessor_id
        new_succ = canonical_id if p.successor_id == duplicate_id else p.successor_id
        key = (new_pred, new_succ, p.source_model)
        if new_pred == new_succ or key in proc_keys:
            session.delete(p)
            deleted += 1
            continue
        p.predecessor_id = new_pred
        p.successor_id = new_succ
        proc_keys.add(key)
        repointed += 1

    # --- forces.affects (JSON list of entity ids) -------------------------
    for f in session.exec(select(ForceRow)).all():
        affects = json.loads(f.affects or "[]")
        if duplicate_id not in affects:
            continue
        # Replace duplicate with canonical, preserving order, deduping.
        seen: set[str] = set()
        new_affects: list[str] = []
        for eid in affects:
            mapped = canonical_id if eid == duplicate_id else eid
            if mapped in seen:
                continue
            seen.add(mapped)
            new_affects.append(mapped)
        f.affects = json.dumps(new_affects)
        repointed += 1

    # --- ifc_pset_recommendations.entity_id -------------------------------
    for row in session.exec(
        select(IFCPropertySetRow).where(IFCPropertySetRow.entity_id == duplicate_id)
    ).all():
        row.entity_id = canonical_id
        repointed += 1

    # --- entity_aliases: repoint aliases already on the duplicate ---------
    for alias in session.exec(
        select(EntityAliasRow).where(EntityAliasRow.entity_id == duplicate_id)
    ).all():
        alias.entity_id = canonical_id
        repointed += 1

    # --- record the duplicate's name as an alias, then mark it merged -----
    if add_alias:
        already = session.exec(
            select(EntityAliasRow).where(
                EntityAliasRow.entity_id == canonical_id,
                EntityAliasRow.alias == duplicate.name,
            )
        ).first()
        if already is None and duplicate.name != canonical.name:
            session.add(EntityAliasRow(entity_id=canonical_id, alias=duplicate.name))

    duplicate.status = "merged"

    return {"repointed": repointed, "deleted": deleted}
