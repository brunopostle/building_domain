"""Pass 11 — Adversarial validation.

Runs across the full accepted+proposed assertion set after Pass 10 completes.
Batches assertions by subject entity (≤BATCH_SIZE per LLM call), asks each
provider to identify exceptions, context limitations, factual errors, and
scope restrictions.

Multi-model merge: same (item_id, finding_type) from ≥2 distinct models
promotes the finding from log-only to state change (for potential_error).

State machine:
  exception          → append to AssertionRow.exceptions (no status change)
  context_limitation → append to AssertionRow.applicability (no status change)
  potential_error    → set status='conflicted' + ReviewDecision(defer) if ≥2 models;
                       log only if single model
  scope_restriction  → ReviewDecision(defer) only; human must confirm via review-pending

Confidence threshold: findings < 0.70 are logged only, no state change.
Resume: per-provider PassProgressRow(pass_number='11', entity_id='__global__').
"""
import json
import uuid
from collections import defaultdict
from datetime import datetime, timezone

import structlog
from sqlmodel import Session, select

from bsos.llm.protocol import LLMProvider
from bsos.persistence.models import (
    AssertionRow,
    EntityRow,
    PassProgressRow,
    ProvenanceLogRow,
    ReviewDecisionRow,
)
from bsos.pipeline.schemas import AdversarialFinding, AdversarialValidationResponse

log = structlog.get_logger()

CONFIDENCE_THRESHOLD = 0.70
BATCH_SIZE = 20

_PROMPT_TEMPLATE = """\
You are an adversarial reviewer for a building domain knowledge base. \
Review the following assertions and identify problems: exceptions where the \
assertion does not hold, context limitations, factual errors, or scope \
restrictions that should narrow or deprecate the assertion.

For each issue found, provide an AdversarialFinding with:
- item_id: the exact UUID shown in brackets
- item_type: "assertion"
- finding_type: "exception" | "context_limitation" | "potential_error" | "scope_restriction"
- detail: specific, concrete description (not generic)
- suggested_action: "add_exception" | "add_condition" | "flag_for_review" | "deprecate"
- confidence: 0.0–1.0

Only report findings you are confident about. Omit assertions that appear correct.

Assertions:
{assertions_text}
"""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _format_batch(rows: list[AssertionRow], entity_names: dict[str, str]) -> str:
    parts = []
    for row in rows:
        subj = entity_names.get(row.subject_id, row.subject_id)
        obj = entity_names.get(row.object_id, row.object_id)
        lines = [f"[{row.id}] {subj} {row.predicate} {obj}"]
        conds = json.loads(row.conditions or "[]")
        if conds:
            lines.append(f"  conditions: {'; '.join(conds)}")
        excs = json.loads(row.exceptions or "[]")
        if excs:
            lines.append(f"  existing exceptions: {'; '.join(excs)}")
        apps = json.loads(row.applicability or "[]")
        if apps:
            lines.append(f"  applicability: {'; '.join(apps)}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def _collect_findings_for_provider(
    engine,
    provider: LLMProvider,
    batch_size: int,
) -> list[tuple[AdversarialFinding, str]]:
    """Run LLM for one provider. Returns list of (finding, model_id). Skips if already done."""
    with Session(engine) as session:
        progress = session.get(PassProgressRow, ("11", "__global__", provider.model_id))
        if progress and progress.status == "completed":
            log.info("pass11_provider_skip_resume", model=provider.model_id)
            return []

        rows = session.exec(
            select(AssertionRow).where(
                AssertionRow.status.in_(["accepted", "proposed"])
            )
        ).all()

        entity_ids = {r.subject_id for r in rows} | {r.object_id for r in rows}
        entity_names: dict[str, str] = {}
        for eid in entity_ids:
            e = session.get(EntityRow, eid)
            if e:
                entity_names[eid] = e.name

    if not rows:
        return []

    by_subject: dict[str, list[AssertionRow]] = defaultdict(list)
    for row in rows:
        by_subject[row.subject_id].append(row)

    findings: list[tuple[AdversarialFinding, str]] = []

    for subject_id, subject_rows in by_subject.items():
        for i in range(0, len(subject_rows), batch_size):
            batch = subject_rows[i : i + batch_size]
            prompt = _PROMPT_TEMPLATE.format(
                assertions_text=_format_batch(batch, entity_names)
            )
            try:
                response = provider.extract(prompt, AdversarialValidationResponse)
                for f in response.findings:
                    findings.append((f, provider.model_id))
                log.debug(
                    "pass11_batch_done",
                    model=provider.model_id,
                    subject=entity_names.get(subject_id, subject_id),
                    found=len(response.findings),
                )
            except Exception as exc:
                log.warning(
                    "pass11_batch_failed",
                    model=provider.model_id,
                    subject=subject_id,
                    error=str(exc),
                )

    return findings


def _apply_findings(
    engine,
    all_findings: list[tuple[AdversarialFinding, str]],
    dry_run: bool,
) -> dict:
    """Group by (item_id, finding_type), apply multi-model merge, write state changes."""
    grouped: dict[tuple[str, str], list[tuple[AdversarialFinding, str]]] = defaultdict(list)
    for finding, model_id in all_findings:
        grouped[(finding.item_id, finding.finding_type)].append((finding, model_id))

    if dry_run:
        for (item_id, finding_type), entries in grouped.items():
            models = {m for _, m in entries}
            log.info(
                "pass11_dry_run_finding",
                item_id=item_id,
                finding_type=finding_type,
                models=sorted(models),
                count=len(entries),
            )
        return {
            "findings_collected": len(all_findings),
            "findings_applied": 0,
            "dry_run": True,
        }

    exceptions_appended = 0
    conditions_appended = 0
    conflicted = 0
    deferred = 0
    logged_only = 0

    with Session(engine) as session:
        for (item_id, finding_type), entries in grouped.items():
            models = {m for _, m in entries}
            multi_model = len(models) >= 2

            # Deduplicate by detail string; pick highest-confidence as canonical
            seen: set[str] = set()
            unique: list[tuple[AdversarialFinding, str]] = []
            for f, m in entries:
                if f.detail not in seen:
                    seen.add(f.detail)
                    unique.append((f, m))
            canonical_f, canonical_model = max(unique, key=lambda x: x[0].confidence)

            if canonical_f.confidence < CONFIDENCE_THRESHOLD:
                log.info(
                    "pass11_below_threshold",
                    item_id=item_id,
                    finding_type=finding_type,
                    confidence=canonical_f.confidence,
                )
                logged_only += 1
                continue

            row = session.get(AssertionRow, item_id)
            if row is None:
                log.debug("pass11_item_not_found", item_id=item_id, finding_type=finding_type)
                continue

            if finding_type == "exception":
                existing = json.loads(row.exceptions or "[]")
                added = 0
                for f, _ in unique:
                    if f.detail not in existing:
                        existing.append(f.detail)
                        added += 1
                if added:
                    row.exceptions = json.dumps(existing)
                    exceptions_appended += added

            elif finding_type == "context_limitation":
                existing = json.loads(row.applicability or "[]")
                added = 0
                for f, _ in unique:
                    if f.detail not in existing:
                        existing.append(f.detail)
                        added += 1
                if added:
                    row.applicability = json.dumps(existing)
                    conditions_appended += added

            elif finding_type == "potential_error":
                if multi_model:
                    old_status = row.status
                    row.status = "conflicted"
                    session.add(ReviewDecisionRow(
                        id=str(uuid.uuid4()),
                        item_id=item_id,
                        item_type="assertion",
                        decision="defer",
                        rationale=canonical_f.detail,
                        reviewer=canonical_model,
                        created_at=_now(),
                    ))
                    session.add(ProvenanceLogRow(
                        id=str(uuid.uuid4()),
                        item_id=item_id,
                        item_type="assertion",
                        old_status=old_status,
                        new_status="conflicted",
                        changed_at=_now(),
                        changed_by=f"pass11/{canonical_model}",
                    ))
                    conflicted += 1
                    log.info("pass11_conflicted", item_id=item_id, models=sorted(models))
                else:
                    log.info(
                        "pass11_potential_error_single_model",
                        item_id=item_id,
                        model=canonical_model,
                        detail=canonical_f.detail,
                    )
                    logged_only += 1

            elif finding_type == "scope_restriction":
                session.add(ReviewDecisionRow(
                    id=str(uuid.uuid4()),
                    item_id=item_id,
                    item_type="assertion",
                    decision="defer",
                    rationale=f"scope_restriction: {canonical_f.detail}",
                    reviewer=canonical_model,
                    created_at=_now(),
                ))
                deferred += 1
                log.info("pass11_scope_deferred", item_id=item_id)

        session.commit()

    return {
        "findings_collected": len(all_findings),
        "exceptions_appended": exceptions_appended,
        "conditions_appended": conditions_appended,
        "conflicted": conflicted,
        "deferred": deferred,
        "logged_only": logged_only,
        "findings_applied": exceptions_appended + conditions_appended + conflicted + deferred,
    }


def run_pass11(
    engine,
    providers: list[LLMProvider],
    run_id: str,
    *,
    dry_run: bool = False,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """Run Pass 11: adversarial validation across accepted+proposed assertions.

    Accepts all providers; multi-model merge combines findings across them.
    Pass is global (entity_id='__global__' per provider in pass_progress).
    Resume: providers with a completed PassProgressRow are skipped.

    Returns summary dict.
    """
    with Session(engine) as session:
        count = len(session.exec(
            select(AssertionRow).where(
                AssertionRow.status.in_(["accepted", "proposed"])
            )
        ).all())

    if count == 0:
        log.info("pass11_no_assertions")
        return {"skipped": True, "reason": "no accepted/proposed assertions"}

    log.info("pass11_start", assertion_count=count, provider_count=len(providers))

    all_findings: list[tuple[AdversarialFinding, str]] = []
    for provider in providers:
        findings = _collect_findings_for_provider(engine, provider, batch_size)
        all_findings.extend(findings)
        log.info("pass11_provider_complete", model=provider.model_id, findings=len(findings))

    result = _apply_findings(engine, all_findings, dry_run)

    if not dry_run:
        now = _now()
        with Session(engine) as session:
            for provider in providers:
                existing = session.get(PassProgressRow, ("11", "__global__", provider.model_id))
                if existing:
                    existing.completed_at = now
                    existing.status = "completed"
                else:
                    session.add(PassProgressRow(
                        pass_number="11",
                        entity_id="__global__",
                        model=provider.model_id,
                        completed_at=now,
                        status="completed",
                    ))
            session.commit()

    log.info("pass11_complete", **result)
    return result
