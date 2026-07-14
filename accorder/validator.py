"""
Accorder — Validator Layer
=============================
Bridges raw extraction output (extractor.RawListingCandidate) to storage
(storage.Grant), via schemas.normalize_and_validate().

This is deliberately a PURE function with respect to the database session
you hand it -- process_candidates() takes a session and a list of
candidates, and does all its work through that session. It doesn't open
connections, doesn't call the network, doesn't know anything about
Playwright. That's what makes it testable with an in-memory SQLite session
and a list of already-parsed fixture candidates (see the smoke test below)
instead of needing a live scrape to test the "did we save this correctly"
logic.

Dedup, for now: exact match on source_url. The architecture doc originally
called for fuzzy matching on (funder_name, grant_name) using rapidfuzz --
that's still the plan, but exact-match-on-URL is a reasonable, honestly-
scoped first version: it's correct as far as it goes (the same grant
scraped twice from the same URL is unambiguously the same grant), it just
doesn't yet catch the harder case of the same grant appearing at two
different URLs across different sources. That harder case doesn't exist
yet anyway, since we only have one source wired up. Worth upgrading when
Site 2 gets added.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy.orm import Session

from .extractor import RawListingCandidate, content_hash, extract_amounts_from_text
from .schemas import (
    ExtractionMethod,
    GrantExtracted,
    SourceMetadata,
    normalize_and_validate,
)
from .storage import Grant, GrantHistoryEntry, ValidationErrorRecord, session_scope

# Fields worth logging a history entry for when an existing grant changes.
# Deliberately a small, meaningful set -- not every column (e.g. skipping
# `description`, which changes on almost every re-scrape due to minor
# whitespace/wording differences and would flood the history table with
# noise that isn't actually useful to a person reading it later).
TRACKED_FIELDS = ["grant_name", "deadline_date", "status", "max_amount", "min_amount"]


@dataclass
class RunStats:
    new_grants: int = 0
    updated_grants: int = 0
    unchanged_grants: int = 0
    validation_failures: int = 0


def candidate_to_extracted(
    candidate: RawListingCandidate, run_id: uuid.UUID, scraped_at: datetime
) -> GrantExtracted:
    """Wraps a raw extraction candidate in the shape Layer 4 expects.

    Two known, honest limitations at this stage (listing-page-only
    extraction, no LLM fallback yet):
      - funder_name isn't distinctly available on the listing page --
        placeholder until we either visit detail pages or add the LLM path.
      - application_url is the fundsforNGOs detail page, not necessarily
        the funder's own external application link -- same limitation.
    """
    hash_input = f"{candidate.grant_name}|{candidate.deadline_raw}|{candidate.description_raw}"
    source = SourceMetadata(
        source_url=candidate.detail_url,
        source_domain=candidate.source_domain,
        run_id=run_id,
        scraped_at=scraped_at,
        content_hash=content_hash(hash_input),
        extraction_method=ExtractionMethod.DETERMINISTIC,
    )
    min_amount_raw, max_amount_raw, currency_raw = extract_amounts_from_text(candidate.description_raw)
    return GrantExtracted(
        grant_name=candidate.grant_name,
        funder_name="Unknown (listing page only — funder not yet extracted)",
        description=candidate.description_raw,
        deadline_raw=candidate.deadline_raw,
        min_amount_raw=min_amount_raw,
        max_amount_raw=max_amount_raw,
        currency_raw=currency_raw,
        application_url=candidate.detail_url,
        source=source,
    )


def _diff_fields(existing: Grant, validated) -> dict[str, tuple]:
    diffs = {}
    for field_name in TRACKED_FIELDS:
        old_val = getattr(existing, field_name)
        new_val = getattr(validated, field_name)
        if isinstance(old_val, Enum):
            old_val = old_val.value
        if isinstance(new_val, Enum):
            new_val = new_val.value
        if old_val != new_val:
            diffs[field_name] = (old_val, new_val)
    return diffs


def process_candidates(
    session: Session,
    candidates: list[RawListingCandidate],
    run_id: uuid.UUID,
    scraped_at: datetime,
) -> RunStats:
    stats = RunStats()

    for candidate in candidates:
        extracted = candidate_to_extracted(candidate, run_id, scraped_at)
        validated, failure = normalize_and_validate(extracted)

        if failure is not None:
            session.add(
                ValidationErrorRecord(
                    raw_payload=failure.raw_payload,
                    errors=failure.errors,
                    source_url=str(failure.source.source_url),
                    source_domain=failure.source.source_domain,
                    run_id=run_id,
                )
            )
            stats.validation_failures += 1
            continue

        existing = (
            session.query(Grant)
            .filter_by(source_url=str(validated.source.source_url))
            .one_or_none()
        )

        if existing is None:
            session.add(
                Grant(
                    grant_name=validated.grant_name,
                    funder_name=validated.funder_name,
                    description=validated.description,
                    max_amount=validated.max_amount,
                    min_amount=validated.min_amount,
                    currency=validated.currency,
                    deadline_date=validated.deadline_date,
                    status=validated.status,
                    funding_type=validated.funding_type,
                    eligibility=validated.eligibility.model_dump(),
                    application_url=str(validated.application_url) if validated.application_url else None,
                    contact_email=validated.contact_email,
                    source_url=str(validated.source.source_url),
                    source_domain=validated.source.source_domain,
                    run_id=run_id,
                    content_hash=validated.source.content_hash,
                    extraction_method=validated.source.extraction_method,
                    first_seen_at=validated.first_seen_at,
                    last_updated_at=validated.last_updated_at,
                )
            )
            stats.new_grants += 1
            continue

        if existing.content_hash == validated.source.content_hash:
            stats.unchanged_grants += 1
            continue

        # Content changed -- log what specifically changed before overwriting,
        # so `grants_history` stays meaningful rather than just "it changed."
        for field_name, (old_val, new_val) in _diff_fields(existing, validated).items():
            session.add(
                GrantHistoryEntry(
                    grant_id=existing.id,
                    field_name=field_name,
                    old_value=str(old_val) if old_val is not None else None,
                    new_value=str(new_val) if new_val is not None else None,
                    run_id=run_id,
                )
            )

        existing.grant_name = validated.grant_name
        existing.description = validated.description
        existing.deadline_date = validated.deadline_date
        existing.status = validated.status
        existing.max_amount = validated.max_amount
        existing.min_amount = validated.min_amount
        existing.content_hash = validated.source.content_hash
        existing.last_updated_at = validated.last_updated_at
        existing.run_id = run_id  # which run last touched this row, for auditability
        stats.updated_grants += 1

    return stats


async def enrich_grants(engine, run_id: uuid.UUID, limit: int = 20) -> int:
    """Opt-in Layer 3 pass: fetches detail pages for grants that still carry
    the placeholder funder_name and uses an LLM to fill in
    funder_name / application_url / eligibility.

    Deliberately scoped to "grants that still need it" rather than "grants
    touched by this specific run" -- process_candidates() only sets run_id
    on brand-new inserts, not on updates, so a run_id-scoped query here
    would miss both updated grants and the backlog of already-scraped
    grants that were never enriched. run_id is still accepted (and now
    recorded on updated grants too, see process_candidates()) for
    auditability, but no longer used to filter the enrichment target list.

    Deliberately separate from process_candidates() and only run when the
    caller asks for it (see cli.py's `scrape --enrich`) -- this costs real
    API calls and real time per grant, so it should never happen silently.

    Returns the count of grants actually enriched (a grant with an
    unparseable detail page or an LLM failure just keeps its existing data
    and isn't counted -- see enrich_one_grant()'s never-raises design).
    """
    from playwright.async_api import async_playwright

    from .domain_throttle import DomainThrottle
    from .llm_extractor import enrich_one_grant, get_openrouter_client

    with session_scope(engine) as session:
        targets = [
            (g.id, g.grant_name, g.source_url, g.source_domain)
            for g in session.query(Grant)
            .filter(Grant.funder_name.like("Unknown%"))
            .limit(limit)
            .all()
        ]

    if not targets:
        return 0

    client = get_openrouter_client()
    throttle = DomainThrottle(max_concurrent_per_domain=2, min_delay_seconds=1.5)
    enriched_count = 0

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            )
        )
        try:
            for grant_id, grant_name, source_url, source_domain in targets:
                async with throttle.throttle(source_domain):
                    result = await enrich_one_grant(page, client, grant_name, source_url)

                if result is None:
                    continue

                with session_scope(engine) as session:
                    grant_row = session.query(Grant).filter_by(id=grant_id).one()
                    if result.get("funder_name"):
                        grant_row.funder_name = result["funder_name"]
                    if result.get("application_url"):
                        grant_row.application_url = result["application_url"]
                    if result.get("eligibility_summary"):
                        eligibility = dict(grant_row.eligibility or {})
                        eligibility["summary"] = result["eligibility_summary"]
                        grant_row.eligibility = eligibility
                enriched_count += 1
        finally:
            await browser.close()

    return enriched_count


# ---------------------------------------------------------------------------
# Smoke test — run directly with `python3 -m accorder.validator`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from .extractor import parse_listing_page, _FIXTURE_HTML
    from .sources import SOURCES
    from .storage import get_engine, init_db, session_scope

    fixture_source_config = SOURCES["fundsforngos"]

    engine = get_engine("sqlite:///:memory:")
    init_db(engine)

    run_id_1 = uuid.uuid4()
    scraped_at_1 = datetime.now().astimezone()
    candidates, _skipped = parse_listing_page(_FIXTURE_HTML, fixture_source_config)
    assert len(candidates) == 4

    # --- Run 1: everything should be new ---
    with session_scope(engine) as session:
        stats_1 = process_candidates(session, candidates, run_id_1, scraped_at_1)
    print(f"Run 1: new={stats_1.new_grants} updated={stats_1.updated_grants} "
          f"unchanged={stats_1.unchanged_grants} failed={stats_1.validation_failures}")
    assert stats_1.new_grants == 4
    assert stats_1.updated_grants == 0

    # --- Run 2: identical candidates again -- everything should be unchanged ---
    run_id_2 = uuid.uuid4()
    scraped_at_2 = datetime.now().astimezone()
    with session_scope(engine) as session:
        stats_2 = process_candidates(session, candidates, run_id_2, scraped_at_2)
    print(f"Run 2 (no changes): new={stats_2.new_grants} updated={stats_2.updated_grants} "
          f"unchanged={stats_2.unchanged_grants} failed={stats_2.validation_failures}")
    assert stats_2.new_grants == 0
    assert stats_2.unchanged_grants == 4

    # --- Run 3: one candidate's deadline changed -- should be 1 update + a history entry ---
    changed_candidates = list(candidates)
    original = changed_candidates[0]
    changed_candidates[0] = RawListingCandidate(
        grant_name=original.grant_name,
        detail_url=original.detail_url,
        deadline_raw="30-Sep-2026",  # pushed back from 19-Jun-2026
        description_raw=original.description_raw,
        source_domain=original.source_domain,
    )
    run_id_3 = uuid.uuid4()
    scraped_at_3 = datetime.now().astimezone()
    with session_scope(engine) as session:
        stats_3 = process_candidates(session, changed_candidates, run_id_3, scraped_at_3)
    print(f"Run 3 (1 deadline changed): new={stats_3.new_grants} updated={stats_3.updated_grants} "
          f"unchanged={stats_3.unchanged_grants} failed={stats_3.validation_failures}")
    assert stats_3.updated_grants == 1
    assert stats_3.unchanged_grants == 3

    with session_scope(engine) as session:
        updated_grant = session.query(Grant).filter_by(source_url=original.detail_url).one()
        # Two fields change here, not one: the deadline itself, AND status --
        # the original fixture deadline (19-Jun-2026) is already in the past
        # relative to today, so derive_status_from_deadline() correctly
        # marked it EXPIRED on first insert. Moving the deadline to a future
        # date correctly flips status back to OPEN too. Both are real,
        # meaningful changes worth a history entry, not a bug.
        changed_field_names = {h.field_name for h in updated_grant.history}
        assert changed_field_names == {"deadline_date", "status"}, changed_field_names
        for h in updated_grant.history:
            print(f"History entry: {h.field_name} changed {h.old_value} -> {h.new_value}")

    print("\nSmoke test passed.")