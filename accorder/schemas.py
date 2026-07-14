"""
Grant Tracker — Data Models
============================
Pydantic v2 schemas for the extraction -> validation -> storage -> API pipeline.

Design:
  GrantExtracted   - lenient "candidate" shape. What the extractor (deterministic
                     parser OR the LLM fallback) produces. Almost everything is
                     optional because a real page might genuinely be missing data,
                     and an LLM might return an unparsed string where a date or
                     number should be.

  GrantValidated   - strict shape. What actually gets written to Postgres.
                     Built from GrantExtracted via normalize_and_validate(),
                     which never raises -- failures become ValidationFailure
                     records instead of crashing a batch run.

  ValidationFailure - dead-letter record. Nothing from extraction is ever
                     silently dropped; it either becomes a GrantValidated
                     or a logged ValidationFailure.

  GrantOut         - what the FastAPI layer actually returns to the frontend.
                     Deliberately separate from GrantValidated so the public
                     API contract can change independently of the DB/storage
                     shape (e.g. we never expose contact_email or content_hash).

Run this file directly (`python schemas.py`) for a smoke test against a
couple of realistic messy inputs.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from dateutil import parser as dateutil_parser
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    field_validator,
    model_validator,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class GrantStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    EXPIRED = "expired"    # deadline passed but source didn't explicitly close it
    ROLLING = "rolling"    # no fixed deadline, accepted year-round
    UNKNOWN = "unknown"    # source didn't state status clearly


class FundingType(str, Enum):
    GRANT = "grant"
    FELLOWSHIP = "fellowship"
    LOAN = "loan"
    PRIZE = "prize"
    IN_KIND = "in_kind"
    UNKNOWN = "unknown"


class ExtractionMethod(str, Enum):
    """Which path produced this record. Kept on every row so you can audit
    LLM-vs-deterministic extraction accuracy over time, and so a spike in
    LLM usage for a domain that used to be deterministic is an early
    signal that the site changed layout."""
    DETERMINISTIC = "deterministic"
    LLM = "llm"
    MANUAL = "manual"


# ---------------------------------------------------------------------------
# Provenance metadata, attached to every record regardless of source layer
# ---------------------------------------------------------------------------

class SourceMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_url: HttpUrl
    source_domain: str
    run_id: UUID
    scraped_at: datetime
    content_hash: str = Field(
        ..., min_length=64, max_length=64,
        description="SHA-256 of the cleaned page text; used to skip reprocessing unchanged pages.",
    )
    extraction_method: ExtractionMethod


# ---------------------------------------------------------------------------
# Nested eligibility block
# ---------------------------------------------------------------------------

class EligibilityCriteria(BaseModel):
    summary: Optional[str] = Field(None, max_length=2000)
    geographic_scope: Optional[str] = None  # e.g. "Nigeria", "West Africa", "Global"
    organization_types: list[str] = Field(default_factory=list)  # e.g. ["nonprofit", "individual"]
    min_years_operating: Optional[int] = Field(None, ge=0)


# ---------------------------------------------------------------------------
# Layer 1: raw extraction candidate — nothing here is trusted yet
# ---------------------------------------------------------------------------

class GrantExtracted(BaseModel):
    """This is the exact shape you hand to the LLM as its target JSON schema,
    and the exact shape a deterministic selector-based parser should also
    produce, so both paths feed the same validation step downstream."""

    model_config = ConfigDict(extra="ignore")  # LLMs sometimes add stray keys; drop, don't fail

    grant_name: Optional[str] = None
    funder_name: Optional[str] = None
    description: Optional[str] = Field(None, max_length=5000)
    max_amount_raw: Optional[str] = None   # e.g. "up to $50,000"
    min_amount_raw: Optional[str] = None
    currency_raw: Optional[str] = None     # e.g. "USD", "$", "NGN"
    deadline_raw: Optional[str] = None     # whatever string the source/LLM gave us
    funding_type_raw: Optional[str] = None
    eligibility: Optional[EligibilityCriteria] = None
    application_url: Optional[str] = None  # validated to HttpUrl only after normalization
    contact_email: Optional[str] = None

    source: SourceMetadata


# ---------------------------------------------------------------------------
# Layer 2: strict, validated shape — this is what gets written to Postgres
# ---------------------------------------------------------------------------

class GrantValidated(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    id: UUID = Field(default_factory=uuid4)

    grant_name: str = Field(..., min_length=3, max_length=300)
    funder_name: str = Field(..., min_length=2, max_length=200)
    description: str = Field(..., max_length=5000)

    max_amount: Optional[Decimal] = Field(None, ge=0)
    min_amount: Optional[Decimal] = Field(None, ge=0)
    currency: str = Field(default="USD", min_length=3, max_length=3)  # ISO 4217

    deadline_date: Optional[date] = None
    status: GrantStatus = GrantStatus.UNKNOWN

    funding_type: FundingType = FundingType.UNKNOWN
    eligibility: EligibilityCriteria = Field(default_factory=EligibilityCriteria)

    application_url: Optional[HttpUrl] = None
    contact_email: Optional[str] = None

    source: SourceMetadata

    first_seen_at: datetime
    last_updated_at: datetime

    @field_validator("currency")
    @classmethod
    def currency_upper(cls, v: str) -> str:
        return v.upper()

    @field_validator("contact_email")
    @classmethod
    def basic_email_shape(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if "@" not in v or "." not in v.split("@")[-1]:
            raise ValueError(f"'{v}' does not look like a valid email")
        return v.lower()

    @model_validator(mode="after")
    def check_amount_order(self) -> "GrantValidated":
        if self.min_amount is not None and self.max_amount is not None:
            if self.min_amount > self.max_amount:
                raise ValueError("min_amount cannot exceed max_amount")
        return self

    @model_validator(mode="after")
    def derive_status_from_deadline(self) -> "GrantValidated":
        """If the source never stated a status explicitly, infer it from the
        deadline. Past deadlines are marked EXPIRED, never rejected outright --
        a tracker needs the historical record."""
        if self.status == GrantStatus.UNKNOWN and self.deadline_date is not None:
            self.status = (
                GrantStatus.EXPIRED if self.deadline_date < date.today() else GrantStatus.OPEN
            )
        return self


# ---------------------------------------------------------------------------
# Dead-letter record — anything that failed validation lands here, with the
# original payload preserved, instead of being silently dropped
# ---------------------------------------------------------------------------

class ValidationFailure(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    raw_payload: dict
    errors: list[str]
    source: SourceMetadata
    failed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Normalization glue: GrantExtracted -> (GrantValidated | ValidationFailure)
# This is the part that does the actual "cleaning" work -- parsing messy
# amount/date strings -- and it NEVER raises, so one bad page can't crash a
# batch run.
# ---------------------------------------------------------------------------

_AMOUNT_RE = re.compile(r"[\d,]+\.?\d*")
_NO_DEADLINE_PHRASES = {"rolling", "ongoing", "no deadline", "n/a", "open until filled"}

_FUNDING_TYPE_MAP: dict[str, FundingType] = {
    "grant": FundingType.GRANT,
    "fellowship": FundingType.FELLOWSHIP,
    "loan": FundingType.LOAN,
    "prize": FundingType.PRIZE,
    "in-kind": FundingType.IN_KIND,
    "in kind": FundingType.IN_KIND,
}


def parse_amount(raw: Optional[str]) -> Optional[Decimal]:
    if not raw:
        return None
    match = _AMOUNT_RE.search(raw.replace(",", ""))
    if not match:
        return None
    try:
        return Decimal(match.group())
    except InvalidOperation:
        return None


def _parse_deadline(raw: Optional[str]) -> Optional[date]:
    if not raw:
        return None
    if raw.strip().lower() in _NO_DEADLINE_PHRASES:
        return None
    try:
        return dateutil_parser.parse(raw, fuzzy=True).date()
    except (ValueError, OverflowError):
        return None


def map_funding_type(raw: Optional[str]) -> FundingType:
    if not raw:
        return FundingType.UNKNOWN
    raw_lower = raw.lower()
    for key, value in _FUNDING_TYPE_MAP.items():
        if key in raw_lower:
            return value
    return FundingType.UNKNOWN


def normalize_and_validate(
    extracted: GrantExtracted,
) -> tuple[Optional[GrantValidated], Optional[ValidationFailure]]:
    """Returns (GrantValidated, None) on success, or (None, ValidationFailure)
    on failure. Always returns exactly one of the two -- callers can branch
    on which is None."""
    now = datetime.now(timezone.utc)
    try:
        validated = GrantValidated(
            grant_name=extracted.grant_name or "",
            funder_name=extracted.funder_name or "",
            description=extracted.description or "",
            max_amount=parse_amount(extracted.max_amount_raw),
            min_amount=parse_amount(extracted.min_amount_raw),
            currency=(extracted.currency_raw or "USD")[:3],
            deadline_date=_parse_deadline(extracted.deadline_raw),
            funding_type=map_funding_type(extracted.funding_type_raw),
            eligibility=extracted.eligibility or EligibilityCriteria(),
            application_url=extracted.application_url,
            contact_email=extracted.contact_email,
            source=extracted.source,
            first_seen_at=now,
            last_updated_at=now,
        )
        return validated, None
    except Exception as e:  # noqa: BLE001 - intentionally broad: never crash a batch run
        failure = ValidationFailure(
            raw_payload=extracted.model_dump(mode="json"),
            errors=[str(e)],
            source=extracted.source,
        )
        return None, failure


# ---------------------------------------------------------------------------
# API-facing models — what FastAPI actually exposes to the Next.js frontend.
# Deliberately narrower than GrantValidated (no contact_email, no source
# provenance internals) so the public contract can evolve independently.
# ---------------------------------------------------------------------------

class GrantOut(BaseModel):
    id: UUID
    grant_name: str
    funder_name: str
    description: str
    max_amount: Optional[Decimal]
    min_amount: Optional[Decimal]
    currency: str
    deadline_date: Optional[date]
    status: GrantStatus
    funding_type: FundingType
    eligibility: EligibilityCriteria
    application_url: Optional[HttpUrl]
    last_updated_at: datetime

    @classmethod
    def from_validated(cls, grant: GrantValidated) -> "GrantOut":
        return cls(**grant.model_dump(exclude={"source", "first_seen_at", "contact_email", "id"}), id=grant.id)


class GrantSearchParams(BaseModel):
    """Query params for GET /grants."""
    q: Optional[str] = None
    status: Optional[GrantStatus] = None
    funding_type: Optional[FundingType] = None
    min_amount_gte: Optional[Decimal] = Field(None, ge=0)
    deadline_before: Optional[date] = None
    deadline_after: Optional[date] = None
    limit: int = Field(default=25, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


# ---------------------------------------------------------------------------
# Smoke test — run directly with `python schemas.py`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_source = SourceMetadata(
        source_url="https://example-foundation.org/grants/youth-fund",
        source_domain="example-foundation.org",
        run_id=uuid4(),
        scraped_at=datetime.now(timezone.utc),
        content_hash="a" * 64,
        extraction_method=ExtractionMethod.LLM,
    )

    # Messy, realistic candidate -- the kind of thing an LLM or a scraper
    # would actually hand back.
    messy_candidate = GrantExtracted(
        grant_name="  Youth Innovation Fund 2026  ",
        funder_name="Example Foundation",
        description="Supports youth-led projects in financial literacy and entrepreneurship.",
        max_amount_raw="up to $25,000 USD",
        min_amount_raw="$5,000",
        currency_raw="usd",
        deadline_raw="31st of December, 2026",
        funding_type_raw="Grant funding",
        eligibility=EligibilityCriteria(
            summary="Open to registered NGOs and youth-led organizations.",
            geographic_scope="Nigeria",
            organization_types=["NGO", "nonprofit"],
        ),
        application_url="https://example-foundation.org/apply",
        contact_email="GRANTS@example-foundation.org",
        source=sample_source,
    )

    validated, failure = normalize_and_validate(messy_candidate)
    assert failure is None, f"Expected success, got failure: {failure}"
    print("Validated grant:")
    print(validated.model_dump_json(indent=2))

    out = GrantOut.from_validated(validated)
    print("\nAPI-facing shape (GrantOut):")
    print(out.model_dump_json(indent=2))

    # A deliberately broken candidate to prove the dead-letter path works.
    broken_candidate = GrantExtracted(
        grant_name="X",  # too short, fails min_length=3
        funder_name="Some Funder",
        description="A grant.",
        source=sample_source,
    )
    validated2, failure2 = normalize_and_validate(broken_candidate)
    assert validated2 is None
    print("\nDead-letter record produced for invalid input:")
    print(failure2.model_dump_json(indent=2))

    print("\nSmoke test passed.")