"""
Accorder — Storage Layer
=========================
SQLAlchemy 2.0 models (declarative, type-hint driven — same pattern as the
Pydantic schemas and the typer CLI: the type hints on each column ARE the
schema, not a separate description of it) plus session handling.

Database selection is entirely runtime, via DATABASE_URL:
    unset                              -> local SQLite file (./accorder.db)
    postgresql+psycopg2://user:pw@host -> Postgres

No async engine, no connection pooling tuning here on purpose -- this is a
single-process CLI tool, not a service handling concurrent requests. See
ARCHITECTURE.md, Layer 5 / design principle 7, for the reasoning.

Tables map directly onto the seven roles from the architecture doc:
    Source           -- portal metadata
    RawPage           -- staging: raw cleaned content per scrape, before parsing
    Grant              -- canonical current state of each grant
    GrantHistoryEntry    -- append-only log of field-level changes
    SiteTemplate          -- cached selector patterns per domain (Layer 3)
    ScrapeRun               -- run metadata (pages attempted/succeeded, LLM calls)
    ValidationErrorRecord     -- dead-letter queue for failed validation
"""

from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    JSON,
    CHAR,
    Boolean,
    Date,
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    Numeric,
    String,
    Text,
    create_engine,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.engine import Engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)
from sqlalchemy.types import TypeDecorator

from .schemas import ExtractionMethod, FundingType, GrantStatus


# ---------------------------------------------------------------------------
# Cross-database UUID type. Postgres has a native UUID column type; SQLite
# doesn't, so we store it as a CHAR(36) string there instead. This lets the
# exact same model definition work against either database without an
# if/else scattered through the rest of the code.
# ---------------------------------------------------------------------------

class GUID(TypeDecorator):
    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if dialect.name == "postgresql":
            return str(value)
        if not isinstance(value, uuid.UUID):
            return str(uuid.UUID(str(value)))
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        if not isinstance(value, uuid.UUID):
            return uuid.UUID(str(value))
        return value


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------

class Source(Base):
    __tablename__ = "sources"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    base_url: Mapped[str] = mapped_column(String(500), nullable=False)
    config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


# ---------------------------------------------------------------------------
# raw_pages -- staging area, written by Layer 1 before any parsing happens
# ---------------------------------------------------------------------------

class RawPage(Base):
    __tablename__ = "raw_pages"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    source_url: Mapped[str] = mapped_column(String(1000), nullable=False)
    source_domain: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    run_id: Mapped[uuid.UUID] = mapped_column(GUID(), nullable=False, index=True)
    scraped_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    cleaned_text: Mapped[str] = mapped_column(Text, nullable=False)
    processed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


# ---------------------------------------------------------------------------
# grants -- canonical current state, mirrors schemas.GrantValidated
# ---------------------------------------------------------------------------

class Grant(Base):
    __tablename__ = "grants"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)

    grant_name: Mapped[str] = mapped_column(String(300), nullable=False)
    funder_name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)

    max_amount: Mapped[Optional[Decimal]] = mapped_column(Numeric(14, 2), nullable=True)
    min_amount: Mapped[Optional[Decimal]] = mapped_column(Numeric(14, 2), nullable=True)
    currency: Mapped[str] = mapped_column(String(3), default="USD")

    deadline_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    status: Mapped[GrantStatus] = mapped_column(
        SAEnum(GrantStatus, native_enum=False, length=20), default=GrantStatus.UNKNOWN
    )
    funding_type: Mapped[FundingType] = mapped_column(
        SAEnum(FundingType, native_enum=False, length=20), default=FundingType.UNKNOWN
    )

    eligibility: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    application_url: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)
    contact_email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # provenance -- flattened from schemas.SourceMetadata rather than a
    # separate joined table, since a grant has exactly one "latest" source
    # and the full history of where it came from lives in grants_history
    source_url: Mapped[str] = mapped_column(String(1000), nullable=False)
    source_domain: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    run_id: Mapped[uuid.UUID] = mapped_column(GUID(), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    extraction_method: Mapped[ExtractionMethod] = mapped_column(
        SAEnum(ExtractionMethod, native_enum=False, length=20)
    )

    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    history: Mapped[list["GrantHistoryEntry"]] = relationship(
        back_populates="grant", cascade="all, delete-orphan"
    )


# ---------------------------------------------------------------------------
# grants_history -- append-only log of what changed on an existing grant
# ---------------------------------------------------------------------------

class GrantHistoryEntry(Base):
    __tablename__ = "grants_history"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    grant_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("grants.id"), nullable=False)

    field_name: Mapped[str] = mapped_column(String(100), nullable=False)
    old_value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    new_value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    run_id: Mapped[uuid.UUID] = mapped_column(GUID(), nullable=False)
    changed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    grant: Mapped["Grant"] = relationship(back_populates="history")


# ---------------------------------------------------------------------------
# site_templates -- cached selector patterns per domain (Layer 3)
# ---------------------------------------------------------------------------

class SiteTemplate(Base):
    __tablename__ = "site_templates"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    domain: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    selectors: Mapped[dict] = mapped_column(JSON, nullable=False)
    learned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    last_verified_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    # how many consecutive runs this template has produced complete records --
    # a rising number of failures here is what triggers falling back to the LLM
    consecutive_failures: Mapped[int] = mapped_column(default=0)


# ---------------------------------------------------------------------------
# scrape_runs -- run-level metadata, one row per `accorder scrape` invocation
# ---------------------------------------------------------------------------

class ScrapeRun(Base):
    __tablename__ = "scrape_runs"

    run_id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    source_name: Mapped[str] = mapped_column(String(100), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    pages_attempted: Mapped[int] = mapped_column(default=0)
    pages_succeeded: Mapped[int] = mapped_column(default=0)
    llm_calls_made: Mapped[int] = mapped_column(default=0)
    validation_failures: Mapped[int] = mapped_column(default=0)
    new_grants: Mapped[int] = mapped_column(default=0)
    updated_grants: Mapped[int] = mapped_column(default=0)

    status: Mapped[str] = mapped_column(String(20), default="running")  # running | success | failed


# ---------------------------------------------------------------------------
# validation_errors -- dead-letter queue, mirrors schemas.ValidationFailure
# ---------------------------------------------------------------------------

class ValidationErrorRecord(Base):
    __tablename__ = "validation_errors"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    raw_payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    errors: Mapped[list] = mapped_column(JSON, nullable=False)
    source_url: Mapped[str] = mapped_column(String(1000), nullable=False)
    source_domain: Mapped[str] = mapped_column(String(255), nullable=False)
    run_id: Mapped[uuid.UUID] = mapped_column(GUID(), nullable=False)
    failed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


# ---------------------------------------------------------------------------
# Engine / session setup
# ---------------------------------------------------------------------------

def get_database_url() -> str:
    return os.environ.get("DATABASE_URL", "sqlite:///accorder.db")


def get_engine(database_url: Optional[str] = None) -> Engine:
    url = database_url or get_database_url()
    # SQLite needs this because, by default, a connection can only be used
    # from the thread that created it -- irrelevant for our single-threaded
    # CLI usage, but harmless to set and avoids a confusing error if any
    # future code path touches the connection from a different thread
    # (e.g. a background asyncio task doing the scraping).
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args)


def init_db(engine: Engine) -> None:
    """Creates all tables that don't already exist. Safe to call on every
    run -- existing tables are left untouched."""
    Base.metadata.create_all(engine)


def get_session_factory(engine: Engine) -> sessionmaker:
    return sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def session_scope(engine: Engine):
    """Usage:
        engine = get_engine()
        with session_scope(engine) as session:
            session.add(some_row)
    Commits on clean exit, rolls back on any exception -- a failed
    operation can never leave a half-written transaction behind."""
    factory = get_session_factory(engine)
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Smoke test — run directly with `python storage.py`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # In-memory SQLite for the smoke test, so it never touches a real file.
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    print(f"Created tables: {sorted(Base.metadata.tables.keys())}")

    run_id = uuid.uuid4()
    now = _utcnow()

    with session_scope(engine) as session:
        grant = Grant(
            grant_name="Youth Innovation Fund 2026",
            funder_name="Example Foundation",
            description="Supports youth-led financial literacy projects.",
            max_amount=Decimal("25000"),
            min_amount=Decimal("5000"),
            currency="USD",
            deadline_date=date(2026, 12, 31),
            status=GrantStatus.OPEN,
            funding_type=FundingType.GRANT,
            eligibility={"geographic_scope": "Nigeria", "organization_types": ["NGO"]},
            application_url="https://example-foundation.org/apply",
            source_url="https://example-foundation.org/grants/youth-fund",
            source_domain="example-foundation.org",
            run_id=run_id,
            content_hash="a" * 64,
            extraction_method=ExtractionMethod.LLM,
            first_seen_at=now,
            last_updated_at=now,
        )
        session.add(grant)

    # New session: prove the row actually persisted and round-trips with
    # correct types (UUID, Decimal, date, Enum) -- not just strings.
    with session_scope(engine) as session:
        fetched = session.query(Grant).filter_by(grant_name="Youth Innovation Fund 2026").one()
        assert isinstance(fetched.id, uuid.UUID), f"id did not round-trip as UUID: {type(fetched.id)}"
        assert isinstance(fetched.max_amount, Decimal), f"max_amount did not round-trip as Decimal: {type(fetched.max_amount)}"
        assert fetched.max_amount == Decimal("25000")
        assert isinstance(fetched.deadline_date, date)
        assert fetched.status == GrantStatus.OPEN
        assert fetched.eligibility["geographic_scope"] == "Nigeria"
        print(f"Fetched grant: {fetched.grant_name} ({fetched.status.value}, {fetched.currency} {fetched.max_amount})")

        # Log a change to grants_history, prove the relationship works
        history_entry = GrantHistoryEntry(
            grant_id=fetched.id,
            field_name="max_amount",
            old_value="20000",
            new_value="25000",
            run_id=run_id,
        )
        session.add(history_entry)

    with session_scope(engine) as session:
        fetched = session.query(Grant).filter_by(grant_name="Youth Innovation Fund 2026").one()
        assert len(fetched.history) == 1
        assert fetched.history[0].field_name == "max_amount"
        print(f"History entry recorded: {fetched.history[0].old_value} -> {fetched.history[0].new_value}")

    print("\nSmoke test passed.")