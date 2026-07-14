"""
Accorder — CLI Layer
========================
Two commands for now, per the build order in ARCHITECTURE.md:
    accorder scrape --source <name> [--max-pages N]
    accorder stats  [--limit N]

`report` and `export` come next, once there's real data worth reporting on.

Notice the shape of `scrape()` below: almost everything it does is call
out to other modules (extractor, validator, storage) and glue their
outputs together. That's intentional -- the CLI command itself has almost
no logic of its own to get wrong. All the actual logic (parsing, dedup,
validation) lives in modules that are independently tested with fixtures.
This command is the one piece of the whole pipeline that genuinely can't
be tested without hitting the live network, which is exactly why it's
been kept this thin.
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Optional

import typer
from jinja2 import Environment, FileSystemLoader, select_autoescape
from rich.console import Console
from rich.table import Table

from .extractor import scrape_source
from .schemas import GrantStatus
from .sources import SOURCES
from .storage import Grant, ScrapeRun, get_engine, init_db, session_scope
from .validator import RunStats, enrich_grants, process_candidates

from dotenv import load_dotenv
load_dotenv()

app = typer.Typer(help="Accorder — a grant/funding portal tracker.")
console = Console()


@app.command()
def scrape(
    source: str = typer.Option(..., help=f"Source to scrape. Configured: {list(SOURCES)}"),
    max_pages: int = typer.Option(1, help="Number of listing pages to fetch"),
    enrich: bool = typer.Option(
        False,
        help="After scraping, visit each new/updated grant's detail page and use an "
             "LLM to fill in funder_name/application_url/eligibility. Costs real API "
             "calls and time per grant -- requires OPENROUTER_API_KEY. Opt-in on purpose.",
    ),
    enrich_limit: int = typer.Option(20, help="Max grants to enrich per run (only used with --enrich)"),
) -> None:
    """Scrape a configured source, extract, validate, and store results."""
    if source not in SOURCES:
        console.print(f"[red]Unknown source '{source}'.[/red] Configured sources: {list(SOURCES)}")
        raise typer.Exit(code=1)

    cfg = SOURCES[source]
    run_id = uuid.uuid4()
    scraped_at = datetime.now(timezone.utc)

    engine = get_engine()
    init_db(engine)

    with session_scope(engine) as session:
        session.add(ScrapeRun(run_id=run_id, source_name=source, status="running"))

    console.print(f"[bold]Scraping[/bold] {source} (run_id={run_id}, max_pages={max_pages})...")
    candidates, skipped = asyncio.run(
        scrape_source(cfg, max_pages=max_pages)
    )
    console.print(
        f"Extracted {len(candidates)} raw candidates "
        f"({len(skipped)} skipped — no matching selector, not yet routed to LLM fallback)."
    )

    with session_scope(engine) as session:
        stats = process_candidates(session, candidates, run_id, scraped_at)
        run_row = session.query(ScrapeRun).filter_by(run_id=run_id).one()
        run_row.pages_attempted = max_pages
        run_row.pages_succeeded = max_pages  # refined once per-page success is tracked individually
        run_row.new_grants = stats.new_grants
        run_row.updated_grants = stats.updated_grants
        run_row.validation_failures = stats.validation_failures
        run_row.finished_at = datetime.now(timezone.utc)
        run_row.status = "success"

    _print_run_summary(source, run_id, stats)

    if enrich:
        console.print(f"\n[bold]Enriching[/bold] up to {enrich_limit} grants via LLM (detail-page visits)...")
        try:
            enriched_count = asyncio.run(enrich_grants(engine, run_id, limit=enrich_limit))
            console.print(f"[green]Enriched {enriched_count} grants with funder_name/application_url/eligibility.[/green]")
        except RuntimeError as e:
            # Specifically OPENROUTER_API_KEY missing -- fail this step
            # loudly but don't undo the successful scrape above.
            console.print(f"[red]Enrichment skipped: {e}[/red]")


@app.command()
def stats(limit: int = typer.Option(10, help="Number of recent runs to show")) -> None:
    """Show recent scrape run history."""
    engine = get_engine()
    init_db(engine)

    with session_scope(engine) as session:
        runs = session.query(ScrapeRun).order_by(ScrapeRun.started_at.desc()).limit(limit).all()
        total_grants = session.query(Grant).count()

    if not runs:
        console.print("[yellow]No scrape runs yet.[/yellow] Try: accorder scrape --source fundsforngos")
        return

    table = Table(title=f"Recent Scrape Runs  (total grants in DB: {total_grants})")
    table.add_column("Started")
    table.add_column("Source")
    table.add_column("New", justify="right")
    table.add_column("Updated", justify="right")
    table.add_column("Failed", justify="right")
    table.add_column("Status")

    for run in runs:
        status_style = {"success": "green", "failed": "red", "running": "yellow"}.get(run.status, "white")
        table.add_row(
            run.started_at.strftime("%Y-%m-%d %H:%M"),
            run.source_name,
            str(run.new_grants),
            str(run.updated_grants),
            str(run.validation_failures),
            f"[{status_style}]{run.status}[/{status_style}]",
        )

    console.print(table)


@app.command()
def report(
    status: Optional[str] = typer.Option(
        None, help=f"Filter by status. Choices: {[s.value for s in GrantStatus]}"
    ),
    funder: Optional[str] = typer.Option(None, help="Filter by funder name (partial match)"),
    limit: int = typer.Option(0, help="Max grants to show (0 = no limit)"),
    html: Optional[str] = typer.Option(
        None, help="Also write a static HTML report to this path (e.g. --html report.html)"
    ),
) -> None:
    """Browse grants currently in the database, with optional filters."""
    rows = _query_grants(status=status, funder=funder, limit=(limit or None))

    if not rows:
        console.print("[yellow]No grants match those filters.[/yellow]")
        return

    _print_report_table(rows)

    if html:
        _write_html_report(rows, html)
        console.print(f"\n[green]HTML report written to {html}[/green]")


@app.command()
def export(
    format: str = typer.Option(..., help="Export format: csv or json"),
    out: str = typer.Option(..., help="Output file path"),
    status: Optional[str] = typer.Option(None, help="Filter by status"),
) -> None:
    """Export current grants to CSV or JSON."""
    if format not in ("csv", "json"):
        console.print(f"[red]Invalid format '{format}'.[/red] Choose 'csv' or 'json'.")
        raise typer.Exit(code=1)

    rows = _query_grants(status=status, funder=None, limit=None)
    if not rows:
        console.print("[yellow]No grants to export.[/yellow]")
        return

    if format == "csv":
        _export_csv(rows, out)
    else:
        _export_json(rows, out)

    console.print(f"[green]Exported {len(rows)} grants to {out}[/green]")


# ---------------------------------------------------------------------------
# Shared helpers for report/export
# ---------------------------------------------------------------------------

def _query_grants(
    status: Optional[str], funder: Optional[str], limit: Optional[int]
) -> list[dict]:
    """Runs the filtered query and converts ORM rows to plain dicts BEFORE
    the session closes. This matters: a SQLAlchemy object is tied to the
    session that loaded it, and accessing its attributes after that session
    closes can fail. Converting to plain dicts inside the `with` block
    sidesteps that entirely -- report(), export(), and the HTML renderer
    all just work with ordinary dicts, no session lifetime to think about."""
    engine = get_engine()
    init_db(engine)

    with session_scope(engine) as session:
        query = session.query(Grant)

        if status:
            try:
                status_enum = GrantStatus(status.lower())
            except ValueError:
                console.print(
                    f"[red]Invalid status '{status}'.[/red] Choose from: {[s.value for s in GrantStatus]}"
                )
                raise typer.Exit(code=1)
            query = query.filter(Grant.status == status_enum)

        if funder:
            query = query.filter(Grant.funder_name.ilike(f"%{funder}%"))

        query = query.order_by(Grant.deadline_date)
        if limit is not None:
            query = query.limit(limit)

        return [_grant_to_dict(g) for g in query.all()]


def _grant_to_dict(g: Grant) -> dict:
    return {
        "grant_name": g.grant_name,
        "funder_name": g.funder_name,
        "description": g.description,
        "max_amount": g.max_amount,
        "min_amount": g.min_amount,
        "currency": g.currency,
        "deadline_date": g.deadline_date,
        "status": g.status.value,
        "funding_type": g.funding_type.value,
        "application_url": g.application_url,
        "source_url": g.source_url,
        "last_updated_at": g.last_updated_at,
    }


def _print_report_table(rows: list[dict]) -> None:
    status_styles = {
        "open": "green", "expired": "red", "closed": "dim",
        "rolling": "cyan", "unknown": "white",
    }
    table = Table(title=f"Grants ({len(rows)})")
    table.add_column("Deadline")
    table.add_column("Status")
    table.add_column("Grant Name", overflow="fold", max_width=55)
    table.add_column("Amount", justify="right")

    for r in rows:
        amount = f"{r['currency']} {r['max_amount']:,.0f}" if r["max_amount"] else "—"
        style = status_styles.get(r["status"], "white")
        table.add_row(
            str(r["deadline_date"]) if r["deadline_date"] else "—",
            f"[{style}]{r['status']}[/{style}]",
            r["grant_name"],
            amount,
        )
    console.print(table)


def _export_csv(rows: list[dict], path: str) -> None:
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("" if v is None else str(v)) for k, v in r.items()})


def _export_json(rows: list[dict], path: str) -> None:
    def _serialize(v):
        if isinstance(v, Decimal):
            return float(v)
        if isinstance(v, (date, datetime)):
            return v.isoformat()
        return v

    serializable = [{k: _serialize(v) for k, v in r.items()} for r in rows]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


def _write_html_report(rows: list[dict], out_path: str) -> None:
    env = Environment(
        loader=FileSystemLoader(_TEMPLATE_DIR),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("report.html.j2")
    rendered = template.render(rows=rows, total=len(rows), generated_at=datetime.now(timezone.utc))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(rendered)


def _print_run_summary(source: str, run_id: uuid.UUID, stats: RunStats) -> None:
    table = Table(title=f"Run Summary — {source} ({run_id})")
    table.add_column("Metric")
    table.add_column("Count", justify="right")
    table.add_row("New grants", str(stats.new_grants))
    table.add_row("Updated grants", str(stats.updated_grants))
    table.add_row("Unchanged grants", str(stats.unchanged_grants))
    table.add_row("Validation failures", str(stats.validation_failures))
    console.print(table)


if __name__ == "__main__":
    app()