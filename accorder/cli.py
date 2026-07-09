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
import uuid
from datetime import datetime, timezone

import typer
from rich.console import Console
from rich.table import Table

from .extractor import scrape_source
from .storage import Grant, ScrapeRun, get_engine, init_db, session_scope
from .validator import RunStats, process_candidates

app = typer.Typer(help="Accorder — a grant/funding portal tracker.")
console = Console()

# Hardcoded for now. Becomes a real config file (sources.yaml or similar)
# once a second source is added -- premature to build that abstraction
# for a single source.
SOURCES = {
    "fundsforngos": {
        "base_url": "https://www2.fundsforngos.org/tag/nigeria/",
        "domain": "www2.fundsforngos.org",
    },
}


@app.command()
def scrape(
    source: str = typer.Option(..., help=f"Source to scrape. Configured: {list(SOURCES)}"),
    max_pages: int = typer.Option(1, help="Number of listing pages to fetch"),
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
    candidates = asyncio.run(
        scrape_source(cfg["base_url"], cfg["domain"], max_pages=max_pages)
    )
    console.print(f"Extracted {len(candidates)} raw candidates.")

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