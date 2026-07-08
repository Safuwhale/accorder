# Accorder

A CLI tool that scrapes grant/funding portals, extracts structured data
through a hybrid deterministic + LLM pipeline, validates and deduplicates
it, tracks changes over time, and reports on it — from the terminal or a
static HTML file. No server, no hosting required.

> **Status:** under active development. See [ARCHITECTURE.md](./ARCHITECTURE.md)
> for the full design and current build progress.

## Why

Grant and funding portals rarely offer a clean feed of what's available or
when it changes. Accorder scrapes a configured set of sources, extracts
structured fields (name, funder, amount, deadline, eligibility) using a
selector cache that falls back to an LLM only when a site's layout is new
or has changed, and keeps a history of what changed between runs.

## Install

```bash
git clone <repo-url>
cd accorder
pip install -e .
playwright install chromium
```

## Usage

```bash
accorder scrape --source <name>
accorder report --status open
accorder export --format csv --out grants.csv
accorder stats
```

(Command details will be filled in as each is built — see ARCHITECTURE.md
for the full command reference.)

## Configuration

Copy `.env.example` to `.env` and fill in:

- `OPENROUTER_API_KEY` — required for the LLM extraction fallback
- `DATABASE_URL` — optional; defaults to a local SQLite file if unset

## Architecture

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the full layer-by-layer design
and the reasoning behind it.

## License

MIT