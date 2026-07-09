"""
Accorder — Extraction Layer (Layer 1), Site 1: fundsforNGOs
==============================================================
Scoped to listing-page-only extraction for the first build pass, per
ARCHITECTURE.md's build order: prove the full pipeline end-to-end on one
source before adding a second, and before adding the LLM fallback.

Two responsibilities are deliberately kept separate:

  parse_listing_page()   -- pure function. HTML string in, list of raw
                             candidates out. No network, no side effects.
                             This is what makes it testable with a fixture,
                             which is exactly what the smoke test below does
                             -- I don't have network access to
                             www2.fundsforngos.org from this environment, so
                             everything network-dependent (fetch_page_html,
                             scrape_source) is written against the real
                             Playwright API but has NOT been run against the
                             live site. Test it on your machine before
                             trusting it against the real portal.

  fetch_page_html() /
  scrape_source()          -- Playwright browser navigation + retry policy.
                               Needs real network access; untestable here.

Selectors target the confirmed DOM structure from the live inspector
(article.post > header.entry-header > h2.entry-title[itemprop="headline"]
> a.entry-title-link, and div.entry-content[itemprop="text"] > p containing
"Deadline: <date> <description>"), with non-itemprop fallback selectors in
case a page's markup varies.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Optional

from bs4 import BeautifulSoup
from playwright.async_api import Page, async_playwright

from .domain_throttle import DomainThrottle
from .retry_utils import BlockedError, TransientScrapeError, retry_blocked, retry_transient

logger = logging.getLogger("accorder.extractor")

# Matches "Deadline: 24-Jul-2026 " at the start of the paragraph text and
# captures just the date token. The rest of the paragraph is the description.
DEADLINE_PREFIX_RE = re.compile(r"^\s*Deadline:\s*(\S+)\s*", re.IGNORECASE)


@dataclass
class RawListingCandidate:
    """What parse_listing_page() produces per grant. Deliberately a plain
    dataclass, not schemas.GrantExtracted directly -- this is pre-cleaning
    data straight off the page, before Layer 4 normalization/validation
    ever sees it."""
    grant_name: str
    detail_url: str
    deadline_raw: Optional[str]
    description_raw: str
    source_domain: str


def parse_listing_page(html: str, source_domain: str) -> list[RawListingCandidate]:
    """Pure function: HTML string -> list of raw candidates. No network
    involved -- this is the part that gets unit tested against a fixture
    instead of the live site."""
    soup = BeautifulSoup(html, "lxml")
    candidates: list[RawListingCandidate] = []

    for article in soup.select("article.post"):
        title_link = (
            article.select_one('h2.entry-title[itemprop="headline"] a.entry-title-link')
            or article.select_one("h2.entry-title a")
        )
        if title_link is None:
            logger.warning("Skipping article: no title link found (selector drift?)")
            continue

        grant_name = title_link.get_text(strip=True)
        detail_url = (title_link.get("href") or "").strip()

        if not grant_name or not detail_url:
            logger.warning(
                f"Skipping article: incomplete data (name={bool(grant_name)}, url={bool(detail_url)})"
            )
            continue

        content_p = (
            article.select_one('div.entry-content[itemprop="text"] p')
            or article.select_one("div.entry-content p")
        )
        raw_text = content_p.get_text(" ", strip=True) if content_p else ""

        deadline_match = DEADLINE_PREFIX_RE.match(raw_text)
        if deadline_match:
            deadline_raw = deadline_match.group(1)
            description_raw = raw_text[deadline_match.end():].strip()
        else:
            # Not every post necessarily leads with "Deadline: ..." -- don't
            # assume, just carry the full text through and let Layer 4's
            # date parsing (which is fuzzy/tolerant) take a pass at it later.
            deadline_raw = None
            description_raw = raw_text

        candidates.append(
            RawListingCandidate(
                grant_name=grant_name,
                detail_url=detail_url,
                deadline_raw=deadline_raw,
                description_raw=description_raw,
                source_domain=source_domain,
            )
        )

    return candidates


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Network-dependent code below. Written against the real Playwright API,
# NOT runnable/tested in this environment -- test against the live site
# before trusting it.
# ---------------------------------------------------------------------------

async def fetch_page_html(page: Page, url: str) -> str:
    """Playwright navigation wrapper. Raises TransientScrapeError or
    BlockedError based on the response, so retry_utils' policies (built and
    tested earlier) can act on the right failure type."""
    try:
        response = await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    except Exception as e:  # noqa: BLE001 - network layer: any failure here is transient by default
        raise TransientScrapeError(f"Navigation failed for {url}: {e}") from e

    if response is None:
        raise TransientScrapeError(f"No response received for {url}")
    if response.status in (403, 429):
        raise BlockedError(f"Blocked (status {response.status}) fetching {url}")
    if response.status >= 500:
        raise TransientScrapeError(f"Server error (status {response.status}) fetching {url}")

    return await page.content()


@retry_blocked
@retry_transient
async def fetch_with_retry(page: Page, url: str) -> str:
    return await fetch_page_html(page, url)


async def scrape_source(
    base_url: str,
    source_domain: str,
    max_pages: int = 1,
    throttle: Optional[DomainThrottle] = None,
) -> list[RawListingCandidate]:
    """Fetches `max_pages` of listing pages (page 1, then /page/2/,
    /page/3/, ... following this site's pagination pattern) and parses
    each with parse_listing_page(). Uses DomainThrottle so pagination
    requests to the same domain are paced, not fired all at once."""
    throttle = throttle or DomainThrottle(max_concurrent_per_domain=2, min_delay_seconds=1.5)
    all_candidates: list[RawListingCandidate] = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            )
        )
        try:
            for page_num in range(1, max_pages + 1):
                url = base_url if page_num == 1 else f"{base_url.rstrip('/')}/page/{page_num}/"
                async with throttle.throttle(source_domain):
                    try:
                        html = await fetch_with_retry(page, url)
                    except (TransientScrapeError, BlockedError) as e:
                        logger.error(f"Giving up on {url} after retries: {e}")
                        continue

                candidates = parse_listing_page(html, source_domain)
                logger.info(f"Parsed {len(candidates)} grants from {url}")
                all_candidates.extend(candidates)
        finally:
            await browser.close()

    return all_candidates


# ---------------------------------------------------------------------------
# Smoke test — parse_listing_page() ONLY, against a fixture built from the
# confirmed live DOM structure. Run directly with `python extractor.py`.
# The network-dependent functions above are NOT exercised here.
# ---------------------------------------------------------------------------

_FIXTURE_HTML = """
<html><body>

<article class="post-477989 post type-post status-publish category-community-development-2 tag-nigeria">
  <header class="entry-header">
    <h2 class="entry-title" itemprop="headline">
      <a class="entry-title-link" rel="bookmark"
         href="https://www2.fundsforngos.org/community-development-2/apply-now-environmental-grants-for-climbing-conservation-projects/">
        Apply Now: Environmental Grants for Climbing Conservation Projects
      </a>
    </h2>
  </header>
  <div class="entry-content" itemprop="text">
    <p>Deadline: 19-Jun-2026 The Environmental Grants program supports climbing
    conservation projects across affected regions, funding up to USD 5,000 per
    project for community-led initiatives.
    <a href="https://www2.fundsforngos.org/leadership/call-for-applications-environmental-grants/" class="more-link">Read more</a></p>
  </div>
  <footer class="entry-footer"></footer>
</article>

<article class="post-477990 post type-post status-publish category-community-development-2 tag-nigeria">
  <header class="entry-header">
    <h2 class="entry-title" itemprop="headline">
      <a class="entry-title-link" rel="bookmark"
         href="https://www2.fundsforngos.org/leadership/call-for-applications-social-impact-grants/">
        Apply Now: Social Impact Grants for Climbing Initiatives
      </a>
    </h2>
  </header>
  <div class="entry-content" itemprop="text">
    <p>Deadline: 24-Jul-2026 The Social Impact Grants program by the Global
    Climbing Initiative provides grants of up to USD 1,000 to support
    community-led climbing projects that promote inclusion, equity, leadership,
    and access for underrepresented groups.
    <a href="https://www2.fundsforngos.org/leadership/call-for-applications-social-impact-grants/" class="more-link">Read more</a></p>
  </div>
  <footer class="entry-footer"></footer>
</article>

<!-- Malformed: no title link at all -- must be skipped, not crash the whole page -->
<article class="post-477991 post type-post status-publish">
  <header class="entry-header"></header>
  <div class="entry-content" itemprop="text">
    <p>Deadline: 01-Aug-2026 A grant with a missing title link, testing the skip-and-log path.</p>
  </div>
</article>

<!-- No itemprop microdata at all -- must still parse via fallback selectors -->
<article class="post-477992 post type-post status-publish">
  <header class="entry-header">
    <h2 class="entry-title">
      <a class="entry-title-link" href="https://www2.fundsforngos.org/other/no-itemprop-grant/">
        Grant Without Itemprop Attributes
      </a>
    </h2>
  </header>
  <div class="entry-content">
    <p>Deadline: 15-Sep-2026 Testing the fallback selector path when itemprop microdata is missing.</p>
  </div>
</article>

</body></html>
"""

if __name__ == "__main__":
    candidates = parse_listing_page(_FIXTURE_HTML, "www2.fundsforngos.org")

    print(f"Parsed {len(candidates)} candidates (fixture has 4 articles, 1 deliberately malformed):\n")
    for c in candidates:
        print(f"  - {c.grant_name}")
        print(f"    deadline_raw: {c.deadline_raw}")
        print(f"    detail_url:   {c.detail_url}")
        print(f"    description:  {c.description_raw[:70]}...")
        print()

    assert len(candidates) == 3, f"expected 3 valid candidates (1 skipped), got {len(candidates)}"

    env_grant = candidates[0]
    assert env_grant.grant_name == "Apply Now: Environmental Grants for Climbing Conservation Projects"
    assert env_grant.deadline_raw == "19-Jun-2026"
    assert env_grant.description_raw.startswith("The Environmental Grants program")
    assert env_grant.detail_url.startswith("https://www2.fundsforngos.org/")

    social_grant = candidates[1]
    assert social_grant.deadline_raw == "24-Jul-2026"
    assert "USD 1,000" in social_grant.description_raw

    fallback_grant = candidates[2]
    assert fallback_grant.grant_name == "Grant Without Itemprop Attributes"
    assert fallback_grant.deadline_raw == "15-Sep-2026"
    print("Fallback selector path (no itemprop microdata) parsed correctly.")

    h1 = content_hash(env_grant.description_raw)
    h2 = content_hash(social_grant.description_raw)
    assert h1 != h2
    assert content_hash(env_grant.description_raw) == h1  # deterministic
    print(f"content_hash() is deterministic and distinguishes different content.")

    print("\nSmoke test passed (parsing logic only — fetch_page_html/scrape_source untested here, no network access).")