"""
Accorder — Extraction Layer (Layer 1)
======================================
Source-agnostic listing-page extraction: selectors, deadline format, and
pagination scheme all come from a SourceConfig (accorder.sources), not
from constants in this module. Adding a new source means adding a new
SourceConfig entry in sources.py — this file shouldn't need to change.

Two responsibilities are deliberately kept separate:

  parse_listing_page()   -- pure function. HTML string + SourceConfig in,
                             list of raw candidates out. No network, no
                             side effects. This is what makes it testable
                             with a fixture, which is exactly what the
                             smoke test below does -- I don't have network
                             access to www2.fundsforngos.org from this
                             environment, so everything network-dependent
                             (fetch_page_html, scrape_source) is written
                             against the real Playwright API but has NOT
                             been run against the live site. Test it on
                             your machine before trusting it against the
                             real portal.

  fetch_page_html() /
  scrape_source()          -- Playwright browser navigation + retry policy.
                               Needs real network access; untestable here.

Selectors for fundsforNGOs specifically (the confirmed DOM structure from
the live inspector: article.post > header.entry-header > h2.entry-title
[itemprop="headline"] > a.entry-title-link, and div.entry-content
[itemprop="text"] > p containing "Deadline: <date> <description>"), with
non-itemprop fallback selectors in case a page's markup varies, now live
in sources.py as that source's SourceConfig entry.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Optional

from bs4 import BeautifulSoup
from playwright.async_api import Page, async_playwright
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from .domain_throttle import DomainThrottle
from .retry_utils import BlockedError, TransientScrapeError, retry_blocked, retry_transient
from .sources import SourceConfig

logger = logging.getLogger("accorder.extractor")

# Best-effort: finds currency+number mentions anywhere in free text and
# treats the smallest/largest as min/max amount. This is a cheap heuristic,
# not real sentence understanding -- it can't distinguish "grants up to
# $50,000" from "a $50,000 penalty for non-compliance," it just grabs
# currency-shaped tokens. Good enough as a first pass on real descriptions;
# the LLM fallback (Layer 3) is the properly correct version of this, since
# it can actually read the sentence. This pattern isn't source-specific --
# it's a generic money-shaped-token heuristic -- so it stays a module-level
# constant rather than moving into SourceConfig.
_CURRENCY_SYMBOL_MAP = {"$": "USD", "€": "EUR", "£": "GBP"}
_AMOUNT_PATTERN = re.compile(
    r"(USD|EUR|GBP|NGN|CAD|AUD|KES|GHS|\$|€|£)\s?([\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)


def extract_amounts_from_text(text: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Returns (min_amount_raw, max_amount_raw, currency_raw), any of which
    may be None.

    Finds ALL currency-shaped mentions, not just the first -- this matters
    for stated ranges like "USD 25,000-USD 50,000", where naively taking
    the first match would silently store the MINIMUM as the maximum. With
    multiple mentions found, the smallest becomes min_amount and the
    largest becomes max_amount; with a single mention, it's just the max
    (open-ended grants are usually phrased as "up to X", i.e. a ceiling).
    """
    matches = _AMOUNT_PATTERN.findall(text)
    if not matches:
        return None, None, None

    parsed: list[tuple[Decimal, str]] = []
    currency: Optional[str] = None
    for currency_token, number_str in matches:
        try:
            value = Decimal(number_str.replace(",", ""))
        except InvalidOperation:
            continue
        parsed.append((value, number_str))
        if currency is None:
            currency = _CURRENCY_SYMBOL_MAP.get(currency_token, currency_token.upper())

    if not parsed:
        return None, None, None

    parsed.sort(key=lambda p: p[0])
    min_raw = parsed[0][1] if len(parsed) > 1 else None
    max_raw = parsed[-1][1]
    return min_raw, max_raw, currency


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


def parse_listing_page(
    html: str, source_config: SourceConfig
) -> tuple[list[RawListingCandidate], list[str]]:
    """Pure function: (HTML string, SourceConfig) -> (candidates, skipped_article_text).

    All selectors and the deadline-prefix pattern come from source_config,
    not from constants in this module -- this is what makes the function
    reusable across sources with different markup.

    skipped_article_text holds the plain text of every article whose
    title link couldn't be found deterministically -- the caller can feed
    these to the LLM fallback (accorder.parser.extract_fields_via_llm)
    instead of just losing them to a log line, which is what happened
    before this function returned them."""
    soup = BeautifulSoup(html, "lxml")
    candidates: list[RawListingCandidate] = []
    skipped_article_text: list[str] = []

    for article in soup.select(source_config.article_selector):
        title_link = article.select_one(source_config.title_selector)
        if title_link is None and source_config.title_selector_fallback:
            title_link = article.select_one(source_config.title_selector_fallback)

        if title_link is None:
            logger.warning(
                f"Skipping article: no title link found (selector drift?) "
                f"— classes: {article.get('class')}"
            )
            skipped_article_text.append(article.get_text(" ", strip=True))
            continue

        grant_name = title_link.get_text(strip=True)
        detail_url = (title_link.get("href") or "").strip()

        if not grant_name or not detail_url:
            logger.warning(
                f"Skipping article: incomplete data (name={bool(grant_name)}, url={bool(detail_url)})"
            )
            skipped_article_text.append(article.get_text(" ", strip=True))
            continue

        content_p = article.select_one(source_config.content_selector)
        if content_p is None and source_config.content_selector_fallback:
            content_p = article.select_one(source_config.content_selector_fallback)
        raw_text = content_p.get_text(" ", strip=True) if content_p else ""

        deadline_match = (
            source_config.deadline_prefix_re.match(raw_text)
            if source_config.deadline_prefix_re
            else None
        )
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
                source_domain=source_config.domain,
            )
        )

    return candidates, skipped_article_text


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Network-dependent code below. Written against the real Playwright API,
# NOT runnable/tested in this environment -- test against the live site
# before trusting it.
# ---------------------------------------------------------------------------

async def fetch_page_html(page: Page, url: str, source_config: SourceConfig) -> str:
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

    # This site (and plenty of others) routes through a client-side "please
    # wait, optimizing your request" interstitial before redirecting to the
    # real page -- domcontentloaded fires on the interstitial ITSELF, before
    # that redirect happens. So instead of trusting a fixed wait strategy,
    # wait explicitly for real content to show up in the DOM, riding through
    # however many redirects that takes. The selector to wait for is
    # source-specific (whatever marks "the real page has loaded" for this
    # source), so it comes from source_config rather than being hardcoded.
    try:
        await page.wait_for_selector(source_config.article_selector, timeout=20_000)
    except PlaywrightTimeoutError:
        # Genuinely no matching content appeared within the timeout --
        # either real selector drift (site changed its markup) or a
        # bot-detection wall we didn't get past. Don't hide this: let it
        # through as an empty/interstitial page so the caller's normal
        # "0 candidates parsed" path surfaces it, rather than silently
        # treating interstitial HTML as if it were real content.
        logger.warning(
            f"No '{source_config.article_selector}' appeared within timeout for {url} "
            f"— possible bot wall or selector drift."
        )

    return await page.content()


@retry_blocked
@retry_transient
async def fetch_with_retry(page: Page, url: str, source_config: SourceConfig) -> str:
    return await fetch_page_html(page, url, source_config)


async def scrape_source(
    source_config: SourceConfig,
    max_pages: int = 1,
    throttle: Optional[DomainThrottle] = None,
) -> tuple[list[RawListingCandidate], list[str]]:
    """Fetches `max_pages` of listing pages for source_config and parses
    each with parse_listing_page(). Returns (candidates, skipped_article_text)
    -- the latter is raw text for articles the deterministic parser
    couldn't handle, meant to be routed to the LLM fallback by the caller."""
    throttle = throttle or DomainThrottle(max_concurrent_per_domain=2, min_delay_seconds=1.5)
    all_candidates: list[RawListingCandidate] = []
    all_skipped: list[str] = []

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
                url = (
                    source_config.base_url
                    if page_num == 1
                    else source_config.pagination_url_fn(source_config.base_url, page_num)
                )
                async with throttle.throttle(source_config.domain):
                    try:
                        html = await fetch_with_retry(page, url, source_config)
                    except (TransientScrapeError, BlockedError) as e:
                        logger.error(f"Giving up on {url} after retries: {e}")
                        continue

                candidates, skipped = parse_listing_page(html, source_config)
                logger.info(f"Parsed {len(candidates)} grants from {url} ({len(skipped)} skipped)")
                all_candidates.extend(candidates)
                all_skipped.extend(skipped)
        finally:
            await browser.close()

    return all_candidates, all_skipped


# ---------------------------------------------------------------------------
# Smoke test — parse_listing_page() ONLY, against a fixture built from the
# confirmed live DOM structure. Run directly with `python -m accorder.extractor`.
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

<!-- Real-world case found on the live site: heading is h4 (not h2) for
     posts with a featured thumbnail, and the title link has NO class
     attribute at all -- only itemprop="headline" is consistent. This is
     exactly the post-478631 case that was silently skipped in production. -->
<article class="post-478631 post type-post status-publish has-post-thumbnail">
  <a href="https://www2.fundsforngos.org/individuals/community-benefit-fund-gambling-research-grant-program-australia/" class="aligncenter" aria-hidden="true" tabindex="-1">
    <img width="300" height="200" alt="thumbnail" />
  </a>
  <header class="entry-header">
    <h4 class="entry-title" itemprop="headline">
      <a href="https://www2.fundsforngos.org/individuals/community-benefit-fund-gambling-research-grant-program-australia/">
        Community Benefit Fund: Gambling Research Grant Program (Australia)
      </a>
    </h4>
  </header>
  <div class="entry-content" itemprop="text">
    <p>Deadline: 20-Oct-2026 Supports community organizations researching the social impact of gambling.</p>
  </div>
</article>

</body></html>
"""

if __name__ == "__main__":
    from .sources import SOURCES

    fixture_source_config = SOURCES["fundsforngos"]
    candidates, skipped = parse_listing_page(_FIXTURE_HTML, fixture_source_config)

    print(f"Parsed {len(candidates)} candidates (fixture has 5 articles, 1 deliberately malformed):\n")
    for c in candidates:
        print(f"  - {c.grant_name}")
        print(f"    deadline_raw: {c.deadline_raw}")
        print(f"    detail_url:   {c.detail_url}")
        print(f"    description:  {c.description_raw[:70]}...")
        print()

    assert len(candidates) == 4, f"expected 4 valid candidates (1 skipped), got {len(candidates)}"
    assert len(skipped) == 1, f"expected 1 skipped article captured for LLM fallback, got {len(skipped)}"
    assert "01-Aug-2026" in skipped[0], "skipped article text should contain its raw content"
    print(f"Captured {len(skipped)} skipped article(s) for LLM fallback:")
    print(f"  {skipped[0][:100]}...")
    print()

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

    h4_grant = candidates[3]
    assert h4_grant.grant_name == "Community Benefit Fund: Gambling Research Grant Program (Australia)"
    assert h4_grant.deadline_raw == "20-Oct-2026"
    print("Real-world regression case (h4 heading, no link class, post-478631) parsed correctly.")

    h1 = content_hash(env_grant.description_raw)
    h2 = content_hash(social_grant.description_raw)
    assert h1 != h2
    assert content_hash(env_grant.description_raw) == h1  # deterministic
    print(f"content_hash() is deterministic and distinguishes different content.")

    print("\nSmoke test passed (parsing logic only — fetch_page_html/scrape_source untested here, no network access).")