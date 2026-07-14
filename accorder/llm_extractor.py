"""
Accorder — LLM Extraction Layer (Layer 3: detail-page enrichment)
=====================================================================
Fills in the two fields the listing page structurally cannot provide:
funder_name and the real external application_url. This is the one place
in the whole pipeline where an LLM earns its keep over a CSS selector --
grant detail pages are free-form human writing, not a consistent template,
so "which link on this page is the real apply link, and who is actually
offering this money" is a judgment call, not a lookup.

As with extractor.py, network-dependent code (fetch_detail_page_text,
call_llm) is kept separate from pure, testable logic (parse_llm_response,
build_user_prompt). I have no network access to OpenRouter or the target
site from this environment -- only the pure logic is tested here. Test the
live path on your machine, same as extractor.py's fetch_page_html was.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from playwright.async_api import Page

from .retry_utils import LLMMalformedResponseError, retry_llm_malformed

logger = logging.getLogger("accorder.llm_extractor")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = os.environ.get("ACCORDER_LLM_MODEL", "anthropic/claude-3.5-sonnet")

SYSTEM_PROMPT = """You are a data extraction assistant. Given the text content of a grant/funding opportunity webpage, extract ONLY the following fields as strict JSON, with no preamble, no markdown formatting, no explanation -- just the raw JSON object:

{
  "funder_name": string or null,        // the organization actually offering the money -- NOT the website hosting the listing
  "application_url": string or null,    // the URL to apply directly, ONLY if explicitly present as a link in the text -- otherwise null
  "eligibility_summary": string or null // 1-2 sentence summary of who can apply, or null if not stated
}

If a field cannot be determined from the text, use null. Do not guess or invent values."""


# ---------------------------------------------------------------------------
# Pure logic -- no network, fully testable with fixtures
# ---------------------------------------------------------------------------

def build_user_prompt(grant_name: str, page_text: str) -> str:
    # Detail pages can be long (related posts, footer links, etc. that
    # trafilatura doesn't always fully strip) -- cap what we send both to
    # control cost and to keep the model focused on the actual grant content
    # near the top of the page rather than trailing boilerplate.
    trimmed = page_text[:6000]
    return f"Grant title: {grant_name}\n\nPage content:\n{trimmed}"


def parse_llm_response(raw_response: str) -> dict:
    """LLM's raw text response -> parsed dict with exactly the 3 expected
    keys. Raises LLMMalformedResponseError (not a bare exception) so
    retry_utils' retry_llm_malformed policy can act on it specifically --
    distinct from a real network failure, which gets a different retry
    policy entirely (see retry_utils.py)."""
    cleaned = raw_response.strip()

    # Models sometimes wrap JSON in markdown fences despite instructions
    # telling them not to -- strip defensively rather than trusting the
    # prompt alone to be followed every time.
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise LLMMalformedResponseError(
            f"LLM response was not valid JSON: {e}. Raw (first 200 chars): {raw_response[:200]!r}"
        ) from e

    if not isinstance(data, dict):
        raise LLMMalformedResponseError(f"LLM response was valid JSON but not an object: {type(data).__name__}")

    expected_keys = {"funder_name", "application_url", "eligibility_summary"}
    missing = expected_keys - set(data.keys())
    if missing:
        raise LLMMalformedResponseError(f"LLM response missing expected keys: {missing}")

    return {k: data.get(k) for k in expected_keys}


# ---------------------------------------------------------------------------
# Network-dependent code below. Written against the real OpenAI-compatible
# and Playwright APIs, NOT runnable/tested in this environment.
# ---------------------------------------------------------------------------

def get_openrouter_client():
    from openai import OpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Add it to your .env file "
            "(get a key at https://openrouter.ai/keys)."
        )
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


@retry_llm_malformed
def call_llm(client, grant_name: str, page_text: str, model: str = DEFAULT_MODEL) -> dict:
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=500,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(grant_name, page_text)},
        ],
    )
    raw = response.choices[0].message.content
    return parse_llm_response(raw)


async def fetch_detail_page_text(page: Page, url: str) -> str:
    """Fetches a grant's detail page and strips it to readable text via
    trafilatura -- same "don't hand the LLM raw HTML" principle as Layer 2,
    applied here to control token cost on detail pages too."""
    import trafilatura

    await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    # Same interstitial-redirect risk as the listing page (see extractor.py) --
    # wait for real content before grabbing it.
    try:
        await page.wait_for_selector("article", timeout=15_000)
    except Exception:
        logger.warning(f"No <article> appeared within timeout for detail page {url}")

    html = await page.content()
    return trafilatura.extract(html) or ""


async def enrich_one_grant(page: Page, client, grant_name: str, detail_url: str, model: str = DEFAULT_MODEL) -> Optional[dict]:
    """Full pipeline for one grant: fetch detail page -> strip to text ->
    LLM extraction -> parsed dict. Returns None (never raises) on failure,
    so one bad detail page can't crash a batch enrichment run -- the
    caller just keeps that grant's existing listing-page-only data."""
    try:
        page_text = await fetch_detail_page_text(page, detail_url)
        if not page_text.strip():
            logger.warning(f"No extractable text from detail page {detail_url}")
            return None
        return call_llm(client, grant_name, page_text, model=model)
    except Exception as e:  # noqa: BLE001 - enrichment is best-effort by design
        logger.error(f"LLM enrichment failed for {detail_url}: {e}")
        return None


# ---------------------------------------------------------------------------
# Smoke test — parse_llm_response() and build_user_prompt() only.
# Run directly with `python3 -m accorder.llm_extractor`.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Clean, well-formed response
    clean = '{"funder_name": "Global Climbing Initiative", "application_url": "https://example.org/apply", "eligibility_summary": "Open to registered NGOs."}'
    result = parse_llm_response(clean)
    assert result["funder_name"] == "Global Climbing Initiative"
    assert result["application_url"] == "https://example.org/apply"
    print("Clean JSON response: parsed correctly.")

    # 2. Response wrapped in markdown fences (a common model habit despite instructions)
    fenced = '```json\n{"funder_name": "Test Fund", "application_url": null, "eligibility_summary": null}\n```'
    result = parse_llm_response(fenced)
    assert result["funder_name"] == "Test Fund"
    assert result["application_url"] is None
    print("Markdown-fenced JSON: stripped and parsed correctly.")

    # 3. All-null response (model genuinely couldn't determine anything -- valid, not an error)
    all_null = '{"funder_name": null, "application_url": null, "eligibility_summary": null}'
    result = parse_llm_response(all_null)
    assert result == {"funder_name": None, "application_url": None, "eligibility_summary": None}
    print("All-null response: correctly treated as valid, not an error.")

    # 4. Malformed JSON -- should raise LLMMalformedResponseError, not crash with a raw JSONDecodeError
    try:
        parse_llm_response("this is not json at all")
        raise AssertionError("should have raised LLMMalformedResponseError")
    except LLMMalformedResponseError:
        print("Malformed JSON: correctly raised LLMMalformedResponseError.")

    # 5. Valid JSON, but missing an expected key -- should also raise, not silently return partial data
    incomplete = '{"funder_name": "Test Fund"}'
    try:
        parse_llm_response(incomplete)
        raise AssertionError("should have raised LLMMalformedResponseError for missing keys")
    except LLMMalformedResponseError:
        print("Incomplete JSON (missing keys): correctly raised LLMMalformedResponseError.")

    # 6. Prompt builder truncates long page text rather than sending it all
    long_text = "x" * 10_000
    prompt = build_user_prompt("Test Grant", long_text)
    assert len(prompt) < 7000, "prompt should be truncated, not send the full 10,000 chars"
    print("Long page text: correctly truncated in the prompt.")

    print("\nSmoke test passed (pure logic only — call_llm/fetch_detail_page_text untested here, no network access).")

    # ---------------------------------------------------------------------
    # Part 2: enrich_one_grant() end-to-end, with a FAKE Playwright page and
    # a FAKE LLM client -- proves the full fetch -> clean -> extract -> map
    # flow works, not just the parsing helper tested above.
    # ---------------------------------------------------------------------
    import asyncio

    class _FakeResponse:
        def __init__(self, status=200):
            self.status = status

    class _FakePage:
        """Implements just enough of Playwright's Page interface for
        fetch_detail_page_text() to work against it."""
        def __init__(self, html: str, status: int = 200):
            self._html = html
            self._status = status

        async def goto(self, url, wait_until=None, timeout=None):
            return _FakeResponse(self._status)

        async def wait_for_selector(self, selector, timeout=None):
            return None  # simulate: content already present, nothing to wait through

        async def content(self):
            return self._html

    class _FakeCompletions:
        def __init__(self, responses):
            self._responses = list(responses)
            self._i = 0

        def create(self, **kwargs):
            r = self._responses[self._i]
            self._i += 1
            if isinstance(r, Exception):
                raise r
            msg = type("_M", (), {"content": r})()
            choice = type("_C", (), {"message": msg})()
            return type("_R", (), {"choices": [choice]})()

    class _FakeClient:
        def __init__(self, responses):
            self.chat = type("_Chat", (), {"completions": _FakeCompletions(responses)})()

    async def run_enrich_test():
        fake_html = """
        <html><body><article>
          <h1>Community Benefit Fund: Gambling Research Grant Program</h1>
          <p>Offered by the Australian Institute for Social Research. Apply at
          https://aisr.example.org/apply/gambling-research-2026. Open to registered
          nonprofits and university research groups in Australia.</p>
        </article></body></html>
        """
        llm_output = json.dumps({
            "funder_name": "Australian Institute for Social Research",
            "application_url": "https://aisr.example.org/apply/gambling-research-2026",
            "eligibility_summary": "Open to registered nonprofits and university research groups in Australia.",
        })

        page = _FakePage(fake_html)
        client = _FakeClient([llm_output])
        result = await enrich_one_grant(page, client, "Community Benefit Fund", "https://example.org/detail-page")

        assert result is not None, "expected a result dict, got None"
        assert result["funder_name"] == "Australian Institute for Social Research"
        assert result["application_url"] == "https://aisr.example.org/apply/gambling-research-2026"
        print("enrich_one_grant() (success case): passed")
        print(f"  funder_name: {result['funder_name']}")
        print(f"  application_url: {result['application_url']}")

        # A page fetch that fails entirely should return None, not raise --
        # enrichment is best-effort, one bad detail page can't crash a batch.
        class _AlwaysErrorsPage(_FakePage):
            async def goto(self, url, wait_until=None, timeout=None):
                raise RuntimeError("simulated network failure")

        broken_page = _AlwaysErrorsPage(fake_html)
        result2 = await enrich_one_grant(broken_page, client, "Some Grant", "https://example.org/broken")
        assert result2 is None
        print("enrich_one_grant() (fetch failure, handled gracefully): passed")

    asyncio.run(run_enrich_test())
    print("\nFull enrich_one_grant() smoke test passed (fake page + fake client — real Playwright/OpenRouter untested here).")