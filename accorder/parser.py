"""
Accorder — LLM Extraction Layer
===================================
The other half of Layer 3's hybrid extraction: when deterministic selectors
can't find a grant's fields on a page (either a genuinely new site with no
selectors written yet, or -- as with the post-478631 case we hit earlier --
an article whose markup doesn't match our assumptions), fall back to asking
an LLM to read the raw text and extract the same fields.

Uses OpenRouter (OpenAI-API-compatible) via the `openai` client library,
pointed at OpenRouter's base_url. Requires OPENROUTER_API_KEY to be set.

TESTABILITY NOTE: the actual network call is wrapped behind
extract_fields_via_llm(client, ...), which takes an already-constructed
client as an ARGUMENT rather than building one internally. This is what
makes the parsing/validation/retry logic testable with a fake client that
returns canned responses (see the smoke test below) -- no real network
access or API key required to prove that logic is correct. I don't have
network access to openrouter.ai from this environment, so the REAL network
call (get_openrouter_client() actually talking to OpenRouter) is untested
here. Test it for real, with a real key, before trusting it against live
data.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from .retry_utils import LLMMalformedResponseError, retry_llm_malformed

logger = logging.getLogger("accorder.parser")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = os.environ.get("ACCORDER_LLM_MODEL", "anthropic/claude-3.5-sonnet")

SYSTEM_PROMPT = """You extract structured grant/funding information from raw web page text.
Respond with ONLY a JSON object, no other text, no markdown code fences, matching exactly this shape:

{
  "grant_name": string or null,
  "funder_name": string or null,
  "description": string or null,
  "eligibility_summary": string or null,
  "deadline_raw": string or null,
  "min_amount_raw": string or null,
  "max_amount_raw": string or null,
  "currency_raw": string or null,
  "application_url": string or null,
  "contact_email": string or null
}

Rules:
- Use null for any field not clearly present in the text. Never guess or invent a value.
- funder_name is the organization GIVING the grant, not the applicant.
- eligibility_summary should describe WHO can apply (organization type, geography, sector) in 1-2 sentences, if stated.
- application_url should be the actual external link to apply, if the page states one that differs from the page's own URL.
- deadline_raw, min_amount_raw, max_amount_raw should be copied verbatim as they appear in the text, not reformatted or calculated.
- Output nothing except the JSON object -- no preamble, no explanation, no markdown fences.
"""

REQUIRED_KEYS = {
    "grant_name", "funder_name", "description", "eligibility_summary", "deadline_raw",
    "min_amount_raw", "max_amount_raw", "currency_raw",
    "application_url", "contact_email",
}


def get_openrouter_client():
    """Constructs the real OpenRouter client. Deliberately separate from
    extract_fields_via_llm() below, which takes a client as an argument --
    that separation is what makes the extraction logic testable without
    a real client."""
    from openai import OpenAI  # imported lazily: don't require `openai` installed just to run offline tests of the parsing logic

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set — required for the LLM extraction fallback."
        )
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


def _parse_llm_response(raw_content: str) -> dict:
    """Parses and validates the LLM's raw text response. Raises
    LLMMalformedResponseError specifically (not a generic exception) so
    retry_llm_malformed can distinguish 'the model said something we can't
    use' from a network-layer failure, and retry accordingly."""
    cleaned = raw_content.strip()

    # Models sometimes wrap JSON in markdown fences despite instructions
    # not to -- strip defensively rather than treating it as malformed.
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise LLMMalformedResponseError(f"LLM response was not valid JSON: {e}") from e

    if not isinstance(data, dict):
        raise LLMMalformedResponseError(f"LLM response was valid JSON but not an object: {type(data).__name__}")

    missing = REQUIRED_KEYS - data.keys()
    if missing:
        raise LLMMalformedResponseError(f"LLM response missing required keys: {missing}")

    return data


@retry_llm_malformed
def extract_fields_via_llm(client, page_text: str, model: str = DEFAULT_MODEL) -> dict:
    """Calls the LLM and returns a validated dict with all of REQUIRED_KEYS
    present (values may be None). `client` is injected rather than
    constructed here specifically so this function is testable with a fake
    client -- see the smoke test below."""
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": page_text[:8000]},  # defensive cap on input size
        ],
    )
    raw_content = response.choices[0].message.content
    return _parse_llm_response(raw_content)


# ---------------------------------------------------------------------------
# Smoke test — run directly with `python3 -m accorder.parser`
# Uses a FAKE client (no real network, no API key needed) to prove the
# parsing, defensive markdown-fence stripping, and retry behavior all work.
# ---------------------------------------------------------------------------

class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    """Returns each item in `responses` in order, one per call. An item
    that's an Exception instance is raised instead of returned -- lets a
    test simulate a transient failure followed by success."""
    def __init__(self, responses: list):
        self._responses = list(responses)
        self._call_count = 0

    def create(self, **kwargs):
        response = self._responses[self._call_count]
        self._call_count += 1
        if isinstance(response, Exception):
            raise response
        return _FakeResponse(response)


class _FakeClient:
    def __init__(self, responses: list):
        self.chat = type("_FakeChat", (), {"completions": _FakeCompletions(responses)})()


if __name__ == "__main__":
    valid_json = json.dumps({
        "grant_name": "Community Benefit Fund: Gambling Research Grant Program",
        "funder_name": "Example Foundation",
        "description": "Supports community organizations researching the social impact of gambling.",
        "eligibility_summary": "Open to registered nonprofits in Australia.",
        "deadline_raw": "20-Oct-2026",
        "min_amount_raw": None,
        "max_amount_raw": "10,000",
        "currency_raw": "AUD",
        "application_url": "https://example-foundation.org/apply/gambling-research",
        "contact_email": "grants@example-foundation.org",
    })

    # --- Case 1: clean valid JSON on the first try ---
    client = _FakeClient([valid_json])
    result = extract_fields_via_llm(client, "some page text")
    assert result["grant_name"] == "Community Benefit Fund: Gambling Research Grant Program"
    assert result["min_amount_raw"] is None
    print("Case 1 (clean JSON): passed")

    # --- Case 2: response wrapped in markdown fences -- must still parse ---
    fenced = f"```json\n{valid_json}\n```"
    client = _FakeClient([fenced])
    result = extract_fields_via_llm(client, "some page text")
    assert result["funder_name"] == "Example Foundation"
    print("Case 2 (markdown-fenced JSON): passed")

    # --- Case 3: malformed once, then valid -- retry_llm_malformed should recover ---
    client = _FakeClient(["not json at all", valid_json])
    result = extract_fields_via_llm(client, "some page text")
    assert result["currency_raw"] == "AUD"
    assert client.chat.completions._call_count == 2
    print("Case 3 (malformed then valid, retried successfully): passed")

    # --- Case 4: persistently malformed -- should exhaust retries and raise ---
    client = _FakeClient(["still not json", "nope", "definitely not json"])
    try:
        extract_fields_via_llm(client, "some page text")
        raise AssertionError("expected LLMMalformedResponseError to be raised")
    except LLMMalformedResponseError:
        print("Case 4 (persistently malformed, retries exhausted): passed")

    # --- Case 5: valid JSON but missing a required key ---
    incomplete = json.dumps({"grant_name": "Some Grant"})  # missing everything else
    client = _FakeClient([incomplete, incomplete, incomplete])
    try:
        extract_fields_via_llm(client, "some page text")
        raise AssertionError("expected LLMMalformedResponseError for missing keys")
    except LLMMalformedResponseError:
        print("Case 5 (missing required keys): passed")

    print("\nSmoke test passed (offline, fake client — real OpenRouter network call untested here).")