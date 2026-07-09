"""
Retry policies for the extraction and AI layers.

The key engineering decision here is that NOT ALL FAILURES ARE THE SAME:

  - A network timeout is almost always worth retrying (transient).
  - A 403/429/CAPTCHA-shaped response means a proxy or session got flagged --
    retrying immediately with the same identity just wastes time and can get
    the whole proxy pool banned. Back off much longer, and rotate identity
    if your proxy provider supports it.
  - A malformed JSON response from the LLM is worth one or two retries
    (sampling noise), but repeated malformed responses mean the prompt or
    schema is the actual problem -- stop retrying and dead-letter it instead
    of burning API calls indefinitely.
  - A Pydantic ValidationError is NEVER worth retrying. The data itself is
    malformed, not the request that fetched it. Retrying just re-fetches
    the same bad data and wastes an LLM call in the process.

Getting this distinction wrong is the most common cost/performance mistake
in scraping pipelines: retrying non-transient failures burns money and time
without ever succeeding, while under-retrying transient ones makes the
pipeline flaky for no reason.
"""

import logging

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

logger = logging.getLogger("grant_tracker.retry")


class TransientScrapeError(Exception):
    """Network timeout, connection reset, 5xx from the target site.
    Worth retrying quickly -- these usually resolve themselves."""


class BlockedError(Exception):
    """403 / 429 / CAPTCHA-shaped response. Needs a much longer backoff,
    and ideally a proxy/session rotation, not an immediate retry."""


class LLMMalformedResponseError(Exception):
    """LLM returned non-JSON or schema-violating output. Worth one or two
    retries for sampling noise -- not indefinite retries."""


# Transient network errors: retry quickly, a handful of times.
retry_transient = retry(
    retry=retry_if_exception_type(TransientScrapeError),
    stop=stop_after_attempt(4),
    wait=wait_exponential_jitter(initial=1, max=20),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)

# Blocked responses: back off aggressively. Hammering a site that just
# flagged you is how you get banned outright, not just delayed.
retry_blocked = retry(
    retry=retry_if_exception_type(BlockedError),
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=30, max=300),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)

# LLM calls: short, cheap retry budget. If it's still malformed after
# 2 retries, the prompt/schema needs fixing, not more attempts.
retry_llm_malformed = retry(
    retry=retry_if_exception_type(LLMMalformedResponseError),
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=2, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


if __name__ == "__main__":
    # Smoke test: prove the transient-error path actually retries and
    # eventually succeeds, and that a non-retryable error passes straight
    # through untouched.

    attempts = {"count": 0}

    @retry_transient
    def flaky_fetch():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise TransientScrapeError(f"simulated timeout on attempt {attempts['count']}")
        return "page content"

    result = flaky_fetch()
    assert result == "page content"
    assert attempts["count"] == 3
    print(f"Transient retry succeeded after {attempts['count']} attempts.")

    @retry_transient
    def always_bad_data():
        raise ValueError("this is a data problem, not a network problem")

    try:
        always_bad_data()
        raise AssertionError("should have raised — ValueError is not retried")
    except ValueError:
        print("Non-retryable error correctly passed through without retrying.")

    print("Smoke test passed.")