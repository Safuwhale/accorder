"""
Per-domain throttling for the extraction layer.

The failure mode this prevents: it's easy to write a scraper with, say, 20
concurrent workers pulling from a shared page queue -- which is fine if
those 20 pages happen to be spread across 3 domains, but is exactly what
gets you blocked if the queue has 15 pages from the same portal back to
back. Anti-bot systems key on request rate PER DOMAIN, not global request
rate across your whole scraper.

DomainThrottle gives you a global worker pool for throughput, but caps
concurrency per domain AND enforces a minimum delay between requests to any
single domain -- independently of how many domains you're scraping at once.
"""

import asyncio
import time
from collections import defaultdict
from contextlib import asynccontextmanager


class DomainThrottle:
    """Share one instance across all your scraper workers.

    max_concurrent_per_domain: how many in-flight requests to the SAME
        domain are allowed at once. Different domains don't share this cap.
    min_delay_seconds: minimum time between the start of two requests to
        the same domain, enforced even if concurrency allows more.
    """

    def __init__(self, max_concurrent_per_domain: int = 2, min_delay_seconds: float = 1.5):
        self.max_concurrent_per_domain = max_concurrent_per_domain
        self.min_delay_seconds = min_delay_seconds
        self._semaphores: dict[str, asyncio.Semaphore] = defaultdict(
            lambda: asyncio.Semaphore(max_concurrent_per_domain)
        )
        self._last_request_time: dict[str, float] = defaultdict(lambda: 0.0)
        self._pace_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    @asynccontextmanager
    async def throttle(self, domain: str):
        """Usage:
            async with throttle.throttle("example-foundation.org"):
                await fetch_page(url)
        """
        semaphore = self._semaphores[domain]
        await semaphore.acquire()
        try:
            async with self._pace_locks[domain]:
                elapsed = time.monotonic() - self._last_request_time[domain]
                if elapsed < self.min_delay_seconds:
                    await asyncio.sleep(self.min_delay_seconds - elapsed)
                self._last_request_time[domain] = time.monotonic()
            yield
        finally:
            semaphore.release()


if __name__ == "__main__":
    # Smoke test: hit two domains concurrently, prove that:
    #   1. Requests to the SAME domain are spaced out by min_delay_seconds
    #   2. Requests to DIFFERENT domains run concurrently, unaffected by
    #      each other's pacing

    async def main():
        throttle = DomainThrottle(max_concurrent_per_domain=1, min_delay_seconds=0.3)
        timeline: list[tuple[str, float]] = []
        start = time.monotonic()

        async def fake_request(domain: str, label: str):
            async with throttle.throttle(domain):
                timeline.append((f"{domain}:{label}", time.monotonic() - start))

        # 3 requests to domain A (should be spaced >= 0.3s apart)
        # 3 requests to domain B (should be spaced >= 0.3s apart, but
        # running concurrently with domain A's requests, not queued behind them)
        await asyncio.gather(
            fake_request("a.org", "1"), fake_request("a.org", "2"), fake_request("a.org", "3"),
            fake_request("b.org", "1"), fake_request("b.org", "2"), fake_request("b.org", "3"),
        )

        timeline.sort(key=lambda t: t[1])
        for label, t in timeline:
            print(f"{t:.3f}s  {label}")

        a_times = sorted(t for label, t in timeline if label.startswith("a.org"))
        b_times = sorted(t for label, t in timeline if label.startswith("b.org"))

        for i in range(1, len(a_times)):
            gap = a_times[i] - a_times[i - 1]
            assert gap >= 0.29, f"domain a.org requests too close together: {gap:.3f}s"
        for i in range(1, len(b_times)):
            gap = b_times[i] - b_times[i - 1]
            assert gap >= 0.29, f"domain b.org requests too close together: {gap:.3f}s"

        # Total wall time should be ~0.9s (3 sequential requests per domain,
        # both domains running in parallel) -- NOT ~1.8s, which is what
        # you'd get if domains were incorrectly serialized against each other.
        total_time = max(t for _, t in timeline)
        assert total_time < 1.2, f"domains appear to be serialized against each other: {total_time:.3f}s total"

        print(f"\nTotal wall time: {total_time:.3f}s (two domains ran concurrently, each internally paced)")
        print("Smoke test passed.")

    asyncio.run(main())