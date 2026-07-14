"""
One-off debug script — NOT part of the package, just for diagnosing why
scrape_source() returned 0 candidates against the live site.

Run with: python3 debug_fetch.py
Then check:
  - debug_page.html       -- open in a browser, or grep it for "article"
  - debug_screenshot.png  -- open it. This is the important one: if it shows
    a CAPTCHA, a "checking your browser" page, or a cookie-consent overlay
    covering the content, that tells us immediately this is a bot-detection
    problem, not a selector problem.
"""

import asyncio

from playwright.async_api import async_playwright


async def debug_fetch():
    url = "https://www2.fundsforngos.org/tag/nigeria/"

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            )
        )
        response = await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        print(f"HTTP status: {response.status if response else 'no response'}")

        html = await page.content()
        print(f"HTML length: {len(html)} characters")

        with open("debug_page.html", "w", encoding="utf-8") as f:
            f.write(html)

        await page.screenshot(path="debug_screenshot.png", full_page=True)

        article_count = html.count("<article")
        post_class_count = html.count('class="post-')
        print(f"Raw count of '<article' tags in HTML: {article_count}")
        print(f"Raw count of 'class=\"post-' occurrences: {post_class_count}")

        await browser.close()

    print("\nSaved debug_page.html and debug_screenshot.png -- check both.")


if __name__ == "__main__":
    asyncio.run(debug_fetch())