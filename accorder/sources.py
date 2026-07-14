"""
Accorder — Source Configuration
================================
Each grant portal has its own DOM shape, deadline text format, and
pagination scheme. A SourceConfig captures exactly that variation so
extractor.py's parsing/fetch logic stays generic. Adding a new source
means adding a new SourceConfig entry here — no changes to extractor.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class SourceConfig:
    name: str
    base_url: str
    domain: str

    # Selectors — primary + fallback, matching the pattern already proven
    # necessary by the post-478631 case (markup varies within one site,
    # let alone across sites).
    article_selector: str
    title_selector: str
    title_selector_fallback: Optional[str]
    content_selector: str
    content_selector_fallback: Optional[str]

    # Compiled regex matching a deadline prefix at the start of the
    # description text, with the date as group(1). None if this source
    # doesn't have a consistent "Deadline: X" prefix — deadline_raw will
    # just be None and Layer 4's fuzzy date parsing gets first crack later.
    deadline_prefix_re: Optional[re.Pattern]

    # Given (base_url, page_num), returns the URL for that listing page.
    # page_num is 1-indexed; callers should skip calling this for page 1
    # if base_url already IS page 1 (matches current scrape_source behavior).
    pagination_url_fn: Callable[[str, int], str]


SOURCES: dict[str, SourceConfig] = {
    "fundsforngos": SourceConfig(
        name="fundsforngos",
        base_url="https://www2.fundsforngos.org/tag/nigeria/",
        domain="www2.fundsforngos.org",
        article_selector="article.post",
        title_selector='[itemprop="headline"] a',
        title_selector_fallback=".entry-title a",
        content_selector='div.entry-content[itemprop="text"] p',
        content_selector_fallback="div.entry-content p",
        deadline_prefix_re=re.compile(r"^\s*Deadline:\s*(\S+)\s*", re.IGNORECASE),
        pagination_url_fn=lambda base, n: f"{base.rstrip('/')}/page/{n}/",
    ),
}