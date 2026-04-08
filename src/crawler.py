"""Polite breadth-first web crawler for the COMP3011 search engine.

The crawler restricts itself to the seed URL's domain, observes a
configurable politeness window between successive HTTP requests,
respects ``robots.txt``, and skips non-HTML responses. URLs are
normalised by stripping the fragment and query string so that pages
which differ only in tracking parameters are not re-fetched.
"""

from __future__ import annotations

import time
import urllib.robotparser
from collections import deque
from typing import Dict, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


class Crawler:
    """Breadth-first crawler that builds a ``{url: html}`` mapping.

    Attributes:
        base_url: The seed URL used to start the crawl. Only URLs in
            the same network location are followed.
        politeness_window: Seconds to sleep after each successful
            request. Defaults to 6 to satisfy the coursework brief.
        visited: URLs that have been enqueued (avoids duplicates).
        pages: ``{url: html}`` for every successfully fetched page.
        max_pages: Optional safety cap on the number of pages fetched
            (``None`` = unlimited). Useful as a defensive bound when
            crawling a site that links to a large number of pages.
    """

    def __init__(
        self,
        base_url: str,
        politeness_window: int = 6,
        max_pages: int | None = None,
        timeout: int = 10,
    ) -> None:
        self.base_url: str = base_url
        self.politeness_window: int = politeness_window
        self.max_pages: int | None = max_pages
        self.timeout: int = timeout
        self.visited: Set[str] = set()
        self.pages: Dict[str, str] = {}
        self.session: requests.Session = requests.Session()
        self.session.headers.update(
            {'User-Agent': 'COMP3011Crawler/1.0'}
        )
        self.rp = urllib.robotparser.RobotFileParser()
        self._load_robots_txt()

    def _load_robots_txt(self) -> None:
        """Fetch and parse ``robots.txt`` for the target site.

        Failure is non-fatal: if the file is unreachable or malformed
        the parser remains permissive and ``_is_allowed`` will treat
        every URL as crawlable.
        """
        robots_url = urljoin(self.base_url, '/robots.txt')
        try:
            self.rp.set_url(robots_url)
            self.rp.read()
        except Exception:
            pass

    def _is_allowed(self, url: str) -> bool:
        """Return True when ``robots.txt`` allows fetching ``url``."""
        try:
            return self.rp.can_fetch('*', url)
        except Exception:
            return True

    def _is_same_domain(self, url: str) -> bool:
        """Return True when ``url`` is on the same netloc as the seed."""
        return urlparse(url).netloc == urlparse(self.base_url).netloc

    def _normalise_url(self, url: str) -> str:
        """Strip fragment and query so equivalent URLs collapse to one."""
        return url.split('#')[0].split('?')[0]

    def crawl(self) -> Dict[str, str]:
        """Run the BFS crawl and return ``{url: html}``.

        Algorithm:
            1. Initialise FIFO queue with the seed URL.
            2. Pop a URL, check ``robots.txt``, GET it.
            3. Skip non-HTML responses.
            4. Store the HTML, parse links, enqueue new same-domain links.
            5. Sleep for ``politeness_window`` before the next request.

        Network and HTTP errors are caught per-page so a single failure
        cannot abort the crawl.

        Returns:
            A dict mapping each successfully fetched URL to its raw
            HTML body.
        """
        queue: deque[str] = deque([self.base_url])
        self.visited.add(self.base_url)

        while queue:
            url = queue.popleft()

            if self.max_pages is not None and len(self.pages) >= self.max_pages:
                break

            if not self._is_allowed(url):
                print(f"Skipping (robots.txt): {url}")
                continue

            try:
                print(f"Crawling: {url}")
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()

                if 'text/html' not in response.headers.get('Content-Type', ''):
                    continue

                self.pages[url] = response.text

                soup = BeautifulSoup(response.text, 'html.parser')
                for tag in soup.find_all('a', href=True):
                    link = urljoin(url, tag['href'])
                    link = self._normalise_url(link)
                    if (link not in self.visited
                            and self._is_same_domain(link)
                            and link.startswith('http')):
                        self.visited.add(link)
                        queue.append(link)

                time.sleep(self.politeness_window)

            except requests.RequestException as e:
                print(f"Error crawling {url}: {e}")

        return self.pages
